"""GatedDeltaNet layer module (hybrid linear-attention layer, e.g. Qwen3.5).

The recurrent state math and kernel dispatch live in kernels.py; this file
contains only the nn.Module wrapper, its projections and speculation state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init

from eole.modules.gdn.gdn_ab import gdn_ab_projections
from eole.modules.gdn.kernels import (
    RMSNormGated,
    _causal_conv1d_dispatch,
    _causal_conv1d_update_dispatch,
    _chunk_gated_delta_rule_dispatch,
    _recurrent_gated_delta_rule_dispatch,
    _torch_speculative_conv1d,
)


class GatedDeltaNet(nn.Module):
    """Gated Delta Network – linear recurrent attention layer.

    Used as the "linear_attention" layer type in hybrid models such as Qwen3.5.

    Weight naming follows the HuggingFace convention exactly so that the HF
    converter can map weights without renaming:

        in_proj_qkv  – joint QKV projection (key_dim*2 + value_dim outputs)
        in_proj_z    – gate projection for the output norm  (value_dim outputs)
        in_proj_b    – beta projection                       (num_v_heads outputs)
        in_proj_a    – alpha projection for A_log decay      (num_v_heads outputs)
        conv1d       – causal depthwise conv over QKV stream
        dt_bias      – bias for dt (time-step)               (num_v_heads,)
        A_log        – log of the decay factor               (num_v_heads,)
        norm         – RMSNormGated applied to value output
        out_proj     – final linear projection               (value_dim → hidden_size)
    """

    def __init__(self, decoder_config, layer_idx: int):
        super().__init__()
        hidden_size = decoder_config.hidden_size
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.num_v_heads = decoder_config.linear_num_value_heads
        self.num_k_heads = decoder_config.linear_num_key_heads
        self.head_k_dim = decoder_config.linear_key_head_dim
        self.head_v_dim = decoder_config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = decoder_config.linear_conv_kernel_dim

        # The conv1d operates on the concatenated QKV stream
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.fused_qkvz = False
        self.in_proj_qkv = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        self.in_proj_z = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.value_dim,
            bias=False,
        )
        self.in_proj_b = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.num_v_heads,
            bias=False,
        )
        self.in_proj_a = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.num_v_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(self.head_v_dim, eps=decoder_config.norm_eps)
        self.out_proj = skip_init(
            nn.Linear,
            in_features=self.value_dim,
            out_features=hidden_size,
            bias=False,
        )

        # Select per invocation so CPU/MPS remain usable with optional GPU libraries installed.
        self._causal_conv1d_fn = _causal_conv1d_dispatch
        self._causal_conv1d_update = _causal_conv1d_update_dispatch
        self._chunk_gated_delta_rule = _chunk_gated_delta_rule_dispatch
        self._recurrent_gated_delta_rule = _recurrent_gated_delta_rule_dispatch

        # Inference state (set by TransformerDecoder._init_cache / _disable_cache)
        self.conv_state = None
        self.recurrent_state = None
        # Transaction used by speculative verification. The persistent state is
        # left untouched until the accepted input prefix is known.
        self._speculating = False
        self._spec_buffers = None
        self._control_gemv_enabled = False
        self.register_buffer("_ab_pair_weight", None, persistent=False)

    def begin_speculation(self, seq_len):
        if self.conv_state is None or self.recurrent_state is None:
            raise RuntimeError("GatedDeltaNet speculation requires initialized decode state")
        batch_size = self.recurrent_state.size(0)
        device = self.recurrent_state.device
        state_dtype = self.recurrent_state.dtype
        if (
            self._spec_buffers is None
            or self._spec_buffers["query"].shape[:2] != (batch_size, seq_len)
            or self._spec_buffers["query"].dtype != state_dtype
            or self._spec_buffers["query"].device != device
            or self._spec_buffers["conv_extended"].dtype != self.conv_state.dtype
            or self._spec_buffers["conv_extended"].device != self.conv_state.device
        ):
            self._spec_buffers = {
                "query": torch.empty(
                    batch_size, seq_len, self.num_v_heads, self.head_k_dim, dtype=state_dtype, device=device
                ),
                "key": torch.empty(
                    batch_size, seq_len, self.num_v_heads, self.head_k_dim, dtype=state_dtype, device=device
                ),
                "value": torch.empty(
                    batch_size, seq_len, self.num_v_heads, self.head_v_dim, dtype=state_dtype, device=device
                ),
                "g": torch.empty(batch_size, seq_len, self.num_v_heads, dtype=torch.float32, device=device),
                "beta": torch.empty(batch_size, seq_len, self.num_v_heads, dtype=state_dtype, device=device),
                "conv_extended": torch.empty(
                    batch_size,
                    self.conv_dim,
                    self.conv_kernel_size + seq_len,
                    dtype=self.conv_state.dtype,
                    device=self.conv_state.device,
                ),
                "final_recurrent_state": torch.empty_like(self.recurrent_state),
            }
        # The legacy four-GEMV fallback was tuned for SM120 only. The paired
        # Triton path resolves its broader hardware eligibility separately in
        # prepare_ab_projections(), outside compilation and graph capture.
        self._control_gemv_enabled = (
            seq_len == 4
            and batch_size == 1
            and device.type == "cuda"
            and torch.cuda.get_device_capability(device) == (12, 0)
            and self.hidden_size == 5120
            and self.num_v_heads == 48
            and isinstance(self.in_proj_a, nn.Linear)
            and isinstance(self.in_proj_b, nn.Linear)
            and self.in_proj_a.weight.shape == (48, 5120)
            and self.in_proj_b.weight.shape == (48, 5120)
        )
        self._speculating = True

    def end_speculation(self):
        self._speculating = False

    def discard_speculation(self):
        self._speculating = False

    def commit_speculation(self, n_kept):
        """Commit the first n inputs consumed by the last speculative pass."""
        spec = self._spec_buffers
        if spec is None:
            raise RuntimeError("GatedDeltaNet has no speculative buffers to commit")
        seq_len = spec["query"].size(1)
        if not 1 <= n_kept <= seq_len:
            raise ValueError(f"n_kept must be in [1, {seq_len}], got {n_kept}")

        if n_kept == seq_len:
            recurrent_state = spec["final_recurrent_state"]
        else:
            # Re-run only this linear-attention layer's short accepted prefix
            # through the fused recurrent kernel. This avoids another full
            # decoder pass and avoids materializing a recurrent-state snapshot
            # for every speculative token.
            _, recurrent_state = self._recurrent_gated_delta_rule(
                spec["query"][:, :n_kept],
                spec["key"][:, :n_kept],
                spec["value"][:, :n_kept],
                spec["g"][:, :n_kept],
                spec["beta"][:, :n_kept],
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        kernel = self.conv_kernel_size
        self.conv_state.copy_(spec["conv_extended"][:, :, n_kept : n_kept + kernel])
        self.recurrent_state.copy_(recurrent_state.to(self.recurrent_state.dtype))

    @staticmethod
    @torch.no_grad()
    def commit_speculation_group(layers, n_kept):
        """Commit a verified prefix for several GDN layers with one replay.

        Partial acceptance needs the recurrent state at an intermediate token
        boundary. The FLA recurrent operator returns only the final state, so
        replay that short accepted prefix. Running one operator per layer
        caused a long chain of tiny launches for hybrid models; concatenate
        same-shaped layer states along the batch dimension and replay them in
        one call instead.
        """
        if not layers:
            return
        if isinstance(n_kept, (list, tuple)):
            if len(set(n_kept)) == 1 and n_kept[0] > 0:
                n_kept = n_kept[0]
            else:
                return GatedDeltaNet._commit_speculation_rows(layers, n_kept)

        specs = [layer._spec_buffers for layer in layers]
        seq_len = specs[0]["query"].size(1)
        if not 1 <= n_kept <= seq_len:
            raise ValueError(f"n_kept must be in [1, {seq_len}], got {n_kept}")

        # Fall back for uncommon heterogeneous GDN layouts. Qwen3.5 layers
        # share shapes, so their replay can be batched as independent rows.
        query_shape = specs[0]["query"].shape[1:]
        can_batch = all(
            all(
                spec[name].shape == specs[0][name].shape
                and spec[name].dtype == specs[0][name].dtype
                and spec[name].device == specs[0][name].device
                for name in ("query", "key", "value", "g", "beta")
            )
            and spec["query"].shape[1:] == query_shape
            and layer.recurrent_state.shape == layers[0].recurrent_state.shape
            and layer.recurrent_state.device == layers[0].recurrent_state.device
            and layer.recurrent_state.dtype == layers[0].recurrent_state.dtype
            for layer, spec in zip(layers, specs)
        )
        if not can_batch:
            for layer in layers:
                layer.commit_speculation(n_kept)
            return

        if n_kept == seq_len:
            # The verifier already computed these states in the persistent
            # dtype. Copy them directly instead of packing a large temporary
            # only to split it into the original per-layer shapes again.
            torch._foreach_copy_(
                [layer.conv_state for layer in layers],
                [
                    spec["conv_extended"][:, :, n_kept : n_kept + layer.conv_kernel_size]
                    for layer, spec in zip(layers, specs)
                ],
            )
            torch._foreach_copy_(
                [layer.recurrent_state for layer in layers],
                [spec["final_recurrent_state"] for spec in specs],
            )
            return
        else:
            # Reuse packing storage across cycles. Accepted prefix lengths
            # vary, but the allocated verifier span stays fixed; a compact
            # view provides the current replay shape without new
            # large allocations on every rejection.
            owner = layers[0]
            signature = tuple(
                (name, tuple(specs[0][name].shape), specs[0][name].dtype, specs[0][name].device)
                for name in ("query", "key", "value", "g", "beta")
            )
            signature += (
                (
                    len(layers),
                    tuple(owner.recurrent_state.shape),
                    owner.recurrent_state.dtype,
                    owner.recurrent_state.device,
                ),
            )
            if getattr(owner, "_spec_replay_signature", None) != signature:
                owner._spec_replay_buffers = {
                    name: torch.empty(
                        (len(layers) * tensor.size(0), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
                    )
                    for name, tensor in specs[0].items()
                    if name in ("query", "key", "value", "g", "beta")
                }
                owner._spec_replay_buffers["state"] = torch.empty(
                    (len(layers) * owner.recurrent_state.size(0), *owner.recurrent_state.shape[1:]),
                    dtype=owner.recurrent_state.dtype,
                    device=owner.recurrent_state.device,
                )
                owner._spec_replay_signature = signature
            packed = owner._spec_replay_buffers
            replay_inputs = {}
            for name in ("query", "key", "value", "g", "beta"):
                buffer = packed[name]
                rows = buffer.size(0)
                # Preserve contiguous inputs for fused kernels that flatten
                # batch and sequence without accepting arbitrary strides.
                compact = buffer.view(-1, *buffer.shape[2:])[: rows * n_kept].view(rows, n_kept, *buffer.shape[2:])
                torch.cat([spec[name][:, :n_kept] for spec in specs], dim=0, out=compact)
                replay_inputs[name] = compact
            torch.cat([layer.recurrent_state for layer in layers], dim=0, out=packed["state"])
            query, key, value, g, beta = (replay_inputs[name] for name in ("query", "key", "value", "g", "beta"))
            initial_state = packed["state"]
            _, recurrent_states = layers[0]._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if recurrent_states is None:
                raise RuntimeError("GatedDeltaNet speculative replay did not return its recurrent state")

        batch_size = layers[0].recurrent_state.size(0)
        conv_sources = []
        recurrent_sources = []
        for index, (layer, spec) in enumerate(zip(layers, specs)):
            kernel = layer.conv_kernel_size
            conv_sources.append(spec["conv_extended"][:, :, n_kept : n_kept + kernel])
            start = index * batch_size
            recurrent_sources.append(recurrent_states[start : start + batch_size])

        # One cast plus foreach copies avoids a separate conversion/copy launch
        # for every recurrent layer after the replay.
        recurrent_dtype = layers[0].recurrent_state.dtype
        if recurrent_states.dtype != recurrent_dtype:
            recurrent_states = recurrent_states.to(recurrent_dtype)
            recurrent_sources = [recurrent_states[i * batch_size : (i + 1) * batch_size] for i in range(len(layers))]
        torch._foreach_copy_([layer.conv_state for layer in layers], conv_sources)
        torch._foreach_copy_([layer.recurrent_state for layer in layers], recurrent_sources)

    @staticmethod
    @torch.no_grad()
    def _commit_speculation_rows(layers, counts):
        """Replay different accepted prefixes without a decoder pass or state history.

        After a row's prefix, g=0 and beta=0 make recurrence updates identities.
        Thus a fixed-width replay can commit heterogeneous row lengths in one
        FLA call, with no per-token recurrent-state snapshots.
        """
        specs = [layer._spec_buffers for layer in layers]
        batch, span = specs[0]["query"].shape[:2]
        if len(counts) != batch or any(not isinstance(n, int) or not 0 <= n <= span for n in counts):
            raise ValueError("One accepted prefix in [0, verifier span] is required per cache row")
        if not any(counts):
            return
        if any(
            layer.recurrent_state.shape != layers[0].recurrent_state.shape
            or layer.recurrent_state.dtype != layers[0].recurrent_state.dtype
            or layer.recurrent_state.device != layers[0].recurrent_state.device
            or any(
                spec[name].shape != specs[0][name].shape
                or spec[name].dtype != specs[0][name].dtype
                or spec[name].device != specs[0][name].device
                for name in ("query", "key", "value", "g", "beta")
            )
            for layer, spec in zip(layers, specs)
        ):
            for layer in layers:
                GatedDeltaNet._commit_speculation_rows([layer], counts)
            return
        lengths = torch.tensor(counts, device=layers[0].recurrent_state.device, dtype=torch.long)
        recurrent = None
        if any(0 < n < span for n in counts):
            owner = layers[0]
            signature = (
                len(layers),
                tuple(owner.recurrent_state.shape),
                owner.recurrent_state.dtype,
                owner.recurrent_state.device,
                span,
            )
            if getattr(owner, "_row_commit_signature", None) != signature:
                owner._row_commit_buffers = {
                    name: torch.empty((len(layers) * batch, *value.shape[1:]), dtype=value.dtype, device=value.device)
                    for name, value in specs[0].items()
                    if name in ("query", "key", "value", "g", "beta")
                }
                owner._row_commit_buffers["state"] = torch.empty(
                    (len(layers) * batch, *owner.recurrent_state.shape[1:]),
                    dtype=owner.recurrent_state.dtype,
                    device=owner.recurrent_state.device,
                )
                owner._row_commit_signature = signature
            packed = owner._row_commit_buffers
            for name in ("query", "key", "value", "g", "beta"):
                torch.cat([spec[name] for spec in specs], dim=0, out=packed[name])
            torch.cat([layer.recurrent_state for layer in layers], dim=0, out=packed["state"])
            mask = torch.arange(span, device=lengths.device)[None, :, None] < lengths.repeat(len(layers))[:, None, None]
            _, recurrent = owner._recurrent_gated_delta_rule(
                packed["query"],
                packed["key"],
                packed["value"],
                torch.where(mask, packed["g"], 0.0),
                torch.where(mask, packed["beta"], 0.0),
                initial_state=packed["state"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            recurrent = recurrent.to(owner.recurrent_state.dtype)
        conv_sources, recurrent_sources = [], []
        for index, (layer, spec) in enumerate(zip(layers, specs)):
            indices = lengths[:, None, None] + torch.arange(layer.conv_kernel_size, device=lengths.device)
            conv_sources.append(spec["conv_extended"].gather(2, indices.expand(-1, layer.conv_dim, -1)))
            partial = layer.recurrent_state if recurrent is None else recurrent[index * batch : (index + 1) * batch]
            source = torch.where(lengths[:, None, None, None].eq(span), spec["final_recurrent_state"], partial)
            source = torch.where(lengths[:, None, None, None].eq(0), layer.recurrent_state, source)
            recurrent_sources.append(source)
        torch._foreach_copy_([layer.conv_state for layer in layers], conv_sources)
        torch._foreach_copy_([layer.recurrent_state for layer in layers], recurrent_sources)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fuse_QKVZ(self):
        """Pack compatible Marlin QKV and output-gate weights before repacking."""
        if self.fused_qkvz or getattr(self, "_qkvz_fusion_skipped", False):
            return
        from eole.modules.quant.autoround_linear import fuse_autoround_projections

        handled, fused = fuse_autoround_projections([self.in_proj_qkv, self.in_proj_z], name="GDN QKV/Z")
        if handled:
            if fused is None:
                self._qkvz_fusion_skipped = True
                return
            self.in_proj_qkvz = fused
            del self.in_proj_qkv, self.in_proj_z
            self.fused_qkvz = True

    def _project_qkvz(self, hidden_states):
        if self.fused_qkvz:
            qkvz = self.in_proj_qkvz(hidden_states)
            mixed_qkv, z = qkvz.split((self.conv_dim, self.value_dim), dim=-1)
        else:
            mixed_qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)
        return mixed_qkv.transpose(1, 2), z

    def prepare_ab_projections(self):
        """Pack BF16 A/B projections for NVIDIA Ampere+; other cases keep the existing path."""
        import importlib.util

        self._ab_pair_weight = None
        if type(self.in_proj_a) is not nn.Linear or type(self.in_proj_b) is not nn.Linear:
            return
        a, b = self.in_proj_a, self.in_proj_b
        if (
            not self.training
            and a.weight.is_cuda
            and torch.version.cuda is not None
            # BF16 tensor-core MMA is available on NVIDIA Ampere and newer.
            # Triton lowers the same split-K algorithm for the target GPU.
            and torch.cuda.get_device_capability(a.weight.device)[0] >= 8
            and a.weight.shape == b.weight.shape
            and a.weight.shape in ((48, 5120), (32, 2048))
            and a.weight.dtype == b.weight.dtype == torch.bfloat16
            and a.weight.device == b.weight.device
            and a.bias is None
            and b.bias is None
            and importlib.util.find_spec("triton") is not None
        ):
            with torch.no_grad():
                self._ab_pair_weight = torch.cat((b.weight, a.weight), dim=0)

    def _should_split_control_rows(self, hidden_states):
        return (
            self._speculating
            and self._control_gemv_enabled
            and not self.training
            and hidden_states.shape[0] == 1
            and hidden_states.shape[1] == 4
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and isinstance(self.in_proj_b, nn.Linear)
            and isinstance(self.in_proj_a, nn.Linear)
        )

    def _project_control(self, projection, hidden_states, split_rows=False):
        if split_rows:
            # Use four GEMVs instead of one skinny GEMM. Inductor incorporates
            # the stack into subsequent pointwise work in the verifier graph.
            return torch.stack([projection(hidden_states[:, i, :]) for i in range(4)], dim=1)
        return projection(hidden_states)

    def _should_pair_ab_projections(self, hidden_states):
        return (
            self._ab_pair_weight is not None
            and not self.training
            and hidden_states.ndim == 3
            and hidden_states.shape[0] == 1
            and hidden_states.shape[2] == self._ab_pair_weight.shape[1]
            and (hidden_states.shape[1] == 1 or (self._speculating and hidden_states.shape[1] == 4))
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_cuda
            and hidden_states.is_contiguous()
        )

    def _project_ab(self, hidden_states):
        """Return in_proj_b and in_proj_a outputs, in that order."""
        if self._should_pair_ab_projections(hidden_states):
            projections = gdn_ab_projections(hidden_states, self._ab_pair_weight)
            b, a = projections.split(self.num_v_heads, dim=-1)
        else:
            split_control_rows = self._should_split_control_rows(hidden_states)
            b = self._project_control(self.in_proj_b, hidden_states, split_control_rows)
            a = self._project_control(self.in_proj_a, hidden_states, split_control_rows)
        return b, a

    def forward(self, hidden_states, attn_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Mask padding to zero so padding tokens don't corrupt conv/recurrent state
        if attn_mask is not None and attn_mask.shape[1] > 1:
            hidden_states = (hidden_states * attn_mask[:, :, None]).to(hidden_states.dtype)

        use_precomputed = self.conv_state is not None and self.recurrent_state is not None and seq_len == 1
        use_speculative = self._speculating and seq_len > 1

        mixed_qkv, z = self._project_qkvz(hidden_states)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b, a = self._project_ab(hidden_states)

        if use_precomputed:
            mixed_qkv = self._causal_conv1d_update(
                mixed_qkv,
                self.conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                "silu",
            )
        elif use_speculative:
            # Extend the cached raw QKV history and compute exactly the same
            # causal-convolution outputs as sequential decode. The extra first
            # valid convolution output belongs to the cache-only history.
            conv_extended = torch.cat([self.conv_state, mixed_qkv.to(self.conv_state.dtype)], dim=-1)
            if torch.compiler.is_compiling():
                mixed_qkv = _torch_speculative_conv1d(conv_extended, self.conv1d.weight, self.conv1d.bias).to(
                    hidden_states.dtype
                )
            else:
                conv_out = F.conv1d(
                    conv_extended,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    groups=self.conv_dim,
                )
                mixed_qkv = F.silu(conv_out[:, :, 1:]).to(hidden_states.dtype)
        else:
            # Cached prefill can arrive in several chunks. Include the raw
            # convolution history, just as single-token updates do. Initial
            # cache state is zero, so the first chunk uses the same padding.
            history = self.conv_kernel_size if self.conv_state is not None else 0
            if history:
                mixed_qkv = torch.cat((self.conv_state, mixed_qkv.to(self.conv_state.dtype)), dim=-1)
                self.conv_state.copy_(mixed_qkv[:, :, -history:])
            mixed_qkv = self._causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
            )
            if history:
                mixed_qkv = mixed_qkv[:, :, history:]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, S, conv_dim)

        # Split into Q, K, V streams
        query = mixed_qkv[:, :, : self.key_dim]  # (B, S, key_dim)
        key = mixed_qkv[:, :, self.key_dim : self.key_dim * 2]  # (B, S, key_dim)
        value = mixed_qkv[:, :, self.key_dim * 2 :]  # (B, S, value_dim)

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Expand K/Q heads to match V heads (GQA-style for GDA layers, mirrors HF implementation)
        if self.num_v_heads // self.num_k_heads > 1:
            n_rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(n_rep, dim=2)
            key = key.repeat_interleave(n_rep, dim=2)

        # Discretise decay / gate
        # dt = F.softplus(a + self.dt_bias)  # (B, S, num_v_heads)
        # A = -torch.exp(self.A_log.float())  # (num_v_heads,)
        # g = dt * A.view(1, 1, -1)  # (B, S, num_v_heads)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        beta = torch.sigmoid(b)  # (B, S, num_v_heads)

        # At padding positions, q/k/v are already zeroed (hidden_states was zeroed above).
        # But g and beta are still non-trivial: g = softplus(dt_bias)*A (decay < 1) and
        # beta = sigmoid(0) = 0.5.  This would decay the recurrent state at padding positions
        # even though no content is written, corrupting the state for shorter sequences when
        # batch_size > 1.  Fix: force g=0 (exp(0)=1, identity decay) and beta=0 (no write)
        # at padding positions so they are true no-ops for the recurrent state.
        if attn_mask is not None and attn_mask.shape[1] > 1:
            pad_mask = ~attn_mask  # (B, S), True = padding position
            g = g.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            beta = beta.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Compute linear attention
        if use_precomputed:
            output, new_recurrent_state = self._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if new_recurrent_state is not None:
                self.recurrent_state.copy_(new_recurrent_state.to(self.recurrent_state.dtype))
        elif use_speculative:
            output, new_recurrent_state = self._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if new_recurrent_state is None:
                raise RuntimeError("GatedDeltaNet speculative forward did not return its recurrent state")
            buffers = self._spec_buffers
            buffers["query"].copy_(query)
            buffers["key"].copy_(key)
            buffers["value"].copy_(value)
            buffers["g"].copy_(g)
            buffers["beta"].copy_(beta)
            buffers["conv_extended"].copy_(conv_extended)
            buffers["final_recurrent_state"].copy_(new_recurrent_state)
        else:
            output, new_recurrent_state = self._chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=(self.recurrent_state is not None),
                use_qk_l2norm_in_kernel=True,
            )
            if new_recurrent_state is not None and self.recurrent_state is not None:
                self.recurrent_state.copy_(new_recurrent_state.to(self.recurrent_state.dtype))

        # output: (B, S, num_v_heads, head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)
        return output
