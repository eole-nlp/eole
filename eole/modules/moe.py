"""MoE mixture of experts"."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from eole.modules.transformer_mlp import MLP
from torch.distributed import all_reduce

try:
    from eole.triton.fused_moe import fused_experts_impl
    from eole.triton.fused_moe_int4 import fused_experts_int4_impl
    from eole.modules.fused_moe_marlin import fused_experts_marlin_impl

    _triton_moe = True
except ImportError:
    _triton_moe = False

from eole.modules.moe_quant_utils import (
    detect_expert_quant_type,
    stack_gptq_moe_weights,
    stack_awq_moe_weights,
    stack_marlin_moe_weights,
)


def naive_moe(x, topk_weights, topk_ids, K, experts):
    """
    Original EOLE Code
    x: (N = BT*K, hidden)  tokens in flat ordering: token0:k0, token0:k1, token1:k0, ...
    topk_weights: (BT, K)
    topk_ids: (BT, K)
    K: num_experts_per_token
    experts: nn.ModuleList
    """
    flat_topk_ids = topk_ids.view(-1)  # (BT*K)
    y = torch.empty_like(x)
    for i, expert in enumerate(experts):
        if torch.any(flat_topk_ids == i):
            y[flat_topk_ids == i] = expert(x[flat_topk_ids == i].unsqueeze(0)).to(dtype=y.dtype)
    y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
    return y


def vectorized_moe(x, topk_weights, topk_ids, K, experts):
    """
    Vectorized version x1.6 faster
    x: (N = BT*K, hidden)  tokens in flat ordering: token0:k0, token0:k1, token1:k0, ...
    topk_weights: (BT, K)
    topk_ids: (BT, K)
    K: num_experts_per_token
    experts: nn.ModuleList
    """
    flat_topk_ids = topk_ids.view(-1)  # (BT*K)
    BTK = flat_topk_ids.size(0)
    hidden = x.size(1)
    if BTK == 0:
        return x.new_empty((0, hidden))

    sort_order = torch.argsort(flat_topk_ids)
    sorted_experts = flat_topk_ids[sort_order].tolist()
    run_starts = [0] + [i for i in range(1, BTK) if sorted_experts[i] != sorted_experts[i - 1]]
    run_ends = run_starts[1:] + [BTK]

    y_flat = torch.empty_like(x)  # (BT*K, hidden)

    for s, e in zip(run_starts, run_ends):
        expert_id = sorted_experts[s]
        routed_positions = sort_order[s:e]
        y_flat[routed_positions] = experts[expert_id](x[routed_positions])

    y = y_flat.view(BTK // K, K, hidden)
    y = (y * topk_weights.unsqueeze(-1)).sum(dim=1)
    return y


class MoE(nn.Module):
    def __init__(
        self,
        model_config,
        running_config,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                MLP(
                    model_config,
                    running_config,
                    moe_transformer_ff=model_config.moe_transformer_ff,
                )
                for i in range(model_config.num_experts)
            ]
        )
        self.gate = nn.Linear(model_config.hidden_size, model_config.num_experts, bias=False)
        self.num_experts_per_tok = model_config.num_experts_per_tok
        self.parallel_gpu = running_config.parallel_gpu
        if model_config.num_shared_experts > 0:
            self.shared_experts = MLP(
                model_config,
                running_config=running_config,
                moe_transformer_ff=model_config.moe_transformer_ff * model_config.num_shared_experts,
            )
            if model_config.shared_expert_gate:
                self.shared_expert_gate = nn.Linear(model_config.hidden_size, 1, bias=False)
            else:
                self.shared_expert_gate = None
        else:
            self.shared_experts = None
            self.shared_expert_gate = None
        self.moe_softmax_after = model_config.moe_softmax_after
        self.moe_renormalize = model_config.moe_renormalize
        self._gates_fused = False
        self._grouped = GroupedExperts(self.experts)

        self.activation_function = next(
            (a for a in ("gelu", "relu", "silu") if a in model_config.mlp_activation_fn), None
        )

        # fp16 fast-path cache (plain nn.Linear)
        self._w1 = None
        self._w2 = None

        # int4 fast-path cache (GPTQ / AWQ)
        self._w1_qweight = None
        self._w1_scales = None
        self._w1_qzeros = None
        self._w2_qweight = None
        self._w2_scales = None
        self._w2_qzeros = None
        self._group_size = None
        self._kpacked = None  # True = GPTQ, False = AWQ

        self._quant_weights_initialized = False

        # ── int4 intermediate buffer cache (grown lazily) ─────────────────────
        # Eliminates 5 GPU allocations per MoE layer per decode step:
        #   intermediate  (num_pairs, I)
        #   final_output  (M, H)
        #   expert_ids    (num_pairs,) int32
        #   token_ids     (num_pairs,) int32
        #   weights       (num_pairs,) dtype
        self._int4_intermediate_buf = None
        self._int4_output_buf = None
        self._int4_expert_ids_buf = None
        self._int4_token_ids_buf = None
        self._int4_weights_buf = None
        self._int4_buf_num_pairs = 0

        # ── Marlin MoE weight cache ───────────────────────────────────────────
        self._w1_marlin = None  # (E, K//16, 4*I) int32
        self._w1_marlin_scales = None  # (E, K//gs, 2*I) fp16
        self._w1_marlin_qzeros = None  # (E, 0) – empty for sym quant
        self._w1_marlin_g_idx = None  # (E, 0) – empty
        self._w1_marlin_perm = None  # (E, 0) – empty
        self._w2_marlin = None  # (E, I//16, 2*K) int32
        self._w2_marlin_scales = None  # (E, I//gs, K) fp16
        self._w2_marlin_qzeros = None  # (E, 0) – empty
        self._w2_marlin_g_idx = None  # (E, 0) – empty
        self._w2_marlin_perm = None  # (E, 0) – empty
        self._marlin_workspace = None  # shared Marlin workspace tensor
        self._marlin_scalar_type_id = None  # int, e.g. uint4b8.id
        self._marlin_num_experts = None  # int
        self._marlin_I = None  # per-stream intermediate size
        self._marlin_K = None  # model hidden size

        # ── Marlin intermediate buffer cache (grown lazily) ───────────────────
        # cache13: flat buffer shared by W1 output (M*topk, 2*I) and
        #          W2 output (M*topk, K) via non-overlapping views.
        # cache2:  activation output (M*topk, I).
        # out_buf: final reduced output (BT, K_mar).
        self._marlin_cache13 = None  # flat buffer: cap ≥ M*topk*max(2*I, K)
        self._marlin_cache2 = None  # flat buffer: cap ≥ M*topk*I
        self._marlin_out_buf = None  # (BT, K_mar) output buffer
        self._marlin_buf_M_topk = 0  # M*topk capacity of current caches

        self._marlin_topk_ids_i32 = None  # (BT, topk) int32 – avoids .to(int32) alloc
        self._marlin_routing_cap = 0  # capacity of topk_ids_i32 buffer (BT*topk)

    # ── Buffer growth helpers ─────────────────────────────────────────────────

    def _maybe_grow_int4_buffers(self, M: int, K: int, I: int, H: int, dev, dt):  # noqa: E741
        """Grow int4 intermediate buffers if the current batch exceeds capacity."""
        num_pairs = M * K
        if (
            self._int4_buf_num_pairs < num_pairs
            or self._int4_intermediate_buf is None
            or self._int4_intermediate_buf.dtype != dt
        ):
            self._int4_intermediate_buf = torch.empty(num_pairs * I, device=dev, dtype=dt)
            # output_buf is zeroed in-place before each use (W2 uses atomic_add).
            # Allocate with empty rather than zeros; zero_ is called per-call.
            self._int4_output_buf = torch.empty(M, H, device=dev, dtype=dt)
            self._int4_expert_ids_buf = torch.empty(num_pairs, dtype=torch.int32, device=dev)
            self._int4_token_ids_buf = torch.empty(num_pairs, dtype=torch.int32, device=dev)
            self._int4_weights_buf = torch.empty(num_pairs, device=dev, dtype=dt)
            self._int4_buf_num_pairs = num_pairs

    def _maybe_grow_marlin_buffers(self, BT: int, topk_val: int, dev, dt):
        """Grow intermediate and routing buffers if the current batch exceeds capacity."""
        M_topk = BT * topk_val
        I_mar = self._marlin_I
        K_mar = self._marlin_K

        # ── Intermediate GEMM buffers ─────────────────────────────────────────
        if self._marlin_buf_M_topk < M_topk or self._marlin_cache13 is None or self._marlin_cache13.dtype != dt:
            self._marlin_cache13 = torch.empty(M_topk * max(2 * I_mar, K_mar), device=dev, dtype=dt)
            self._marlin_cache2 = torch.empty(M_topk * I_mar, device=dev, dtype=dt)
            self._marlin_out_buf = torch.empty(BT, K_mar, device=dev, dtype=dt)
            self._marlin_buf_M_topk = M_topk

        # ── topk_ids int32 buffer ─────────────────────────────────────────────
        if self._marlin_routing_cap < BT * topk_val or self._marlin_topk_ids_i32 is None:
            self._marlin_topk_ids_i32 = torch.empty(BT, topk_val, dtype=torch.int32, device=dev)
            self._marlin_routing_cap = BT * topk_val

    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_fuse_gates(self):
        """Fuse all expert gates, executed only once."""
        if self._gates_fused:
            return

        for expert in self.experts:
            if hasattr(expert, "_fuse_gate") and getattr(expert, "up_proj", None) is not None:
                expert._fuse_gate()

        if self.shared_experts is not None and getattr(self.shared_experts, "up_proj", None) is not None:
            self.shared_experts._fuse_gate()

        self._gates_fused = True

    def _maybe_fused_moe_weights(self, device, dtype):
        if self._w1 is not None:
            return  # already initialized

        w1_list = []
        w2_list = []

        for e in self.experts:
            Wg = e.gate_up_proj.weight.to(device=device, dtype=dtype)
            Wu = getattr(e, "up_proj", None)
            if Wu is not None:
                Wu = Wu.weight.to(device=device, dtype=dtype)
                W1 = torch.cat([Wg, Wu], dim=0)  # (2*ffn, hidden)
            else:
                W1 = Wg  # Only gate_up_proj is available
            w1_list.append(W1.unsqueeze(0))

            W2 = e.down_proj.weight.to(device=device, dtype=dtype)
            w2_list.append(W2.unsqueeze(0))

        self._w1 = torch.cat(w1_list, dim=0)
        self._w2 = torch.cat(w2_list, dim=0)

    def _maybe_init_quant_weights(self, device, dtype):
        """Detect quantisation and populate the appropriate weight cache. Runs once."""
        if self._quant_weights_initialized or not _triton_moe:
            return
        self._quant_weights_initialized = True

        if not self.experts:
            return

        quant_type = detect_expert_quant_type(self.experts)

        if quant_type == "fp16":
            if type(self.experts[0].gate_up_proj) is torch.nn.Linear:
                self._maybe_fused_moe_weights(device, dtype)

        elif quant_type == "gptq":
            (
                self._w1_qweight,
                self._w1_scales,
                self._w1_qzeros,
                self._w2_qweight,
                self._w2_scales,
                self._w2_qzeros,
                self._group_size,
            ) = stack_gptq_moe_weights(self.experts, device)
            self._kpacked = True

        elif quant_type == "awq":
            (
                self._w1_qweight,
                self._w1_scales,
                self._w1_qzeros,
                self._w2_qweight,
                self._w2_scales,
                self._w2_qzeros,
                self._group_size,
            ) = stack_awq_moe_weights(self.experts, device)
            self._kpacked = False

        elif quant_type == "marlin":
            (
                self._w1_marlin,
                self._w1_marlin_scales,
                self._w1_marlin_qzeros,
                self._w1_marlin_g_idx,
                self._w1_marlin_perm,
                self._w2_marlin,
                self._w2_marlin_scales,
                self._w2_marlin_qzeros,
                self._w2_marlin_g_idx,
                self._w2_marlin_perm,
                self._marlin_workspace,
                _num_bits,
                self._marlin_scalar_type_id,
                _group_size,
            ) = stack_marlin_moe_weights(self.experts, device)
            self._marlin_num_experts = len(self.experts)

            # w1_marlin: (E, K//16, 4*I) → K = shape[1]*16 (per-expert hidden size)
            # w2_marlin: (E, I//16, 2*K) → I = shape[1]*16 (per-stream intermediate)
            self._marlin_K = self._w1_marlin.shape[1] * 16
            self._marlin_I = self._w2_marlin.shape[1] * 16

    def forward(self, x):

        if self.training:
            self._maybe_fuse_gates()
        else:
            self._maybe_init_quant_weights(x.device, x.dtype)

        B, T, C = x.shape
        K = self.num_experts_per_tok

        # flatten only token dimension, but keep track
        x_flat = x.view(-1, C)

        # ── Gate ──────────────────────────────────────────────────────────────
        scores = self.gate(x_flat)  # (BT, num_experts)

        if self.parallel_gpu > 1:
            all_reduce(scores)

        if not self.moe_softmax_after:
            scores = scores.softmax(dim=-1)

        expert_weights, expert_indices = torch.topk(scores, K, dim=-1, sorted=False)  # both are (BT, K)

        if self.moe_softmax_after:
            # Mixtral-style: softmax is applied *after* topk selection.
            expert_weights = expert_weights.softmax(dim=-1)
        elif self.moe_renormalize:
            # Qwen-style (default): softmax is applied over all experts before topk.
            # After topk selection the remaining weights no longer sum to 1, so we
            # re-normalize to restore a proper probability distribution.
            # clamp prevents a theoretical zero-sum when all selected logits are tiny.
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # ── Path 1: fp16 Triton ───────────────────────────────────────────────
        if not self.training and self._w1 is not None:
            y = fused_experts_impl(
                hidden_states=x_flat,
                w1=self._w1,
                w2=self._w2,
                topk_weights=expert_weights,
                topk_ids=expert_indices,
                activation=self.activation_function,
            ).view(B, T, C)
        # ── Path 2: Marlin MoE ───────────────────────────────────────────
        elif not self.training and self._w1_marlin is not None:
            topk_val = K  # number of experts per token
            BT = x_flat.shape[0]  # number of (batch * seq) tokens

            self._maybe_grow_marlin_buffers(BT, topk_val, x_flat.device, x_flat.dtype)

            y = fused_experts_marlin_impl(
                hidden_states=x_flat,
                w1_qweight=self._w1_marlin,
                w1_scales=self._w1_marlin_scales,
                w1_qzeros=self._w1_marlin_qzeros,
                w1_g_idx=self._w1_marlin_g_idx,
                w1_perm=self._w1_marlin_perm,
                w2_qweight=self._w2_marlin,
                w2_scales=self._w2_marlin_scales,
                w2_qzeros=self._w2_marlin_qzeros,
                w2_g_idx=self._w2_marlin_g_idx,
                w2_perm=self._w2_marlin_perm,
                workspace=self._marlin_workspace,
                topk_weights=expert_weights,
                topk_ids=expert_indices,
                b_q_type_id=self._marlin_scalar_type_id,
                num_experts=self._marlin_num_experts,
                activation=self.activation_function or "silu",
                cache13=self._marlin_cache13,
                cache2=self._marlin_cache2,
                out=self._marlin_out_buf[:BT, :],
                topk_ids_i32=self._marlin_topk_ids_i32[:BT, :],
            ).view(B, T, C)
        # ── Path 3: int4 Triton (GPTQ / AutoRound / AWQ) ─────────────────────
        elif not self.training and self._w1_qweight is not None:
            BT = x_flat.shape[0]
            topk_val = K
            # Derive I from w2_qweight shape:
            #   kpacked (GPTQ): (E, I//8, H)  → I = shape[1] * 8
            #   awq:            (E, I,    H//8)→ I = shape[1]
            I_int4 = self._w2_qweight.shape[1] * 8 if self._kpacked else self._w2_qweight.shape[1]
            self._maybe_grow_int4_buffers(BT, topk_val, I_int4, C, x_flat.device, x_flat.dtype)

            y = fused_experts_int4_impl(
                hidden_states=x_flat,
                w1_qweight=self._w1_qweight,
                w1_scales=self._w1_scales,
                w1_qzeros=self._w1_qzeros,
                w2_qweight=self._w2_qweight,
                w2_scales=self._w2_scales,
                w2_qzeros=self._w2_qzeros,
                topk_weights=expert_weights,
                topk_ids=expert_indices,
                group_size=self._group_size,
                kpacked=self._kpacked,
                activation=self.activation_function or "silu",
                intermediate_buf=self._int4_intermediate_buf,
                output_buf=self._int4_output_buf,
                expert_ids_buf=self._int4_expert_ids_buf,
                token_ids_buf=self._int4_token_ids_buf,
                weights_buf=self._int4_weights_buf,
            ).view(B, T, C)
        # ── Path 4: vectorized fallback ───────────────────────────────────────
        else:
            x_flat = x_flat.repeat_interleave(K, dim=0)  # (BTK, C)
            y = vectorized_moe(x_flat, expert_weights, expert_indices, K, self.experts).view(B, T, C)
            """
            For educational purpose, we leave here the original implementation
            y = naive_moe(x_flat, expert_weights, expert_indices, K, self.experts).view(B, T, C)
            and a faster vectorized version
            y = self._grouped.forward_tokens(x_flat, expert_weights, expert_indices, K).view(B, T, C)
            """

        # ── Optional shared experts ───────────────────────────────────────────
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x)
            if self.shared_expert_gate is not None:
                x_flat_for_gate = x.view(-1, x.shape[-1])
                gate_val = torch.sigmoid(self.shared_expert_gate(x_flat_for_gate)).view(B, T, 1)
                y = y + gate_val * shared_out
            else:
                y = y + shared_out

        return y


class GroupedExperts:
    """
    Memory-safe grouped expert executor using only F.linear (no expert.forward calls).
    Provides debug mode to compare per-expert intermediate tensors vs naive expert.forward.
    [2025-11-22 12:31:41,110 INFO] Subsequent prediction time including all (s): 58.98
    [2025-11-22 12:31:41,110 INFO] Average prediction time (ms): 2681.0
    [2025-11-22 12:31:41,110 INFO] Tokens per second: 274.7
    """

    def __init__(self, experts: torch.nn.ModuleList, debug: bool = False):
        assert len(experts) > 0
        self.experts = experts
        self.num_experts = len(experts)
        self.debug = debug

        # detect gated vs simple from first expert
        first = experts[0]
        self.gated = getattr(first, "up_proj", None) is not None
        self.ffn = first.gate_up_proj.weight.size(0)
        self.hidden = first.gate_up_proj.weight.size(1)

        # store references to per-expert weight/bias tensors (kept as lists)
        # we'll not stack into huge tensors — we will store refs and .to() them per-forward

        self.W_gate = [e.gate_up_proj.weight for e in experts]  # (ffn, hidden)
        self.b_gate = [
            (e.gate_up_proj.bias if getattr(e.gate_up_proj, "bias", None) is not None else None) for e in experts
        ]

        self.W_down = [e.down_proj.weight for e in experts]  # (hidden, ffn)
        self.b_down = [(e.down_proj.bias if getattr(e.down_proj, "bias", None) is not None else None) for e in experts]

        if self.gated:
            self.W_up = [e.up_proj.weight for e in experts]  # (ffn, hidden)
            self.b_up = [(e.up_proj.bias if getattr(e.up_proj, "bias", None) is not None else None) for e in experts]
        else:
            self.W_up = [None] * self.num_experts
            self.b_up = [None] * self.num_experts

        # activation from first expert
        self.activation = getattr(first, "activation", F.gelu)
        # track device/dtype to move/cast weights once per forward
        self._cached_device = None
        self._cached_dtype = None

    def _ensure_weights_on(self, device: torch.device, dtype: torch.dtype):
        """
        Move/cast per-expert weight/bias tensors to requested device/dtype.
        We replace list elements (storing device-dtype tensors) so subsequent calls are fast.
        """
        if self._cached_device == device and self._cached_dtype == dtype:
            return

        # move/cast each tensor in lists
        self.W_gate = [w.to(device=device, dtype=dtype) for w in self.W_gate]
        self.b_gate = [(b.to(device=device, dtype=dtype) if b is not None else None) for b in self.b_gate]
        self.W_down = [w.to(device=device, dtype=dtype) for w in self.W_down]
        self.b_down = [(b.to(device=device, dtype=dtype) if b is not None else None) for b in self.b_down]
        if self.gated:
            self.W_up = [w.to(device=device, dtype=dtype) for w in self.W_up]
            self.b_up = [(b.to(device=device, dtype=dtype) if b is not None else None) for b in self.b_up]
            # Concatenate once and store for fused gate
            self.W_gate_up = [torch.cat([Wg, Wup], dim=0) for Wg, Wup in zip(self.W_gate, self.W_up)]
            del self.W_up, self.W_gate
            # Pre-concat biases with full None-support
            self.b_gate_up = []
            for bg, bup in zip(self.b_gate, self.b_up):
                if bg is None and bup is None:
                    self.b_gate_up.append(None)
                elif bg is None:
                    # gate part is missing, so fill zeros of same size as W_gate rows
                    bg = torch.zeros_like(bup)
                    self.b_gate_up.append(torch.cat([bg, bup], dim=0))
                elif bup is None:
                    bup = torch.zeros_like(bg)
                    self.b_gate_up.append(torch.cat([bg, bup], dim=0))
                else:
                    self.b_gate_up.append(torch.cat([bg, bup], dim=0))
            del self.b_up, self.b_gate

        self._cached_device = device
        self._cached_dtype = dtype

    def forward_tokens(self, x, topk_weights, topk_ids, K):
        """
        x: (N = BT*K, hidden)  tokens in flat ordering: token0:k0, token0:k1, token1:k0, ...
        topk_weights: (BT, K)
        topk_ids: (BT, K)
        K: num_experts_per_token
        """
        flat_topk_ids = topk_ids.view(-1)  # (BT*K)
        BTK = flat_topk_ids.size(0)

        if x.numel() == 0:
            return x.new_empty((0, x.shape[-1]))

        self._ensure_weights_on(x.device, x.dtype)

        # ---- Sort tokens by expert ID ----
        sort_order = torch.argsort(flat_topk_ids, stable=False)
        sorted_experts = flat_topk_ids[sort_order].tolist()  # Python list
        run_starts = [0] + [i for i in range(1, BTK) if sorted_experts[i] != sorted_experts[i - 1]]
        run_ends = run_starts[1:] + [BTK]

        y_flat = torch.empty_like(x)  # (BT*K, hidden)

        for s, e in zip(run_starts, run_ends):
            expert_id = sorted_experts[s]
            routed_positions = sort_order[s:e]
            inp_tokens = x[routed_positions]  # (n_e, hidden)

            if self.gated:  # fused gate/up
                gate_up = F.linear(inp_tokens, self.W_gate_up[expert_id], self.b_gate_up[expert_id])
                gate, up = gate_up.split(self.ffn, dim=1)
                gate = self.activation(gate)
                hidden_rep = gate * up
            else:
                gate = F.linear(inp_tokens, self.W_gate[expert_id], self.b_gate[expert_id])
                hidden_rep = self.activation(gate)

            y_flat[routed_positions] = F.linear(hidden_rep, self.W_down[expert_id], self.b_down[expert_id])

        # ---- to (BT, K, hidden) and weighted combine ----
        y = y_flat.view(BTK // K, K, self.hidden)
        y = (y * topk_weights.unsqueeze(-1)).sum(dim=1)
        return y
