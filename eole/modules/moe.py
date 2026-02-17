"""MoE mixture of experts"."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from eole.modules.transformer_mlp import MLP
from torch.distributed import all_reduce

try:
    from eole.triton.fused_moe import fused_experts_impl
except ImportError:
    fused_experts_impl = None


def naive_moe(x, topk_weights, topk_ids, K, experts):
    """
        Original EOLE Code
        x: (N = BT*K, hidden)  tokens in flat ordering: token0:k0, token0:k1, token1:k0, ...
        topk_weights: (BT, K)
        topk_ids: (BT, K)
        K: num_experts_per_token
        experts: nn.ModuleList

    [2025-11-21 13:35:29,773 INFO] Subsequent prediction time including all (s): 152.35
    [2025-11-21 13:35:29,773 INFO] Average prediction time (ms): 6925.0
    [2025-11-21 13:35:29,773 INFO] Tokens per second: 106.2
    with fused gate
    [2025-11-22 13:12:40,557 INFO] Subsequent prediction time including all (s): 140.64
    [2025-11-22 13:12:40,557 INFO] Average prediction time (ms): 6392.7
    [2025-11-22 13:12:40,557 INFO] Tokens per second: 115.2
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

    [2025-11-22 12:16:20,890 INFO] Subsequent prediction time including all (s): 73.50
    [2025-11-22 12:16:20,890 INFO] Average prediction time (ms): 3340.9
    [2025-11-22 12:16:20,890 INFO] Tokens per second: 220.1
    with fused gate
    [2025-11-22 12:24:38,864 INFO] Subsequent prediction time including all (s): 64.12
    [2025-11-22 12:24:38,864 INFO] Average prediction time (ms): 2914.6
    [2025-11-22 12:24:38,864 INFO] Tokens per second: 252.6
    """

    flat_topk_ids = topk_ids.view(-1)  # (BT*K)
    BTK = flat_topk_ids.size(0)
    hidden = x.size(1)
    if BTK == 0:
        return x.new_empty((0, hidden))

    # ---- Sort tokens by expert ID ----
    sort_order = torch.argsort(flat_topk_ids)
    sorted_experts = flat_topk_ids[sort_order].tolist()
    run_starts = [0] + [i for i in range(1, BTK) if sorted_experts[i] != sorted_experts[i - 1]]
    run_ends = run_starts[1:] + [BTK]

    # ---- Buffer for outputs in sorted order ----
    y_flat = torch.empty_like(x)  # (BT*K, hidden)

    # ---- Process each expert once per run ----
    for s, e in zip(run_starts, run_ends):
        expert_id = sorted_experts[s]
        routed_positions = sort_order[s:e]
        inp = x[routed_positions]
        y_flat[routed_positions] = experts[expert_id](inp)

    # ---- to (BT, K, hidden) and weighted combine ----
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
        else:
            self.shared_experts = None
        self.moe_softmax_after = model_config.moe_softmax_after
        self._gates_fused = False
        self._grouped = GroupedExperts(self.experts)

        self.activation_function = next(
            (a for a in ("gelu", "relu", "silu") if a in model_config.mlp_activation_fn), None
        )
        self._w1 = None
        self._w2 = None

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
            # --- W1 (gate_up + up) ---
            Wg = e.gate_up_proj.weight.to(device=device, dtype=dtype)
            Wu = getattr(e, "up_proj", None)
            if Wu is not None:
                Wu = Wu.weight.to(device=device, dtype=dtype)
                W1 = torch.cat([Wg, Wu], dim=0)  # (2*ffn, hidden)
            else:
                W1 = Wg  # Only gate_up_proj is available
            w1_list.append(W1.unsqueeze(0))

            # --- W2 (down_proj) ---
            W2 = e.down_proj.weight.to(device=device, dtype=dtype)
            w2_list.append(W2.unsqueeze(0))

        # stack all experts -> (num_experts, ...)
        self._w1 = torch.cat(w1_list, dim=0)
        self._w2 = torch.cat(w2_list, dim=0)

    def forward(self, x):

        if not self.training:
            self._maybe_fused_moe_weights(x.device, x.dtype)
        else:
            self._maybe_fuse_gates()

        B, T, C = x.shape
        K = self.num_experts_per_tok

        # flatten only token dimension, but keep track
        x_flat = x.view(-1, C)

        # ---- GATE ----
        scores = self.gate(x_flat)  # (BT, num_experts)

        if self.parallel_gpu > 1:
            all_reduce(scores)

        if not self.moe_softmax_after:
            scores = scores.softmax(dim=-1)

        expert_weights, expert_indices = torch.topk(scores, K, dim=-1, sorted=False)  # both are (BT, K)

        if self.moe_softmax_after:
            expert_weights = expert_weights.softmax(dim=-1)

        if not self.training and fused_experts_impl is not None:
            y = fused_experts_impl(
                hidden_states=x_flat,
                w1=self._w1,
                w2=self._w2,
                topk_weights=expert_weights,
                topk_ids=expert_indices,
                activation=self.activation_function,
            ).view(B, T, C)
        else:
            x_flat = x_flat.repeat_interleave(K, dim=0)  # (BTK, C)
            y = vectorized_moe(x_flat, expert_weights, expert_indices, K, self.experts).view(B, T, C)
            """
            For educational purpose, we leave here the original implementation
            y = naive_moe(x_flat, expert_weights, expert_indices, K, self.experts).view(B, T, C)
            and a faster vectorized version
            y = self._grouped.forward_tokens(x_flat, expert_weights, expert_indices, K).view(B, T, C)
            """

        # optional shared experts
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

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
