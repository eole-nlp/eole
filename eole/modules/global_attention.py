"""Global attention modules (Luong / Bahdanau)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from eole.modules.sparse_activations import sparsemax


class GlobalAttention(nn.Module):
    r"""
    Global attention mechanism for sequence-to-sequence models.

    Computes a parameterized convex combination of encoder hidden states
    based on decoder queries.

    Supports three attention scoring functions:
        - **dot**: score(H_j, q) = H_j^T * q
        - **general**: score(H_j, q) = H_j^T * W_a * q
        - **mlp**: score(H_j, q) = v_a^T * tanh(W_a*q + U_a*h_j)

    Args:
        dim: Dimensionality of query and key vectors
        coverage: Whether to use coverage mechanism
        attn_type: Type of attention scoring ('dot', 'general', or 'mlp')
        attn_func: Normalization function ('softmax' or 'sparsemax')

    References:
        - Luong et al. (2015): Effective Approaches to Attention-based NMT
        - Bahdanau et al. (2015): Neural Machine Translation by Jointly Learning to Align
    """

    VALID_ATTN_TYPES = {"dot", "general", "mlp"}
    VALID_ATTN_FUNCS = {"softmax", "sparsemax"}

    def __init__(self, dim: int, coverage: bool = False, attn_type: str = "dot", attn_func: str = "softmax"):
        super().__init__()

        self.dim = dim
        self._validate_params(attn_type, attn_func)
        self.attn_type = attn_type
        self.attn_func = attn_func

        # Initialize attention-specific layers
        self._build_attention_layers(dim, coverage)

    def _validate_params(self, attn_type: str, attn_func: str) -> None:
        """Validate attention type and function parameters."""
        if attn_type not in self.VALID_ATTN_TYPES:
            raise ValueError(f"Invalid attention type '{attn_type}'. " f"Must be one of {self.VALID_ATTN_TYPES}")
        if attn_func not in self.VALID_ATTN_FUNCS:
            raise ValueError(f"Invalid attention function '{attn_func}'. " f"Must be one of {self.VALID_ATTN_FUNCS}")

    def _build_attention_layers(self, dim: int, coverage: bool) -> None:
        """Initialize layers based on attention type."""
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # Output projection (with bias only for MLP attention)
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        # Coverage layer (optional)
        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)
        else:
            self.linear_cover = None

    def _score_dot(self, h_t: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """Compute dot-product attention scores."""
        # (batch, tgt_len, dim) x (batch, dim, src_len) -> (batch, tgt_len, src_len)
        return torch.bmm(h_t, h_s.transpose(1, 2))

    def _score_general(self, h_t: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """Compute general (bilinear) attention scores."""
        h_t_transformed = self.linear_in(h_t)
        return torch.bmm(h_t_transformed, h_s.transpose(1, 2))

    def _score_mlp(self, h_t: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """Compute MLP (additive/Bahdanau) attention scores."""
        batch, tgt_len, dim = h_t.size()
        src_len = h_s.size(1)

        # Project query and expand
        wq = self.linear_query(h_t)  # (batch, tgt_len, dim)
        wq = wq.unsqueeze(2).expand(batch, tgt_len, src_len, dim)

        # Project context and expand
        uh = self.linear_context(h_s)  # (batch, src_len, dim)
        uh = uh.unsqueeze(1).expand(batch, tgt_len, src_len, dim)

        # Combine and score
        wquh = torch.tanh(wq + uh)  # (batch, tgt_len, src_len, dim)
        return self.v(wquh).squeeze(-1)  # (batch, tgt_len, src_len)

    def score(self, h_t: torch.Tensor, h_s: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between queries and keys.

        Args:
            h_t: Query vectors (batch, tgt_len, dim)
            h_s: Key vectors (batch, src_len, dim)

        Returns:
            Unnormalized attention scores (batch, tgt_len, src_len)
        """
        if self.attn_type == "dot":
            return self._score_dot(h_t, h_s)
        elif self.attn_type == "general":
            return self._score_general(h_t, h_s)
        else:  # mlp
            return self._score_mlp(h_t, h_s)

    def _apply_coverage(self, enc_out: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        """Apply coverage mechanism to encoder outputs."""
        coverage_input = coverage.view(-1, 1)  # Flatten for linear layer
        coverage_feature = self.linear_cover(coverage_input)
        coverage_feature = coverage_feature.view_as(enc_out)
        return torch.tanh(enc_out + coverage_feature)

    def _normalize_attention(self, align: torch.Tensor, batch: int, target_l: int, src_l: int) -> torch.Tensor:
        """Apply softmax or sparsemax to normalize attention weights."""
        align_flat = align.view(batch * target_l, src_l)

        if self.attn_func == "softmax":
            align_vectors = F.softmax(align_flat, dim=-1)
        else:  # sparsemax
            align_vectors = sparsemax(align_flat, dim=-1)

        return align_vectors.view(batch, target_l, src_l)

    def forward(
        self,
        src: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        coverage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vectors.

        Args:
            src: Decoder queries (batch, tgt_len, dim) or (batch, dim) for single step
            enc_out: Encoder hidden states (batch, src_len, dim)
            src_pad_mask: Source pad mask (batch,)
            coverage: Coverage vector (batch, src_len) - optional

        Returns:
            Tuple of:
                - Context-aware output vectors (batch, tgt_len, dim)
                - Attention distributions (batch, tgt_len, src_len)
        """
        # Handle single-step decoding
        one_step = src.dim() == 2
        if one_step:
            src = src.unsqueeze(1)

        batch, src_l, dim = enc_out.size()
        target_l = src.size(1)

        # Apply coverage if provided
        if coverage is not None and self.linear_cover is not None:
            enc_out = self._apply_coverage(enc_out, coverage)

        # Compute attention scores
        align = self.score(src, enc_out)  # (batch, target_l, src_l)

        # Apply padding mask if provided
        if src_pad_mask is not None:
            align.masked_fill_(src_pad_mask, -float("inf"))

        # Normalize attention weights
        align_vectors = self._normalize_attention(align, batch, target_l, src_l)

        # Compute weighted context vectors
        context = torch.bmm(align_vectors, enc_out)  # (batch, target_l, dim)

        # Concatenate context with query and project
        concat_c = torch.cat([context, src], dim=2)  # (batch, target_l, dim*2)
        concat_c = concat_c.view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)

        # Apply tanh for dot/general attention types
        if self.attn_type in {"dot", "general"}:
            attn_h = torch.tanh(attn_h)

        # Remove extra dimension for single-step decoding
        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

        return attn_h, align_vectors
