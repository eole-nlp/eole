"""Multi-Token Prediction (MTP) auxiliary heads.

Each :class:`MTPHead` predicts the token ``t+k+1`` by combining the hidden
state at position ``t`` with the embedding of token ``t+k``.  Following the
DeepSeek-V3 paper the auxiliary heads share the embedding table with the main
model and receive a **detached** copy of the main hidden states so that their
gradients do not back-propagate into the core decoder.

Reference: https://arxiv.org/abs/2412.19437
"""

import torch
import torch.nn as nn

from eole.constants import LayerNorm
from eole.decoders.transformer import TransformerDecoderLayer


class MTPHead(nn.Module):
    """Single Multi-Token Prediction auxiliary head.

    Qwen3.5 normalizes the hidden and token-embedding streams, concatenates
    ``[embedding, hidden]``, and applies a fused projection. DeepSeek-style
    heads retain their additive hidden-projection and embedding formulation.
    Both variants then run one
    :class:`~eole.decoders.transformer.TransformerDecoderLayer` and a final
    layer norm. The result is projected by the shared generator (lm_head).

    Args:
        decoder_config: :class:`~eole.config.models.TransformerDecoderConfig`
            — reuses the same architecture hyperparameters as the main decoder.
        running_config: Training or inference config (passed through to the
            underlying :class:`TransformerDecoderLayer`).
    """

    def __init__(self, decoder_config, running_config=None):
        super().__init__()
        hidden_size = decoder_config.hidden_size

        # Qwen3.5 projects the concatenation of the two normalized input
        # streams. The other MTP family uses a hidden-only projection.
        projection_size = hidden_size * (2 if decoder_config.mtp_emb_norm else 1)
        self.proj = nn.Linear(projection_size, hidden_size, bias=False)

        # Embedding normalisation before combining with target embeddings.
        self.enorm = LayerNorm[decoder_config.layer_norm](hidden_size, eps=decoder_config.norm_eps)
        self.emb_norm = (
            LayerNorm[decoder_config.layer_norm](hidden_size, eps=decoder_config.norm_eps)
            if decoder_config.mtp_emb_norm
            else None
        )

        # Single transformer layer reusing the main decoder's architecture.
        # Use the last-layer index so that first_k_dense_replace is respected:
        # MTP heads in MoE models (e.g. DeepSeek-V3) should be MoE layers, not
        # dense layers.  layer_types is cleared so the MTP layer is always a
        # standard full-attention layer (GatedDeltaNet variants are not used).
        _last_idx = max(0, decoder_config.layers - 1)
        _cfg = decoder_config.model_copy(update={"layer_types": None, "with_cross_attn": False})
        self.layer = TransformerDecoderLayer(_cfg, idx=_last_idx, running_config=running_config)

        # Final layer norm applied to the transformer output.
        self.norm = LayerNorm[decoder_config.layer_norm](hidden_size, eps=decoder_config.norm_eps)

    def forward(self, hidden_states, tgt_emb_k, attn_mask=None, **kwargs):
        """Run the MTP head.

        Args:
            hidden_states (Tensor): Main decoder hidden states
                ``(batch, seq_len, hidden_size)``.  **Must already be
                detached** from the main computation graph before being passed
                here (enforced by :meth:`DecoderModel.forward`).
            tgt_emb_k (Tensor): Target token embeddings shifted by ``k``
                positions, ``(batch, seq_len, hidden_size)``.  Obtained by
                embedding ``tgt[:, k : k + seq_len]``.
            attn_mask (Tensor, optional): Causal attention mask reused from
                the main decoder pass.
            **kwargs: Forwarded to :class:`TransformerDecoderLayer`.

        Returns:
            Tensor: MTP head output ``(batch, seq_len, hidden_size)``, ready
            to be projected through the shared ``generator`` (lm_head).
        """
        # Qwen3.5 MTP normalizes each stream, concatenates [embedding,
        # hidden], then applies the checkpoint's full 2H -> H projection.
        if self.emb_norm is not None:
            combined = self.proj(torch.cat([self.emb_norm(tgt_emb_k), self.enorm(hidden_states)], dim=-1))
        else:
            # DeepSeek-style MTP uses a projected hidden stream plus embedding.
            combined = self.enorm(self.proj(hidden_states)) + tgt_emb_k

        # 3. Transformer layer (no cross-attention).
        layer_out, _ = self.layer(combined, attn_mask=attn_mask, **kwargs)

        # 4. Final norm.
        return self.norm(layer_out)

    def update_dropout(self, dropout, attention_dropout):
        self.layer.update_dropout(dropout, attention_dropout)
