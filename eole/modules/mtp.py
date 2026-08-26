"""Multi-Token Prediction (MTP) auxiliary heads.

Each :class:`MTPHead` predicts the token ``t+k`` from the hidden state at
position ``t``.  Following the DeepSeek-V3 paper the auxiliary heads share the
embedding table with the main model and receive a **detached** copy of the
main hidden states so that their gradients do not back-propagate into the core
decoder.

Reference: https://arxiv.org/abs/2412.19437
"""

import torch
import torch.nn as nn

from eole.constants import LayerNorm
from eole.decoders.transformer import TransformerDecoderLayer


class MTPHead(nn.Module):
    """Single Multi-Token Prediction auxiliary head.

    Architecture (per DeepSeek-V3):
    1. Project the detached main hidden state: ``h' = enorm(linear(h.detach()))``
    2. Add the embedding of the shifted target token:
       ``combined = h' + emb(tgt[:, k-1:-1])``
    3. Run a single :class:`~eole.decoders.transformer.TransformerDecoderLayer`.
    4. Apply a final layer norm.
    5. The output is fed to the *shared* generator (lm_head) in the loss
       computation — no separate projection needed here.

    Args:
        decoder_config: :class:`~eole.config.models.TransformerDecoderConfig`
            — reuses the same architecture hyperparameters as the main decoder.
        running_config: Training or inference config (passed through to the
            underlying :class:`TransformerDecoderLayer`).
    """

    def __init__(self, decoder_config, running_config=None):
        super().__init__()
        hidden_size = decoder_config.hidden_size

        # Linear projection applied to the (detached) main hidden states.
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Embedding normalisation before combining with target embeddings.
        self.enorm = LayerNorm[decoder_config.layer_norm](hidden_size, eps=decoder_config.norm_eps)

        # Single transformer layer shared architecture with the main decoder.
        # We pass idx=0; the layer_types list (if any) is intentionally not
        # forwarded so the MTP layer is always a standard full-attention layer.
        _cfg = decoder_config.model_copy(update={"layer_types": None, "with_cross_attn": False})
        self.layer = TransformerDecoderLayer(_cfg, idx=0, running_config=running_config)

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
        # 1. Project main hidden states.
        h = self.enorm(self.proj(hidden_states))

        # 2. Combine with shifted target embeddings.
        combined = h + tgt_emb_k

        # 3. Transformer layer (no cross-attention).
        layer_out, _ = self.layer(combined, attn_mask=attn_mask, **kwargs)

        # 4. Final norm.
        return self.norm(layer_out)
