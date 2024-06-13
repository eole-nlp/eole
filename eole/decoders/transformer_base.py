"""
Implementation of "Attention is All You Need" and of
subsequent transformer based architectures
"""

import torch
import torch.nn as nn
from eole.decoders.decoder import DecoderBase
from eole.modules.multi_headed_attn import SelfMHA
from eole.modules.transformer_mlp import MLP
from eole.modules.moe import MoE
from eole.constants import LayerNorm


class TransformerDecoderLayerBase(nn.Module):
    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        """
        Args:
            model_config (eole.config.TransformerDecoderConfig): full decoder config
        """
        super(TransformerDecoderLayerBase, self).__init__()

        self.parallel_residual = model_config.parallel_residual
        self.shared_layer_norm = model_config.shared_layer_norm
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0]
        self.full_context_alignment = model_config.full_context_alignment
        self.alignment_heads = model_config.alignment_heads
        self.sliding_window = model_config.sliding_window

        self.input_layernorm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
        self.self_attn = SelfMHA(
            model_config,
            running_config=running_config,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.post_attention_layernorm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
        if model_config.parallel_residual and not model_config.shared_layer_norm:
            self.residual_layernorm = LayerNorm[model_config.layer_norm](
                model_config.hidden_size, eps=model_config.norm_eps
            )
        if model_config.num_experts > 0:
            self.mlp = MoE(model_config, running_config)
        else:
            self.mlp = MLP(
                model_config,
                running_config=running_config,
            )

    def forward(self, *args, **kwargs):
        """Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward, of which
            with_align (bool): needed to compute attn_align
            return_attn (bool): to force MHA to return attns

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * layer_out ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        with_align = kwargs.pop("with_align", False)
        layer_out, attns = self._forward(*args, **kwargs)
        top_attn = None if attns is None else attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, : self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return layer_out, top_attn, attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.mlp.update_dropout(dropout)
        self.dropout.p = dropout

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:
            # Add triangular future_mask and pad_mask, result mask in (B, T, T).
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.tril_(0)
            if self.sliding_window > 0:
                future_mask = future_mask.triu_(-self.sliding_window)
            future_mask = future_mask.bool()
            future_mask = ~future_mask.view(1, tgt_len, tgt_len)
            # Patch for scaled dot product attention.
            patch_mask = ~torch.all(
                tgt_pad_mask + future_mask, dim=2, keepdim=True
            ).expand_as(tgt_pad_mask + future_mask)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            dec_mask = torch.logical_and(dec_mask, patch_mask)
        else:
            # Only mask padding, result mask in (B, 1, T).
            dec_mask = tgt_pad_mask
        return dec_mask


class TransformerDecoderBase(DecoderBase):
    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(TransformerDecoderBase, self).__init__()

        self.alignment_layer = model_config.alignment_layer

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        return cls(
            model_config,
            running_config=running_config,
        )

    def init_state(self, **kwargs):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        # if self.state["src"] is not None:
        #    self.state["src"] = fn(self.state["src"], 0)
        for layer in self.transformer_layers:
            if hasattr(layer, "context_attn"):
                if layer.context_attn.layer_cache[1]["keys"].numel() != 0:
                    x = fn(layer.context_attn.layer_cache[1]["keys"], 0)
                    y = fn(layer.context_attn.layer_cache[1]["values"], 0)
                    layer.context_attn.layer_cache = True, {"keys": x, "values": y}
            if layer.self_attn.layer_cache[1]["keys"].numel() != 0:
                x = fn(layer.self_attn.layer_cache[1]["keys"], 0)
                y = fn(layer.self_attn.layer_cache[1]["values"], 0)
                if layer.self_attn.layer_cache[1].get("key_pad_mask", None) is not None:
                    z = fn(layer.self_attn.layer_cache[1]["key_pad_mask"], 0)
                else:
                    z = None
                layer.self_attn.layer_cache = True, {
                    "keys": x,
                    "values": y,
                    "key_pad_mask": z,
                }

    def detach_state(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
