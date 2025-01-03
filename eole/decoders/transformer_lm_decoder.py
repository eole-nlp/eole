"""
Implementation of "Attention is All You Need" and of
subsequent transformer based architectures
"""

import torch
import torch.nn as nn

from eole.decoders.transformer_base import (
    TransformerDecoderLayerBase,
    TransformerDecoderBase,
)
from eole.constants import LayerNorm


class TransformerLMDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder only layer block in GPT style.
    Args:
         See TransformerDecoderLayerBase
    """

    def _forward(
        self,
        layer_in,
        pad_mask,
        step=None,
        future=False,
        return_attn=False,
        position_embeddings=None,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            layer_in (FloatTensor): ``(batch_size, T, model_dim)``
            pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.
            return_attn (bool): If set True return attn
            position_embeddings (FloatTensor): rotary position encodings, if any

        Returns:
            (FloatTensor, FloatTensor):

            * layer_out ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, T)``

        """
        attn_mask = None

        if layer_in.size(1) > 1:
            # Masking is necessary when sequence length is greater than one
            # The decoding has not started yet,
            # we compute the scores on the source tokens in one shot.
            attn_mask = self._compute_attn_mask(pad_mask, future)
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.expand(-1, -1, attn_mask.size(3), -1)
            # mask now are (batch x 1 x tlen x tlen)
            # 1 = heads to be expanded in MHA

        norm_layer_in = self.input_layernorm(layer_in)

        attn_output, attns = self.self_attn(
            norm_layer_in,
            attn_mask=attn_mask,
            step=step,
            return_attn=return_attn,
            position_embeddings=position_embeddings,
        )
        if self.dropout_p > 0:
            attn_output = self.dropout(attn_output)
        if self.parallel_residual:
            # we apply residual with un-normed
            if not self.shared_layer_norm:
                norm_res_layer_in = self.residual_layernorm(layer_in)
                ff_in = norm_res_layer_in
            else:
                ff_in = norm_layer_in
        else:
            ff_in = self.post_attention_layernorm(attn_output + layer_in)
        # add residual to mlp output
        layer_out = self.mlp(ff_in) + layer_in + attn_output

        return layer_out, attns


class TransformerLMDecoder(TransformerDecoderBase):
    """The Transformer decoder from GPT-2
    Args:
        model_config (eole.config.TransformerDecoderConfig): full decoder config
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(TransformerLMDecoder, self).__init__(model_config)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerLMDecoderLayer(
                    model_config,
                    running_config=running_config,
                )
                for i in range(model_config.layers)
            ]
        )
        # This is the Decoder out layer norm
        self.layer_norm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
        self._clear_cache()

    def forward(self, emb, **kwargs):
        """Decode, possibly stepwise."""

        assert emb.dim() == 3  # batch x len x embedding_dim
        pad_mask = kwargs.pop("tgt_pad_mask", None)
        assert pad_mask is not None, "TransformerLMDecoder requires a pad mask"
        step = kwargs.pop("step", None)
        with_align = kwargs.pop("with_align", False)
        return_attn = kwargs.pop("return_attn", False)
        return_attn = with_align or return_attn
        assert not with_align, "TransformerLMDecoder does not support align"
        position_embeddings = kwargs.pop("position_embeddings", None)

        if step == 0:
            self._init_cache(emb.device, pad_mask)

        for layer in self.transformer_layers:
            emb, attn, _ = layer(
                emb,
                pad_mask,
                step=step,
                with_align=with_align,
                return_attn=return_attn,
                position_embeddings=position_embeddings,
            )

        emb = self.layer_norm(emb)

        attns = {"std": attn}

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return emb, attns

    def _init_cache(self, device, pad_mask):
        for layer in self.transformer_layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_cache = (
                    True,
                    {
                        "keys": torch.tensor([], device=device),
                        "values": torch.tensor([], device=device),
                        "key_pad_mask": pad_mask,
                    },
                )

    def _clear_cache(self):
        for layer in self.transformer_layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_cache = (
                    False,
                    {
                        "keys": torch.tensor([]),
                        "values": torch.tensor([]),
                        "key_pad_mask": None,
                    },
                )
