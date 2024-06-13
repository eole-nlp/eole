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
from eole.modules.multi_headed_attn import ContextMHA
from eole.constants import LayerNorm


class TransformerDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(TransformerDecoderLayer, self).__init__(
            model_config,
            running_config=running_config,
        )

        self.precontext_layernorm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
        self.context_attn = ContextMHA(
            model_config,
            running_config=running_config,
        )

    def update_dropout(self, dropout, attention_dropout):
        super(TransformerDecoderLayer, self).update_dropout(dropout, attention_dropout)
        self.context_attn.update_dropout(attention_dropout)

    def _forward(
        self,
        layer_in,
        enc_out,
        src_pad_mask,
        tgt_pad_mask,
        step=None,
        future=False,
        return_attn=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            layer_in (FloatTensor): ``(batch_size, T, model_dim)``
            enc_out (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.
            return_attn (bool) : if set True requires attns output

        Returns:
            (FloatTensor, FloatTensor):

            * layer_out ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None
        src_pad_mask = src_pad_mask.unsqueeze(1)  # [B,1,1,slen]

        if layer_in.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
            dec_mask = dec_mask.unsqueeze(1)
            dec_mask = dec_mask.expand(-1, -1, dec_mask.size(3), -1)
            src_pad_mask = src_pad_mask.expand(-1, -1, dec_mask.size(3), -1)
            # mask now are (batch x 1 x tlen x s or t len)
            # 1 = heads to be expanded in MHA

        norm_layer_in = self.input_layernorm(layer_in)

        self_attn, _ = self.self_attn(
            norm_layer_in,
            mask=dec_mask,
            sliding_window=self.sliding_window,
            step=step,
            return_attn=return_attn,
        )

        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        if self.parallel_residual:
            ctx_attn, attns = self.context_attn(
                enc_out,
                enc_out,
                norm_layer_in,
                mask=src_pad_mask,
                return_attn=return_attn,
            )
            if not self.shared_layer_norm:
                norm_res_layer_in = self.residual_layernorm(layer_in)
                ff_in = norm_res_layer_in
            else:
                ff_in = norm_layer_in
        else:
            norm_query = self.precontext_layernorm(self_attn + layer_in)
            ctx_attn, attns = self.context_attn(
                enc_out, enc_out, norm_query, mask=src_pad_mask, return_attn=return_attn
            )
            if self.dropout_p > 0:
                ctx_attn = self.dropout(ctx_attn)
            ff_in = self.post_attention_layernorm(ctx_attn + self_attn + layer_in)
        # we apply residual with un-normed
        layer_out = self.mlp(ff_in) + layer_in + self_attn + ctx_attn
        return layer_out, attns


class TransformerDecoder(TransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        model_config (eole.config.TransformerDecoderConfig): full decoder config
        embeddings (eole.modules.Embeddings):
            embeddings to use, should have positional encodings
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(TransformerDecoder, self).__init__(
            model_config, running_config=running_config
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
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

    def forward(self, emb, **kwargs):
        """Decode, possibly stepwise."""

        assert emb.dim() == 3  # batch x len x embedding_dim
        enc_out = kwargs.pop("enc_out", None)

        tgt_pad_mask = kwargs.pop("tgt_pad_mask", None)
        assert tgt_pad_mask is not None, "TransformerDecoder requires a tgt pad mask"
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        assert src_pad_mask is not None, "TransformerDecoder requires a src pad mask"

        step = kwargs.pop("step", None)

        if enc_out is None:
            enc_out = emb
        if step == 0:
            self._init_cache(enc_out.device)
        elif step is None:
            for layer in self.transformer_layers:
                layer.self_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )
                layer.context_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )

        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or kwargs.pop("return_attn", False)

        attn_aligns = []

        for layer in self.transformer_layers:
            emb, attn, attn_align = layer(
                emb,
                enc_out,
                src_pad_mask,
                tgt_pad_mask,
                step=step,
                with_align=with_align,
                return_attn=return_attn,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)

        emb = self.layer_norm(emb)

        attns = {"std": attn}
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return emb, attns

    def _init_cache(self, device):

        for layer in self.transformer_layers:
            # first value set to True triggered by the beginning of decoding
            # layer_cache becomes active in the MultiHeadedAttention fwd
            layer.context_attn.layer_cache = (
                True,
                {
                    "keys": torch.tensor([], device=device),
                    "values": torch.tensor([], device=device),
                },
            )
            layer.self_attn.layer_cache = (
                True,
                {
                    "keys": torch.tensor([], device=device),
                    "values": torch.tensor([], device=device),
                },
            )
            if hasattr(layer.self_attn, "rope"):
                layer.self_attn.rope = layer.self_attn.rope.to(device)
                layer.self_attn.cos = layer.self_attn.cos.to(device)
                layer.self_attn.sin = layer.self_attn.sin.to(device)
