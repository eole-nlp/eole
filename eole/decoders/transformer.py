"""
Implementation of "Attention is All You Need" Transformer Decoder
"""

import torch
import torch.nn as nn
from eole.decoders.decoder import DecoderBase
from eole.modules.multi_headed_attn import SelfMHA, ContextMHA
from eole.modules.transformer_mlp import MLP
from eole.modules.moe import MoE
from eole.constants import LayerNorm


class TransformerDecoderLayer(nn.Module):
    """Single layer of a transformer decoder

    Args:
        model_config (eole.config.TransformerEncoderConfig): full encoder config
        running_config (TrainingConfig / InferenceConfig)
        with_cross_attn (True when used with an encoder)
    """

    def __init__(
        self,
        model_config,
        running_config=None,
        with_cross_attn=False,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.parallel_residual = model_config.parallel_residual
        self.shared_layer_norm = model_config.shared_layer_norm
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0]
        self.full_context_alignment = model_config.full_context_alignment
        self.alignment_heads = model_config.alignment_heads
        self.ffn_layernorm = model_config.ffn_layernorm

        # order of layers corresponds to forward flow of tensors
        self.input_layernorm = LayerNorm[model_config.layer_norm](model_config.hidden_size, eps=model_config.norm_eps)
        self.self_attn = SelfMHA(
            model_config,
            running_config=running_config,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        if with_cross_attn:
            self.precontext_layernorm = LayerNorm[model_config.layer_norm](
                model_config.hidden_size, eps=model_config.norm_eps
            )
            self.context_attn = ContextMHA(
                model_config,
                running_config=running_config,
            )
        else:
            self.context_attn = None

        if self.ffn_layernorm:
            self.pre_feedforward_layernorm = LayerNorm[model_config.layer_norm](
                model_config.hidden_size, eps=model_config.norm_eps
            )
            self.post_feedforward_layernorm = LayerNorm[model_config.layer_norm](
                model_config.hidden_size, eps=model_config.norm_eps
            )

        if model_config.parallel_residual and not model_config.shared_layer_norm:
            self.residual_layernorm = LayerNorm[model_config.layer_norm](
                model_config.hidden_size, eps=model_config.norm_eps
            )
        self.post_attention_layernorm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
        if model_config.num_experts > 0:
            self.mlp = MoE(model_config, running_config)
        else:
            self.mlp = MLP(
                model_config,
                running_config=running_config,
            )

    def _mlp(self, hidden_states):
        if self.ffn_layernorm:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states

    def forward(self, layer_in, **kwargs):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, tgt_len, model_dim)``
            **kwargs
                enc_out: (only when using encoder/decoder)
                src_pad_mask: (only when using encoder/decoder)
                attn_mask (BoolTensor): ``(batch_size, 1, src/tgt_len, tgt_len)``
                step: int
                return_attention: bool
                position_embeddings (FloatTensor): rotary position encodings, if any

        Returns:
            (FloatTensor, FloatTensor):
            * layer_out ``(batch_size, tgt_len, model_dim)``
            * attns
        """

        enc_out = kwargs.pop("enc_out", None)
        attn_mask = kwargs.pop("attn_mask", None)
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        step = kwargs.pop("step", None)
        return_attn = kwargs.pop("return_attn", False)
        position_embeddings = kwargs.pop("position_embeddings", None)

        norm_layer_in = self.input_layernorm(layer_in)

        self_attn, attns = self.self_attn(
            norm_layer_in,
            attn_mask=attn_mask,
            step=step,
            return_attn=return_attn,
            position_embeddings=position_embeddings,
        )

        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        if self.ffn_layernorm:
            # NOTE: this case was added for Gemma2 support and might be extended for further ctx_attn support
            ff_in = layer_in + self.post_attention_layernorm(self_attn)
            layer_out = ff_in + self._mlp(ff_in)
            return layer_out, attns

        if self.parallel_residual:
            if self.context_attn:
                ctx_attn, attns = self.context_attn(
                    enc_out,
                    enc_out,
                    norm_layer_in,
                    attn_mask=~src_pad_mask,
                    return_attn=return_attn,
                )
            else:
                ctx_attn = 0
            if not self.shared_layer_norm:
                norm_res_layer_in = self.residual_layernorm(layer_in)
                ff_in = norm_res_layer_in
            else:
                ff_in = norm_layer_in
        else:
            if self.context_attn:
                norm_query = self.precontext_layernorm(self_attn + layer_in)
                ctx_attn, attns = self.context_attn(
                    enc_out,
                    enc_out,
                    norm_query,
                    attn_mask=~src_pad_mask,
                    return_attn=return_attn,
                )
                if self.dropout_p > 0:
                    ctx_attn = self.dropout(ctx_attn)
            else:
                ctx_attn = 0
            ff_in = self.post_attention_layernorm(ctx_attn + self_attn + layer_in)
        # we apply residual with un-normed
        layer_out = self.mlp(ff_in) + layer_in + self_attn + ctx_attn

        return layer_out, attns

    def get_attn_align(self, layer_in, **kwargs):
        """:cite:`garg2019jointly`."""
        attns = kwargs.pop("attns", None)
        if self.full_context_alignment:
            # return _, (B, Q_len, K_len)
            _, attns = self.forward(layer_in, **kwargs)
        if self.alignment_heads > 0:
            attns = attns[:, : self.alignment_heads, :, :].contiguous()
        # layer average attention across heads, get ``(B, Q, K)``
        # Case 1: no full_context, no align heads -> layer avg baseline
        # Case 2: no full_context, 1 align heads -> guided align
        # Case 3: full_context, 1 align heads -> full cte guided align
        attn_align = attns.mean(dim=1)

        return attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        if self.context_attn:
            self.context_attn.update_dropout(attention_dropout)
        self.mlp.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        model_config (eole.config.TransformerEncoderConfig): full encoder config
        running_config (TrainingConfig / InferenceConfig)
        with_cross_attn (True when used with an encoder)
    """

    def __init__(
        self,
        model_config,
        running_config=None,
        with_cross_attn=False,
    ):
        super(TransformerDecoder, self).__init__()
        self.alignment_layer = model_config.alignment_layer
        self.with_cross_attn = with_cross_attn
        self.sliding_window = model_config.sliding_window

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    model_config,
                    running_config=running_config,
                    with_cross_attn=with_cross_attn,
                )
                for i in range(model_config.layers)
            ]
        )
        self.layer_norm = LayerNorm[model_config.layer_norm](model_config.hidden_size, eps=model_config.norm_eps)
        self._disable_cache()

    @classmethod
    def from_config(cls, model_config, running_config=None, with_cross_attn=False):
        """Alternate constructor."""
        return cls(
            model_config,
            running_config=running_config,
            with_cross_attn=with_cross_attn,
        )

    def init_state(self, **kwargs):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        z = fn(self.left_pad_mask, 0)
        self.left_pad_mask = z
        for layer in self.transformer_layers:
            if self.with_cross_attn:
                if layer.context_attn.layer_cache[1]["keys"].numel() != 0:
                    x = fn(layer.context_attn.layer_cache[1]["keys"], 0)
                    y = fn(layer.context_attn.layer_cache[1]["values"], 0)
                    layer.context_attn.layer_cache = True, {"keys": x, "values": y}
            if layer.self_attn.layer_cache[1]["keys"].numel() != 0:
                x = fn(layer.self_attn.layer_cache[1]["keys"], 0)
                y = fn(layer.self_attn.layer_cache[1]["values"], 0)
                layer.self_attn.layer_cache = True, {"keys": x, "values": y}

    def update_dropout(self, dropout, attention_dropout):
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)

    def _causal_attn_mask(self, tgt_pad_mask):
        tgt_len = tgt_pad_mask.size(-1)
        # Add triangular future_mask and pad_mask, result mask in (B, T, T).
        future_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt_pad_mask.device, dtype=torch.bool),
            diagonal=0,
        )
        if self.sliding_window > 0:
            future_mask = future_mask.triu_(-self.sliding_window)
        attn_mask = ~tgt_pad_mask & future_mask.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(1)  # (batch x 1 x 1 x tgt_len)
        # dim 1 (heads) and 2 (tgt_len) will be broadcasted automatically in MHA
        return attn_mask

    def forward(self, emb, **kwargs):
        """Decode, possibly stepwise.

        Args:
            emb (FloatTensor): ``(batch_size, tgt_len, model_dim)``
            **kwargs
                enc_out: (only when using encoder/decoder)
                src_pad_mask: (only when using encoder/decoder)
                tgt_pad_mask (BoolTensor): ``(batch_size, tgt_len)``
                    used to build the attention mask
                step: int
                with_aligh: bool
                return_attention: bool, set by `attn_debug`
                position_embeddings (FloatTensor): rotary position encodings, if any

        Returns:
            (FloatTensor, FloatTensor):
            * emb: decoder_out ``(batch_size, tgt_len, model_dim)``
            * attns: standard and align if set
        """
        assert emb.dim() == 3  # batch x len x embedding_dim
        enc_out = kwargs.pop("enc_out", None)
        tgt_pad_mask = kwargs.pop("tgt_pad_mask", None)
        assert tgt_pad_mask is not None, "TransformerDecoder requires a tgt pad mask"
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        if self.with_cross_attn:
            assert src_pad_mask is not None, "TransformerDecoder requires a src pad mask"
            src_pad_mask = src_pad_mask.unsqueeze(1)  # (batch x 1 x 1 x src_len)
            # dim 1 (heads) and 2 (tgt_len) will be broadcasted automatically in MHA
        left_pad = kwargs.pop("left_pad", False)
        step = kwargs.pop("step", None)
        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or kwargs.pop("return_attn", False)
        position_embeddings = kwargs.pop("position_embeddings", None)
        attn_aligns = []

        if step == 0:
            self._enable_cache(emb.device, tgt_pad_mask)

        if emb.size(1) > 1:
            # training or first step decoding
            attn_mask = self._causal_attn_mask(tgt_pad_mask)
        else:
            # step by step decoding
            # we rebuild the pad mask of previous steps (left padded prompt)
            if left_pad:
                pad_mask = torch.zeros([emb.size(0), 1, step + 1], dtype=torch.bool, device=emb.device)
                pad_mask[:, :, : self.left_pad_mask.size(2)] = self.left_pad_mask
                attn_mask = ~pad_mask.unsqueeze(1)
            else:
                attn_mask = None

        for layer in self.transformer_layers:
            emb, attn = layer(
                emb,
                enc_out=enc_out if enc_out is not None else emb,
                src_pad_mask=src_pad_mask,
                attn_mask=attn_mask,
                step=step,
                return_attn=return_attn,
                position_embeddings=position_embeddings,
            )
            if with_align:
                attn_align = layer.get_attn_align(
                    emb,
                    enc_out=enc_out if enc_out is not None else emb,
                    src_pad_mask=src_pad_mask,
                    attn_mask=~tgt_pad_mask,
                    step=step,
                    return_attn=return_attn,
                    position_embeddings=position_embeddings,
                    attns=attn,
                )
                if attn_align is not None:
                    attn_aligns.append(attn_align)

        emb = self.layer_norm(emb)

        # we take the first head
        top_attn = None if attn is None else attn[:, 0, :, :].contiguous()
        attns = {"std": top_attn}
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`

        return emb, attns

    def _enable_cache(self, device, pad_mask):
        self.left_pad_mask = pad_mask
        for layer in self.transformer_layers:
            # first value set to True triggered by the beginning of decoding
            # layer_cache becomes active in the MultiHeadedAttention fwd
            layer.self_attn.layer_cache = (
                True,
                {
                    "keys": torch.tensor([], device=device),
                    "values": torch.tensor([], device=device),
                },
            )
            if layer.context_attn:
                layer.context_attn.layer_cache = (
                    True,
                    {
                        "keys": torch.tensor([], device=device),
                        "values": torch.tensor([], device=device),
                    },
                )

    def _disable_cache(self):
        self.left_pad_mask = torch.tensor([])
        for layer in self.transformer_layers:
            layer.self_attn.layer_cache = (
                False,
                {
                    "keys": torch.tensor([]),
                    "values": torch.tensor([]),
                },
            )
            if layer.context_attn:
                layer.context_attn.layer_cache = (
                    False,
                    {"keys": torch.tensor([]), "values": torch.tensor([])},
                )
