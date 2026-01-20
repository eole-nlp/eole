"""
Implementation of "Attention is All You Need" Transformer Decoder
"""

import torch
import torch.nn as nn
from eole.decoders.decoder import DecoderBase
from eole.modules.multi_headed_attn import SelfMHA, ContextMHA
from eole.modules.transformer_mlp import MLP
from eole.modules.moe import MoE
from eole.modules.rope import build_rope
from eole.constants import LayerNorm


class TransformerDecoderLayer(nn.Module):
    """Single layer of a transformer decoder

    Args:
        decoder_config (eole.config.TransformerDecoderConfig): full decoder config
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(self, decoder_config, idx: int, running_config=None):
        super(TransformerDecoderLayer, self).__init__()
        self.parallel_residual = decoder_config.parallel_residual
        self.shared_layer_norm = decoder_config.shared_layer_norm
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0]
        self.full_context_alignment = decoder_config.full_context_alignment
        self.alignment_heads = decoder_config.alignment_heads
        self.ffn_layernorm = decoder_config.ffn_layernorm

        # order of layers corresponds to forward flow of tensors
        self.input_layernorm = LayerNorm[decoder_config.layer_norm](
            decoder_config.hidden_size, eps=decoder_config.norm_eps
        )
        self.self_attn = SelfMHA(
            decoder_config,
            running_config=running_config,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        if decoder_config.with_cross_attn:
            self.precontext_layernorm = LayerNorm[decoder_config.layer_norm](
                decoder_config.hidden_size, eps=decoder_config.norm_eps
            )
            self.context_attn = ContextMHA(
                decoder_config,
                running_config=running_config,
            )
        else:
            self.context_attn = None

        if self.ffn_layernorm:
            self.pre_feedforward_layernorm = LayerNorm[decoder_config.layer_norm](
                decoder_config.hidden_size, eps=decoder_config.norm_eps
            )
            self.post_feedforward_layernorm = LayerNorm[decoder_config.layer_norm](
                decoder_config.hidden_size, eps=decoder_config.norm_eps
            )

        if decoder_config.parallel_residual and not decoder_config.shared_layer_norm:
            self.residual_layernorm = LayerNorm[decoder_config.layer_norm](
                decoder_config.hidden_size, eps=decoder_config.norm_eps
            )
        if self.ffn_layernorm or not self.parallel_residual:
            self.post_attention_layernorm = LayerNorm[decoder_config.layer_norm](
                decoder_config.hidden_size, eps=decoder_config.norm_eps
            )
        if decoder_config.num_experts > 0:
            if idx >= decoder_config.first_k_dense_replace:
                self.mlp = MoE(decoder_config, running_config)
            else:
                self.mlp = MLP(
                    decoder_config,
                    running_config=running_config,
                )
        else:
            self.mlp = MLP(
                decoder_config,
                running_config=running_config,
            )

        if self.ffn_layernorm:
            self._forward = self._forward_ffn_layernorm
        elif self.parallel_residual:
            self._forward = self._forward_parallel_residual
        elif self.context_attn:
            self._forward = self._forward_cross_attn
        else:
            self._forward = self._forward_no_cross_attn

    def _forward_ffn_layernorm(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        ff_in = layer_in + self.post_attention_layernorm(self_attn)
        ff_in2 = self.pre_feedforward_layernorm(ff_in)
        ff_in2 = self.mlp(ff_in2)
        layer_out = ff_in + self.post_feedforward_layernorm(ff_in2)
        return layer_out, attns

    def _forward_parallel_residual(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        enc_out = kwargs.pop("enc_out", None)
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        return_attn = kwargs.pop("return_attn", False)
        self_attn.add_(layer_in)

        if self.context_attn:
            ctx_attn, attns = self.context_attn(
                enc_out,
                enc_out,
                norm_layer_in,
                attn_mask=~src_pad_mask,
                return_attn=return_attn,
            )
            self_attn = self_attn + ctx_attn
        if not self.shared_layer_norm:
            ff_in = self.residual_layernorm(layer_in)
        else:
            ff_in = norm_layer_in

        return self_attn + self.mlp(ff_in), attns

    def _forward_cross_attn(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        enc_out = kwargs.pop("enc_out", None)
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        return_attn = kwargs.pop("return_attn", False)
        self_attn.add_(layer_in)
        norm_query = self.precontext_layernorm(self_attn)
        ctx_attn, attns = self.context_attn(
            enc_out,
            enc_out,
            norm_query,
            attn_mask=~src_pad_mask,
            return_attn=return_attn,
        )
        if self.dropout_p > 0:
            ctx_attn = self.dropout(ctx_attn)
        self_attn = self_attn + ctx_attn

        ff_in = self.post_attention_layernorm(self_attn)
        # we apply residual with un-normed
        return self_attn + self.mlp(ff_in), attns

    def _forward_no_cross_attn(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        self_attn.add_(layer_in)
        ff_in = self.post_attention_layernorm(self_attn)
        return self_attn + self.mlp(ff_in), attns

    def forward(self, layer_in, **kwargs):
        """
        Args:
            layer_in (Tensor):
                Input hidden states of shape
                ``(batch_size, tgt_len, hidden_size)``.

            **kwargs:
                enc_out (Tensor, optional):
                    Encoder outputs.
                src_pad_mask (Tensor, optional):
                    Source padding mask.
                attn_mask (Tensor, optional):
                    Attention mask.
                step (int, optional):
                    Decoding step.
                return_attn (bool):
                    Whether to return attention weights.

        Returns:
            (Tensor, Tensor):

            * layer_out:
                Output hidden states of shape
                ``(batch_size, tgt_len, hidden_size)``.

            * attns:
                Attention weights of shape
                ``(batch_size, num_heads, tgt_len, src_len)``,
                or ``None``.
        """

        attn_mask = kwargs.pop("attn_mask", None)
        step = kwargs.pop("step", None)
        return_attn = kwargs.pop("return_attn", False)
        position_embeddings = kwargs.pop("position_embeddings", None)
        pos_ids = kwargs.pop("pos_ids", None)

        norm_layer_in = self.input_layernorm(layer_in)

        self_attn, attns = self.self_attn(
            norm_layer_in,
            attn_mask=attn_mask,
            step=step,
            return_attn=return_attn,
            position_embeddings=position_embeddings,
            pos_ids=pos_ids,
        )

        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        return self._forward(layer_in, norm_layer_in, self_attn, attns, **kwargs)

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
        decoder_config (eole.config.TransformerEncoderConfig): full encoder config
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(
        self,
        decoder_config,
        running_config=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.alignment_layer = decoder_config.alignment_layer
        self.with_cross_attn = decoder_config.with_cross_attn
        self.sliding_window = decoder_config.sliding_window
        self.rope = build_rope(decoder_config)
        if hasattr(decoder_config, "rope_config") and getattr(decoder_config.rope_config, "interleave_local", 0) > 0:
            self.rope_local = build_rope(decoder_config, variant="local")
        else:
            self.rope_local = None
        self.interleave_local = getattr(decoder_config.rope_config, "interleave_local", 0) or 1
        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    decoder_config,
                    running_config=running_config,
                    idx=i,
                )
                for i in range(decoder_config.layers)
            ]
        )
        self.layer_norm = LayerNorm[decoder_config.layer_norm](decoder_config.hidden_size, eps=decoder_config.norm_eps)
        self.LM_type = getattr(decoder_config, "LM_type", "causal")
        self.self_attn_backend = getattr(running_config, "self_attn_backend", "flash")
        self.flash = False
        self.max_length = getattr(running_config, "max_length", 1024)
        self.left_pad_mask = None
        self._disable_cache()

    def init_state(self, **kwargs):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        if self.left_pad_mask is not None:
            self.left_pad_mask = fn(self.left_pad_mask)
        for layer in self.transformer_layers:
            if self.with_cross_attn:
                if layer.context_attn.kcache is not None:
                    layer.context_attn.kcache = fn(layer.context_attn.kcache)
                    layer.context_attn.vcache = fn(layer.context_attn.vcache)
            if layer.self_attn.kcache is not None:
                layer.self_attn.kcache = fn(layer.self_attn.kcache)
                layer.self_attn.vcache = fn(layer.self_attn.vcache)
            if layer.self_attn.cache_leftpad is not None:
                layer.self_attn.cache_leftpad = fn(layer.self_attn.cache_leftpad)

    def update_dropout(self, dropout, attention_dropout):
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)

    def _causal_attn_mask(self, tgt_pad_mask, prefix_len=None):
        batch_size, _, tgt_len = tgt_pad_mask.size()
        device = tgt_pad_mask.device
        # Add triangular future_mask and pad_mask, result mask in (B, T, T).
        future_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt_pad_mask.device, dtype=torch.bool),
            diagonal=0,
        )  # Shape: (tgt_len, tgt_len)
        if self.sliding_window > 0:
            future_mask = future_mask.triu_(-self.sliding_window)
        if self.LM_type == "prefix" and prefix_len is not None:
            idx = torch.arange(tgt_len, device=device)
            block = (idx.view(1, tgt_len, 1) < prefix_len.view(batch_size, 1, 1)) & (
                idx.view(1, 1, tgt_len) < prefix_len.view(batch_size, 1, 1)
            )
            future_mask = torch.where(block, torch.ones_like(future_mask), future_mask)  # (B, T, T)
            attn_mask = ~(tgt_pad_mask | tgt_pad_mask.transpose(2, 1)) & future_mask
        else:
            future_mask = future_mask.unsqueeze(0)
            attn_mask = ~tgt_pad_mask & future_mask
        return attn_mask.unsqueeze(1)  # (batch x 1 x 1 x tgt_len)

    def _update_causal_mask(self, attn_mask, image_locations):
        # replicating HF code, can probably be simplified
        token_type_ids = torch.where(image_locations, torch.tensor(1), torch.tensor(0))
        token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
        token_type_mask[token_type_ids == 0] = False
        token_type_mask = token_type_mask.unsqueeze(1).to(attn_mask.device, dtype=torch.bool)
        attn_mask = attn_mask.masked_fill(token_type_mask, True)
        return attn_mask

    def forward(self, emb, **kwargs):
        """
        Decode a sequence using transformer layers.

        Args:
            emb (Tensor):
                Target embeddings of shape
                ``(batch_size, tgt_len, hidden_size)``.

            **kwargs:
                enc_out (Tensor, optional):
                    Encoder outputs.
                tgt_pad_mask (Tensor):
                    Target padding mask.
                src_pad_mask (Tensor, optional):
                    Source padding mask.
                step (int, optional):
                    Decoding step.
                with_align (bool, optional):
                    Whether to return alignment attention.

        Returns:
            (Tensor, Dict[str, Tensor]):

            * dec_outs:
                Decoder outputs of shape
                ``(batch_size, tgt_len, hidden_size)``.

            * attns:
                Dictionary of attention tensors.

                - ``attns["std"]``:
                  Standard attention of shape
                  ``(batch_size, tgt_len, src_len)``.
                - ``attns["align"]`` (optional):
                  Alignment attention of shape
                  ``(batch_size, tgt_len, src_len)``.
        """
        assert emb.dim() == 3  # batch x len x embedding_dim
        seq_len = emb.size(1)
        enc_out = kwargs.pop("enc_out", None)
        tgt_pad_mask = kwargs.pop("tgt_pad_mask", None)
        assert tgt_pad_mask is not None, "TransformerDecoder requires a tgt pad mask"
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        if self.with_cross_attn:
            assert src_pad_mask is not None, "TransformerDecoder requires a src pad mask"
            src_pad_mask = src_pad_mask.unsqueeze(1)  # (batch x 1 x 1 x src_len)
            # dim 1 (heads) and 2 (tgt_len) will be broadcasted automatically in MHA

        step = kwargs.pop("step", None)
        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or kwargs.pop("return_attn", False)

        position_embeddings = self.rope.update(seq_len)

        if self.rope_local is not None:
            position_embeddings_local = self.rope_local.update(seq_len)
            pos_emb_list = [
                position_embeddings_local if (i + 1) % self.interleave_local else position_embeddings
                for i in range(len(self.transformer_layers))
            ]
        else:
            pos_emb_list = [position_embeddings] * len(self.transformer_layers)

        image_locations = kwargs.pop("image_locations", None)
        prefix_len = kwargs.pop("prefix_len", None)
        pos_ids = kwargs.pop("pos_ids", None)

        attn_aligns = []
        attn_mask = None

        if not self.flash:
            if seq_len > 1:
                # training or first step decoding, combine pad and causal mask
                attn_mask = self._causal_attn_mask(tgt_pad_mask, prefix_len=prefix_len)
                if image_locations is not None:
                    attn_mask = self._update_causal_mask(attn_mask, image_locations)
            else:
                # step by step decoding, no causal but sdpa needs pad information
                if self.left_pad_mask is not None:
                    attn_mask = self.left_pad_mask[:, :, :, : step + 1]
                else:
                    attn_mask = None

            if self.sliding_window > 0 and (step or 0) >= self.sliding_window and attn_mask is not None:
                attn_mask = attn_mask[:, :, :, -self.sliding_window :]

        for layer, pos_emb in zip(self.transformer_layers, pos_emb_list):
            emb, attn = layer(
                emb,
                enc_out=enc_out,
                src_pad_mask=src_pad_mask,
                attn_mask=attn_mask,
                step=step,
                return_attn=return_attn,
                position_embeddings=pos_emb,
                pos_ids=pos_ids,
            )
            if with_align:
                attn_align = layer.get_attn_align(
                    emb,
                    enc_out=enc_out,
                    src_pad_mask=src_pad_mask,
                    attn_mask=attn_mask,
                    step=step,
                    return_attn=return_attn,
                    position_embeddings=pos_emb,
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

    def _init_cache(self, prompt, pad_mask):
        b, l, d = prompt.size()
        device = prompt.device
        dtype = prompt.dtype

        self.flash = self.self_attn_backend == "flash"  # we set it if backend is set and step decoding
        # We find the index of the first non-pad token (the first '0' or 'False')
        # If the first token is NOT a pad, this returns 0.
        mask_int = pad_mask.squeeze(1).to(torch.int8)
        left_pad_counts = mask_int.argmin(dim=-1).to(torch.int32)
        # Check if the VERY FIRST token is actually a pad.
        # This boolean tensor tells us if left-padding actually exists.
        is_left_padded = pad_mask[:, 0, 0]
        cache_leftpad = torch.where(is_left_padded, left_pad_counts, 0)
        total_seq_len = self.max_length + pad_mask.size(2)
        position_indices = torch.arange(total_seq_len, device=device).unsqueeze(0)
        # If cache_leftpad is 0, this mask will effectively do nothing
        self.left_pad_mask = position_indices >= cache_leftpad.view(-1, 1)
        self.left_pad_mask = self.left_pad_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.transformer_layers:
            heads_kv = layer.self_attn.heads_kv
            dph = layer.self_attn.dim_per_head
            layer.self_attn.kcache = torch.zeros((b, l + self.max_length, heads_kv, dph), dtype=dtype, device=device)
            layer.self_attn.vcache = torch.zeros((b, l + self.max_length, heads_kv, dph), dtype=dtype, device=device)
            layer.self_attn.cache_leftpad = cache_leftpad
            if layer.context_attn:
                layer.context_attn.kcache = torch.empty(0, device=device)
                layer.context_attn.vcache = torch.empty(0, device=device)

    def _disable_cache(self):
        self.left_pad_mask = None
        for layer in self.transformer_layers:
            layer.self_attn.kcache, layer.self_attn.vcache = None, None
            if layer.context_attn:
                layer.context_attn.kcache, layer.self_attn.vcache = None, None
