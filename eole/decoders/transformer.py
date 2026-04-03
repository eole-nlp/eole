"""
Implementation of "Attention is All You Need" Transformer Decoder
"""

import torch
import torch.nn as nn
from eole.decoders.decoder import DecoderBase
from eole.modules.multi_headed_attn import SelfMHA, ContextMHA
from eole.modules.transformer_mlp import MLP
from eole.modules.moe import MoE
from eole.modules.gated_delta_net import GatedDeltaNet
from eole.modules.rope import build_rope
from eole.modules.prefill_cache import PrefillCache
from eole.constants import LayerNorm, PositionEncodingType
from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE
from eole.ops import _FLASH_ATTN_AVAILABLE


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

        # Determine the layer type for hybrid architectures (e.g. Qwen3.5, Gemma4).
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is not None and idx < len(layer_types):
            self.layer_type = layer_types[idx]
        else:
            self.layer_type = "full_attention"

        # order of layers corresponds to forward flow of tensors
        self.input_layernorm = LayerNorm[decoder_config.layer_norm](
            decoder_config.hidden_size, eps=decoder_config.norm_eps
        )

        if self.layer_type == "linear_attention":
            # Replace the standard self-attention with a GatedDeltaNet layer.
            self.linear_attn = GatedDeltaNet(decoder_config, layer_idx=idx)
            self.self_attn = None
            self.context_attn = None
        else:
            # For full_attention layers in Gemma4-style hybrid models, use a
            # modified config with global_head_dim / global_heads_kv / no sliding window.
            if self.layer_type == "full_attention" and getattr(decoder_config, "global_head_dim", None) is not None:
                update = {
                    "head_dim": getattr(decoder_config, "global_head_dim", None),
                    "heads_kv": getattr(decoder_config, "global_heads_kv", None),
                    "sliding_window": 0,
                }
                rope_config = getattr(decoder_config, "rope_config", None)
                if rope_config is not None:
                    rotary_dim_global = getattr(rope_config, "rotary_dim_global", 0)
                    if rotary_dim_global > 0:
                        new_rope_config = rope_config.model_copy(update={"rotary_dim": rotary_dim_global})
                        update["rope_config"] = new_rope_config
                attn_config = decoder_config.model_copy(update=update)
            else:
                attn_config = decoder_config
            self.self_attn = SelfMHA(
                attn_config,
                running_config=running_config,
            )
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

        self.dropout = nn.Dropout(self.dropout_p)

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

        if self.layer_type == "linear_attention":
            self._forward = self._forward_linear_attn
        elif self.ffn_layernorm:
            self._forward = self._forward_ffn_layernorm
        elif self.parallel_residual:
            self._forward = self._forward_parallel_residual
        elif self.context_attn:
            self._forward = self._forward_cross_attn
        else:
            self._forward = self._forward_no_cross_attn

        self.compiled_shapes = set()
        self.hidden_size = decoder_config.hidden_size

        # Gemma4 per-layer scalar (initialized to 1 so it is a no-op for all other models).
        # In HF it is a registered buffer applied as `hidden_states *= self.layer_scalar`
        # at the very end of every decoder layer's forward pass.
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["2", "3"]:
            self._forward_compile = torch.compile(
                self._forward_eager,
                fullgraph=True,
                dynamic=False,
                options={
                    "guard_filter_fn": lambda guards: [g.guard_type == "TENSOR_MATCH" for g in guards],
                    "triton.cudagraphs": EOLE_COMPILE_MODE == "2",
                },
            )

    def _compile_decoder(
        self,
        layer_in=None,
        enc_out=None,
        src_pad_mask=None,
        position_embeddings=None,
        cache_seqlens=None,
    ):
        if layer_in is not None:
            # assumes that _init_cache was before warmup and tiling along beam_size
            B = cache_seqlens.size(0)
            S = enc_out.size(1) if enc_out is not None else 0
            if (B, S) not in self.compiled_shapes:
                dummy_layer_in = torch.randn(B, 1, self.hidden_size, device=layer_in.device, dtype=layer_in.dtype)
                self._forward_compile(
                    dummy_layer_in,
                    enc_out=enc_out,
                    src_pad_mask=src_pad_mask,
                    position_embeddings=position_embeddings,
                    cache_seqlens=cache_seqlens,
                )
            self.compiled_shapes.add((B, S))

    def _forward_ffn_layernorm(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        """Gemma style"""
        kwargs.pop("return_attn", None)
        ff_in = layer_in + self.post_attention_layernorm(self_attn)
        ff_in2 = self.pre_feedforward_layernorm(ff_in)
        ff_in2 = self.mlp(ff_in2)
        layer_out = ff_in + self.post_feedforward_layernorm(ff_in2)
        layer_out = layer_out * self.layer_scalar
        return layer_out, attns

    def _forward_parallel_residual(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        """Phi style"""
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
        """EncoderDecoder style"""
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

    def _forward_linear_attn(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        """Qwen3.5 style (GatedDeltaNet) layers."""
        # Note: self_attn here is already the output of linear_attn (set by _forward_eager)
        self_attn.add_(layer_in)
        ff_in = self.post_attention_layernorm(self_attn)
        return self_attn + self.mlp(ff_in), attns

    def _forward_no_cross_attn(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        """Regular decoder only"""
        kwargs.pop("return_attn", None)
        self_attn.add_(layer_in)
        ff_in = self.post_attention_layernorm(self_attn)
        return self_attn + self.mlp(ff_in), attns

    def forward(self, layer_in, **kwargs):
        if layer_in.size(1) > 1:
            # prefill or training no compile
            return self._forward_eager(layer_in, **kwargs)
        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["2", "3"]:
            return self._forward_compile(layer_in, **kwargs)

        return self._forward_eager(layer_in, **kwargs)

    def _forward_eager(self, layer_in, **kwargs):
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
        return_attn = kwargs.pop("return_attn", False)
        position_embeddings = kwargs.pop("position_embeddings", None)
        cache_seqlens = kwargs.pop("cache_seqlens", None)
        cache_slice = kwargs.pop("cache_slice", None)
        pos_ids_2d = kwargs.pop("pos_ids_2d", None)
        norm_layer_in = self.input_layernorm(layer_in)

        if self.layer_type == "linear_attention":
            self_attn = self.linear_attn(norm_layer_in, attn_mask=attn_mask)
            attns = None
        else:
            self_attn, attns = self.self_attn(
                norm_layer_in,
                attn_mask=attn_mask,
                return_attn=return_attn,
                position_embeddings=position_embeddings,
                cache_seqlens=cache_seqlens,
                cache_slice=cache_slice,
                pos_ids_2d=pos_ids_2d,
            )

        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        return self._forward(layer_in, norm_layer_in, self_attn, attns, return_attn=return_attn, **kwargs)

    def get_attn_align(self, layer_in, **kwargs):
        """:cite:`garg2019jointly`."""
        if self.layer_type == "linear_attention":
            return None
        attns = kwargs.pop("attns", None)
        if self.full_context_alignment:
            # return _, (B, Q_len, K_len)
            _, attns = self.forward(layer_in, **kwargs)
        if attns is None:
            return None
        if self.alignment_heads > 0:
            attns = attns[:, : self.alignment_heads, :, :].contiguous()
        # layer average attention across heads, get ``(B, Q, K)``
        # Case 1: no full_context, no align heads -> layer avg baseline
        # Case 2: no full_context, 1 align heads -> guided align
        # Case 3: full_context, 1 align heads -> full cte guided align
        attn_align = attns.mean(dim=1)

        return attn_align

    def update_dropout(self, dropout, attention_dropout):
        if self.self_attn is not None:
            self.self_attn.update_dropout(attention_dropout)
        if self.context_attn:
            self.context_attn.update_dropout(attention_dropout)
        self.mlp.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        decoder_config (eole.config.TransformerDecoderConfig): full encoder config
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
        context_length = getattr(running_config, "context_length", 0) or 0
        _model_max_pos = getattr(decoder_config, "max_position_embeddings", 8192) or 8192
        if context_length > _model_max_pos:
            raise ValueError(
                f"context_length ({context_length}) exceeds the model's "
                f"max_position_embeddings ({_model_max_pos}). "
                "Reduce context_length to be within the model's capacity."
            )
        self.kvcache_maxsize = context_length if context_length > 0 else _model_max_pos
        self.rope = build_rope(decoder_config, ctx_len=self.kvcache_maxsize)
        if hasattr(decoder_config, "rope_config") and getattr(decoder_config.rope_config, "interleave_local", 0) > 0:
            self.rope_local = build_rope(decoder_config, variant="local", ctx_len=self.kvcache_maxsize)
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
        self.has_linear_attn = any(
            getattr(layer, "layer_type", "full_attention") == "linear_attention" for layer in self.transformer_layers
        )
        self.has_sliding_attn = any(
            getattr(layer, "layer_type", "") == "sliding_attention" for layer in self.transformer_layers
        )
        self.self_attn_backend = getattr(running_config, "self_attn_backend", "flash")
        self.position_encoding_type = decoder_config.position_encoding_type
        self.flash = False
        self.dynamic_shapes = getattr(running_config, "dynamic_shapes", not EOLE_TORCH_COMPILE)
        if self.dynamic_shapes is None:
            self.dynamic_shapes = not EOLE_TORCH_COMPILE
        self.left_pad_attn_mask = None
        self.position_indices = None
        self.cache_seqlens = None
        self.hidden_size = decoder_config.hidden_size
        self.compiled_shapes = set()

        # Chunked prefill configuration.
        # When sliding_window > 0 the window size is the natural chunk boundary;
        # prefill_chunk_size provides a separate knob for non-sliding-window models.
        self.prefill_chunk_size = getattr(running_config, "prefill_chunk_size", 0)

        # Prefill KV cache: stores per-chunk KV tensors on CPU so that
        # repeated prompt prefixes can be served without recomputation.
        prefill_cache_size = getattr(running_config, "prefill_cache_size", 0)
        self._prefill_cache = PrefillCache(max_entries=prefill_cache_size) if prefill_cache_size > 0 else None

        self._disable_cache()

        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["0", "1"]:
            self._forward_compile = torch.compile(
                self._forward_eager,
                fullgraph=True,
                dynamic=False,
                options={
                    "guard_filter_fn": lambda guards: [g.guard_type == "TENSOR_MATCH" for g in guards],
                    "triton.cudagraphs": EOLE_COMPILE_MODE == "0",
                },
            )

    def _compile_decoder(self, emb=None, enc_out=None, src_pad_mask=None, tgt_pad_mask=None, fn_tile=None):

        if emb is not None and tgt_pad_mask is not None:
            B = self.cache_seqlens.size(0)
            S = enc_out.size(1) if enc_out is not None else 0
            if (B, S) not in self.compiled_shapes:
                dummy_emb = torch.randn(B, 1, self.hidden_size, device=emb.device, dtype=emb.dtype)
                dummy_pad_mask = torch.zeros((B, 1), dtype=torch.bool, device=emb.device).unsqueeze(1)
                self._forward_compile(
                    dummy_emb,
                    enc_out=enc_out,
                    src_pad_mask=src_pad_mask,
                    tgt_pad_mask=dummy_pad_mask,
                )
                self.compiled_shapes.add((B, S))

    def init_state(self, **kwargs):
        """Initialize decoder state."""
        pass

    def map_state(self, fn):
        if self.left_pad_attn_mask is not None:
            self.left_pad_attn_mask = fn(self.left_pad_attn_mask)
        if self.cache_seqlens is not None:
            self.cache_seqlens = fn(self.cache_seqlens)
        for layer in self.transformer_layers:
            if layer.layer_type == "linear_attention":
                if layer.linear_attn.conv_state is not None:
                    layer.linear_attn.conv_state = fn(layer.linear_attn.conv_state)
                    layer.linear_attn.recurrent_state = fn(layer.linear_attn.recurrent_state)
            else:
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

    def _causal_attn_mask(self, tgt_pad_mask, prefix_len=None, use_sliding_window=True):
        B, _, seq_len = tgt_pad_mask.size()
        device = tgt_pad_mask.device
        MAX_T = seq_len if (self.training or self.dynamic_shapes) else self.kvcache_maxsize

        # Add triangular future_mask and pad_mask, result mask in (B, T, T).
        future_mask = torch.tril(
            torch.ones((MAX_T, MAX_T), device=tgt_pad_mask.device, dtype=torch.bool),
            diagonal=0,
        )
        if use_sliding_window and self.sliding_window > 0:
            # Use "- sliding_window + 1" to match the chunked-prefill path
            # (_chunk_attn_mask: k >= q - sliding_window + 1) and the single-step
            # decoding path (start = current_step - sliding_window + 1), giving a
            # window of exactly sliding_window tokens (not sliding_window + 1).
            future_mask = future_mask.triu_(-(self.sliding_window - 1))

        future_mask = future_mask.unsqueeze(0).expand(B, -1, -1)  # (B, MAX_T, MAX_T)
        future_mask = future_mask[:, :seq_len, :]  # (B, seq_len, MAX_T)

        if self.LM_type == "prefix" and prefix_len is not None:
            idx = torch.arange(MAX_T, device=device)
            block = (idx.view(1, MAX_T, 1) < prefix_len.view(B, 1, 1)) & (
                idx.view(1, 1, MAX_T) < prefix_len.view(B, 1, 1)
            )
            future_mask = torch.where(block, torch.ones_like(future_mask), future_mask)

        tgt_pad_mask_full = torch.zeros((B, MAX_T), dtype=torch.bool, device=device)
        tgt_pad_mask_full[:, :seq_len] = tgt_pad_mask[:, 0, :].bool()
        tgt_pad_mask_full = tgt_pad_mask_full[:, None, :]  # (B, 1, MAX_T)
        attn_mask = future_mask & ~tgt_pad_mask_full  # True = attend, False = pad

        return attn_mask.unsqueeze(1)  # (B, 1, seq_len, MAX_T)

    def _update_causal_mask(self, attn_mask, image_locations, k_image_locations=None):
        """Update attn_mask so that image tokens can attend to other image tokens.

        Args:
            attn_mask (Tensor): ``(B, 1, q_len, k_len)`` bool mask, True = attend.
            image_locations (Tensor): ``(B, q_len)`` bool, True = image token in
                the query sequence.
            k_image_locations (Tensor, optional): ``(B, k_loc_len)`` bool, True =
                image token in the key sequence.  ``k_loc_len`` must satisfy
                ``k_loc_len <= k_len``.  When ``None`` (the default), defaults to
                ``image_locations`` — the square-mask case used during training
                where query and key sequences are identical.

        Returns:
            Tensor: updated ``attn_mask`` with image-to-image positions forced to
                ``True``.
        """
        B, _, q_len, k_len = attn_mask.shape
        device = attn_mask.device

        if k_image_locations is None:
            k_image_locations = image_locations

        is_q_img = image_locations  # (B, q_len)
        is_k_img = k_image_locations  # (B, k_loc_len)
        k_loc_len = is_k_img.size(1)

        # (B, q_len, k_loc_len): True where both tokens are image tokens
        img_block = is_q_img.unsqueeze(2) & is_k_img.unsqueeze(1)

        # Place into a (B, 1, q_len, k_len) mask, padding the remaining key
        # positions (positions k_loc_len..k_len-1) with False.  This handles
        # the training case where k_len == MAX_T >= seq_len == k_loc_len.
        full_mask = torch.zeros((B, 1, q_len, k_len), dtype=torch.bool, device=device)
        full_mask[:, :, :, :k_loc_len] = img_block.unsqueeze(1)

        return attn_mask | full_mask

    def _chunk_attn_mask(self, chunk_size, current_step, tgt_pad_mask, prefix_len=None, use_sliding_window=True):
        """Create an absolute-position attention mask for a prefill chunk.

        Handles positional/causal masking only.  Image-token type masking is
        applied separately via ``_update_causal_mask`` in ``_forward_eager``.

        Used for all chunks during prefill when a KV cache is present (both
        ``current_step == 0`` for the first chunk and ``current_step > 0`` for
        subsequent ones).  Building the mask with explicit absolute positions
        ensures the key dimension always equals ``cache_len_tgt`` regardless of
        chunk size, and avoids device-to-host syncs that would arise from
        comparing a 0-d CUDA ``current_step`` tensor to a Python scalar.

        Args:
            chunk_size (int): number of query tokens in this chunk.
            current_step (int or Tensor): absolute position of the first query
                token.  A 0-d scalar tensor (``self.cache_seqlens[0]``) or a
                plain Python ``int`` are both accepted; all arithmetic remains
                on-device to avoid unnecessary host synchronisation.
            tgt_pad_mask (Tensor): target padding mask ``(B, 1, chunk_size)``,
                True = padding token.
            prefix_len (Tensor, optional): ``(B,)`` int tensor with per-batch
                prefix lengths.  When provided and ``self.LM_type == "prefix"``
                prefix tokens are allowed to attend bidirectionally to every
                other prefix token, mirroring the logic of
                ``_causal_attn_mask``.
            use_sliding_window (bool): Whether to apply the sliding window
                constraint.  Pass ``False`` to get a full causal mask (used
                for full_attention layers in hybrid models like Gemma4).

        Returns:
            Tensor: boolean mask ``(B, 1, chunk_size, cache_len_tgt)``,
                True = position can be attended to.
        """
        B = tgt_pad_mask.size(0)
        device = tgt_pad_mask.device
        cache_len = self.cache_len_tgt

        # Absolute positions of each query token: shape (chunk_size, 1)
        q_pos = (current_step + torch.arange(chunk_size, device=device)).view(-1, 1)
        # Absolute positions of each cached key: shape (1, cache_len)
        k_pos = torch.arange(cache_len, device=device).view(1, -1)

        # Causal constraint: key position must not exceed query position
        mask = k_pos <= q_pos  # (chunk_size, cache_len)

        if use_sliding_window and self.sliding_window > 0:
            # Sliding window: key position must be within the window.
            # Use "- sliding_window + 1" to match the single-step decoding path
            # (start = current_step - sliding_window + 1), giving a window of
            # exactly sliding_window tokens (not sliding_window + 1).
            window_start = q_pos - self.sliding_window + 1
            mask = mask & (k_pos >= window_start)

        # Apply prefix-LM mask: prefix tokens attend to each other
        # bidirectionally regardless of the causal / sliding-window constraint.
        if self.LM_type == "prefix" and prefix_len is not None:
            plen = prefix_len.view(B, 1, 1)  # (B, 1, 1)
            # q_pos: (chunk_size, 1) -> (1, chunk_size, 1) for broadcasting
            # k_pos: (1, cache_len)  -> (1, 1, cache_len) for broadcasting
            q_in_prefix = q_pos.view(1, chunk_size, 1) < plen  # (B, chunk_size, 1)
            k_in_prefix = k_pos.view(1, 1, cache_len) < plen  # (B, 1, cache_len)
            # Guard: only allow the prefix bidirectional bonus for key positions
            # that have already been written to the KV cache by this chunk.
            # Positions k >= current_step + chunk_size are uninitialized and
            # must remain blocked, even if they fall inside the prefix.
            # Shape: (1, cache_len) — broadcasts over the (B, chunk_size) dims.
            filled_k = k_pos < (current_step + chunk_size)  # (1, cache_len)
            # (chunk_size, cache_len) | (B, chunk_size, cache_len) = (B, chunk_size, cache_len)
            mask = mask.unsqueeze(0) | (q_in_prefix & k_in_prefix & filled_k)
        else:
            mask = mask.unsqueeze(0)  # (1, chunk_size, cache_len) for broadcast below

        # Apply the left-padding mask stored from _init_cache:
        # left_pad_attn_mask[b, j] is True when position j is not a pad token.
        # left_pad_attn_mask unsqueeze(1) is (B, 1, cache_len);
        # broadcasts with mask (B or 1, chunk_size, cache_len).
        mask = mask & self.left_pad_attn_mask[:, :cache_len].unsqueeze(1)
        # mask is now (B, chunk_size, cache_len)

        return mask.unsqueeze(1)  # (B, 1, chunk_size, cache_len)

    def _forward_chunked_prefill(self, emb, src_ids=None, **kwargs):
        """Process a long prefill sequence in chunks.

        When ``sliding_window > 0`` and the input sequence is longer than the
        window, processing the full sequence as a single forward pass would
        allow early tokens to attend to positions they cannot see during
        generation.  When ``prefill_chunk_size > 0`` (and no sliding window is
        active), this method is also used to split long prompts so that their
        KV computations can be cached and reused across requests.

        The sequence is split into consecutive chunks of ``chunk_size`` tokens
        (the last chunk may be shorter) and ``_forward_eager`` is run on each
        chunk so that every token attends only to the positions inside its
        causal window.

        When a :class:`~eole.modules.prefill_cache.PrefillCache` is attached to
        this decoder (``self._prefill_cache is not None``), the output embeddings
        and per-layer state tensors for each chunk are cached on CPU.  On
        subsequent calls whose token prefix produces the same rolling hash, the
        cached data is restored directly into the running cache buffers so that
        the transformer forward pass is skipped entirely for those chunks.

        **What is cached per layer:**

        * Standard attention layers: position-indexed KV slices
          ``(kcache[b, start:end], vcache[b, start:end])``.
        * Linear attention (GatedDeltaNet) layers: the accumulated
          ``conv_state[b]`` and ``recurrent_state[b]`` *after* this chunk.
          These states are a deterministic function of the entire prefix, so
          two requests sharing the same prefix will have identical states at
          every chunk boundary and restoration is safe.

        **Batch size > 1:** cache keys are computed independently for every
        sequence in the batch.  If *all* B sequences produce a cache hit for
        a given chunk, ``_forward_eager`` is skipped for that chunk entirely.
        If any sequence misses, ``_forward_eager`` runs for the whole batch
        and all B results are stored (or overwritten) in the cache.

        The only case where caching is unconditionally disabled is when
        image-token sequences are present, because their attention patterns
        depend on pixel-layout information that is not captured by the
        token-ID hash.

        The KV cache is filled incrementally: after chunk *k* the cache holds
        keys/values for positions ``0 … (k+1)*chunk_size - 1``.

        Args:
            emb (Tensor): ``(B, S, hidden_size)`` prefill embeddings.
            src_ids (Tensor, optional): ``(B, S)`` integer token IDs
                corresponding to *emb*.  Required for cache key computation;
                if ``None``, caching is skipped for the entire prefill.
            **kwargs: forwarded to ``_forward_eager`` unchanged except that
                ``tgt_pad_mask`` is sliced per chunk, ``image_locations`` is
                always passed as the full ``(B, S_full)`` tensor so that
                ``_update_causal_mask`` (called inside ``_forward_eager``) can
                correctly compute cross-chunk image-to-image attention using
                absolute query positions, and ``prefix_len`` is forwarded to
                every chunk unchanged.

        Returns:
            (Tensor, dict): concatenated hidden-state output over all chunks
                and an attention dict whose tensors are concatenated across
                all chunks along the query (tgt) dimension.  Chunks served
                from cache contribute no attention tensors.
        """
        B, S, _ = emb.size()
        # Sliding window takes priority as the chunk size because it is a
        # correctness constraint; prefill_chunk_size is an optional knob.
        chunk_size = self.sliding_window if self.sliding_window > 0 else self.prefill_chunk_size

        tgt_pad_mask_full = kwargs.pop("tgt_pad_mask")
        # Extract image_locations and prefix_len so we can distribute them
        # correctly to each chunk (see per-chunk handling below).
        image_locations = kwargs.pop("image_locations", None)
        prefix_len = kwargs.pop("prefix_len", None)
        return_attn = kwargs.get("return_attn", False) or kwargs.get("with_align", False)

        # Caching is disabled only when image tokens are present (their
        # attention patterns depend on pixel-layout info not captured by
        # the token-ID hash) or when no token IDs were supplied for hashing.
        # Batch size > 1 and hybrid linear-attention models are both
        # supported: see docstring for the full strategy.
        use_cache = (
            self._prefill_cache is not None and src_ids is not None and image_locations is None and not return_attn
        )

        all_emb_chunks = []
        all_std_attns = []
        all_align_attns = []
        all_cross_attns_layers = None  # list-of-lists, one inner list per layer

        # Per-sequence rolling hash keys — one entry per batch element.
        # Starts as None for every sequence; updated after each chunk so that
        # chunk k+1's key depends on chunk k's key (rolling hash).
        prev_keys = [None] * B

        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            chunk_len = end - start

            # ----------------------------------------------------------
            # Try to serve this chunk from the prefill KV cache.
            # ----------------------------------------------------------
            cache_keys = None
            if use_cache:
                src_ids_chunk_cpu = src_ids[:, start:end].cpu()
                # Compute an independent rolling-hash key for each sequence.
                cache_keys = [self._prefill_cache.compute_key(src_ids_chunk_cpu[b], prev_keys[b]) for b in range(B)]
                # A "full hit" means every sequence in the batch has a valid
                # cache entry so _forward_eager can be skipped entirely.
                cached_results = [self._prefill_cache.get(k) for k in cache_keys]
                if all(r is not None for r in cached_results):
                    device = emb.device
                    dtype = emb.dtype
                    emb_outs = []
                    for b, cached in enumerate(cached_results):
                        emb_out_cpu, kv_slices = cached
                        emb_outs.append(emb_out_cpu.to(device=device, dtype=dtype))
                        for layer_idx, layer in enumerate(self.transformer_layers):
                            layer_state = kv_slices[layer_idx]
                            if layer_state is None:
                                continue
                            if layer.layer_type == "linear_attention":
                                # Restore accumulated conv and recurrent states.
                                conv_cpu, rec_cpu = layer_state
                                layer.linear_attn.conv_state[b] = conv_cpu.to(device=device, dtype=dtype)
                                layer.linear_attn.recurrent_state[b] = rec_cpu.to(device=device, dtype=dtype)
                            else:
                                # Restore position-indexed KV slices.
                                k_cpu, v_cpu = layer_state
                                layer.self_attn.kcache[b, start:end, :, :] = k_cpu.to(device=device, dtype=dtype)
                                layer.self_attn.vcache[b, start:end, :, :] = v_cpu.to(device=device, dtype=dtype)
                    # Reassemble batch dimension: list of (chunk_len, hidden)
                    # → (B, chunk_len, hidden).
                    emb_chunk_out = torch.stack(emb_outs, dim=0)
                    all_emb_chunks.append(emb_chunk_out)
                    self.cache_seqlens.add_(chunk_len)
                    prev_keys = list(cache_keys)
                    continue  # skip _forward_eager for this chunk

            # ----------------------------------------------------------
            # Cache miss (or caching disabled): run the forward pass.
            # ----------------------------------------------------------
            emb_chunk = emb[:, start:end, :]
            pad_mask_chunk = tgt_pad_mask_full[:, :, start:end]

            chunk_kwargs = dict(kwargs)
            chunk_kwargs["tgt_pad_mask"] = pad_mask_chunk

            if image_locations is not None:
                # Always pass the full image_locations tensor; _forward_eager
                # will slice query positions using absolute indices when it
                # calls _update_causal_mask for cross-chunk image-to-image
                # attention.
                chunk_kwargs["image_locations"] = image_locations

            if prefix_len is not None:
                # prefix_len is passed to every chunk unchanged; _chunk_attn_mask
                # uses absolute positions so prefix tokens beyond the first chunk
                # are handled correctly in all chunks.
                chunk_kwargs["prefix_len"] = prefix_len

            emb_chunk_out, chunk_attns = self._forward_eager(emb_chunk, **chunk_kwargs)
            all_emb_chunks.append(emb_chunk_out)

            # ----------------------------------------------------------
            # Store each sequence's result in the prefill KV cache.
            # For partial hits (some sequences had a cache entry, some did
            # not), all B results are stored; existing entries are
            # overwritten with the same values (idempotent).
            # ----------------------------------------------------------
            if cache_keys is not None:
                for b in range(B):
                    kv_slices = []
                    for layer in self.transformer_layers:
                        if layer.layer_type == "linear_attention":
                            # Cache end-of-chunk linear-attention states.
                            # These represent the fully accumulated conv and
                            # recurrent state after processing tokens 0..end-1.
                            conv_slice = layer.linear_attn.conv_state[b].clone()
                            rec_slice = layer.linear_attn.recurrent_state[b].clone()
                            kv_slices.append((conv_slice, rec_slice))
                        else:
                            k_slice = layer.self_attn.kcache[b, start:end, :, :].clone()
                            v_slice = layer.self_attn.vcache[b, start:end, :, :].clone()
                            kv_slices.append((k_slice, v_slice))
                    self._prefill_cache.put(cache_keys[b], emb_chunk_out[b].detach(), kv_slices)

            if cache_keys is not None:
                prev_keys = list(cache_keys)

            if chunk_attns.get("std") is not None:
                all_std_attns.append(chunk_attns["std"])
            if chunk_attns.get("align") is not None:
                all_align_attns.append(chunk_attns["align"])
            if chunk_attns.get("cross_attns") is not None:
                if all_cross_attns_layers is None:
                    all_cross_attns_layers = [[] for _ in chunk_attns["cross_attns"]]
                for li, layer_attn in enumerate(chunk_attns["cross_attns"]):
                    all_cross_attns_layers[li].append(layer_attn)

        # Merge per-chunk attention tensors along the query (tgt) dimension.
        attns = {"std": torch.cat(all_std_attns, dim=1) if all_std_attns else None}
        if all_align_attns:
            attns["align"] = torch.cat(all_align_attns, dim=1)
        if all_cross_attns_layers is not None:
            attns["cross_attns"] = [torch.cat(layer_chunks, dim=2) for layer_chunks in all_cross_attns_layers]

        return torch.cat(all_emb_chunks, dim=1), attns

    def forward(self, emb, **kwargs):
        B, S, _ = emb.size()
        src_ids = kwargs.pop("src_ids", None)

        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["0", "1"] and S == 1:
            return self._forward_compile(emb, **kwargs)
        # Determine the effective chunk size for chunked prefill.
        # Sliding window takes precedence (it is a correctness requirement);
        # prefill_chunk_size is an optional performance knob for non-sliding models.
        effective_chunk = self.sliding_window if self.sliding_window > 0 else self.prefill_chunk_size
        if effective_chunk > 0 and S > effective_chunk and self.cache_seqlens is not None:
            return self._forward_chunked_prefill(emb, src_ids=src_ids, **kwargs)
        return self._forward_eager(emb, **kwargs)

    def _forward_eager(self, emb, **kwargs):
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
        _, S, _ = emb.size()
        enc_out = kwargs.pop("enc_out", None)
        tgt_pad_mask = kwargs.pop("tgt_pad_mask", None)
        assert tgt_pad_mask is not None, "TransformerDecoder requires a tgt pad mask"
        src_pad_mask = kwargs.pop("src_pad_mask", None)
        if self.with_cross_attn:
            assert src_pad_mask is not None, "TransformerDecoder requires a src pad mask"
            src_pad_mask = src_pad_mask.unsqueeze(1)  # (batch x 1 x 1 x src_len)
            # dim 1 (heads) and 2 (tgt_len) will be broadcasted automatically in MHA

        with_align = kwargs.pop("with_align", False)
        return_attn = with_align or kwargs.pop("return_attn", False)
        collect_cross_attns = kwargs.pop("collect_cross_attns", False)
        all_cross_attns = [] if (return_attn and collect_cross_attns) else None

        if self.cache_seqlens is not None:
            # prefill & decoding
            current_step = self.cache_seqlens[0]
        else:
            # training
            current_step = 0

        pos_ids_1d = current_step + torch.arange(S, device=emb.device)
        position_embeddings = self.rope.cos_sin

        if self.rope_local is not None:
            position_embeddings_local = self.rope_local.cos_sin
            if self.has_sliding_attn:
                # Gemma4-style: use layer_types to select rope per layer
                # "sliding_attention" → local rope, "full_attention" → global rope
                pos_emb_list = [
                    (
                        position_embeddings_local[pos_ids_1d]
                        if layer.layer_type == "sliding_attention"
                        else position_embeddings[pos_ids_1d]
                    )
                    for layer in self.transformer_layers
                ]
            else:
                # Legacy interleave_local approach (e.g. Gemma3)
                pos_emb_list = [
                    (
                        position_embeddings_local[pos_ids_1d]
                        if (i + 1) % self.interleave_local
                        else position_embeddings[pos_ids_1d]
                    )
                    for i in range(len(self.transformer_layers))
                ]
        elif position_embeddings is not None:
            pos_emb_list = [position_embeddings[pos_ids_1d]] * len(self.transformer_layers)
        else:
            pos_emb_list = [None] * len(self.transformer_layers)

        if not self.flash:
            cache_slice = pos_ids_1d  # tensor of position indices to update
            if S > 1:
                # training or prefill decoding, combine pad and causal mask
                image_locations = kwargs.pop("image_locations", None)
                prefix_len = kwargs.pop("prefix_len", None)

                if self.cache_seqlens is not None:
                    # Prefill with KV cache (including chunked prefill chunk 0):
                    # use an absolute-position mask so the key dimension always
                    # matches cache_len_tgt regardless of chunk size.  This also
                    # avoids a device-to-host sync that `current_step > 0` would
                    # trigger when current_step is a 0-d CUDA tensor.
                    # _chunk_attn_mask handles positional/causal masking only;
                    # image-type masking is applied separately below via
                    # _update_causal_mask with correct absolute query positions.
                    attn_mask = self._chunk_attn_mask(
                        S,
                        current_step,
                        tgt_pad_mask,
                        prefix_len=prefix_len,
                    )
                    # For hybrid sliding/full attention models, also compute a full
                    # (no-sliding-window) mask for full_attention layers.
                    if self.has_sliding_attn:
                        attn_mask_full = self._chunk_attn_mask(
                            S, current_step, tgt_pad_mask, prefix_len=prefix_len, use_sliding_window=False
                        )
                    if image_locations is not None:
                        # Query tokens occupy absolute positions
                        # [current_step, current_step + S); key tokens span the
                        # full [0, cache_len_tgt).  Only key positions that have
                        # already been written to the cache are eligible for the
                        # image-to-image bonus: positions >= current_step+S are
                        # in future chunks (uninitialized) and must be blocked
                        # even if they are flagged as image tokens.  Use an
                        # on-device boolean mask to avoid a host sync on the
                        # 0-d CUDA tensor current_step.
                        cache_len = self.cache_len_tgt
                        q_abs_positions = current_step + torch.arange(S, device=tgt_pad_mask.device)
                        k_positions = torch.arange(cache_len, device=tgt_pad_mask.device)
                        filled_mask = k_positions < (current_step + S)  # (cache_len,)
                        # image_locations may be shorter than cache_len (e.g., static
                        # cache where cache_len_tgt == max_length but image_locations
                        # has shape (B, seq_len)).  Pad to (B, cache_len) before
                        # AND-ing to avoid shape/broadcast errors.
                        img_loc_len = image_locations.size(1)
                        if img_loc_len < cache_len:
                            k_img_full = torch.zeros(
                                image_locations.size(0),
                                cache_len,
                                dtype=torch.bool,
                                device=image_locations.device,
                            )
                            k_img_full[:, :img_loc_len] = image_locations
                        else:
                            k_img_full = image_locations[:, :cache_len]
                        attn_mask = self._update_causal_mask(
                            attn_mask,
                            image_locations[:, q_abs_positions],
                            k_img_full & filled_mask,
                        )
                        if self.has_sliding_attn:
                            attn_mask_full = self._update_causal_mask(
                                attn_mask_full,
                                image_locations[:, q_abs_positions],
                                k_img_full & filled_mask,
                            )
                else:
                    # Training path (no KV cache): _causal_attn_mask handles
                    # the full sequence with MAX_T = seq_len (training) or
                    # self.max_length (static-shape eval without cache).
                    attn_mask = self._causal_attn_mask(tgt_pad_mask, prefix_len=prefix_len)
                    if self.has_sliding_attn:
                        attn_mask_full = self._causal_attn_mask(
                            tgt_pad_mask, prefix_len=prefix_len, use_sliding_window=False
                        )
                    if image_locations is not None:
                        attn_mask = self._update_causal_mask(attn_mask, image_locations)
                        if self.has_sliding_attn:
                            attn_mask_full = self._update_causal_mask(attn_mask_full, image_locations)
                lin_attn_mask = ~tgt_pad_mask[:, 0, :] if self.has_linear_attn else None
            else:
                # at decoding _init_cache must be called and init these
                valid = self.position_indices <= self.cache_seqlens.view(-1, 1)
                valid = valid & self.left_pad_attn_mask
                if self.sliding_window > 0:
                    start = torch.clamp(current_step - self.sliding_window + 1, min=0)
                    valid = valid & (self.position_indices >= start)
                attn_mask = valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, MAX_T or Dynamic Cache Len)
                if self.has_sliding_attn:
                    # full_attention layers: no sliding window restriction
                    valid_full = self.position_indices <= self.cache_seqlens.view(-1, 1)
                    valid_full = valid_full & self.left_pad_attn_mask
                    attn_mask_full = valid_full.unsqueeze(1).unsqueeze(2)
                lin_attn_mask = None
        else:
            attn_mask = None
            attn_mask_full = None
            cache_slice = None  # triggers flash decoding
            if S > 1 and self.has_linear_attn:
                lin_attn_mask = ~tgt_pad_mask[:, 0, :]
            else:
                lin_attn_mask = None

        pos_ids_2d = kwargs.pop("pos_ids_2d", None)
        attn_aligns = []

        for layer, pos_emb in zip(self.transformer_layers, pos_emb_list):
            if lin_attn_mask is not None and layer.layer_type == "linear_attention":
                layer_attn_mask = lin_attn_mask
            elif self.has_sliding_attn and layer.layer_type == "full_attention":
                layer_attn_mask = attn_mask_full
            else:
                layer_attn_mask = attn_mask

            emb, attn = layer(
                emb.clone() if (EOLE_COMPILE_MODE == "2" and EOLE_TORCH_COMPILE) else emb,
                enc_out=enc_out,
                src_pad_mask=src_pad_mask,
                attn_mask=layer_attn_mask,
                return_attn=return_attn,
                position_embeddings=pos_emb,
                cache_seqlens=self.cache_seqlens,
                cache_slice=cache_slice,
                pos_ids_2d=pos_ids_2d,
            )
            if all_cross_attns is not None and attn is not None:
                all_cross_attns.append(attn)
            if with_align:
                attn_align = layer.get_attn_align(
                    emb.clone() if (EOLE_COMPILE_MODE == "2" and EOLE_TORCH_COMPILE) else emb,
                    enc_out=enc_out,
                    src_pad_mask=src_pad_mask,
                    attn_mask=layer_attn_mask,
                    return_attn=return_attn,
                    position_embeddings=pos_emb,
                    cache_seqlens=self.cache_seqlens,
                    cache_slice=cache_slice,
                    pos_ids_2d=pos_ids_2d,
                    attns=attn,
                )
                if attn_align is not None:
                    attn_aligns.append(attn_align)

        emb = self.layer_norm(emb)
        if self.cache_seqlens is not None:
            self.cache_seqlens.add_(S)

        # we take the first head
        top_attn = None if attn is None else attn[:, 0, :, :].contiguous()
        attns = {"std": top_attn}
        if with_align and self.alignment_layer < len(attn_aligns):
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
        if all_cross_attns is not None:
            attns["cross_attns"] = all_cross_attns

        return emb, attns

    def _init_cache(self, emb, pad_mask, enc_out=None):
        b, l, d = emb.size()
        device = emb.device
        dtype = emb.dtype
        if torch.is_autocast_enabled(device.type):
            dtype = torch.get_autocast_dtype(device.type)

        self.flash = (
            _FLASH_ATTN_AVAILABLE
            and self.self_attn_backend == "flash"
            and self.position_encoding_type not in [PositionEncodingType.Relative, PositionEncodingType.Alibi]
            and dtype in [torch.float16, torch.bfloat16]
            and device != torch.device("cpu")
        )

        if self.dynamic_shapes:
            self.cache_len_tgt = l  # kv cache starts at target length and grows
        else:
            self.cache_len_tgt = self.kvcache_maxsize  # kv cache is set to max and remains static

        # We find the index of the first non-pad token (the first '0' or 'False')
        # If the first token is NOT a pad, this returns 0.
        mask_int = pad_mask.squeeze(1).to(torch.int8)
        left_pad_counts = mask_int.argmin(dim=-1).to(torch.int32)
        # Check if the VERY FIRST token is actually a pad.
        # This boolean tensor tells us if left-padding actually exists.
        is_left_padded = pad_mask[:, 0, 0]
        cache_leftpad = torch.where(is_left_padded, left_pad_counts, 0)
        self.position_indices = torch.arange(self.cache_len_tgt, device=device).unsqueeze(0)
        # If cache_leftpad is 0, this mask will effectively do nothing
        self.left_pad_attn_mask = self.position_indices >= cache_leftpad.view(-1, 1)
        self.cache_seqlens = torch.zeros((b,), device=device, dtype=torch.int32)

        for layer in self.transformer_layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.conv_state = torch.zeros(
                    b,
                    layer.linear_attn.conv_dim,
                    layer.linear_attn.conv_kernel_size,
                    dtype=dtype,
                    device=device,
                )
                layer.linear_attn.recurrent_state = torch.zeros(
                    b,
                    layer.linear_attn.num_v_heads,
                    layer.linear_attn.head_k_dim,
                    layer.linear_attn.head_v_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                heads_kv = layer.self_attn.heads_kv
                dph = layer.self_attn.dim_per_head
                layer.self_attn.kcache = torch.zeros((b, self.cache_len_tgt, heads_kv, dph), dtype=dtype, device=device)
                layer.self_attn.vcache = torch.zeros((b, self.cache_len_tgt, heads_kv, dph), dtype=dtype, device=device)
                layer.self_attn.cache_leftpad = cache_leftpad
                if layer.context_attn:
                    # prefill context KV with encoder out
                    layer.context_attn._prefill_cache(enc_out)

    def _extend_cache(self, threshold=1, addzeros=32):
        if self.dynamic_shapes and (self.cache_len_tgt - self.cache_seqlens[0].item()) < threshold:
            for layer in self.transformer_layers:
                if layer.layer_type == "linear_attention":
                    continue
                b, l, h, d = layer.self_attn.kcache.shape
                extend = torch.zeros(
                    b, addzeros, h, d, device=layer.self_attn.kcache.device, dtype=layer.self_attn.kcache.dtype
                )
                layer.self_attn.kcache = torch.cat([layer.self_attn.kcache, extend], dim=1)
                layer.self_attn.vcache = torch.cat([layer.self_attn.vcache, extend], dim=1)
            self.cache_len_tgt += addzeros
            self.position_indices = torch.arange(self.cache_len_tgt, device=layer.self_attn.kcache.device).unsqueeze(0)
            b, _ = self.left_pad_attn_mask.shape
            extend = torch.ones(b, addzeros, device=self.left_pad_attn_mask.device, dtype=torch.bool)
            self.left_pad_attn_mask = torch.cat([self.left_pad_attn_mask, extend], dim=1)

    def _disable_cache(self):
        self.left_pad_attn_mask = None
        self.cache_seqlens = None
        self.flash = False
        for layer in self.transformer_layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.conv_state = None
                layer.linear_attn.recurrent_state = None
            else:
                layer.self_attn.kcache, layer.self_attn.vcache = None, None
                layer.self_attn.cache_leftpad = None
                if layer.context_attn:
                    layer.context_attn.kcache, layer.context_attn.vcache = None, None
