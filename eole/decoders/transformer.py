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

        # Determine the layer type for hybrid architectures (e.g. Qwen3.5).
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
            self.self_attn = SelfMHA(
                decoder_config,
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

        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["2", "3"]:
            self._compile_decoder()

    def _compile_decoder(
        self,
        layer_in=None,
        enc_out=None,
        src_pad_mask=None,
        position_embeddings=None,
        cache_seqlens=None,
    ):
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
        kwargs.pop("return_attn", None)
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

    def _forward_linear_attn(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        """Forward pass for linear attention (GatedDeltaNet) layers."""
        # Note: self_attn here is already the output of linear_attn (set by _forward_eager)
        self_attn.add_(layer_in)
        ff_in = self.post_attention_layernorm(self_attn)
        return self_attn + self.mlp(ff_in), attns

    def _forward_no_cross_attn(self, layer_in, norm_layer_in, self_attn, attns, **kwargs):
        kwargs.pop("return_attn", None)
        self_attn.add_(layer_in)
        ff_in = self.post_attention_layernorm(self_attn)
        return self_attn + self.mlp(ff_in), attns

    def forward(self, layer_in, **kwargs):
        if layer_in.size(1) > 1:
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
        self.has_linear_attn = any(
            getattr(layer, "layer_type", "full_attention") == "linear_attention" for layer in self.transformer_layers
        )
        self.self_attn_backend = getattr(running_config, "self_attn_backend", "flash")
        self.position_encoding_type = decoder_config.position_encoding_type
        self.flash = False
        self.dynamic_shapes = getattr(running_config, "dynamic_shapes", None)
        self.max_length = getattr(running_config, "max_length", 256)
        self.left_pad_attn_mask = None
        self.position_indices = None
        self.cache_seqlens = None
        self.hidden_size = decoder_config.hidden_size
        self.compiled_shapes = set()
        self._disable_cache()

        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["0", "1"]:
            self._compile_decoder()

    def _compile_decoder(self, emb=None, enc_out=None, src_pad_mask=None, tgt_pad_mask=None, fn_tile=None):
        self._forward_compile = torch.compile(
            self._forward_eager,
            fullgraph=True,
            dynamic=False,
            options={
                "guard_filter_fn": lambda guards: [g.guard_type == "TENSOR_MATCH" for g in guards],
                "triton.cudagraphs": EOLE_COMPILE_MODE == "0",
            },
        )

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

    def _causal_attn_mask(self, tgt_pad_mask, prefix_len=None):
        B, _, seq_len = tgt_pad_mask.size()
        device = tgt_pad_mask.device
        MAX_T = seq_len if (self.training or self.dynamic_shapes) else self.max_length

        # Add triangular future_mask and pad_mask, result mask in (B, T, T).
        future_mask = torch.tril(
            torch.ones((MAX_T, MAX_T), device=tgt_pad_mask.device, dtype=torch.bool),
            diagonal=0,
        )
        if self.sliding_window > 0:
            future_mask = future_mask.triu_(-self.sliding_window)

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

        is_q_img = image_locations            # (B, q_len)
        is_k_img = k_image_locations          # (B, k_loc_len)
        k_loc_len = is_k_img.size(1)

        # (B, q_len, k_loc_len): True where both tokens are image tokens
        img_block = is_q_img.unsqueeze(2) & is_k_img.unsqueeze(1)

        # Place into a (B, 1, q_len, k_len) mask, padding the remaining key
        # positions (positions k_loc_len..k_len-1) with False.  This handles
        # the training case where k_len == MAX_T >= seq_len == k_loc_len.
        full_mask = torch.zeros((B, 1, q_len, k_len), dtype=torch.bool, device=device)
        full_mask[:, :, :, :k_loc_len] = img_block.unsqueeze(1)

        return attn_mask | full_mask

    def _chunk_attn_mask(self, chunk_size, current_step, tgt_pad_mask, prefix_len=None):
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

        if self.sliding_window > 0:
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
            k_in_prefix = k_pos.view(1, 1, cache_len) < plen   # (B, 1, cache_len)
            # (chunk_size, cache_len) | (B, chunk_size, cache_len) = (B, chunk_size, cache_len)
            mask = mask.unsqueeze(0) | (q_in_prefix & k_in_prefix)
        else:
            mask = mask.unsqueeze(0)  # (1, chunk_size, cache_len) for broadcast below

        # Apply the left-padding mask stored from _init_cache:
        # left_pad_attn_mask[b, j] is True when position j is not a pad token.
        # left_pad_attn_mask unsqueeze(1) is (B, 1, cache_len);
        # broadcasts with mask (B or 1, chunk_size, cache_len).
        mask = mask & self.left_pad_attn_mask[:, :cache_len].unsqueeze(1)
        # mask is now (B, chunk_size, cache_len)

        return mask.unsqueeze(1)  # (B, 1, chunk_size, cache_len)

    def _forward_chunked_prefill(self, emb, **kwargs):
        """Process a long prefill sequence in chunks of ``sliding_window`` size.

        When ``sliding_window > 0`` and the input sequence is longer than the
        window, processing the full sequence as a single forward pass would
        allow early tokens to attend to positions they cannot see during
        generation.  This method splits the sequence into consecutive chunks
        of exactly ``sliding_window`` tokens (the last chunk may be shorter)
        and runs ``_forward_eager`` on each chunk so that every token attends
        only to the positions inside its window.

        The KV cache is filled incrementally: after chunk *k* the cache holds
        keys/values for positions ``0 … (k+1)*sliding_window - 1``.

        Args:
            emb (Tensor): ``(B, S, hidden_size)`` with ``S > sliding_window``.
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
                all chunks along the query (tgt) dimension.
        """
        _, S, _ = emb.size()
        chunk_size = self.sliding_window

        tgt_pad_mask_full = kwargs.pop("tgt_pad_mask")
        # Extract image_locations and prefix_len so we can distribute them
        # correctly to each chunk (see per-chunk handling below).
        image_locations = kwargs.pop("image_locations", None)
        prefix_len = kwargs.pop("prefix_len", None)

        all_emb_chunks = []
        all_std_attns = []
        all_align_attns = []
        all_cross_attns_layers = None  # list-of-lists, one inner list per layer

        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
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
        if EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE in ["0", "1"] and S == 1:
            return self._forward_compile(emb, **kwargs)
        # When sliding window attention is active and the prefill sequence is
        # longer than the window, split it into window-sized chunks so that
        # every token's effective receptive field matches what it will see
        # during token-by-token generation.
        if self.sliding_window > 0 and S > self.sliding_window and self.cache_seqlens is not None:
            return self._forward_chunked_prefill(emb, **kwargs)
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
                    if image_locations is not None:
                        # Query tokens occupy absolute positions
                        # [current_step, current_step + S); key tokens span the
                        # full [0, cache_len_tgt).  Pass both sides explicitly so
                        # cross-chunk image-to-image attention is handled
                        # correctly (advanced indexing keeps current_step on
                        # device — no host sync).
                        cache_len = self.cache_len_tgt
                        idx = current_step + torch.arange(S, device=tgt_pad_mask.device)
                        attn_mask = self._update_causal_mask(
                            attn_mask,
                            image_locations[:, idx],
                            image_locations[:, :cache_len],
                        )
                else:
                    # Training path (no KV cache): _causal_attn_mask handles
                    # the full sequence with MAX_T = seq_len (training) or
                    # self.max_length (static-shape eval without cache).
                    attn_mask = self._causal_attn_mask(tgt_pad_mask, prefix_len=prefix_len)
                    if image_locations is not None:
                        attn_mask = self._update_causal_mask(attn_mask, image_locations)
                lin_attn_mask = ~tgt_pad_mask[:, 0, :] if self.has_linear_attn else None
            else:
                # at decoding _init_cache must be called and init these
                valid = self.position_indices <= self.cache_seqlens.view(-1, 1)
                valid = valid & self.left_pad_attn_mask
                if self.sliding_window > 0:
                    start = torch.clamp(current_step - self.sliding_window + 1, min=0)
                    valid = valid & (self.position_indices >= start)
                attn_mask = valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, MAX_T or Dynamic Cache Len)
                lin_attn_mask = None
        else:
            attn_mask = None
            cache_slice = None  # triggers flash decoding
            if S > 1 and self.has_linear_attn:
                lin_attn_mask = ~tgt_pad_mask[:, 0, :] if self.has_linear_attn else None
            else:
                lin_attn_mask = None

        pos_ids_2d = kwargs.pop("pos_ids_2d", None)
        attn_aligns = []

        for layer, pos_emb in zip(self.transformer_layers, pos_emb_list):
            layer_attn_mask = (
                lin_attn_mask if (lin_attn_mask is not None and layer.layer_type == "linear_attention") else attn_mask
            )
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
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
        if all_cross_attns is not None:
            attns["cross_attns"] = all_cross_attns

        return emb, attns

    def _init_cache(self, emb, pad_mask, enc_out=None):
        b, l, d = emb.size()
        device = emb.device
        dtype = emb.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()

        self.flash = (
            _FLASH_ATTN_AVAILABLE
            and self.self_attn_backend == "flash"
            and self.position_encoding_type not in [PositionEncodingType.Relative, PositionEncodingType.Alibi]
            and dtype in [torch.float16, torch.bfloat16]
            and device != torch.device("cpu")
        )

        if self.dynamic_shapes is None:
            self.dynamic_shapes = not EOLE_TORCH_COMPILE
        if self.dynamic_shapes:
            self.cache_len_tgt = l  # kv cache starts at target length and grows
        else:
            self.cache_len_tgt = self.max_length  # kv cache is set to max and remains static

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
                layer.self_attn.causal = True
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
