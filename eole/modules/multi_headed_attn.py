""" Multi-Head Attention module """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import skip_init
from torch.distributed import all_reduce
from importlib import import_module
from eole.constants import PositionEncodingType
from .relative_position_bias import relative_matmul, gen_relative_positions, compute_bias
from .alibi_position_bias import AlibiPositionalBias
from .rope import apply_rotary_emb, apply_rotary_pos_emb_xdrope
from eole.constants import LayerNorm


# Help functions to split model dim per head
def bld_to_blhd(x: Tensor, dim_per_head: int) -> Tensor:
    """[batchsize x length x modeldim]
    -> [batchsize x length x heads x dimperhead]
    """
    x_0, x_1, _ = x.size()
    return x.view(x_0, x_1, -1, dim_per_head)


def blhd_to_bld(x: Tensor) -> Tensor:
    """[batchsize x length x heads x dimperhead]
    -> [batchsize x length x modeldim]
    """
    x_0, x_1, x_2, x_3 = x.size()
    return x.reshape(x_0, x_1, x_2 * x_3)


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
        model_config: model_config: eole.config.models.ModelConfig object
        running_config: TrainingConfig or InferenceConfig derived from RunningConfig
        is_decoder: bool, true if called by the Decoder layers
    """

    def __init__(self, model_config, running_config=None, is_decoder: bool = True) -> None:
        super(MultiHeadedAttention, self).__init__()
        self.model_config = model_config
        self.fused_kvq = False
        self.dim_per_head = model_config.dim_per_head
        self.heads = model_config.heads
        self.heads_kv = model_config.heads_kv if model_config.heads_kv is not None else model_config.heads
        self.attn_scaling = model_config.attn_scaling
        self.parallel_gpu = getattr(running_config, "parallel_gpu", 1)

        assert (
            self.dim_per_head * self.heads_kv
        ) % self.parallel_gpu == 0, "Model dimension must be divisible by the number of partitions"
        self.linear_keys = skip_init(
            nn.Linear,
            in_features=model_config.hidden_size,
            out_features=self.dim_per_head * self.heads_kv // self.parallel_gpu,
            bias=model_config.add_qkvbias,
        )
        self.linear_values = skip_init(
            nn.Linear,
            in_features=model_config.hidden_size,
            out_features=self.dim_per_head * self.heads_kv // self.parallel_gpu,
            bias=model_config.add_qkvbias,
        )
        self.linear_query = skip_init(
            nn.Linear,
            in_features=model_config.hidden_size,
            out_features=self.dim_per_head * self.heads // self.parallel_gpu,
            bias=model_config.add_qkvbias,
        )
        self.dropout_p = getattr(running_config, "attention_dropout", [0.0])[0]
        self.dropout = nn.Dropout(self.dropout_p)

        # introduced for gemma3 / Hunyuan-MT
        if model_config.query_norm:
            self.q_norm = LayerNorm[model_config.layer_norm](model_config.head_dim, eps=model_config.norm_eps)
        if model_config.key_norm:
            self.k_norm = LayerNorm[model_config.layer_norm](model_config.head_dim, eps=model_config.norm_eps)
        self.qk_norm_post_rope = model_config.qk_norm_post_rope

        self.final_linear = skip_init(
            nn.Linear,
            in_features=self.dim_per_head * self.heads // self.parallel_gpu,
            out_features=model_config.hidden_size,
            bias=model_config.add_final_linear_bias,
        )
        self.is_decoder = is_decoder
        self.scale = self.attn_scaling**-0.5 if self.is_decoder and self.attn_scaling is not None else None
        self.relative_positions_buckets = model_config.relative_positions_buckets
        self.kcache, self.vcache = None, None
        self.sliding_window = model_config.sliding_window
        # TODO find a cleaner way to initialize?
        self.relative_positions_embeddings = None
        self.relative_attention_bias = None
        self.rotary_interleave = None  # for flash_kvcache without rotary
        self.xdrope_section = None
        if self.relative_positions_buckets > 0:
            self.relative_attention_bias = nn.Embedding(self.relative_positions_buckets, self.heads)
        elif self.position_encoding_type == PositionEncodingType.Relative:
            # https://arxiv.org/pdf/1803.02155.pdf
            # in the paper they suggest either two embeds
            # relative_key / relative_value or only
            # relative_key. We implemented the same embed
            # for both.
            vocab_size = self.n_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(vocab_size, self.dim_per_head)
        elif self.position_encoding_type == PositionEncodingType.Rotary:
            if model_config.rope_config.rotary_dim == 0:
                self.rotary_dim = self.dim_per_head
            else:
                self.rotary_dim = model_config.rope_config.rotary_dim
            self.rotary_interleave = model_config.rope_config.rotary_interleave
            self.xdrope_section = model_config.rope_config.xdrope_section
        elif self.position_encoding_type == PositionEncodingType.Alibi:
            self.alibi = AlibiPositionalBias(self.heads)

        self.maybe_ckpt = checkpoint if "mha" in getattr(running_config, "use_ckpting", []) else lambda f, x: f(x)

        if getattr(running_config, "self_attn_backend", "") == "flash":
            flash_pack = import_module("flash_attn")
            self.flash_attn_func = getattr(flash_pack, "flash_attn_func")
            self.flash_attn_with_kvcache = getattr(flash_pack, "flash_attn_with_kvcache")
            self.flash = True
        else:
            self.flash = False

    def update_dropout(self, dropout: float) -> None:
        self.dropout.p = dropout
        self.dropout_p = dropout

    def _fuse_KVQ(self) -> None:
        if hasattr(self.linear_keys, "weight"):
            self.linear_kvq = skip_init(
                nn.Linear,
                in_features=self.model_config.hidden_size,
                out_features=self.dim_per_head * (self.heads_kv * 2 + self.heads) // self.parallel_gpu,
                bias=self.model_config.add_qkvbias,
            )
            self.linear_kvq.weight = nn.Parameter(
                torch.cat([self.linear_keys.weight, self.linear_values.weight, self.linear_query.weight], 0)
            )
            if self.model_config.add_qkvbias:
                self.linear_kvq.bias = nn.Parameter(
                    torch.cat([self.linear_keys.bias, self.linear_values.bias, self.linear_query.bias], 0)
                )

        elif (
            hasattr(self.linear_keys, "qweight")
            and hasattr(self.linear_values, "qweight")
            and hasattr(self.linear_query, "qweight")
        ):
            w_bit = self.linear_keys.w_bit
            group_size = self.linear_keys.group_size

            awq_cls = self.linear_keys.__class__
            dim = 1 if awq_cls.__name__ == "WQLinear_GEMM" else 0

            self.linear_kvq = awq_cls(
                w_bit=w_bit,
                group_size=group_size,
                in_features=self.model_config.hidden_size,
                out_features=self.dim_per_head * (self.heads_kv * 2 + self.heads) // self.parallel_gpu,
                bias=self.model_config.add_qkvbias,
                dev=self.linear_keys.qweight.device,
            )
            self.linear_kvq.qweight = torch.cat(
                [self.linear_keys.qweight, self.linear_values.qweight, self.linear_query.qweight], dim
            )
            self.linear_kvq.qzeros = torch.cat(
                [self.linear_keys.qzeros, self.linear_values.qzeros, self.linear_query.qzeros], dim
            )
            self.linear_kvq.scales = torch.cat(
                [self.linear_keys.scales, self.linear_values.scales, self.linear_query.scales], dim
            )

        del self.linear_keys
        del self.linear_values
        del self.linear_query
        self.fused_kvq = True

    def _prepare_inputs(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        step: Optional[int] = 0,
        position_embeddings=None,
        pos_ids=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepare inputs for attention computation.
        This method performs the following steps:
        1. Applies linear transformations to key, value, and query inputs.
        2. Reshapes the tensors to the multi-head attention format.
        3. Applies rotary position encoding if configured.

        Args:
            key (Tensor): The key tensor of shape [batch_size, seq_len, hidden_size].
            value (Tensor): The value tensor of shape [batch_size, seq_len, hidden_size].
            query (Tensor): The query tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Processed key, value, and query tensors, each of shape
            [batch_size, seq_len, num_heads, dim_per_head].
        """
        # Retrieve keys and values from linear layers (training mode).
        if self.fused_kvq:  # only applicable for self_MHA and inference
            kvq = self.maybe_ckpt(self.linear_kvq, query)
            kvdim = self.dim_per_head * self.heads_kv // self.parallel_gpu
            key = kvq[:, :, :kvdim]
            value = kvq[:, :, kvdim : 2 * kvdim]
            query = kvq[:, :, 2 * kvdim :]
        else:
            key = self.maybe_ckpt(self.linear_keys, key)
            value = self.maybe_ckpt(self.linear_values, value)
            query = self.maybe_ckpt(self.linear_query, query)
        key = bld_to_blhd(key, self.dim_per_head)
        value = bld_to_blhd(value, self.dim_per_head)
        query = bld_to_blhd(query, self.dim_per_head)

        if not self.qk_norm_post_rope:
            if hasattr(self, "q_norm"):
                query = self.q_norm(query)
            if hasattr(self, "k_norm"):
                key = self.k_norm(key)

        if self.position_encoding_type == PositionEncodingType.Rotary:
            seq_len = query.size(1)
            cos_sin = position_embeddings[step : step + seq_len]
            # XDRoPE is only applied at step 0 (initial forward pass) because it is designed for full-sequence encoding.
            # For subsequent steps (e.g., during autoregressive decoding), standard RoPE is used.
            if step == 0 and self.xdrope_section is not None and pos_ids is not None:
                query, key = apply_rotary_pos_emb_xdrope(query, key, cos_sin, pos_ids, self.xdrope_section)
            else:
                query, key = apply_rotary_emb(query, key, cos_sin, interleave=self.rotary_interleave)
            if self.qk_norm_post_rope:
                if hasattr(self, "q_norm"):
                    query = self.q_norm(query)
                if hasattr(self, "k_norm"):
                    key = self.k_norm(key)

        return key, value, query

    def _compute_attention(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attn: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the context vector and the attention vectors.

        Args:
           key (Tensor): set of `key_len`
               key vectors ``(batch, key_len, head, dim)``
           value (Tensor): set of `key_len`
               value vectors ``(batch, key_len, head, dim)``
           query (Tensor): set of `query_len`
               query vectors  ``(batch, query_len, head, dim)``
           attn_mask (bool Tensor): True = position needs attention
                   ``(batch, 1, query_len, key_len)``
        Returns:
           (Tensor, Tensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        b, l, h, d = key.size()
        if self.heads_kv < self.heads:
            qh = query.size(2)
            # expand key on heads dimension when it's less than query heads (multi-query variant)
            key = key.view(b, l, -1, 1, d).repeat(1, 1, 1, qh // h, 1).view(b, l, qh, d)

            # expand value on heads dimension when it's less than query heads (multi-query variant)
            value = value.view(b, l, -1, 1, d).repeat(1, 1, 1, qh // h, 1).view(b, l, qh, d)

        # 2) When standard pos. enc. or rotary, use flash attention or SDPA
        if (
            self.position_encoding_type not in [PositionEncodingType.Relative, PositionEncodingType.Alibi]
            and not return_attn
            and query.device.type != "cpu"
        ):

            # Apply pytorch scaled_dot_product_attention (expects b, h, l, d)
            attn_output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout_p,
                scale=self.scale,
            ).transpose(1, 2)
            # Transpose back to b, l, h, d
            attn = None
        else:
            # Manual attention computation
            scale = self.dim_per_head**-0.5 if self.scale is None else self.scale

            scores = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(2, 3)) * scale

            if self.relative_attention_bias is not None:
                q_len = key.size(1) if self.kcache is not None else query.size(1)
                relative_position_bucket = compute_bias(
                    q_len,
                    key.size(1),
                    self.is_decoder,
                    self.n_positions,
                    self.relative_positions_buckets,
                    device=key.device,
                )
                values = self.relative_attention_bias(
                    relative_position_bucket
                )  # shape (query_length, key_length, num_heads)
                position_bias = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
                if self.kcache is not None:
                    position_bias = position_bias[:, :, -query.size(1) :, :]
                scores.add_(position_bias)

            elif self.relative_positions_embeddings is not None:
                key_len = key.size(1)
                # 1 or key_len x key_len
                relative_positions_matrix = gen_relative_positions(
                    key_len,
                    self.n_positions,
                    cache=self.kcache is not None,
                    device=key.device,
                )
                #  1 or key_len x key_len x dim_per_head
                relations_keys = self.relative_positions_embeddings(relative_positions_matrix)
                scores.add_(relative_matmul(query.transpose(1, 2), relations_keys, True))
            elif self.position_encoding_type == PositionEncodingType.Alibi:
                scores = self.alibi(scores)

            scores = scores.float()

            if attn_mask is not None:
                if len(attn_mask.size()) == 2:
                    attn_mask = attn_mask[None, None, :, :]
                # not 100% necessary but expand to nb of heads
                attn_mask = attn_mask.expand(-1, self.heads // self.parallel_gpu, -1, -1)
                # now mask and scores have the same shape
                scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

            # 3) Apply attention dropout and compute context vectors.
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            drop_attn = self.dropout(attn) if self.dropout_p > 0 else attn

            attn_output = torch.matmul(drop_attn, value.transpose(1, 2)).transpose(1, 2)

            if self.relative_positions_embeddings is not None:
                # We use the same embeddings for key and value
                relations_values = relations_keys
                attn_output.add_(relative_matmul(drop_attn, relations_values, False).transpose(1, 2))

        context = blhd_to_bld(attn_output)

        if self.kcache is not None:
            attn_output = self.final_linear(context)
        else:
            attn_output = self.maybe_ckpt(self.final_linear, context)

        if self.parallel_gpu > 1:
            # all_reduce is an inplace op - not easily backprop
            attn_output1 = attn_output.detach().clone()
            all_reduce(attn_output1)
            attn_output.copy_(attn_output1 + (attn_output - attn_output.detach()))

        return attn_output, attn


class SelfMHA(MultiHeadedAttention):
    def __init__(self, model_config, running_config=None, is_decoder: bool = True) -> None:
        self.position_encoding_type = model_config.position_encoding_type
        self.n_positions = model_config.n_positions
        super(SelfMHA, self).__init__(model_config, running_config, is_decoder)

    def _expand_cache(self, add_length: int, step: int, key: Tensor) -> int:
        if step == 0:
            self.kcache, self.vcache = key.new_zeros(key.shape), key.new_zeros(key.shape)
        b, l, h, dph = self.kcache.shape

        if step >= l:
            ktype = self.kcache.dtype
            kdev = self.kcache.device
            self.kcache = torch.cat(
                (
                    self.kcache,
                    torch.zeros((b, add_length, h, dph), device=kdev, dtype=ktype),
                ),
                dim=1,
            )
            self.vcache = torch.cat(
                (
                    self.vcache,
                    torch.zeros((b, add_length, h, dph), device=kdev, dtype=ktype),
                ),
                dim=1,
            )
        if self.sliding_window > 0 and step > self.sliding_window - 1:
            self.kcache = self.kcache[:, 1:, :, :]
            self.vcache = self.vcache[:, 1:, :, :]
            return self.sliding_window
        else:
            return step + 1

    def _update_cache_w_inputs(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        step: Optional[int] = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if step == 0:
            # init cache with initial cat(key, value) on batch_size dim
            self.kcache, self.vcache = key, value
            return key, value, query
        else:
            cache_len = self._expand_cache(32, step, key)
            self.kcache[:, cache_len - 1, :, :] = key[:, 0, :, :]
            self.vcache[:, cache_len - 1, :, :] = value[:, 0, :, :]
            return self.kcache[:, :cache_len, :, :], self.vcache[:, :cache_len, :, :], query

    def forward(
        self,
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
        position_embeddings=None,
        pos_ids=None,
    ) -> Tuple[Tensor, Tensor]:

        key, value, query = super()._prepare_inputs(
            query,
            query,
            query,
            step=0 if step is None else step,
            position_embeddings=position_embeddings,
            pos_ids=pos_ids,
        )
        if self.kcache is not None:
            # Inference step decoding
            if (
                not self.flash
                or self.position_encoding_type in [PositionEncodingType.Relative, PositionEncodingType.Alibi]
                or query.dtype not in [torch.float16, torch.bfloat16]  # to match with flash
                or query.device == torch.device("cpu")
            ):
                key, value, query = self._update_cache_w_inputs(
                    query,
                    key,
                    value,
                    step=step,
                )

            else:
                # Fast path with flash_attn_with_kvcache
                cache_len = self._expand_cache(32, step, key)
                # restore initial tgt_pad_mask - migth be better to store it instead.
                cache_leftpad = (
                    (~torch.any(attn_mask, dim=-2).squeeze(1)).sum(dim=1).to(torch.int32)
                    if attn_mask is not None
                    else None
                )
                context = self.flash_attn_with_kvcache(
                    query,
                    self.kcache[:, :, :, :],
                    self.vcache[:, :, :, :],
                    key,
                    value,
                    rotary_cos=None,
                    rotary_sin=None,
                    cache_seqlens=cache_len - 1,
                    cache_leftpad=cache_leftpad,
                    softmax_scale=self.scale,
                    causal=step == 0,
                    rotary_interleaved=self.rotary_interleave,
                )
                attn_output = self.final_linear(blhd_to_bld(context))
                if self.parallel_gpu > 1:
                    # all_reduce is an inplace op - not easily backprop
                    attn_output1 = attn_output.detach().clone()
                    all_reduce(attn_output1)
                    attn_output.copy_(attn_output1 + (attn_output - attn_output.detach()))
                return attn_output, None

        return super()._compute_attention(
            key,
            value,
            query,
            attn_mask=attn_mask,
            return_attn=return_attn,
        )


class ContextMHA(MultiHeadedAttention):
    def __init__(
        self,
        model_config,
        running_config=None,
    ) -> None:
        self.position_encoding_type = None
        self.n_positions = 0
        super(ContextMHA, self).__init__(model_config, running_config, True)

    def _prepare_inputs_w_cache(self, key: Tensor, value: Tensor, query: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        query = self.linear_query(query)
        query = bld_to_blhd(query, self.dim_per_head)
        if self.kcache.numel() == 0:
            key, value = self.linear_keys(key), self.linear_values(value)
            self.kcache, self.vcache = bld_to_blhd(key, self.dim_per_head), bld_to_blhd(value, self.dim_per_head)
        key, value = self.kcache, self.vcache
        return key, value, query

    def forward(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
    ) -> Tuple[Tensor, Tensor]:
        if self.kcache is not None:
            # inference: we fill the cross-attention cache only once
            key, value, query = self._prepare_inputs_w_cache(key, value, query)
        else:
            # training: we project key, value query
            key, value, query = super()._prepare_inputs(key, value, query)

        return super()._compute_attention(
            key,
            value,
            query,
            attn_mask=attn_mask,
            return_attn=return_attn,
        )
