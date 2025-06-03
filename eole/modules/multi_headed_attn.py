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
from .rope import apply_rotary_emb

from eole.constants import LayerNorm


# Help functions to split model dim per head
def shape(x: Tensor, dim_per_head: int) -> Tensor:
    """[batchsize x length x modeldim]
    -> [batchsize x heads x length x dimperhead]
    """
    # patch for images
    if len(x.size()) == 2:
        x_0 = 1
        x_1, _ = x.size()
    else:
        x_0, x_1, _ = x.size()
    return x.view(x_0, x_1, -1, dim_per_head).transpose(1, 2)


def unshape(x: Tensor) -> Tensor:
    """[batchsize x heads x length x dimperhead]
    -> [batchsize x length x modeldim]
    """
    x_0, x_1, _, x_3 = x.size()
    return x.transpose(1, 2).reshape(x_0, -1, x_1 * x_3)


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

        # introduced for gemma3
        if model_config.query_norm:
            self.q_norm = LayerNorm[model_config.layer_norm](model_config.dim_per_head, eps=model_config.norm_eps)
        if model_config.key_norm:
            self.k_norm = LayerNorm[model_config.layer_norm](model_config.dim_per_head, eps=model_config.norm_eps)

        self.final_linear = skip_init(
            nn.Linear,
            in_features=self.dim_per_head * self.heads // self.parallel_gpu,
            out_features=model_config.hidden_size,
            bias=model_config.add_final_linear_bias,
        )
        
        self.is_decoder = is_decoder
        self.image_generation = getattr(running_config, "image_generation", False)

        if self.image_generation and self.is_decoder:
            # initialize MOE GEN params
            self.linear_keys_moe_gen = skip_init(
                nn.Linear,
                in_features=model_config.hidden_size,
                out_features=self.dim_per_head * self.heads_kv // self.parallel_gpu,
                bias=model_config.add_qkvbias,
            )
            self.linear_values_moe_gen = skip_init(
                nn.Linear,
                in_features=model_config.hidden_size,
                out_features=self.dim_per_head * self.heads_kv // self.parallel_gpu,
                bias=model_config.add_qkvbias,
            )
            self.linear_query_moe_gen = skip_init(
                nn.Linear,
                in_features=model_config.hidden_size,
                out_features=self.dim_per_head * self.heads // self.parallel_gpu,
                bias=model_config.add_qkvbias,
            )
            if model_config.query_norm:
                self.q_norm_moe_gen = LayerNorm[model_config.layer_norm](model_config.dim_per_head, eps=model_config.norm_eps)
            if model_config.key_norm:
                self.k_norm_moe_gen = LayerNorm[model_config.layer_norm](model_config.dim_per_head, eps=model_config.norm_eps)
            
            self.final_linear_moe_gen = skip_init(
                nn.Linear,
                in_features=self.dim_per_head * self.heads // self.parallel_gpu,
                out_features=model_config.hidden_size,
                bias=model_config.add_final_linear_bias,
            )


        
        self.scale = self.attn_scaling**-0.5 if self.is_decoder and self.attn_scaling is not None else None
        self.relative_positions_buckets = model_config.relative_positions_buckets
        self.kcache, self.kcache = None, None
        self.sliding_window = model_config.sliding_window
        # TODO find a cleaner way to initialize?
        self.relative_positions_embeddings = None
        self.relative_attention_bias = None
        self.rotary_interleave = None  # for flash_kvcache without rotary
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

    def _prepare_inputs(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        position_embeddings=None,
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
            [batch_size, num_heads, seq_len, dim_per_head].
        """
        # Retrieve keys and values from linear layers (training mode).
        key = self.maybe_ckpt(self.linear_keys, key)
        value = self.maybe_ckpt(self.linear_values, value)
        query = self.maybe_ckpt(self.linear_query, query)
        key = shape(key, self.dim_per_head)
        value = shape(value, self.dim_per_head)
        query = shape(query, self.dim_per_head)

        if self.position_encoding_type == PositionEncodingType.Rotary:
            seqlen = query.size(2)
            cos, sin = position_embeddings[0][:seqlen], position_embeddings[1][:seqlen]
            query, key = apply_rotary_emb(query, key, (cos, sin), interleave=self.rotary_interleave)

        return key, value, query

    def _compute_attention(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        return_attn: Optional[bool] = False,
        text_indices: Optional[Tensor] = None,
        image_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the context vector and the attention vectors.

        Args:
           key (Tensor): set of `key_len`
               key vectors ``(batch, head, key_len, dim)``
           value (Tensor): set of `key_len`
               value vectors ``(batch, head, key_len, dim)``
           query (Tensor): set of `query_len`
               query vectors  ``(batch, head, query_len, dim)``
           attn_mask (bool Tensor): True = position needs attention
                   ``(batch, 1, query_len, key_len)``
        Returns:
           (Tensor, Tensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        b, h, l, d = key.size()
        # replaced by enable_gqa?
        if self.heads_kv < self.heads:
            qh = query.size(1)
            # expand key on heads dimension when it's less than query heads (multi-query variant)
            key = key.view(b, -1, 1, l, d).repeat(1, 1, qh // h, 1, 1).view(b, qh, l, d)

            # expand value on heads dimension when it's less than query heads (multi-query variant)
            value = value.view(b, -1, 1, l, d).repeat(1, 1, qh // h, 1, 1).view(b, qh, l, d)

        # 2) When standard pos. enc. or rotary, use flash attention or SDPA
        if (
            self.position_encoding_type not in [PositionEncodingType.Relative, PositionEncodingType.Alibi]
            and not return_attn
            and query.device.type != "cpu"
        ):

            # Apply pytorch scaled_dot_product_attention.
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                # is_causal=False,
                dropout_p=self.dropout_p,
                scale=self.scale,
                # enable_gqa=True, # not memory efficient -- https://github.com/pytorch/pytorch/issues/154363
            )

            # print("ATTN OUTPUT sdpa:", attn_output.size(), attn_output.sum(), attn_output)
            # print("ATTN OUTPUT sdpa img:", attn_output[:, :, -4098:, :].size(), attn_output[:, :, -4098:, :].sum(), attn_output[:, :, -4098:, :])

            attn = None
        else:
            # batch x num_heads x query_len x key_len
            scale = self.dim_per_head**-0.5 if self.scale is None else self.scale
            scores = torch.matmul(query, key.transpose(2, 3)) * scale

            if self.relative_attention_bias is not None:
                q_len = key.size(2) if self.kcache is not None else query.size(2)
                relative_position_bucket = compute_bias(
                    q_len,
                    key.size(2),
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
                    position_bias = position_bias[:, :, -query.size(2) :, :]
                scores.add_(position_bias)

            elif self.relative_positions_embeddings is not None:
                key_len = key.size(2)
                # 1 or key_len x key_len
                relative_positions_matrix = gen_relative_positions(
                    key_len,
                    self.n_positions,
                    cache=self.kcache is not None,
                    device=key.device,
                )
                #  1 or key_len x key_len x dim_per_head
                relations_keys = self.relative_positions_embeddings(relative_positions_matrix)
                scores.add_(relative_matmul(query, relations_keys, True))
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

            attn_output = torch.matmul(drop_attn, value)

            if self.relative_positions_embeddings is not None:
                # We use the same embeddings for key and value
                relations_values = relations_keys
                attn_output.add_(relative_matmul(drop_attn, relations_values, False))

        context = unshape(attn_output)

        # TODO: enable MOE stuff here

        if self.image_generation:
            assert text_indices is not None and image_indices is not None, "text_indices and image_indices must be provided for image generation"
            attn_output = torch.zeros_like(context)
            attn_output[:, text_indices, :] = self.final_linear(context[:, text_indices, :])
            attn_output[:, image_indices, :] = self.final_linear_moe_gen(context[:, image_indices, :])
        elif self.kcache is not None:
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
        b, h, l, dph = self.kcache.shape

        if step >= l:
            ktype = self.kcache.dtype
            kdev = self.kcache.device
            self.kcache = torch.cat(
                (
                    self.kcache,
                    torch.zeros((b, h, add_length, dph), device=kdev, dtype=ktype),
                ),
                dim=2,
            )
            self.vcache = torch.cat(
                (
                    self.vcache,
                    torch.zeros((b, h, add_length, dph), device=kdev, dtype=ktype),
                ),
                dim=2,
            )
        if self.sliding_window > 0 and step > self.sliding_window - 1:
            self.kcache = self.kcache[:, :, 1:, :]
            self.vcache = self.vcache[:, :, 1:, :]
            return self.sliding_window
        else:
            return step + 1

    def _prepare_inputs_w_cache(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        step: Optional[int] = 0,
        position_embeddings=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if self.position_encoding_type == PositionEncodingType.Rotary:
            seqlen = query.size(2)
            cos, sin = position_embeddings[0][step : step + seqlen].to(query.dtype), position_embeddings[1][step : step + seqlen].to(query.dtype)
            query, key = apply_rotary_emb(query, key, (cos, sin), interleave=self.rotary_interleave)

            _query = query.transpose(1, 2).squeeze(0)
            _key = key.transpose(1, 2).squeeze(0)

            # print("ROTARY QUERY:", _query.size(), _query.sum(), _query)
            # print("ROTARY KEY:", _key.size(), _key.sum(), _key)
            # print("ROTARY QUERY img:", _query[-4098:, :, :].size(), _query[-4098:, :, :].sum(), _query[-4098:, :, :])
            # print("ROTARY KEY img:", _key[-4098:, :, :].size(), _key[-4098:, :, :].sum(), _key[-4098:, :, :])

        if step == 0:
            # init cache with initial cat(key, value) on batch_size dim
            self.kcache, self.vcache = key, value
            return key, value, query
        else:
            cache_len = self._expand_cache(32, step, key)
            self.kcache[:, :, cache_len - 1, :] = key[:, :, 0, :]
            self.vcache[:, :, cache_len - 1, :] = value[:, :, 0, :]
            return self.kcache[:, :, :cache_len, :], self.vcache[:, :, :cache_len, :], query

    def forward(
        self,
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        step: Optional[int] = 0,
        return_attn: Optional[bool] = False,
        position_embeddings=None,
        text_indices: Optional[Tensor] = None,
        image_indices: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if self.kcache is not None:
            # Inference step decoding
            if self.image_generation:

                # _query = query.transpose(1, 2).squeeze(0)
                _query = query.squeeze(0)
                # print("INPUT QUERY:", _query.size(), _query.sum(), _query)
                # print("INPUT QUERY:", _query[:76].size(), _query[:76].sum(), _query[:76])
                # print("INPUT QUERY img:", _query[-4098:, :].size(), _query[-4098:, :].sum(), _query[-4098:, :])

                text_query = query[:, text_indices, :]
                image_query = query[:, image_indices, :]

                # print("TEXT QUERY:", text_query.size(), text_query.sum(), text_query)
                # print("IMAGE QUERY:", image_query.size(), image_query.sum(), image_query)

                text_key = self.linear_keys(text_query)
                image_key = self.linear_keys_moe_gen(image_query)
                text_value = self.linear_values(text_query)
                image_value = self.linear_values_moe_gen(image_query)
                text_query = self.linear_query(text_query)
                image_query = self.linear_query_moe_gen(image_query)
                
                # prompt_text_key = self.linear_keys(prompt_text_query)
                # prompt_text_value = self.linear_values(prompt_text_query)
                # prompt_text_query = self.linear_query(prompt_text_query)
                # print("PROMPT TEXT QUERY:", prompt_text_query.size(), prompt_text_query.sum(), prompt_text_query)
                # print("PROMPT TEXT KEY:", prompt_text_key.size(), prompt_text_key.sum(), prompt_text_key)
                # print("PROMPT TEXT VALUE:", prompt_text_value.size(), prompt_text_value.sum(), prompt_text_value)

                # prompt_text_query = shape(prompt_text_query, self.dim_per_head).transpose(1, 2).squeeze(0)
                # prompt_text_key = shape(prompt_text_key, self.dim_per_head).transpose(1, 2).squeeze(0)
                # prompt_text_value = shape(prompt_text_value, self.dim_per_head).transpose(1, 2).squeeze(0)
                # print("PROMPT TEXT QUERY:", prompt_text_query.size(), prompt_text_query.sum(), prompt_text_query)
                # print("PROMPT TEXT KEY:", prompt_text_key.size(), prompt_text_key.sum(), prompt_text_key)
                # print("PROMPT TEXT VALUE:", prompt_text_value.size(), prompt_text_value.sum(), prompt_text_value)

                text_query = shape(text_query, self.dim_per_head)
                image_query = shape(image_query, self.dim_per_head)
                text_key = shape(text_key, self.dim_per_head)
                image_key = shape(image_key, self.dim_per_head)
                text_value = shape(text_value, self.dim_per_head)
                image_value = shape(image_value, self.dim_per_head)

                # print("TEXT QUERY:", text_query.size(), text_query.sum(), text_query)
                # print("IMAGE QUERY:", image_query.size(), image_query.sum(), image_query)
                # print("TEXT KEY:", text_key.size(), text_key.sum(), text_key)
                # print("IMAGE KEY:", image_key.size(), image_key.sum(), image_key)
                # print("TEXT VALUE:", text_value.size(), text_value.sum(), text_value)
                # print("IMAGE VALUE:", image_value.size(), image_value.sum(), image_value)


                text_query_ = text_query.transpose(1, 2).squeeze(0)
                text_key_ = text_key.transpose(1, 2).squeeze(0)
                text_value_ = text_value.transpose(1, 2).squeeze(0)

                # print("PROMPT TEXT QUERY:", text_query_[:76].size(), text_query_[:76].sum(), text_query_[:76])
                # print("PROMPT TEXT KEY:", text_key_[:76].size(), text_key_[:76].sum(), text_key_[:76])
                # print("PROMPT TEXT VALUE:", text_value_[:76].size(), text_value_[:76].sum(), text_value_[:76])
                
                key = query.new_zeros((query.size(0), self.heads_kv, query.size(1), self.dim_per_head // self.parallel_gpu))
                value = query.new_zeros((query.size(0), self.heads_kv, query.size(1), self.dim_per_head // self.parallel_gpu))
                query = query.new_zeros((query.size(0), self.heads, query.size(1), self.dim_per_head // self.parallel_gpu))

                key[:, :, text_indices, :] = text_key
                key[:, :, image_indices, :] = image_key
                value[:, :, text_indices, :] = text_value
                value[:, :, image_indices, :] = image_value

                query[:, :, text_indices, :] = text_query
                query[:, :, image_indices, :] = image_query

                # print("QUERY:", query.size(), query.sum(), query)
                # print("KEY:", key.size(), key.sum(), key)
                # print("VALUE:", value.size(), value.sum(), value)

                query_ = query.transpose(1, 2).squeeze(0)
                key_ = key.transpose(1, 2).squeeze(0)
                value_ = value.transpose(1, 2).squeeze(0)

                # print("QUERY img:", query_[-4098:, :, :].size(), query_[-4098:, :, :].sum(), query_[-4098:, :, :])
                # print("KEY img:", key_[-4098:, :, :].size(), key_[-4098:, :, :].sum(), key_[-4098:, :, :])
                # print("VALUE img:", value_[-4098:, :, :].size(), value_[-4098:, :, :].sum(), value_[-4098:, :, :])

                # print("MERGED KEY:", key.size(), key.sum(), key)
                # print("MERGED VALUE:", value.size(), value.sum(), value)
                # print("MERGED QUERY:", query.size(), query.sum(), query)

                # key = shape(key, self.dim_per_head)
                # value = shape(value, self.dim_per_head)
                # query = shape(query, self.dim_per_head)

                if hasattr(self, "q_norm") and hasattr(self, "q_norm_moe_gen"):
                    query[:, :, text_indices, :] = self.q_norm(query[:, :, text_indices, :])
                    query[:, :, image_indices, :] = self.q_norm_moe_gen(query[:, :, image_indices, :])
                if hasattr(self, "k_norm") and hasattr(self, "k_norm_moe_gen"):
                    key[:, :, text_indices, :] = self.k_norm(key[:, :, text_indices, :])
                    key[:, :, image_indices, :] = self.k_norm_moe_gen(key[:, :, image_indices, :])

                _key = key.transpose(1, 2).squeeze(0)
                _value = value.transpose(1, 2).squeeze(0)
                _query = query.transpose(1, 2).squeeze(0)

                # print("PROMPT TEXT KEY:", _key[:76].size(), _key[:76].sum(), _key[:76])
                # print("PROMPT TEXT VALUE:", _value[:76].size(), _value[:76].sum(), _value[:76])
                # print("PROMPT TEXT QUERY:", _query[:76].size(), _query[:76].sum(), _query[:76])

                # print("QUERY NORM:", _query.size(), _query.sum(), _query)
                # print("KEY NORM:", _key.size(), _key.sum(), _key)
                # print("QUERY NORM img:", _query[-4098:, :, :].size(), _query[-4098:, :, :].sum(), _query[-4098:, :, :])
                # print("KEY NORM img:", _key[-4098:, :, :].size(), _key[-4098:, :, :].sum(), _key[-4098:, :, :])
                # print("VALUE:", _value.size(), _value.sum(), _value)

            else:
                key = self.linear_keys(query)
                value = self.linear_values(query)
                query = self.linear_query(query)

                key = shape(key, self.dim_per_head)
                value = shape(value, self.dim_per_head)
                query = shape(query, self.dim_per_head)

                if hasattr(self, "q_norm"):
                    query = self.q_norm(query)
                if hasattr(self, "k_norm"):
                    key = self.k_norm(key)

            if (
                not self.flash
                or self.position_encoding_type in [PositionEncodingType.Relative, PositionEncodingType.Alibi]
                or query.dtype not in [torch.float16, torch.bfloat16]  # to match with flash
                or query.device == torch.device("cpu")
            ):
                key, value, query = self._prepare_inputs_w_cache(
                    query,
                    key,
                    value,
                    step=step,
                    position_embeddings=position_embeddings,
                )
            else:
                # Fast path with flash_attn_with_kvcache
                cache_len = self._expand_cache(32, step, key)
                if position_embeddings is not None:
                    rotdim = self.rotary_dim // 2
                    cos = position_embeddings[0][:, :rotdim].to(query.dtype).contiguous()
                    sin = position_embeddings[1][:, :rotdim].to(query.dtype).contiguous()
                else:
                    cos, sin = None, None
                # restore initial tgt_pad_mask - migth be better to store it instead.
                cache_leftpad = (
                    (~torch.any(attn_mask, dim=-2).squeeze(1)).sum(dim=1).to(torch.int32)
                    if attn_mask is not None
                    else None
                )

                context = self.flash_attn_with_kvcache(
                    query.transpose(1, 2),
                    self.kcache[:, :, :, :].transpose(1, 2),
                    self.vcache[:, :, :, :].transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    rotary_cos=cos,
                    rotary_sin=sin,
                    cache_seqlens=cache_len - 1,
                    cache_leftpad=cache_leftpad,
                    softmax_scale=self.scale,
                    causal=step == 0,
                    rotary_interleaved=self.rotary_interleave,
                ).transpose(1, 2)


                # print("CONTEXT:", context.size(), context.sum(), context)

                attn_output = self.final_linear(unshape(context))
                if self.parallel_gpu > 1:
                    # all_reduce is an inplace op - not easily backprop
                    attn_output1 = attn_output.detach().clone()
                    all_reduce(attn_output1)
                    attn_output.copy_(attn_output1 + (attn_output - attn_output.detach()))
                return attn_output, None

        else:
            key, value, query = super()._prepare_inputs(query, query, query, position_embeddings=position_embeddings)

        return super()._compute_attention(
            key,
            value,
            query,
            attn_mask=attn_mask,
            return_attn=return_attn,
            text_indices=text_indices,
            image_indices=image_indices,
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
        query = shape(query, self.dim_per_head)
        if self.kcache.numel() == 0:
            key, value = self.linear_keys(key), self.linear_values(value)
            self.kcache, self.vcache = shape(key, self.dim_per_head), shape(value, self.dim_per_head)
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
            # training: we project key, value query and apply rotary if required
            key, value, query = super()._prepare_inputs(key, value, query)

        return super()._compute_attention(
            key,
            value,
            query,
            attn_mask=attn_mask,
            return_attn=return_attn,
        )
