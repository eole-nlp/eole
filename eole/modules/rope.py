import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Tuple, List
from eole.constants import PositionEncodingType, _vllm_available


class NoOpPosition:
    """A no-op position encoding callable."""

    def update(self, *args, **kwargs):
        return None


def build_rope(model_config, mode="1d", variant="global"):
    """Build RoPE with optional vLLM acceleration."""
    if model_config.position_encoding_type == PositionEncodingType.Rotary:
        return RotaryPosition(model_config, mode=mode, variant=variant)
    else:
        return NoOpPosition()


def apply_rotary_pos_emb_xdrope(
    query: Tensor,
    key: Tensor,
    cos_sin: Tensor,
    position_ids: Tensor,
    xdrope_section: List[int],
) -> Tuple[Tensor, Tensor]:
    """
    query, key: (bs, seqlen, heads, rot_dim)
    cos_sin: (max_pos, x_dim, rot_dim_per_x)
    """
    bs, seqlen, heads, rot_dim = query.size()

    x_dim = len(xdrope_section)
    cos = cos_sin[:, : cos_sin.size(1) // 2]
    sin = cos_sin[:, cos_sin.size(1) // 2 :]

    # ---- THIS PERMUTE IS REQUIRED ----
    cos = (
        cos[position_ids, ...]  # (bs, seqlen, x_dim, dim_per_x)
        .permute(0, 2, 1, 3)  # (bs, x_dim, seqlen, dim_per_x)
        .reshape(bs, seqlen, x_dim, -1)  # XD-friendly layout
        .contiguous()
    )
    sin = (
        sin[position_ids, ...]  # (bs, seqlen, x_dim, dim_per_x)
        .permute(0, 2, 1, 3)  # (bs, x_dim, seqlen, dim_per_x)
        .reshape(bs, seqlen, x_dim, -1)  # XD-friendly layout
        .contiguous()
    )

    assert sum(xdrope_section) == cos.shape[-1], "Illegal partition for xd rope"

    # XD concat
    cos = torch.cat(
        [m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(xdrope_section, dim=-1))],
        dim=-1,
    )
    sin = torch.cat(
        [m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(xdrope_section, dim=-1))],
        dim=-1,
    )

    # broadcast over heads (new layout)
    cos = cos.view(bs, seqlen, 1, -1)
    sin = sin.view(bs, seqlen, 1, -1)

    origin_dtype = query.dtype
    query = query.float()
    key = key.float()
    cos = cos.float()
    sin = sin.float()

    # Split query/key into halves
    half = rot_dim // 2
    q1, q2 = query[..., :half], query[..., half:]
    k1, k2 = key[..., :half], key[..., half:]

    # Apply rotation
    q_out = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k_out = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)

    return q_out.to(origin_dtype), k_out.to(origin_dtype)


def apply_rotary_emb(query: Tensor, key: Tensor, cos_sin: Tensor, interleave: bool) -> Tuple[Tensor, Tensor]:

    B, S, H, D = query.shape

    if _vllm_available and query.device.type == "cuda":
        num_tok = B * S
        query = query.view(num_tok, -1, D)
        key = key.view(num_tok, -1, D)
        positions = torch.arange(S, device=query.device).repeat(B)
        # vllm ops change query key in-place
        torch.ops._C.rotary_embedding(positions, query, key, D, cos_sin, not interleave)
        return query.view(B, S, -1, D), key.view(B, S, -1, D)

    else:
        Rd = cos_sin.size(1)
        q_rot, q_pass = query[..., :Rd], query[..., Rd:]
        k_rot, k_pass = key[..., :Rd], key[..., Rd:]

        # Reshape based on interleave mode
        if interleave:
            # Split into pairs: (B, S, H, Rd) -> (B, S, H, Rd//2, 2)
            q1, q2 = q_rot.view(B, S, H, -1, 2).unbind(-1)
            k1, k2 = k_rot.view(B, S, H, -1, 2).unbind(-1)
        else:
            # Split into halves
            half = Rd // 2
            q1, q2 = q_rot[..., :half], q_rot[..., half:]
            k1, k2 = k_rot[..., :half], k_rot[..., half:]

        # Broadcast cos/sin to match
        cos = cos_sin[None, :, None, : Rd // 2]
        sin = cos_sin[None, :, None, Rd // 2 :]

        # Apply rotation (same formula for both modes)
        qr = q1 * cos - q2 * sin
        qi = q1 * sin + q2 * cos
        kr = k1 * cos - k2 * sin
        ki = k1 * sin + k2 * cos

        # Recombine based on interleave mode
        if interleave:
            q_rotated = torch.stack((qr, qi), dim=-1).flatten(-2)
            k_rotated = torch.stack((kr, ki), dim=-1).flatten(-2)
        else:
            q_rotated = torch.cat((qr, qi), dim=-1)
            k_rotated = torch.cat((kr, ki), dim=-1)

        return (
            torch.cat((q_rotated, q_pass), dim=-1),
            torch.cat((k_rotated, k_pass), dim=-1),
        )


class RotaryPosition(nn.Module):
    """
    Handles rotary position embeddings for transformer models.

    This module was refactored from multi-headed attention for improved clarity
    and to support future enhancements, such as additional scaling types.
    """

    def __init__(self, model_config, mode="1d", variant="global"):
        """
        Initializes the RotaryPosition module.

        Args:
            model_config: Configuration object that contains model parameters,
                          including rotary embedding settings.

        Attributes:
            model_config: The configuration object passed during initialization.
            dim_per_head: The dimensionality of each attention head, computed
                          as `hidden_size // heads`.
            rotary_interleave: Boolean flag to determine if head dimensions should
                               be interleaved or split when applying rotary embeddings.
            rotary_theta: The base frequency for rotary embeddings.
            inv_freq: Inverse frequency values used to calculate the rotary embeddings.

        Notes:
            - If `rotary_dim` is set to 0 in the configuration, it defaults to
              `dim_per_head`.
            - Additional scaling types can be added in the future by extending this class.
        """
        super(RotaryPosition, self).__init__()
        self.model_config = model_config
        self.mode = mode
        self.dim_per_head = model_config.dim_per_head
        if model_config.rope_config.rotary_dim == 0:
            rotary_dim = self.dim_per_head
        else:
            rotary_dim = model_config.rope_config.rotary_dim
        self.rotary_interleave = model_config.rope_config.rotary_interleave
        if variant == "global":
            self.rotary_theta = model_config.rope_config.rotary_theta
        else:
            self.rotary_theta = model_config.rope_config.rotary_theta_local
        if getattr(self.model_config.rope_config, "scaling_type", None) in ["dynamic", "xdrope"] and getattr(
            self.model_config.rope_config, "alpha", None
        ):
            base = self.rotary_theta * getattr(self.model_config.rope_config, "alpha", None) ** (
                rotary_dim / (rotary_dim - 2)
            )
        else:
            base = self.rotary_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        if mode == "2d":
            inv_freq = self.init_2d_inv_freq(inv_freq)
        self.inv_freq = inv_freq
        # TODO: extend with other scaling types
        if getattr(self.model_config.rope_config, "scaling_type", None) == "llama3":
            self.llama3_scaling()
        if getattr(self.model_config.rope_config, "scaling_type", None) == "gemma3" and variant == "global":
            self.gemma3_scaling()

        cos_sin = self.update(1024)
        self.register_buffer("cos_sin", cos_sin, persistent=False)

    def init_2d_inv_freq(self, inv_freq):
        """
        Initialize 2d inverse frequencies for pixtral rotary embeddings.
        """
        max_patches_per_side = self.model_config.image_size // self.model_config.patch_size
        h = torch.arange(max_patches_per_side, device=inv_freq.device)
        w = torch.arange(max_patches_per_side, device=inv_freq.device)
        freqs_h = torch.outer(h, inv_freq[::2]).float()
        freqs_w = torch.outer(w, inv_freq[1::2]).float()
        inv_freq = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
            ],
            dim=-1,
        ).reshape(-1, self.dim_per_head // 2)
        return inv_freq

    def llama3_scaling(self):
        """
        Applies the LLaMA3.1-specific scaling to the inverse frequencies.

        This scaling is based on LLaMA3.1's handling of different frequency components
        within rotary embeddings. The method modifies `self.inv_freq` in place.

        Notes:
            - Original values for `factor`, `low_freq_factor`, `high_freq_factor`,
              and `original_max_position_embeddings` are taken from the configuration.
            - The scaling factors are applied conditionally based on the wavelength
              derived from the inverse frequencies.
        """
        rope_config = self.model_config.rope_config
        factor = rope_config.scaling_factor  # `8` in the original implementation
        low_freq_factor = rope_config.low_freq_factor  # `1` in the original implementation
        high_freq_factor = rope_config.high_freq_factor  # `4` in the original implementation
        old_context_len = rope_config.original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / self.inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq)

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        self.inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    def gemma3_scaling(self):
        rope_config = self.model_config.rope_config
        factor = rope_config.scaling_factor  # `8` in the original implementation
        self.inv_freq /= factor

    def forward_1d(self, maxseqlen, step=0, prefetch=1024, offset=32):
        maxseqlen += prefetch
        device = self.cos_sin.device if hasattr(self, "cos_sin") else torch.device("cpu")
        dtype = self.cos_sin.dtype if hasattr(self, "cos_sin") else torch.float32

        tmax = torch.arange(max(offset + step, 0) + maxseqlen, device=device)
        tmax += self.model_config.rope_config.tmax_index
        rope = torch.outer(tmax, self.inv_freq.to(device))
        cos = torch.cos(rope).to(dtype)
        sin = torch.sin(rope).to(dtype)

        return cos, sin

    # TODO: investigate if this is useful / properly implemented
    # def _dynamic_frequency_update(self, position_ids, device):
    #     """
    #     dynamic RoPE layers should recompute `inv_freq` in the following situations:
    #     1 - growing beyond the cached sequence length (allow scaling)
    #     2 - the current sequence length is in the original scale
    #           (avoid losing precision with small sequences)
    #     """
    #     seq_len = torch.max(position_ids) + 1
    #     if seq_len > self.max_seq_len_cached:  # growth
    #         inv_freq, self.attention_scaling = self.rope_init_fn(
    #             self.config, device, seq_len=seq_len, **self.rope_kwargs
    #         )
    #         self.inv_freq = inv_freq
    #         self.max_seq_len_cached = seq_len

    #     if (seq_len < self.original_max_seq_len
    #     and self.max_seq_len_cached > self.original_max_seq_len):  # reset
    #         self.inv_freq = self.original_inv_freq
    #         self.max_seq_len_cached = self.original_max_seq_len

    def forward_2d(self, maxseqlen, step=0, prefetch=1024, offset=32, positions=None):
        # TODO: maybe do scaling here
        device = self.cos_sin.device if hasattr(self, "cos_sin") else torch.device("cpu")
        dtype = self.cos_sin.dtype if hasattr(self, "cos_sin") else torch.float32

        if positions is None:
            tmax = torch.arange(maxseqlen, device=self.inv_freq.device)
        else:
            tmax = positions.to(self.inv_freq.device)
        rope = self.inv_freq[tmax].to(device)
        # rope is now matrix [maxseqlen, dim/2]
        # if device is not None:
        #     rope = rope.to(device)
        cos = rope.cos().to(dtype)
        sin = rope.sin().to(dtype)

        return cos, sin

    def update(self, maxseqlen, step=0, prefetch=1024, reset=False, positions=None):
        """
        Computes the rotary position embeddings for a given input.
        Args:
            maxseqlen: max seq length of the input embeddings.
            step: The current step or position within the sequence. Defaults to 0.
            offset: An optional offset to apply to the position indices.
                    This is used for the specific `flash_attn_with_kvcache` path,
                    which requires processes by chunks of 32 tokens. Defaults to 0.
        Returns:
            torch.Tensor: A tensor containing the computed rotary embeddings.
        Notes:
            - The returned tensor contains cosine and sine values representing the
              rotary embeddings, concatenated along the last dimension.
            - The output tensor's dimensions are `[maxseqlen, dim]`, where `dim` is
              twice the size of the original inverse frequency tensor (`inv_freq`).
        """
        if reset:
            self.rope = None
        offset = 32  # make sure we have at least 32 positions for flash_attn_with_kvcache
        if step == 0:
            maxseqlen = max(maxseqlen, 1024)  # reset as in init() with self.update(1024)
        elif hasattr(self, "cos") and self.cos.size(0) >= max(offset + (step or 0), 0) + maxseqlen:
            return self.cos, self.sin
        if self.mode == "1d":
            cos, sin = self.forward_1d(maxseqlen, step=(step or 0), prefetch=prefetch, offset=offset)
        elif self.mode == "2d":
            cos, sin = self.forward_2d(maxseqlen, step=(step or 0), prefetch=prefetch, positions=positions)
        else:
            raise NotImplementedError
        cos_sin = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin", cos_sin, persistent=False)
        return cos_sin
