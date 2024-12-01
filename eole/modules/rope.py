import torch
import torch.nn as nn
import math


class RotaryPosition(nn.Module):
    """
    Handles rotary position embeddings for transformer models.

    This module was refactored from multi-headed attention for improved clarity
    and to support future enhancements, such as additional scaling types.
    """

    def __init__(self, model_config, mode="1d"):
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
        self.rotary_theta = model_config.rope_config.rotary_theta
        self.inv_freq = 1.0 / (
            self.rotary_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
        )
        # TODO: extend with other scaling types
        if getattr(self.model_config.rope_config, "scaling_type", None) == "llama3":
            self.llama3_scaling()
        # cache rope tensor to limit unnecessary computations
        self.rope = None
        # specific pixtral handling, to refactor properly
        if mode == "2d":
            max_patches_per_side = model_config.image_size // model_config.patch_size
            h = torch.arange(max_patches_per_side, device=self.inv_freq.device)
            w = torch.arange(max_patches_per_side, device=self.inv_freq.device)
            freqs_h = torch.outer(h, self.inv_freq[::2]).float()
            freqs_w = torch.outer(w, self.inv_freq[1::2]).float()
            inv_freq = torch.cat(
                [
                    freqs_h[:, None, :].repeat(1, max_patches_per_side, 1),
                    freqs_w[None, :, :].repeat(max_patches_per_side, 1, 1),
                ],
                dim=-1,
            ).reshape(-1, self.dim_per_head // 2)
            self.inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)

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
        low_freq_factor = (
            rope_config.low_freq_factor
        )  # `1` in the original implementation
        high_freq_factor = (
            rope_config.high_freq_factor
        )  # `4` in the original implementation
        old_context_len = (
            rope_config.original_max_position_embeddings
        )  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / self.inv_freq
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq
        )

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        self.inv_freq = inv_freq_llama

    def forward_1d(self, emb, step=0, device=None, prefetch=1024):
        if step is None:
            step = 0
        maxseqlen = emb.size(1)
        offset = (
            32  # make sure we have at least 32 positions for flash_attn_with_kvcache
        )
        # This could probably a bit cleaner/homogenized with the offset case
        if self.rope is not None:
            if self.rope.size(0) >= max(offset + step, 0) + maxseqlen:
                return self.rope
            else:
                maxseqlen = maxseqlen + prefetch
        tmax = torch.arange(
            max(offset + step, 0) + maxseqlen, device=self.inv_freq.device
        )
        rope = torch.outer(tmax, self.inv_freq).float()
        # rope is now matrix [maxseqlen, dim/2]
        rope = torch.polar(torch.ones_like(rope), rope)
        rope = torch.cat((rope, rope), dim=1)
        if device is not None:
            rope = rope.to(device)
        self.rope = rope
        return rope

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

    def forward_2d(self, emb, step=0, device=None, prefetch=1024, positions=None):
        # TODO: maybe do scaling here
        if step is None:
            step = 0
        maxseqlen = emb.size(1)
        if positions is None:
            tmax = torch.arange(maxseqlen, device=self.inv_freq.device)
        else:
            tmax = positions
        rope = self.inv_freq[tmax]
        # rope is now matrix [maxseqlen, dim/2]
        rope = torch.polar(torch.ones_like(rope), rope)
        if device is not None:
            rope = rope.to(device)
        self.rope = rope
        return rope

    def forward(
        self, emb, step=0, device=None, prefetch=1024, reset=False, positions=None
    ):
        """
        Computes the rotary position embeddings for a given input.

        Args:
            emb: The input embeddings to which rotary embeddings will be applied.
            step: The current step or position within the sequence. Defaults to 0.
            device: The device on which the computations should be performed.
                    If None, defaults to the device of `self.inv_freq`.
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
        if self.mode == "1d":
            return self.forward_1d(emb, step=step, device=device, prefetch=prefetch)
        elif self.mode == "2d":
            return self.forward_2d(
                emb, step=step, device=device, prefetch=prefetch, positions=positions
            )
        else:
            raise NotImplementedError
