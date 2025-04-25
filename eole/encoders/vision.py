"""
Base class for vision encoders.
We'll start with Pixtral encoder, but we can build up on it later.
https://github.com/mistralai/mistral-inference/pull/217/files#diff-5cf8b83293298d0bdda514a77cae34565cb54571cddab02801d99a52f0f71a66
"""

import torch
import torch.nn as nn
from typing import Optional

# from eole.modules.multi_headed_attn import SelfMHA
from eole.modules.rmsnorm import RMSNorm, GemmaRMSNorm
from eole.encoders.transformer import TransformerEncoderLayer
from eole.modules.rope import build_rope
from eole.constants import PositionEncodingType


class PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        hidden_size = model_config.encoder.hidden_size
        self.spatial_merge_size = model_config.spatial_merge_size
        self.patch_size = model_config.encoder.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]
        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        flattened_features = image_features.view(-1, d)
        permuted_tensor = []
        for image_index, image_tokens in enumerate(flattened_features.split(tokens_per_image)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features.unsqueeze(0))

        return image_features


def position_ids_in_meshgrid(patch_embeds_list, max_width, flatten=True):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    if flatten:
        return torch.cat(positions)
    else:
        return torch.stack(positions)


def create_block_diagonal_mask(lengths, device):
    """
    Create a block diagonal mask based on sequence lengths.
    Args:
        lengths (list of int): List of lengths for each block.
    Returns:
        torch.Tensor: A 2D boolean tensor with True in the
            block diagonal regions and False elsewhere.
    """
    total_length = sum(lengths)
    mask = torch.zeros((total_length, total_length), dtype=torch.bool)

    start = 0
    for length in lengths:
        end = start + length
        mask[start:end, start:end] = True
        start = end

    return mask.to(device)


class VisionEncoder(nn.Module):
    def __init__(self, encoder_config, running_config=None):
        super(VisionEncoder, self).__init__()
        self.encoder_config = encoder_config
        if encoder_config.position_encoding_type == PositionEncodingType.Learned:
            self.position_embeddings = nn.Embedding(
                encoder_config.n_positions,
                encoder_config.hidden_size,
            )
        else:
            self.rope = build_rope(encoder_config, mode="2d")
        self.patch_conv = nn.Conv2d(
            in_channels=encoder_config.num_channels,
            out_channels=encoder_config.hidden_size,
            kernel_size=encoder_config.patch_size,
            stride=encoder_config.patch_size,
            bias=encoder_config.patch_conv_bias,
        )
        if encoder_config.layernorm_pre:
            self.ln_pre = RMSNorm(encoder_config.hidden_size, eps=1e-5)
        else:
            self.ln_pre = None
        self.transformer_layers = torch.nn.ModuleList()
        for _ in range(encoder_config.layers):
            self.transformer_layers.append(TransformerEncoderLayer(encoder_config, running_config=running_config))

        if self.ln_pre is None:  # then post layer norm (gemma3)
            self.post_layernorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.norm_eps)
        else:
            self.post_layernorm = None

        head_dim = encoder_config.hidden_size // encoder_config.heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, encoder_config, running_config=None):
        """Alternate constructor."""
        return cls(
            encoder_config,
            running_config,
        )

    @property
    def max_patches_per_side(self):
        return self.encoder_config.image_size // self.encoder_config.patch_size

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, images):
        """
        Args:
            images: list of N_img images of variable sizes, each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for all tokens of all images of
                shape (N_img, Seqlen, D)
        """

        # TODO add as @property somewhere
        dtype = next(self.parameters()).dtype

        # pass images through initial convolution independently (because they may have different sizes)
        patch_embeds_list = [self.patch_conv(img.to(dtype)) for img in images]

        if self.ln_pre is not None:  # pixtral / mistral
            # flatten H+W then change to (H+W, C) and stack all images of ex
            patch_embeds = torch.cat([p.flatten(1).permute(1, 0) for p in patch_embeds_list], dim=0)
            patch_embeds = self.ln_pre(patch_embeds)
            # create a fake batch dim
            patch_embeds = patch_embeds.unsqueeze(0)
            mask = create_block_diagonal_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
                device=self.device,
            )  # block diagonal mask delimiting images
            # revert for proper downstream compatibility
            mask = ~mask
        else:  # gemma3
            patch_embeds = torch.cat([p.unsqueeze(0) for p in patch_embeds_list], dim=0)
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            mask = None

        # positional embeddings
        positions = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.encoder_config.image_size // self.encoder_config.patch_size,
            flatten=self.ln_pre is not None,  # dirty flag need to improve
        ).to(self.device)
        # TODO: make this cleaner
        if hasattr(self, "position_embeddings"):
            # this is only used for rope
            position_embeddings = None
            patch_embeds += self.position_embeddings(positions)
        else:
            position_embeddings = self.rope.update(
                patch_embeds.size(1),
                step=0,
                reset=True,
                positions=positions,
            )

        out = patch_embeds
        for i, layer in enumerate(self.transformer_layers):
            out = layer(out, pad_mask=mask, position_embeddings=position_embeddings)

        if self.post_layernorm is not None:
            out = self.post_layernorm(out)

        return out


# Multi-Modal Projector
class VisionLanguageAdapter(nn.Module):
    def __init__(self, model_config):
        super(VisionLanguageAdapter, self).__init__()
        in_dim = model_config.encoder.hidden_size
        out_dim = model_config.decoder.hidden_size
        bias = getattr(model_config, "adapter_bias", False)
        self.has_patch = False
        if model_config.spatial_merge_size > 1:
            self.has_patch = True
            self.layernorm = RMSNorm(in_dim, eps=1e-5)
            self.patch_merger = PatchMerger(model_config)
        self.w_in = nn.Linear(in_dim, out_dim, bias=bias)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(out_dim, out_dim, bias=bias)

    def forward(self, x, image_sizes=None):
        if self.has_patch:
            x = self.layernorm(x)
            x = self.patch_merger(x, image_sizes)
        return self.w_out(self.gelu(self.w_in(x)))

    @classmethod
    def from_config(cls, model_config, running_config=None):
        return cls(
            model_config,
        )


class Gemma3MultiModalProjector(nn.Module):
    # https://github.com/huggingface/transformers/blob/071a161d3e38f56dbda2743b979f0afeed2cd4f1/src/transformers/models/gemma3/modular_gemma3.py#L717
    def __init__(self, in_dim, out_dim, image_size, patch_size, mm_tokens_per_image):
        super(Gemma3MultiModalProjector, self).__init__()
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = GemmaRMSNorm(in_dim)
        self.patches_per_image = int(image_size / patch_size)
        self.tokens_per_side = int(mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, x, image_sizes):
        batch_size, _, hidden_size = x.size()
        reshaped = (
            x.transpose(1, 2)
            .reshape(batch_size, hidden_size, self.patches_per_image, self.patches_per_image)
            .contiguous()
        )
        pooled = self.avg_pool(reshaped).flatten(2).transpose(1, 2)
        normed = self.norm(pooled)
        projected = self.w_in(normed)
        return projected.type_as(x)

    @classmethod
    def from_config(cls, model_config, running_config=None):
        return cls(
            in_dim=model_config.encoder.hidden_size,
            out_dim=model_config.decoder.hidden_size,
            image_size=model_config.encoder.image_size,
            patch_size=model_config.encoder.patch_size,
            mm_tokens_per_image=model_config.encoder.mm_tokens_per_image,
        )


str2adapter = {
    "llava": VisionLanguageAdapter,
    "gemma3": Gemma3MultiModalProjector,
}
