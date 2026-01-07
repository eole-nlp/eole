"""Vision adapters for multimodal models."""

import torch
import torch.nn as nn

from eole.constants import LayerNorm


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


# TODO refactor the 3 projectors below for unique code path
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
            self.layernorm = LayerNorm[model_config.encoder.layer_norm](in_dim, eps=1e-5)
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


# Multi-Modal Projector
class Gemma3MultiModalProjector(nn.Module):
    # https://github.com/huggingface/transformers/blob/071a161d3e38f56dbda2743b979f0afeed2cd4f1/src/transformers/models/gemma3/modular_gemma3.py#L717
    def __init__(self, in_dim, out_dim, image_size, patch_size, mm_tokens_per_image):
        super(Gemma3MultiModalProjector, self).__init__()
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = LayerNorm["gemma-rms"](in_dim)  # forced because no value in config file
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


# Multi-Modal Projector
class DeepSeekOCRProjector(nn.Module):
    def __init__(self, model_config):
        super(DeepSeekOCRProjector, self).__init__()
        bias = getattr(model_config, "adapter_bias", False)
        self.w_in = nn.Linear(model_config.encoder.hidden_size * 2, model_config.decoder.hidden_size, bias=bias)

    def forward(self, x, image_sizes=None):
        return self.w_in(x)

    @classmethod
    def from_config(cls, model_config, running_config=None):
        return cls(
            model_config,
        )


# Multi-Modal Projector + Patch merger
class HunYuanVisionPatchMerger(nn.Module):
    def __init__(
        self,
        model_config,
    ):
        super().__init__()
        in_channels = model_config.encoder.hidden_size
        out_channels = model_config.decoder.hidden_size
        rms_norm_eps = getattr(model_config.encoder, "norm_eps", 1e-5)
        self.patch_size = model_config.encoder.patch_size
        self.spatial_merge_size = model_config.spatial_merge_size
        embed_std = out_channels**-0.5

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            ),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_channels * 4, out_channels)

        self.image_newline = nn.Parameter(torch.randn(in_channels * 4) * embed_std)
        self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = LayerNorm["rms"](in_channels, eps=rms_norm_eps)
        self.after_rms = LayerNorm["rms"](out_channels, eps=rms_norm_eps)

    def forward(self, x, image_sizes):
        """
        x: Tensor (1, sum_i N_i, D)
        image_sizes: (B, 2) = (height_px, width_px) for each image
        """
        x = x.squeeze(0)
        patch_h = image_sizes[:, 0] // self.patch_size
        patch_w = image_sizes[:, 1] // self.patch_size
        patch_counts = (patch_h * patch_w).tolist()

        xs = x.split(patch_counts)
        outputs = []

        # ---- 2) Process each image independently ----
        for i, xi in enumerate(xs):
            h, w = int(patch_h[i]), int(patch_w[i])
            xi = self.before_rms(xi)  # (N_i, D)
            # (N_i, D) -> (1, D, h, w)
            xi = xi.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            # Convolution / projection
            xi = self.proj(xi)  # (1, C, h, w)
            _, c, h, w = xi.shape
            # Add image_newline
            newline = self.image_newline.reshape(1, c, 1, 1).to(xi.dtype)
            xi = torch.cat([xi, newline.expand(1, c, h, 1)], dim=-1)  # (1, C, h, w+1)
            # Flatten back to sequence
            xi = xi.flatten(2).permute(0, 2, 1)  # (1, L_i, C)
            # MLP
            xi = self.mlp(xi)  # (1, L_i, C)
            # Add begin / end tokens
            begin = self.image_begin.reshape(1, 1, -1).to(xi.dtype)
            end = self.image_end.reshape(1, 1, -1).to(xi.dtype)
            xi = torch.cat([begin, xi, end], dim=1)  # (1, L_i+2, C)
            xi = self.after_rms(xi)  # (1, L_i+2, C)
            outputs.append(xi.squeeze(0))  # (L_i+2, C)

        # ---- 3) Return concatenation over patch dimension ----
        return torch.cat(outputs, dim=0).unsqueeze(0)  # (1, sum_i (L_i+2), C)

    @classmethod
    def from_config(cls, model_config, running_config=None):
        return cls(
            model_config,
        )
