"""Vision adapters for multimodal models."""

import torch
import torch.nn as nn

from eole.constants import LayerNorm


class BaseVisionAdapter(nn.Module):
    """
    Base class for all vision-to-language adapters.

    Adapters take visual token embeddings produced by a vision encoder and
    transform them into a sequence of embeddings consumable by a language
    model decoder.

    All subclasses must implement:
        forward(x, image_sizes)

    Expected input format:
        x: Tensor of shape (B, N, D) or (1, sum_i N_i, D)
        image_sizes: Tensor of shape (B, 2), containing (height_px, width_px)
                     for each image in the batch.
    """

    def forward(self, x, image_sizes=None):
        raise NotImplementedError


def compute_patch_grid(image_sizes, patch_size):
    """
    Compute patch grid dimensions for each image.

    Args:
        image_sizes: Iterable of (height_px, width_px).
        patch_size: Size of one vision patch in pixels.

    Returns:
        List of (height_in_patches, width_in_patches).
    """
    grids = []
    for h, w in image_sizes:
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"Image size ({h}, {w}) not divisible by patch_size {patch_size}")
        grids.append((h // patch_size, w // patch_size))
    return grids


def split_by_image(x, image_sizes, patch_size):
    """
    Split a concatenated patch sequence into per-image tensors.

    Args:
        x: Tensor of shape (sum_i N_i, D).
        image_sizes: Tensor of shape (B, 2).
        patch_size: Vision patch size in pixels.

    Returns:
        xs: Tuple of tensors, one per image.
        grids: List of (height, width) patch grids.
    """
    grids = compute_patch_grid(image_sizes, patch_size)
    counts = [h * w for h, w in grids]
    return x.split(counts), grids


class LinearPatchMerger(nn.Module):
    """
    Spatial patch merger using learned linear projection.

    This module merges non-overlapping spatial blocks of size
    (spatial_merge_size x spatial_merge_size) from a vision encoder's
    patch grid into a single token.

    The merging is performed per image, but the final linear projection
    is applied once globally, matching the behavior of the original
    PatchMerger implementation.

    Input:
        image_features: Tensor of shape (1, sum_i N_i, D)
            Concatenated patch embeddings for a batch of images.

        image_sizes: Tensor of shape (B, 2)
            Image sizes in pixels as (height, width).

    Output:
        Tensor of shape (1, sum_i (N_i / spatial_merge_size^2), D)
    """

    def __init__(self, hidden_size, patch_size, spatial_merge_size):
        """
        Args:
            hidden_size: Dimensionality of vision encoder embeddings (D).
            patch_size: Size of a single vision patch in pixels.
            spatial_merge_size: Number of patches to merge per spatial dimension.
        """
        super().__init__()
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.merging_layer = nn.Linear(
            hidden_size * spatial_merge_size**2,
            hidden_size,
            bias=False,
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        """
        Merge spatial patches into larger visual tokens.

        Args:
            image_features: (1, sum_i N_i, D)
            image_sizes: (B, 2), image sizes in pixels

        Returns:
            Tensor of shape (1, sum_i L_i, D), where
            L_i = (H_i / S) * (W_i / S)
        """
        D = image_features.shape[-1]
        xs, grids = split_by_image(image_features.squeeze(0), image_sizes, self.patch_size)

        outputs = []
        for xi, (h, w) in zip(xs, grids):
            # (N_i, D) → (1, D, h, w)
            xi = xi.view(h, w, D).permute(2, 0, 1).unsqueeze(0)

            # unfold → (1, D*S*S, L)
            xi = torch.nn.functional.unfold(
                xi,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )

            # (L, D*S*S)
            xi = xi.view(D * self.spatial_merge_size**2, -1).t()
            outputs.append(xi)

        # Concatenate all images on token axis
        merged = torch.cat(outputs, dim=0)  # (sum L_i, D*S*S)

        # Single projection, identical to original
        merged = self.merging_layer(merged).unsqueeze(0)  # (1, sum L_i, D)
        return merged


class VisionLanguageAdapter(BaseVisionAdapter):
    """
    Generic vision-language adapter used by LLaVA-style architectures.

    This adapter optionally performs spatial patch merging followed by
    a two-layer MLP projection into the decoder embedding space.

    When spatial merging is disabled, the adapter reduces to a simple
    MLP projector.
    """

    def __init__(self, model_config, running_config=None):
        """
        Args:
            model_config: Model configuration object containing encoder and
                          decoder parameters.
            running_config: Optional runtime configuration (unused).
        """
        super().__init__()
        enc, dec = model_config.encoder, model_config.decoder
        self.has_patch = model_config.spatial_merge_size > 1

        if self.has_patch:
            self.layernorm = LayerNorm[enc.layer_norm](enc.hidden_size, eps=1e-5)
            self.patch_merger = LinearPatchMerger(
                enc.hidden_size,
                enc.patch_size,
                model_config.spatial_merge_size,
            )
        bias = getattr(model_config, "adapter_bias", False)
        self.w_in = nn.Linear(enc.hidden_size, dec.hidden_size, bias=bias)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dec.hidden_size, dec.hidden_size, bias=bias)

    def forward(self, x, image_sizes=None):
        """
        Args:
            x: Vision encoder outputs of shape (1, N, D).
            image_sizes: Tensor of shape (B, 2) containing image sizes.

        Returns:
            Projected visual tokens of shape (1, M, decoder_hidden_size).
        """
        if self.has_patch:
            x = self.layernorm(x)
            x = self.patch_merger(x, image_sizes)
        return self.w_out(self.gelu(self.w_in(x)))


class Gemma3MultiModalProjector(BaseVisionAdapter):
    """
    Multi-modal projector used in Gemma 3 models.

    This projector downsamples the vision encoder patch grid using
    average pooling to a fixed number of tokens per image, followed by
    RMS normalization and linear projection.
    """

    def __init__(self, model_config, running_config=None):
        """
        Args:
            model_config: Model configuration containing encoder and decoder
                          parameters, including mm_tokens_per_image.
            running_config: Optional runtime configuration (unused).
        """
        super().__init__()
        in_dim = model_config.encoder.hidden_size
        out_dim = model_config.decoder.hidden_size
        image_size = model_config.encoder.image_size
        patch_size = model_config.encoder.patch_size
        mm_tokens = model_config.encoder.mm_tokens_per_image

        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = LayerNorm["gemma-rms"](in_dim)  # forced because no value in config file

        patches = image_size // patch_size
        tokens_side = int(mm_tokens**0.5)
        kernel = patches // tokens_side
        self.pool = nn.AvgPool2d(kernel, stride=kernel)

    def forward(self, x, image_sizes=None):
        """
        Args:
            x: Vision encoder output of shape (B, N, D), where N is assumed
               to form a square grid.
            image_sizes: Unused (kept for interface compatibility).

        Returns:
            Tensor of shape (B, mm_tokens_per_image, decoder_hidden_size).
        """
        b, n, d = x.shape
        h = int(n**0.5)
        x = x.transpose(1, 2).reshape(b, d, h, h)
        x = self.pool(x).flatten(2).transpose(1, 2)
        return self.w_in(self.norm(x)).type_as(x)


class DeepSeekOCRProjector(BaseVisionAdapter):
    """
    Projector used in DeepSeek OCR models.

    Expects concatenated visual features (e.g., multi-branch encoder
    outputs) and applies a single linear projection to decoder space.
    """

    def __init__(self, model_config, running_config=None):
        """
        Args:
            model_config: Model configuration object.
            running_config: Optional runtime configuration (unused).
        """
        super().__init__()
        self.w_in = nn.Linear(
            model_config.encoder.hidden_size * 2,
            model_config.decoder.hidden_size,
            bias=getattr(model_config, "adapter_bias", False),
        )

    def forward(self, x, image_sizes=None):
        """
        Args:
            x: Tensor of shape (..., 2 * encoder_hidden_size).
            image_sizes: Unused.

        Returns:
            Tensor projected to decoder hidden size.
        """
        return self.w_in(x)


class HunYuanVisionPatchMerger(BaseVisionAdapter):
    """
    Vision adapter used by HunYuan-style multimodal models.

    This adapter:
    - processes each image independently
    - applies convolutional spatial merging
    - injects special image tokens (begin, end, newline)
    - projects features into the decoder embedding space
    """

    def __init__(self, model_config, running_config=None):
        """
        Args:
            model_config: Model configuration containing encoder, decoder,
                          and spatial merge parameters.
            running_config: Optional runtime configuration (unused).
        """
        super().__init__()

        enc, dec = model_config.encoder, model_config.decoder
        self.patch_size = enc.patch_size
        spatial_merge_size = model_config.spatial_merge_size
        rms_norm_eps = getattr(model_config.encoder, "norm_eps", 1e-5)

        scale = dec.hidden_size**-0.5

        self.proj = nn.Sequential(
            nn.Conv2d(
                enc.hidden_size,
                enc.hidden_size * 2,
                kernel_size=spatial_merge_size,
                stride=spatial_merge_size,
            ),
            nn.GELU(),
            nn.Conv2d(enc.hidden_size * 2, enc.hidden_size * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(enc.hidden_size * 4, dec.hidden_size)

        self.image_newline = nn.Parameter(torch.randn(enc.hidden_size * 4) * scale)
        self.image_begin = nn.Parameter(torch.randn(dec.hidden_size) * scale)
        self.image_end = nn.Parameter(torch.randn(dec.hidden_size) * scale)
        self.image_sep = nn.Parameter(torch.randn(dec.hidden_size) * scale)

        self.before_rms = LayerNorm["rms"](enc.hidden_size, eps=rms_norm_eps)
        self.after_rms = LayerNorm["rms"](dec.hidden_size, eps=rms_norm_eps)

    def forward(self, x, image_sizes):
        """
        Args:
            x: Tensor of shape (1, sum_i N_i, encoder_hidden_size).
            image_sizes: Tensor of shape (B, 2) with image sizes in pixels.

        Returns:
            Tensor of shape (1, sum_i L_i, decoder_hidden_size), where each
            image contributes its own begin/end tokens.
        """
        x = x.squeeze(0)
        xs, grids = split_by_image(x, image_sizes, self.patch_size)
        outputs = []

        for xi, (h, w) in zip(xs, grids):
            xi = self.before_rms(xi)
            xi = xi.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
            xi = self.proj(xi)

            newline = self.image_newline.view(1, -1, 1, 1)
            xi = torch.cat([xi, newline.expand(1, -1, xi.shape[2], 1)], dim=-1)

            xi = xi.flatten(2).transpose(1, 2)
            xi = self.mlp(xi)

            xi = torch.cat([self.image_begin[None, None, :], xi, self.image_end[None, None, :]], dim=1)

            outputs.append(self.after_rms(xi).squeeze(0))

        return torch.cat(outputs, dim=0).unsqueeze(0)
