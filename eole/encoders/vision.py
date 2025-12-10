"""
Base class for vision encoders.
We'll start with Pixtral encoder, but we can build up on it later.
https://github.com/mistralai/mistral-inference/pull/217/files#diff-5cf8b83293298d0bdda514a77cae34565cb54571cddab02801d99a52f0f71a66
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from eole.encoders.transformer import TransformerEncoderLayer
from eole.modules.rope import build_rope
from eole.constants import PositionEncodingType, LayerNorm


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


def position_ids_in_meshgrid(patch_embeds_list, max_width):
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return positions


def create_block_diagonal_mask(lengths, device):
    """
    Create a block diagonal mask based on sequence lengths.
    Args:
        lengths (list of int): List of lengths for each block.
    Returns:
        torch.Tensor: A 2D boolean tensor with False in the
            block diagonal regions and True elsewhere.
    """
    total_length = sum(lengths)
    mask = torch.ones((total_length, total_length), dtype=torch.bool)

    start = 0
    for length in lengths:
        end = start + length
        mask[start:end, start:end] = False
        start = end

    return mask.to(device)


def get_abs_pos(abs_pos, tgt_size):
    dim = abs_pos.size(-1)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]

    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).contiguous()
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


# Clip() based vision encoder
class VisionEncoder(nn.Module):
    def __init__(self, encoder_config, running_config=None):
        super(VisionEncoder, self).__init__()
        self.encoder_config = encoder_config
        if encoder_config.encoder_sam:
            from eole.encoders.deepseek_sam import build_sam_vit_b

            self.sam = build_sam_vit_b()
        else:
            self.sam = None
        if encoder_config.position_encoding_type == PositionEncodingType.Learned:
            n_positions = encoder_config.n_positions
            self.interpolate_mode = encoder_config.interpolate_mode
            if self.interpolate_mode is not None:
                self.max_patch_per_side = int(n_positions**0.5)
                n_positions += 1
            self.position_embeddings = nn.Embedding(
                n_positions,
                encoder_config.hidden_size,
            )
            self.register_buffer("position_ids", torch.arange(n_positions).expand((1, -1)))
        else:
            self.rope = build_rope(encoder_config, mode="2d")
        self.patch_conv = nn.Conv2d(
            in_channels=encoder_config.num_channels,
            out_channels=encoder_config.hidden_size,
            kernel_size=encoder_config.patch_size,
            stride=encoder_config.patch_size,
            bias=encoder_config.patch_conv_bias,
        )
        if encoder_config.use_class_embedding:
            self.class_embedding = torch.nn.Embedding(1, encoder_config.hidden_size)
        else:
            self.class_embedding = None
        if encoder_config.layernorm_pre:
            self.ln_pre = LayerNorm[encoder_config.layer_norm](encoder_config.hidden_size, eps=1e-5)
        else:
            self.ln_pre = None
        self.transformer_layers = torch.nn.ModuleList()
        for _ in range(encoder_config.layers):
            self.transformer_layers.append(TransformerEncoderLayer(encoder_config, running_config=running_config))

        if encoder_config.layernorm_post:
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
    def device(self):
        return next(self.parameters()).device

    def forward(self, images, sam_patches=None):
        """
        Args:
            images: list of N_img images of variable sizes, each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for all tokens of all images of
                shape (N_img, Seqlen, D)
        """

        # TODO add as @property somewhere
        dtype = next(self.parameters()).dtype
        mask = None
        position_embeddings = None

        if sam_patches is not None:
            patch_embeds = sam_patches
            # flatten H+W then change to (H+W, C)
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            if self.class_embedding is not None:  # deepseekocr
                bs = patch_embeds.size(0)
                class_embeds = self.class_embedding.weight.expand(bs, -1, -1)
                patch_embeds = torch.cat([class_embeds, patch_embeds], dim=1)
                patch_embeds += get_abs_pos(self.position_embeddings(self.position_ids), patch_embeds.size(1))
            if self.ln_pre is not None:
                patch_embeds = self.ln_pre(patch_embeds)
        else:
            patch_embeds_list = [self.patch_conv(img.to(dtype)) for img in images]
            positions = position_ids_in_meshgrid(
                patch_embeds_list,
                max_width=self.encoder_config.image_size // self.encoder_config.patch_size,
            )
            if hasattr(self, "position_embeddings"):
                # Learned positions (Gemma3, Classic Clip)
                if self.interpolate_mode is None:  # Gemma3 / DeepseekOCR
                    # all images have the same size.
                    patch_pos_embed = self.position_embeddings(torch.stack(positions).to(self.device))
                    patch_embeds = torch.cat([p.unsqueeze(0) for p in patch_embeds_list], dim=0)
                    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
                    patch_embeds += patch_pos_embed
                else:  # HunyuanOCR
                    # images may have a different size
                    all_patch_embeds = []
                    for patch_embeds in patch_embeds_list:
                        D, H, W = patch_embeds.size()
                        # flatten this image's patches → (H*W, D)
                        patch_embeds_flat = patch_embeds.flatten(1).transpose(0, 1)  # (N, D)
                        patch_pos_shape = (1, self.max_patch_per_side, self.max_patch_per_side, D)
                        patch_pos_embed = (
                            self.position_embeddings.weight[1:, :].reshape(patch_pos_shape).permute(0, 3, 1, 2).float()
                        )
                        h0 = H + 0.1  # taken from the HunyuanOCR implementation
                        w0 = W + 0.1
                        patch_pos_embed = nn.functional.interpolate(
                            patch_pos_embed,
                            scale_factor=((h0 / self.max_patch_per_side), (w0 / self.max_patch_per_side)),
                            mode=self.interpolate_mode,
                            align_corners=False,
                        )
                        patch_pos_embed = patch_pos_embed.reshape(D, -1).transpose(0, 1)  # -> (N, D)
                        patch_embeds_flat = patch_embeds_flat + patch_pos_embed.to(patch_embeds_flat.dtype)
                        all_patch_embeds.append(patch_embeds_flat)
                    patch_embeds = torch.cat(all_patch_embeds, dim=0).unsqueeze(0)
                    mask = create_block_diagonal_mask(
                        [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
                        device=self.device,
                    )  # block diagonal mask delimiting images

            else:  # rope embedding / pixtral / Mistral
                patch_embeds = torch.cat([p.flatten(1).permute(1, 0) for p in patch_embeds_list], dim=0)
                if self.ln_pre is not None:
                    patch_embeds = self.ln_pre(patch_embeds)
                # create a fake batch dim
                patch_embeds = patch_embeds.unsqueeze(0)
                mask = create_block_diagonal_mask(
                    [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
                    device=self.device,
                )  # block diagonal mask delimiting images
                position_embeddings = self.rope.update(
                    patch_embeds.size(1),
                    step=0,
                    reset=True,
                    positions=torch.cat(positions).to(self.device),
                )

        out = patch_embeds

        for i, layer in enumerate(self.transformer_layers):
            out = layer(out, pad_mask=mask, position_embeddings=position_embeddings)

        if self.post_layernorm is not None:
            out = self.post_layernorm(out)

        return out


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


str2adapter = {
    "llava": VisionLanguageAdapter,
    "gemma3": Gemma3MultiModalProjector,
    "deepseekocr": DeepSeekOCRProjector,
    "hunyuanocr": HunYuanVisionPatchMerger,
}
