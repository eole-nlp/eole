"""Vision encoder for multimodal models."""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from eole.encoders.encoder import EncoderBase
from eole.encoders.transformer import TransformerEncoderLayer
from eole.modules.rope import build_rope
from eole.constants import PositionEncodingType, LayerNorm


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
    mask = torch.zeros((total_length, total_length), dtype=torch.bool)

    start = 0
    for length in lengths:
        end = start + length
        mask[start:end, start:end] = True
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


class VisionEncoder(EncoderBase):
    """
    Vision encoder for processing images into token representations.

    Supports various vision architectures:
    - CLIP-style with learned positional embeddings
    - Pixtral with RoPE 2D embeddings
    - SAM (Segment Anything Model) preprocessing

    Args:
        encoder_config: Vision encoder configuration
        running_config: Runtime configuration (optional)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()

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

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(encoder_config, running_config=running_config)
                for _ in range(encoder_config.layers)
            ]
        )

        if encoder_config.layernorm_post:
            self.post_layernorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.norm_eps)
        else:
            self.post_layernorm = None

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        emb: List[torch.Tensor],  # List of images for vision
        pad_mask: Optional[torch.Tensor] = None,  # Not used directly
        sam_patches: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """
        Encode images into token representations.

        Args:
            emb: List of N images of variable sizes, each (C, H, W)
            pad_mask: Not used for vision encoder (uses block diagonal masks)
            sam_patches: Pre-computed SAM patches (optional)
            **kwargs: Additional arguments

        Returns:
            Tuple of:
                - Encoded image features (N_img, total_tokens, hidden_size)
                - None (vision encoders don't return hidden states)
        """
        dtype = next(self.parameters()).dtype
        mask = None
        position_embeddings = None

        if sam_patches is not None:
            patch_embeds = self._process_sam_patches(sam_patches)
        else:
            images = emb  # Rename for clarity in vision context
            patch_embeds, mask, position_embeddings = self._process_images(images, dtype)

        # Apply transformer layers
        out = patch_embeds
        for layer in self.transformer_layers:
            out = layer(out, attn_mask=mask, position_embeddings=position_embeddings)

        # Post-processing
        if self.post_layernorm is not None:
            out = self.post_layernorm(out)

        return out

    def _process_sam_patches(self, sam_patches: torch.Tensor) -> torch.Tensor:
        """Process pre-computed SAM patches."""
        # Flatten H+W then transpose to (batch, H*W, C)
        patch_embeds = sam_patches.flatten(2).transpose(1, 2)

        if self.class_embedding is not None:
            batch_size = patch_embeds.size(0)
            class_embeds = self.class_embedding.weight.expand(batch_size, -1, -1)
            patch_embeds = torch.cat([class_embeds, patch_embeds], dim=1)

            # Add positional embeddings
            patch_embeds += get_abs_pos(self.position_embeddings(self.position_ids), patch_embeds.size(1))

        if self.ln_pre is not None:
            patch_embeds = self.ln_pre(patch_embeds)

        return patch_embeds

    def _process_images(
        self, images: List[torch.Tensor], dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process list of images into patch embeddings.

        Returns:
            Tuple of (patch_embeds, mask, position_embeddings)
        """
        # Extract patches from all images
        patch_embeds_list = [self.patch_conv(img.to(dtype)) for img in images]
        positions = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.encoder_config.image_size // self.encoder_config.patch_size,
        )
        if hasattr(self, "position_embeddings"):
            return self._process_with_learned_positions(patch_embeds_list, positions)
        else:
            return self._process_with_rope(patch_embeds_list, positions)

    def _process_with_learned_positions(
        self, patch_embeds_list: List[torch.Tensor], positions
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """Process patches with learned positional embeddings."""

        if self.interpolate_mode is None:
            # Fixed size images (Gemma3 / DeepseekOCR)
            patch_pos_embed = self.position_embeddings(torch.stack(positions).to(self.device))
            patch_embeds = torch.cat([p.unsqueeze(0) for p in patch_embeds_list], dim=0)
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
            patch_embeds += patch_pos_embed
            mask = None
        else:
            # Variable size images (HunyuanOCR)
            patch_embeds = self._interpolate_positions(patch_embeds_list)
            mask = create_block_diagonal_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], device=self.device
            )

        return patch_embeds, mask, None

    def _interpolate_positions(self, patch_embeds_list: List[torch.Tensor]) -> torch.Tensor:
        """Interpolate positional embeddings for variable-sized images."""
        all_patch_embeds = []

        for patch_embeds in patch_embeds_list:
            D, H, W = patch_embeds.size()

            # Flatten patches
            patch_embeds_flat = patch_embeds.flatten(1).transpose(0, 1)  # (H*W, D)

            # Interpolate position embeddings to match image size
            patch_pos_shape = (1, self.max_patch_per_side, self.max_patch_per_side, D)
            patch_pos_embed = (
                self.position_embeddings.weight[1:, :].reshape(patch_pos_shape).permute(0, 3, 1, 2).float()
            )

            # Scale factors with slight offset (from HunyuanOCR implementation)
            h_scale = (H + 0.1) / self.max_patch_per_side
            w_scale = (W + 0.1) / self.max_patch_per_side

            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed, scale_factor=(h_scale, w_scale), mode=self.interpolate_mode, align_corners=False
            )

            patch_pos_embed = patch_pos_embed.reshape(D, -1).transpose(0, 1)  # (H*W, D)
            patch_embeds_flat = patch_embeds_flat + patch_pos_embed.to(patch_embeds_flat.dtype)
            all_patch_embeds.append(patch_embeds_flat)

        return torch.cat(all_patch_embeds, dim=0).unsqueeze(0)

    def _process_with_rope(
        self, patch_embeds_list: List[torch.Tensor], positions
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process patches with RoPE 2D embeddings (Pixtral/Mistral style)."""

        # Concatenate all patches
        patch_embeds = torch.cat([p.flatten(1).permute(1, 0) for p in patch_embeds_list], dim=0)

        if self.ln_pre is not None:
            patch_embeds = self.ln_pre(patch_embeds)

        # Add batch dimension
        patch_embeds = patch_embeds.unsqueeze(0)

        # Create block diagonal mask to separate images
        mask = create_block_diagonal_mask([p.shape[-2] * p.shape[-1] for p in patch_embeds_list], device=self.device)

        # Compute RoPE embeddings
        position_embeddings = self.rope.update(
            patch_embeds.size(1), step=0, reset=True, positions=torch.cat(positions).to(self.device)
        )

        return patch_embeds, mask, position_embeddings

    def update_dropout(self, dropout: float, attention_dropout: float) -> None:
        """Update dropout rates for all transformer layers."""
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
