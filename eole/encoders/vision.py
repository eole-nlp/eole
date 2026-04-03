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


def position_ids_in_meshgrid(
    patch_embeds_list: List[torch.Tensor],
    max_width: int,
) -> List[torch.Tensor]:
    """
    Compute flattened 2D position indices for patch grids.

    For each image's patch embedding tensor of shape (D, H, W), this function:
    - creates a 2D meshgrid of (row, column) indices
    - flattens the grid in row-major (raster) order
    - converts 2D coordinates into 1D position IDs using:
        position_id = row * max_width + column

    This produces position indices compatible with learned absolute
    positional embeddings and block-diagonal attention masks, even for
    variable-sized images.

    Args:
        patch_embeds_list:
            List of patch embedding tensors, one per image.
            Each tensor has shape (hidden_dim, height, width).
        max_width:
            Maximum number of patches per row, used to compute unique
            linearized position IDs across images.

    Returns:
        List[torch.Tensor]:
            A list of 1D tensors, one per image.
            Each tensor has shape (height * width,) and contains integer
            position IDs corresponding to flattened patch locations.
    """
    positions: List[torch.Tensor] = []

    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]

        h, w = torch.meshgrid(
            torch.arange(height, device=patch.device),
            torch.arange(width, device=patch.device),
            indexing="ij",
        )

        pos_ids = (h * max_width + w).reshape(-1)
        positions.append(pos_ids)

    return positions


def create_block_diagonal_mask(
    lengths: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a block-diagonal attention mask for concatenated sequences.

    The mask is a 2D boolean tensor where each diagonal block corresponds
    to a contiguous subsequence and is marked as `True`, while all
    off-diagonal positions are `False`.

    This is typically used to prevent attention across different images
    or segments when multiple sequences are concatenated along the
    sequence dimension.

    Args:
        lengths:
            List of sequence lengths for each block (e.g., number of
            patches per image).
        device:
            Device on which to allocate the mask tensor.

    Returns:
        torch.Tensor:
            A boolean tensor of shape (total_length, total_length),
            where `total_length = sum(lengths)`.
            Positions within the same block are `True` and positions
            across different blocks are `False`.
    """
    total_length = sum(lengths)
    mask = torch.zeros((total_length, total_length), dtype=torch.bool, device=device)

    start = 0
    for length in lengths:
        end = start + length
        mask[start:end, start:end] = True
        start = end

    return mask


def get_abs_pos(
    abs_pos: torch.Tensor,
    tgt_size: int,
) -> torch.Tensor:
    """
    Resize absolute positional embeddings for a new target grid size.

    This function adapts learned ViT-style absolute positional embeddings
    when the number of image patches changes (e.g., due to different image
    resolutions at inference time).

    The input embedding is expected to contain a leading [CLS] token
    followed by flattened 2D patch embeddings. Patch embeddings are
    reshaped into a square grid, resized via bicubic interpolation,
    and then flattened back.

    Args:
        abs_pos:
            Absolute positional embeddings of shape (1, N + 1, D),
            where N is the number of patch positions and D is the
            embedding dimension. The first position corresponds to
            the [CLS] token.
        tgt_size:
            Target number of patch positions (excluding the [CLS] token).
            Must be a perfect square.

    Returns:
        torch.Tensor:
            Resized absolute positional embeddings of shape
            (1, tgt_size + 1, D), including the [CLS] token.
            If the source and target grid sizes match, the input
            tensor is returned unchanged.
    """
    dim = abs_pos.size(-1)

    abs_pos = abs_pos.squeeze(0)
    cls_token, patch_pos = abs_pos[:1], abs_pos[1:]

    src_size = int(math.sqrt(patch_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size == tgt_size:
        return abs_pos.unsqueeze(0)

    # Reshape to (1, D, H, W) for interpolation
    patch_pos = patch_pos.view(1, src_size, src_size, dim).permute(0, 3, 1, 2)
    patch_pos = patch_pos.to(torch.float32)

    patch_pos = F.interpolate(
        patch_pos,
        size=(tgt_size, tgt_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(dtype)

    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(tgt_size * tgt_size, dim)

    out = torch.cat([cls_token, patch_pos], dim=0)
    return out.unsqueeze(0)


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
            from eole.encoders.deepseek_sam import ImageEncoderViT

            self.sam = ImageEncoderViT()
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

        # Qwen3.5 VL / Qwen3 VL: optional absolute position embedding table
        # used together with 2D RoPE (position_encoding_type=Rotary)
        if encoder_config.num_position_embeddings and encoder_config.num_position_embeddings > 0:
            self.pos_embed = nn.Embedding(
                encoder_config.num_position_embeddings,
                encoder_config.hidden_size,
            )
        else:
            self.pos_embed = None

        # Gemma4: learnable 2D position embedding table of shape
        # [2, position_embedding_size, hidden_size] added to patch embeddings.
        # Each axis (x and y) is looked up independently via one-hot and summed.
        # Initialized with ones to match HF's Gemma4PreTrainedModel._init_weights
        # (pre-trained weights will overwrite this at load time).
        pos_emb_size = getattr(encoder_config, "position_embedding_size", None)
        if pos_emb_size:
            self.position_embedding_table = nn.Parameter(torch.ones(2, pos_emb_size, encoder_config.hidden_size))
        else:
            self.position_embedding_table = None

        # Patch conv: Conv3D when temporal_patch_size > 1, Conv2D otherwise
        if encoder_config.temporal_patch_size > 1:
            self.patch_conv = nn.Conv3d(
                in_channels=encoder_config.num_channels,
                out_channels=encoder_config.hidden_size,
                kernel_size=(
                    encoder_config.temporal_patch_size,
                    encoder_config.patch_size,
                    encoder_config.patch_size,
                ),
                stride=(
                    encoder_config.temporal_patch_size,
                    encoder_config.patch_size,
                    encoder_config.patch_size,
                ),
                bias=encoder_config.patch_conv_bias,
            )
        else:
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

        # Gemma4 standardization (std_bias/std_scale) is applied after pooling
        # in the adapter (Gemma4MultiModalProjector), not here in the encoder.

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

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
        mask = None
        position_embeddings = None

        if sam_patches is not None:
            patch_embeds = self._process_sam_patches(sam_patches)
        else:
            images = emb  # Rename for clarity in vision context
            patch_embeds, mask, position_embeddings = self._process_images(images, self.dtype)

        # Apply transformer layers
        out = patch_embeds
        for layer in self.transformer_layers:
            out = layer(out, attn_mask=mask, position_embeddings=position_embeddings)

        # Post-processing
        if self.post_layernorm is not None:
            out = self.post_layernorm(out)

        return out, None

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
        if isinstance(self.patch_conv, nn.Conv3d):
            patch_embeds_list = self._apply_conv3d(images)
        else:
            patch_embeds_list = [self.patch_conv(img.to(self.dtype)) for img in images]
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

            patch_pos_embed = F.interpolate(
                patch_pos_embed, scale_factor=(h_scale, w_scale), mode=self.interpolate_mode, align_corners=False
            )

            patch_pos_embed = patch_pos_embed.reshape(D, -1).transpose(0, 1)  # (H*W, D)
            patch_embeds_flat = patch_embeds_flat + patch_pos_embed.to(patch_embeds_flat.dtype)
            all_patch_embeds.append(patch_embeds_flat)

        return torch.cat(all_patch_embeds, dim=0).unsqueeze(0)

    def _apply_conv3d(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply Conv3D patch embedding to images by treating each as a 2-frame video.

        Qwen3.5 VL uses ``nn.Conv3d`` with ``temporal_patch_size`` frames.
        For still images the same image is duplicated along the temporal axis.

        Args:
            images: List of images, each of shape (C, H, W).

        Returns:
            List of patch-embedding tensors, each of shape
            (hidden_size, H//patch_size, W//patch_size).
        """
        temporal = self.encoder_config.temporal_patch_size
        patch_embeds_list = []
        for img in images:
            img = img.to(self.dtype)
            # (C, H, W) → (1, C, temporal, H, W) by duplicating across time
            img_t = img.unsqueeze(1).expand(-1, temporal, -1, -1)  # (C, T, H, W)
            img_t = img_t.unsqueeze(0)  # (1, C, T, H, W)
            # Conv3d output: (1, hidden_size, 1, H//patch, W//patch)
            pe = self.patch_conv(img_t)
            # Drop temporal dim (always 1 after pooling) → (hidden_size, H//patch, W//patch)
            pe = pe.squeeze(0).squeeze(1)
            patch_embeds_list.append(pe)
        return patch_embeds_list

    def _interp_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """
        Bilinear interpolation of the absolute position-embedding table for an
        arbitrary patch grid of size ``(h, w)``.

        Matches HF's ``fast_pos_embed_interpolate``: patch coordinates are
        mapped *uniformly* to ``[0, num_grid - 1]`` (via ``linspace``) and the
        2-D embedding table is sampled with bilinear interpolation.  This
        ensures that the full extent of the learned table is used for any
        image resolution rather than only the top-left corner.

        Args:
            h: height in patches.
            w: width in patches.

        Returns:
            Tensor of shape ``(h * w, hidden_size)`` on the encoder device,
            in row-major patch order.
        """
        num_grid = int(self.pos_embed.num_embeddings**0.5)
        D = self.pos_embed.embedding_dim
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        # Fast path: exact-size images need no interpolation
        if h == num_grid and w == num_grid:
            return self.pos_embed.weight  # (num_grid^2, D)

        # Degenerate table (single entry)
        if num_grid <= 1:
            return self.pos_embed.weight.expand(h * w, -1)

        # Map [0, h-1] uniformly to [0, num_grid-1] (and same for w)
        h_idxs = torch.linspace(0, num_grid - 1, h, device=device)
        w_idxs = torch.linspace(0, num_grid - 1, w, device=device)
        h_pos, w_pos = torch.meshgrid(h_idxs, w_idxs, indexing="ij")  # (h, w)

        # Normalise to [-1, 1] for F.grid_sample (align_corners=True)
        h_norm = 2.0 * h_pos / (num_grid - 1) - 1.0
        w_norm = 2.0 * w_pos / (num_grid - 1) - 1.0
        # grid_sample convention: last dim is (x=col, y=row)
        grid = torch.stack([w_norm, h_norm], dim=-1).unsqueeze(0)  # (1, h, w, 2)

        # Reshape embedding table to (1, D, num_grid, num_grid) for interpolation
        embed_grid = self.pos_embed.weight.reshape(num_grid, num_grid, D).permute(2, 0, 1).unsqueeze(0).float()

        interp = F.grid_sample(embed_grid, grid, mode="bilinear", padding_mode="border", align_corners=True)
        # (1, D, h, w) → (h*w, D)
        return interp.squeeze(0).permute(1, 2, 0).reshape(h * w, D).to(dtype)

    def _process_with_rope(
        self, patch_embeds_list: List[torch.Tensor], positions
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process patches with RoPE 2D embeddings (Pixtral/Mistral/Qwen3.5 VL style)."""

        # Concatenate all patches
        patch_embeds = torch.cat([p.flatten(1).permute(1, 0) for p in patch_embeds_list], dim=0)

        if self.ln_pre is not None:
            patch_embeds = self.ln_pre(patch_embeds)

        # Qwen3.5 VL / Qwen3 VL: add absolute position embeddings on top of RoPE.
        # Use per-image bilinear interpolation to uniformly sample the embedding
        # table for any image resolution (matches HF fast_pos_embed_interpolate).
        if self.pos_embed is not None:
            pos_emb = torch.cat(
                [self._interp_pos_embed(p.shape[-2], p.shape[-1]) for p in patch_embeds_list],
                dim=0,
            )
            patch_embeds = patch_embeds + pos_emb

        # Gemma4: add learned 2D position embeddings via position_embedding_table.
        # positions is a list of 1D tensors with flat indices (row * max_width + col)
        # produced by position_ids_in_meshgrid.  Convert back to (row, col) coords
        # so each axis can be looked up independently in the [2, P, H] table.
        if self.position_embedding_table is not None:
            all_flat = torch.cat(positions, dim=0).to(self.device)  # (total_patches,)
            max_width = self.encoder_config.image_size // self.encoder_config.patch_size
            P = self.position_embedding_table.shape[1]
            row = all_flat // max_width  # height / row coordinate
            col = all_flat % max_width  # width / col coordinate
            row_hot = F.one_hot(row, num_classes=P).to(self.position_embedding_table.dtype)  # (N, P)
            col_hot = F.one_hot(col, num_classes=P).to(self.position_embedding_table.dtype)  # (N, P)
            # HF's position_embedding_table: index 0 = x/col, index 1 = y/row
            pos_emb = col_hot @ self.position_embedding_table[0] + row_hot @ self.position_embedding_table[1]
            patch_embeds = patch_embeds + pos_emb

        # Add batch dimension
        patch_embeds = patch_embeds.unsqueeze(0)

        # Create block diagonal mask to separate images
        mask = create_block_diagonal_mask([p.shape[-2] * p.shape[-1] for p in patch_embeds_list], device=self.device)

        # Compute RoPE embeddings
        position_embeddings = self.rope.update(
            patch_embeds.size(1), reset=True, positions=torch.cat(positions).to(self.device)
        )

        return patch_embeds, mask, position_embeddings

    def update_dropout(self, dropout: float, attention_dropout: float) -> None:
        """Update dropout rates for all transformer layers."""
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
