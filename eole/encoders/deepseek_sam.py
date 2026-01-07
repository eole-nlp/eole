"""
DeepSeek SAM (Segment Anything Model) Vision Encoder.

Based on Meta's SAM ViT architecture with modifications for DeepSeek integration.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from eole.encoders.encoder import EncoderBase


def interpolate_abs_pos(
    abs_pos: torch.Tensor,
    tgt_size: int,
) -> torch.Tensor:
    """
    Interpolate absolute positional embeddings to match target size.

    Args:
        abs_pos: Positional embeddings (1, src_size, src_size, dim)
        tgt_size: Target spatial dimension

    Returns:
        Interpolated embeddings (1, tgt_size, tgt_size, dim)
    """
    dtype = abs_pos.dtype
    src_size = abs_pos.size(1)

    if src_size == tgt_size:
        return abs_pos

    # Permute to (1, dim, src_size, src_size) for interpolation
    abs_pos = abs_pos.permute(0, 3, 1, 2).float()

    abs_pos = F.interpolate(
        abs_pos,
        size=(tgt_size, tgt_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(dtype)

    # Permute back to (1, tgt_size, tgt_size, dim)
    return abs_pos.permute(0, 2, 3, 1)


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization for convolutional features.

    Applies layer normalization over channel dimension while preserving
    spatial dimensions (H, W).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Normalized tensor with same shape
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    """
    Image to patch embedding via convolution.

    Converts images to non-overlapping patches using strided convolution.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (batch, channels, height, width)

        Returns:
            Patch embeddings (batch, num_patches_h, num_patches_w, embed_dim)
        """
        x = self.proj(x)
        # Permute from (B, C, H, W) to (B, H, W, C)
        return x.permute(0, 2, 3, 1)


class MLPBlock(nn.Module):
    """
    Simple two-layer MLP with configurable activation.
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type = nn.GELU,
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class RelativePositionAttention(nn.Module):
    """
    Multi-head attention with relative position embeddings.

    Implements decomposed relative positional bias for 2D spatial attention
    as described in MViTv2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            assert input_size is not None, "input_size required for relative position encoding"
            # Relative positional embeddings for height and width
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, height, width, dim)

        Returns:
            Attention output (batch, height, width, dim)
        """
        B, H, W, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv.unbind(0)

        # Reshape for attention computation
        q = q.view(B, self.num_heads, H * W, self.head_dim)
        k = k.view(B, self.num_heads, H * W, self.head_dim)
        v = v.view(B, self.num_heads, H * W, self.head_dim)

        # Compute attention with optional relative position bias
        if self.use_rel_pos:
            attn_bias = self._compute_rel_pos_bias(q, (H, W))
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        # Reshape and project output
        x = x.view(B, self.num_heads, H, W, self.head_dim)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        return self.proj(x)

    def _compute_rel_pos_bias(
        self,
        q: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute relative position bias for 2D spatial attention.

        Args:
            q: Query tensor (B, num_heads, H*W, head_dim)
            spatial_size: (height, width) of feature map

        Returns:
            Attention bias (B, num_heads, H*W, H*W)
        """
        H, W = spatial_size
        B, num_heads, _, _ = q.shape

        # Get relative position embeddings
        rel_h = self._get_rel_pos(H, H, self.rel_pos_h)
        rel_w = self._get_rel_pos(W, W, self.rel_pos_w)

        # Compute relative position attention
        q_reshaped = q.view(B, num_heads, H, W, self.head_dim)

        # Height-wise attention
        rel_h_attn = torch.einsum("bnhwc,hkc->bnhwk", q_reshaped, rel_h)
        rel_h_attn = rel_h_attn.unsqueeze(-1).reshape(B, num_heads, H * W, H, 1)

        # Width-wise attention
        rel_w_attn = torch.einsum("bnhwc,wkc->bnhwk", q_reshaped, rel_w)
        rel_w_attn = rel_w_attn.unsqueeze(-2).reshape(B, num_heads, H * W, 1, W)

        # Combine height and width biases
        attn_bias = (rel_h_attn + rel_w_attn).view(B, num_heads, H * W, H * W)

        return attn_bias

    @staticmethod
    def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Extract relative positional embeddings for given query/key sizes.

        Args:
            q_size: Query spatial size
            k_size: Key spatial size
            rel_pos: Relative position embeddings (L, C)

        Returns:
            Extracted embeddings (q_size, k_size, C)
        """
        max_rel_dist = 2 * max(q_size, k_size) - 1

        # Interpolate if needed
        if rel_pos.shape[0] != max_rel_dist:
            dtype = rel_pos.dtype
            rel_pos = rel_pos.float()
            rel_pos = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            ).to(dtype)
            rel_pos = rel_pos.squeeze(0).T

        # Get coordinates and compute relative positions
        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos[relative_coords.long()]


def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: Input tensor (batch, height, width, channels)
        window_size: Window size

    Returns:
        Tuple of:
            - Windowed tensor (batch * num_windows, window_size, window_size, channels)
            - Padded dimensions (height_padded, width_padded)
    """
    B, H, W, C = x.shape

    # Pad to make dimensions divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    # Reshape to windows
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Unpartition windows back to original feature map.

    Args:
        windows: Windowed tensor (batch * num_windows, window_size, window_size, channels)
        window_size: Window size
        pad_hw: Padded dimensions (height_padded, width_padded)
        hw: Original dimensions (height, width)

    Returns:
        Unpartitioned tensor (batch, height, width, channels)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    # Reshape windows back to feature map
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, Hp, Wp, -1)

    # Remove padding if needed
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]

    return x


class TransformerBlock(nn.Module):
    """
    Transformer block with optional windowed attention.

    Supports both global and windowed (local) attention patterns.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type = nn.LayerNorm,
        act_layer: type = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = RelativePositionAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(dim, mlp_hidden_dim, act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, height, width, channels)

        Returns:
            Output tensor with same shape
        """
        shortcut = x
        x = self.norm1(x)

        # Apply windowed attention if window_size > 0
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attn(x)

        # First residual connection
        x = shortcut + x

        # Second residual connection with MLP
        x = x + self.mlp(self.norm2(x))

        return x


class ImageEncoderViT(EncoderBase):
    """
    Vision Transformer encoder for SAM (Segment Anything Model).

    Processes images through patch embedding, transformer blocks, and
    outputs multi-scale features via convolutional neck.

    SAM uses a fixed architecture (ViT-B) and is not configurable.

    Args:
        encoder_config: Only used to check if SAM is enabled (optional)
        running_config: Runtime configuration (unused for SAM)
    """

    # SAM ViT-B architecture is fixed
    DEFAULT_CONFIG = {
        "img_size": 1024,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "out_chans": 256,
        "window_size": 14,
        "global_attn_indexes": (2, 5, 8, 11),
    }

    def __init__(self, encoder_config=None, running_config=None):
        super().__init__()

        # SAM has a fixed architecture - use defaults
        config = self.DEFAULT_CONFIG

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        embed_dim = config["embed_dim"]
        depth = config["depth"]
        num_heads = config["num_heads"]
        mlp_ratio = config["mlp_ratio"]
        out_chans = config["out_chans"]
        window_size = config["window_size"]
        global_attn_indexes = config["global_attn_indexes"]

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            in_chans=config["in_chans"],
            embed_dim=embed_dim,
        )

        # Absolute positional embedding
        num_patches = self.img_size // self.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    use_rel_pos=True,
                    rel_pos_zero_init=True,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(num_patches, num_patches),
                )
                for i in range(depth)
            ]
        )

        # Neck: project to output channels
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

        # Additional downsampling layers
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    @property
    def dtype(self):
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype

    def forward(
        self, emb: Union[torch.Tensor, list], pad_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """
        Process images through SAM encoder.

        Args:
            emb: List of N images of variable sizes, each (C, H, W)
            pad_mask: Not used for SAM (for EncoderBase compatibility)
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of:
                - Multi-scale features (batch, 1024, height/4, width/4)
                - None (SAM doesn't return hidden states)
        """

        x = torch.stack(emb, dim=0)

        # Patch embedding
        x = self.patch_embed(x.to(self.dtype))

        # Add positional embedding
        x = x + interpolate_abs_pos(self.pos_embed, x.size(1))

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply neck and downsampling
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = self.neck(x)
        x = self.net_2(x)
        x = self.net_3(x)

        return x, None
