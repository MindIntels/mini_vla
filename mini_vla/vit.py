"""ViT Vision Encoder (ViT-L/14 style).

Implements a standard Vision Transformer with:
- Patch embedding via Conv2d
- CLS token + learnable position embeddings
- Pre-LN Transformer encoder blocks
- Patch token extraction (drop CLS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_vla.config import MiniVLAConfig


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings using Conv2d."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            3, config.vit_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.num_patches = config.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, N_p, d_vit)
        x = self.proj(x)  # (B, d_vit, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N_p, d_vit)
        return x


class ViTAttention(nn.Module):
    """Multi-Head Self-Attention for ViT."""

    def __init__(self, dim: int, num_heads: int, drop_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d_head)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTMLP(nn.Module):
    """MLP block for ViT with GELU activation."""

    def __init__(self, dim: int, mlp_dim: int, drop_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViTBlock(nn.Module):
    """Pre-LN Transformer block for ViT."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, num_heads, drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_dim, drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder (ViT-L/14 style).

    Pipeline:
        Image (B,3,H,W) -> PatchEmbed -> [CLS]+PE -> TransformerBlocks -> patch_tokens (B,N_p,d_vit)
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vit_dim))

        # Position embedding (learnable, for CLS + patches)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.vit_dim)
        )

        self.pos_drop = nn.Dropout(config.vit_drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=config.vit_dim,
                num_heads=config.vit_heads,
                mlp_dim=config.vit_mlp_dim,
                drop_rate=config.vit_drop_rate,
            )
            for _ in range(config.vit_depth)
        ])

        self.norm = nn.LayerNorm(config.vit_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            (B, N_p, d_vit) patch token features (CLS removed)
        """
        B = x.shape[0]

        # Patch embedding: (B, N_p, d_vit)
        x = self.patch_embed(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N_p+1, d_vit)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Remove CLS token, return patch tokens only
        patch_tokens = x[:, 1:, :]  # (B, N_p, d_vit)
        return patch_tokens
