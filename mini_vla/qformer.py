"""Q-Former Visual Bridge (BLIP-2 style).

Implements the Querying Transformer with:
- Learnable queries (N_q=32, d=768)
- Self-Attention among queries
- Cross-Attention from queries to ViT patch tokens
- FFN
- Output projection to LLM dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_vla.config import MiniVLAConfig


class QFormerSelfAttention(nn.Module):
    """Multi-Head Self-Attention among Q-Former queries."""

    def __init__(self, dim: int, num_heads: int, drop_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class QFormerCrossAttention(nn.Module):
    """Cross-Attention: queries attend to ViT features."""

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, drop_rate: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.attn_drop = nn.Dropout(drop_rate)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_q) learnable queries
            kv: (B, N_p, d_kv) ViT patch features
        Returns:
            (B, N_q, d_q)
        """
        B, N_q, C = query.shape
        N_kv = kv.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        return x


class QFormerFFN(nn.Module):
    """Feed-Forward Network for Q-Former."""

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


class QFormerLayer(nn.Module):
    """Single Q-Former layer with Self-Attention + Cross-Attention + FFN."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        d_q = config.qformer_dim

        # Self-Attention
        self.norm_sa = nn.LayerNorm(d_q)
        self.self_attn = QFormerSelfAttention(
            d_q, config.qformer_heads, config.qformer_drop_rate,
        )

        # Cross-Attention (query -> ViT features)
        self.norm_ca = nn.LayerNorm(d_q)
        self.cross_attn = QFormerCrossAttention(
            q_dim=d_q,
            kv_dim=config.vit_dim,
            num_heads=config.qformer_heads,
            drop_rate=config.qformer_drop_rate,
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(d_q)
        self.ffn = QFormerFFN(d_q, config.qformer_mlp_dim, config.qformer_drop_rate)

    def forward(self, query: torch.Tensor, vit_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_q)
            vit_features: (B, N_p, d_vit)
        Returns:
            (B, N_q, d_q)
        """
        # Self-Attention among queries
        query = query + self.self_attn(self.norm_sa(query))

        # Cross-Attention: queries attend to ViT features
        query = query + self.cross_attn(self.norm_ca(query), vit_features)

        # FFN
        query = query + self.ffn(self.norm_ffn(query))

        return query


class QFormer(nn.Module):
    """Q-Former: Querying Transformer for visual information compression.

    Pipeline:
        ViT features (B, N_p, d_vit) + learnable queries (N_q, d_q)
        -> Q-Former layers (SA + CA + FFN) x depth
        -> compressed visual tokens (B, N_q, d_q)
        -> linear projection to LLM dim (B, N_q, d_llm)
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        # Learnable queries
        self.queries = nn.Parameter(
            torch.zeros(1, config.qformer_num_queries, config.qformer_dim)
        )
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(config) for _ in range(config.qformer_depth)
        ])

        self.norm = nn.LayerNorm(config.qformer_dim)

        # Output projection: d_q -> d_llm
        self.proj = nn.Linear(config.qformer_dim, config.llm_dim)

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vit_features: (B, N_p, d_vit) patch tokens from ViT

        Returns:
            (B, N_q, d_llm) compressed visual tokens projected to LLM dim
        """
        B = vit_features.shape[0]

        # Expand learnable queries for batch
        query = self.queries.expand(B, -1, -1)  # (B, N_q, d_q)

        # Pass through Q-Former layers
        for layer in self.layers:
            query = layer(query, vit_features)

        query = self.norm(query)

        # Project to LLM dimension
        visual_tokens = self.proj(query)  # (B, N_q, d_llm)
        return visual_tokens
