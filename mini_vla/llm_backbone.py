"""LLM Backbone (Qwen2.5VL-7B style, simulated).

Implements a simulated Qwen2.5VL backbone with:
- Token embedding
- GQA (Grouped Query Attention) with RoPE
- SwiGLU FFN
- RMSNorm (Pre-Norm)
- Action condition extraction (last token hidden state)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_vla.config import MiniVLAConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0, device=None):
    """Build RoPE (Rotary Position Embedding) cos/sin cache."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, head_dim/2)
    cos = torch.cos(angles)  # (seq_len, head_dim/2)
    sin = torch.sin(angles)  # (seq_len, head_dim/2)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor.

    Args:
        x: (B, H, T, d_head)
        cos: (T, d_head/2)
        sin: (T, d_head/2)
    """
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return x * cos_full + rotated * sin_full


class GQAttention(nn.Module):
    """Grouped Query Attention (GQA).

    Q heads = llm_heads (28), KV heads = llm_kv_heads (4)
    Each KV head is shared by (llm_heads // llm_kv_heads) Q heads.
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.num_q_heads = config.llm_heads
        self.num_kv_heads = config.llm_kv_heads
        self.head_dim = config.llm_head_dim
        self.groups = self.num_q_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.llm_dim, self.num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(config.llm_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.llm_dim, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.llm_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Expand KV heads for GQA
        if self.groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(B, self.num_q_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(B, self.num_q_heads, T, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLMBlock(nn.Module):
    """Single Qwen2.5VL decoder block."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.llm_dim)
        self.attn = GQAttention(config)
        self.norm2 = RMSNorm(config.llm_dim)
        self.ffn = SwiGLU(config.llm_dim, config.llm_ffn_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class LLMBackbone(nn.Module):
    """Simulated Qwen2.5VL-7B backbone.

    Pipeline:
        visual_tokens (B, N_v, d_llm) + text_tokens (B, T_txt)
        -> token embedding -> concat -> decoder layers -> hidden states
        -> extract last token as action condition (B, 1, d_llm)
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        # Token embedding for text
        self.token_emb = nn.Embedding(config.llm_vocab_size, config.llm_dim)

        # Robot state projection (optional)
        self.state_proj = nn.Linear(config.robot_state_dim, config.llm_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            LLMBlock(config) for _ in range(config.llm_depth)
        ])

        self.norm = RMSNorm(config.llm_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_token_ids: torch.Tensor,
        robot_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, N_v, d_llm) from Q-Former
            text_token_ids: (B, T_txt) integer token IDs
            robot_state: (B, d_state) optional robot state

        Returns:
            (B, d_llm) action condition vector (last token hidden state)
        """
        # Text embedding
        text_emb = self.token_emb(text_token_ids)  # (B, T_txt, d_llm)

        # Build input sequence: [visual_tokens, text_emb, (state_emb)]
        parts = [visual_tokens, text_emb]

        if robot_state is not None:
            state_emb = self.state_proj(robot_state).unsqueeze(1)  # (B, 1, d_llm)
            parts.append(state_emb)

        x = torch.cat(parts, dim=1)  # (B, T_total, d_llm)

        # Build RoPE cache
        T_total = x.shape[1]
        cos, sin = _build_rope_cache(
            T_total, self.config.llm_head_dim,
            theta=self.config.llm_rope_theta,
            device=x.device,
        )

        # Decoder layers
        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)

        # Extract last token hidden state as action condition
        action_cond = x[:, -1, :]  # (B, d_llm)
        return action_cond
