"""Tests for Q-Former Visual Bridge."""

import pytest
import torch
from mini_vla.config import MiniVLAConfig
from mini_vla.qformer import (
    QFormerSelfAttention,
    QFormerCrossAttention,
    QFormerFFN,
    QFormerLayer,
    QFormer,
)


@pytest.fixture
def small_config():
    return MiniVLAConfig(
        img_size=56,
        patch_size=14,
        vit_dim=64,
        qformer_num_queries=8,
        qformer_dim=32,
        qformer_depth=2,
        qformer_heads=4,
        qformer_mlp_ratio=2.0,
        llm_dim=48,
    )


class TestQFormerSelfAttention:
    def test_output_shape(self):
        sa = QFormerSelfAttention(dim=32, num_heads=4)
        x = torch.randn(2, 8, 32)
        out = sa(x)
        assert out.shape == (2, 8, 32)


class TestQFormerCrossAttention:
    def test_output_shape(self):
        ca = QFormerCrossAttention(q_dim=32, kv_dim=64, num_heads=4)
        query = torch.randn(2, 8, 32)
        kv = torch.randn(2, 16, 64)
        out = ca(query, kv)
        assert out.shape == (2, 8, 32)

    def test_different_kv_length(self):
        ca = QFormerCrossAttention(q_dim=32, kv_dim=64, num_heads=4)
        query = torch.randn(2, 8, 32)
        kv = torch.randn(2, 100, 64)
        out = ca(query, kv)
        assert out.shape == (2, 8, 32)


class TestQFormerFFN:
    def test_output_shape(self):
        ffn = QFormerFFN(dim=32, mlp_dim=64)
        x = torch.randn(2, 8, 32)
        out = ffn(x)
        assert out.shape == (2, 8, 32)


class TestQFormerLayer:
    def test_output_shape(self, small_config):
        layer = QFormerLayer(small_config)
        query = torch.randn(2, 8, 32)
        vit_feats = torch.randn(2, 16, 64)
        out = layer(query, vit_feats)
        assert out.shape == (2, 8, 32)


class TestQFormer:
    def test_output_shape(self, small_config):
        qf = QFormer(small_config)
        vit_feats = torch.randn(2, 16, 64)
        out = qf(vit_feats)
        # (B, N_q, d_llm) = (2, 8, 48)
        assert out.shape == (2, 8, 48)

    def test_compression_ratio(self, small_config):
        """Q-Former should compress N_p tokens to N_q tokens."""
        qf = QFormer(small_config)
        vit_feats = torch.randn(2, 16, 64)  # 16 patch tokens
        out = qf(vit_feats)
        assert out.shape[1] == small_config.qformer_num_queries  # compressed to 8

    def test_output_dimension_matches_llm(self, small_config):
        qf = QFormer(small_config)
        vit_feats = torch.randn(1, 16, 64)
        out = qf(vit_feats)
        assert out.shape[-1] == small_config.llm_dim

    def test_gradient_flow(self, small_config):
        qf = QFormer(small_config)
        vit_feats = torch.randn(1, 16, 64, requires_grad=True)
        out = qf(vit_feats)
        loss = out.sum()
        loss.backward()
        assert vit_feats.grad is not None

    def test_learnable_queries_updated(self, small_config):
        qf = QFormer(small_config)
        initial_queries = qf.queries.clone()

        optimizer = torch.optim.SGD(qf.parameters(), lr=0.01)
        vit_feats = torch.randn(2, 16, 64)
        out = qf(vit_feats)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        assert not torch.equal(qf.queries, initial_queries)

    def test_variable_input_length(self, small_config):
        """Q-Former should handle variable ViT feature lengths."""
        qf = QFormer(small_config)
        vit_feats_short = torch.randn(1, 8, 64)
        vit_feats_long = torch.randn(1, 100, 64)

        out_short = qf(vit_feats_short)
        out_long = qf(vit_feats_long)

        # Output should always be (B, N_q, d_llm) regardless of input length
        assert out_short.shape == (1, 8, 48)
        assert out_long.shape == (1, 8, 48)
