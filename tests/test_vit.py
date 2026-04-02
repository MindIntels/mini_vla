"""Tests for ViT Vision Encoder."""

import pytest
import torch
from mini_vla.config import MiniVLAConfig
from mini_vla.vit import PatchEmbedding, ViTAttention, ViTMLP, ViTBlock, ViTEncoder


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return MiniVLAConfig(
        img_size=56,
        patch_size=14,
        vit_dim=64,
        vit_depth=2,
        vit_heads=4,
        vit_mlp_ratio=2.0,
    )


class TestPatchEmbedding:
    def test_output_shape(self, small_config):
        pe = PatchEmbedding(small_config)
        x = torch.randn(2, 3, 56, 56)
        out = pe(x)
        # 56/14 = 4, N_p = 4*4 = 16
        assert out.shape == (2, 16, 64)

    def test_different_image_size(self):
        config = MiniVLAConfig(img_size=112, patch_size=14, vit_dim=64)
        pe = PatchEmbedding(config)
        x = torch.randn(1, 3, 112, 112)
        out = pe(x)
        # 112/14 = 8, N_p = 64
        assert out.shape == (1, 64, 64)


class TestViTAttention:
    def test_output_shape(self):
        attn = ViTAttention(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_single_head(self):
        attn = ViTAttention(dim=32, num_heads=1)
        x = torch.randn(1, 8, 32)
        out = attn(x)
        assert out.shape == (1, 8, 32)


class TestViTMLP:
    def test_output_shape(self):
        mlp = ViTMLP(dim=64, mlp_dim=128)
        x = torch.randn(2, 16, 64)
        out = mlp(x)
        assert out.shape == (2, 16, 64)


class TestViTBlock:
    def test_output_shape(self):
        blk = ViTBlock(dim=64, num_heads=4, mlp_dim=128)
        x = torch.randn(2, 16, 64)
        out = blk(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self):
        blk = ViTBlock(dim=64, num_heads=4, mlp_dim=128)
        x = torch.zeros(1, 8, 64)
        # With zero input, output should be non-zero (bias terms)
        out = blk(x)
        assert out.shape == (1, 8, 64)


class TestViTEncoder:
    def test_output_shape(self, small_config):
        vit = ViTEncoder(small_config)
        x = torch.randn(2, 3, 56, 56)
        out = vit(x)
        # CLS removed, only patch tokens: (B, N_p, d_vit)
        assert out.shape == (2, 16, 64)

    def test_cls_removed(self, small_config):
        """Verify CLS token is removed from output."""
        vit = ViTEncoder(small_config)
        x = torch.randn(1, 3, 56, 56)
        out = vit(x)
        # N_p = 16, NOT 17 (CLS removed)
        assert out.shape[1] == small_config.num_patches

    def test_gradient_flow(self, small_config):
        vit = ViTEncoder(small_config)
        x = torch.randn(1, 3, 56, 56, requires_grad=True)
        out = vit(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_independence(self, small_config):
        """Each sample in batch should be processed independently."""
        vit = ViTEncoder(small_config)
        vit.eval()
        x1 = torch.randn(1, 3, 56, 56)
        x2 = torch.randn(1, 3, 56, 56)
        x_batch = torch.cat([x1, x2], dim=0)

        out_batch = vit(x_batch)
        out1 = vit(x1)
        out2 = vit(x2)

        torch.testing.assert_close(out_batch[0], out1[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_batch[1], out2[0], atol=1e-5, rtol=1e-5)

    def test_param_count(self, small_config):
        vit = ViTEncoder(small_config)
        total_params = sum(p.numel() for p in vit.parameters())
        assert total_params > 0
