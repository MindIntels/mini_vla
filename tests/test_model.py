"""Tests for full Mini-VLA Model."""

import pytest
import torch
from mini_vla.config import MiniVLAConfig
from mini_vla.model import MiniVLAModel


@pytest.fixture
def small_config():
    """Minimal config for fast full-model testing."""
    return MiniVLAConfig(
        img_size=28,
        patch_size=14,
        vit_dim=32,
        vit_depth=1,
        vit_heads=4,
        vit_mlp_ratio=2.0,
        qformer_num_queries=4,
        qformer_dim=24,
        qformer_depth=1,
        qformer_heads=4,
        qformer_mlp_ratio=2.0,
        llm_dim=32,
        llm_depth=1,
        llm_heads=4,
        llm_kv_heads=2,
        llm_ffn_dim=64,
        llm_vocab_size=100,
        llm_max_seq_len=64,
        action_dim=7,
        action_chunk_size=4,
        action_head_dim=16,
        action_head_depth=1,
        robot_state_dim=7,
        diffusion_steps=10,
        ddim_inference_steps=3,
        flow_inference_steps=3,
        gmm_num_components=2,
    )


@pytest.fixture
def dummy_inputs(small_config):
    B = 2
    return {
        "images": torch.randn(B, 3, small_config.img_size, small_config.img_size),
        "text_token_ids": torch.randint(0, small_config.llm_vocab_size, (B, 8)),
        "robot_state": torch.randn(B, small_config.robot_state_dim),
        "gt_actions": torch.randn(B, small_config.action_chunk_size, small_config.action_dim),
    }


class TestModelWithDiffusion:
    def test_predict_shape(self, small_config, dummy_inputs):
        small_config.action_head_type = "diffusion"
        model = MiniVLAModel(small_config)
        model.eval()
        actions = model.predict(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
        )
        assert actions.shape == (2, 4, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        small_config.action_head_type = "diffusion"
        model = MiniVLAModel(small_config)
        loss = model.compute_loss(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow_end_to_end(self, small_config, dummy_inputs):
        small_config.action_head_type = "diffusion"
        model = MiniVLAModel(small_config)
        loss = model.compute_loss(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        loss.backward()
        # Check gradients flow to ViT
        assert model.vit.patch_embed.proj.weight.grad is not None
        # Check gradients flow to Q-Former
        assert model.qformer.queries.grad is not None


class TestModelWithFlowMatching:
    def test_predict_shape(self, small_config, dummy_inputs):
        small_config.action_head_type = "flow_matching"
        model = MiniVLAModel(small_config)
        model.eval()
        actions = model.predict(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
        )
        assert actions.shape == (2, 4, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        small_config.action_head_type = "flow_matching"
        model = MiniVLAModel(small_config)
        loss = model.compute_loss(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert loss.item() > 0


class TestModelWithMLP:
    def test_predict_shape(self, small_config, dummy_inputs):
        small_config.action_head_type = "mlp"
        model = MiniVLAModel(small_config)
        model.eval()
        actions = model.predict(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
        )
        assert actions.shape == (2, 4, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        small_config.action_head_type = "mlp"
        model = MiniVLAModel(small_config)
        loss = model.compute_loss(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()


class TestModelWithGMM:
    def test_predict_shape(self, small_config, dummy_inputs):
        small_config.action_head_type = "gmm"
        model = MiniVLAModel(small_config)
        model.eval()
        actions = model.predict(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
        )
        assert actions.shape == (2, 4, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        small_config.action_head_type = "gmm"
        model = MiniVLAModel(small_config)
        loss = model.compute_loss(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)


class TestEncodePipeline:
    def test_encode_vision(self, small_config, dummy_inputs):
        model = MiniVLAModel(small_config)
        visual_tokens = model.encode_vision(dummy_inputs["images"])
        assert visual_tokens.shape == (2, small_config.qformer_num_queries, small_config.llm_dim)

    def test_get_action_condition(self, small_config, dummy_inputs):
        model = MiniVLAModel(small_config)
        cond = model.get_action_condition(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            dummy_inputs["robot_state"],
        )
        assert cond.shape == (2, small_config.llm_dim)

    def test_get_action_condition_no_state(self, small_config, dummy_inputs):
        model = MiniVLAModel(small_config)
        cond = model.get_action_condition(
            dummy_inputs["images"],
            dummy_inputs["text_token_ids"],
            robot_state=None,
        )
        assert cond.shape == (2, small_config.llm_dim)
