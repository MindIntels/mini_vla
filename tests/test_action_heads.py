"""Tests for Action Head variants."""

import pytest
import torch
from mini_vla.config import MiniVLAConfig
from mini_vla.action_heads import (
    SinusoidalTimeEmbedding,
    ConditionEncoder,
    MLPActionHead,
    GMMActionHead,
    DiffusionActionHead,
    FlowMatchingActionHead,
    build_action_head,
)


@pytest.fixture
def small_config():
    return MiniVLAConfig(
        llm_dim=48,
        action_dim=7,
        action_chunk_size=8,
        action_head_dim=32,
        action_head_depth=2,
        robot_state_dim=7,
        diffusion_steps=20,
        ddim_inference_steps=5,
        flow_inference_steps=5,
        gmm_num_components=3,
    )


@pytest.fixture
def dummy_inputs(small_config):
    B = 2
    return {
        "llm_cond": torch.randn(B, small_config.llm_dim),
        "robot_state": torch.randn(B, small_config.robot_state_dim),
        "gt_actions": torch.randn(B, small_config.action_chunk_size, small_config.action_dim),
    }


class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 32)

    def test_different_times_give_different_embeddings(self):
        emb = SinusoidalTimeEmbedding(dim=32)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])


class TestConditionEncoder:
    def test_output_shape(self, small_config):
        enc = ConditionEncoder(small_config)
        llm_cond = torch.randn(2, small_config.llm_dim)
        state = torch.randn(2, small_config.robot_state_dim)
        out = enc(llm_cond, state)
        assert out.shape == (2, small_config.action_head_dim)


class TestMLPActionHead:
    def test_predict_shape(self, small_config, dummy_inputs):
        head = MLPActionHead(small_config)
        out = head.predict(dummy_inputs["llm_cond"], dummy_inputs["robot_state"])
        assert out.shape == (2, 8, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        head = MLPActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self, small_config, dummy_inputs):
        head = MLPActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        loss.backward()
        for p in head.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestGMMActionHead:
    def test_predict_shape(self, small_config, dummy_inputs):
        small_config.action_head_type = "gmm"
        head = GMMActionHead(small_config)
        out = head.predict(dummy_inputs["llm_cond"], dummy_inputs["robot_state"])
        assert out.shape == (2, 8, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        head = GMMActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flow(self, small_config, dummy_inputs):
        head = GMMActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        loss.backward()
        for p in head.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestDiffusionActionHead:
    def test_predict_shape(self, small_config, dummy_inputs):
        head = DiffusionActionHead(small_config)
        out = head.predict(dummy_inputs["llm_cond"], dummy_inputs["robot_state"])
        assert out.shape == (2, 8, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        head = DiffusionActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert loss.item() > 0

    def test_noise_schedule(self, small_config):
        head = DiffusionActionHead(small_config)
        assert head.alphas_cumprod[0] > head.alphas_cumprod[-1]
        assert (head.alphas_cumprod >= 0).all()
        assert (head.alphas_cumprod <= 1).all()

    def test_gradient_flow(self, small_config, dummy_inputs):
        head = DiffusionActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        loss.backward()
        trainable = [p for p in head.parameters() if p.requires_grad]
        assert all(p.grad is not None for p in trainable)


class TestFlowMatchingActionHead:
    def test_predict_shape(self, small_config, dummy_inputs):
        head = FlowMatchingActionHead(small_config)
        out = head.predict(dummy_inputs["llm_cond"], dummy_inputs["robot_state"])
        assert out.shape == (2, 8, 7)

    def test_compute_loss(self, small_config, dummy_inputs):
        head = FlowMatchingActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self, small_config, dummy_inputs):
        head = FlowMatchingActionHead(small_config)
        loss = head.compute_loss(
            dummy_inputs["llm_cond"],
            dummy_inputs["robot_state"],
            dummy_inputs["gt_actions"],
        )
        loss.backward()
        trainable = [p for p in head.parameters() if p.requires_grad]
        assert all(p.grad is not None for p in trainable)


class TestBuildActionHead:
    def test_build_mlp(self, small_config):
        small_config.action_head_type = "mlp"
        head = build_action_head(small_config)
        assert isinstance(head, MLPActionHead)

    def test_build_gmm(self, small_config):
        small_config.action_head_type = "gmm"
        head = build_action_head(small_config)
        assert isinstance(head, GMMActionHead)

    def test_build_diffusion(self, small_config):
        small_config.action_head_type = "diffusion"
        head = build_action_head(small_config)
        assert isinstance(head, DiffusionActionHead)

    def test_build_flow_matching(self, small_config):
        small_config.action_head_type = "flow_matching"
        head = build_action_head(small_config)
        assert isinstance(head, FlowMatchingActionHead)

    def test_invalid_type(self, small_config):
        small_config.action_head_type = "invalid"
        with pytest.raises(ValueError, match="Unknown action_head_type"):
            build_action_head(small_config)
