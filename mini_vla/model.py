"""Full Mini-VLA Model.

Assembles ViT + Q-Former + LLM Backbone + Action Head into
a complete Vision-Language-Action system.
"""

import torch
import torch.nn as nn
from mini_vla.config import MiniVLAConfig
from mini_vla.vit import ViTEncoder
from mini_vla.qformer import QFormer
from mini_vla.llm_backbone import LLMBackbone
from mini_vla.action_heads import build_action_head


class MiniVLAModel(nn.Module):
    """Complete VLA model: ViT + Q-Former + LLM + Action Head.

    Pipeline:
        Image (B,3,H,W)  ──→ ViT ──→ (B,N_p,d_vit)
                                │
                                ▼
                            Q-Former ──→ (B,N_q,d_llm)
                                │
        Text (B,T_txt)  ────────┼──→ LLM Backbone ──→ (B,d_llm) action condition
                                │
        Robot State (B,7) ──────┘         │
                                          ▼
                                    Action Head ──→ (B,T_a,d_a) actions
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config

        self.vit = ViTEncoder(config)
        self.qformer = QFormer(config)
        self.llm = LLMBackbone(config)
        self.action_head = build_action_head(config)

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through ViT + Q-Former.

        Args:
            images: (B, 3, H, W)
        Returns:
            (B, N_q, d_llm) compressed visual tokens
        """
        patch_tokens = self.vit(images)        # (B, N_p, d_vit)
        visual_tokens = self.qformer(patch_tokens)  # (B, N_q, d_llm)
        return visual_tokens

    def get_action_condition(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        robot_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get action condition from LLM backbone.

        Args:
            images: (B, 3, H, W)
            text_token_ids: (B, T_txt)
            robot_state: (B, d_state) optional
        Returns:
            (B, d_llm) action condition vector
        """
        visual_tokens = self.encode_vision(images)
        action_cond = self.llm(visual_tokens, text_token_ids, robot_state)
        return action_cond

    def compute_loss(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        robot_state: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            images: (B, 3, H, W)
            text_token_ids: (B, T_txt)
            robot_state: (B, d_state)
            gt_actions: (B, T_a, d_a) ground truth action trajectories
        Returns:
            scalar loss
        """
        action_cond = self.get_action_condition(images, text_token_ids, robot_state)
        loss = self.action_head.compute_loss(action_cond, robot_state, gt_actions)
        return loss

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict actions (inference mode).

        Args:
            images: (B, 3, H, W)
            text_token_ids: (B, T_txt)
            robot_state: (B, d_state)
        Returns:
            (B, T_a, d_a) predicted action trajectories
        """
        action_cond = self.get_action_condition(images, text_token_ids, robot_state)
        actions = self.action_head.predict(action_cond, robot_state)
        return actions
