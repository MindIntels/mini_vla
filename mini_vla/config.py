"""Configuration for Mini-VLA model."""

from dataclasses import dataclass, field


@dataclass
class MiniVLAConfig:
    """Complete configuration for the Mini-VLA system."""

    # --- ViT Vision Encoder ---
    img_size: int = 448
    patch_size: int = 14
    vit_dim: int = 1024
    vit_depth: int = 24
    vit_heads: int = 16
    vit_mlp_ratio: float = 4.0
    vit_drop_rate: float = 0.0

    # --- Q-Former ---
    qformer_num_queries: int = 32
    qformer_dim: int = 768
    qformer_depth: int = 12
    qformer_heads: int = 12
    qformer_mlp_ratio: float = 4.0
    qformer_drop_rate: float = 0.0

    # --- LLM Backbone ---
    llm_dim: int = 3584
    llm_depth: int = 28
    llm_heads: int = 28
    llm_kv_heads: int = 4
    llm_ffn_dim: int = 18944
    llm_vocab_size: int = 32000
    llm_max_seq_len: int = 2048
    llm_rope_theta: float = 1000000.0

    # --- Action Head ---
    action_head_type: str = "diffusion"  # mlp, gmm, diffusion, flow_matching
    action_dim: int = 7  # 6DoF + gripper
    action_chunk_size: int = 16  # T_a
    action_head_dim: int = 256  # d_act
    action_head_depth: int = 4  # number of DiT blocks

    # Diffusion specific
    diffusion_steps: int = 100
    ddim_inference_steps: int = 10

    # GMM specific
    gmm_num_components: int = 5

    # Flow Matching specific
    flow_inference_steps: int = 10

    # --- Robot State ---
    robot_state_dim: int = 7  # 6 joints + gripper

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def vit_mlp_dim(self) -> int:
        return int(self.vit_dim * self.vit_mlp_ratio)

    @property
    def qformer_mlp_dim(self) -> int:
        return int(self.qformer_dim * self.qformer_mlp_ratio)

    @property
    def llm_head_dim(self) -> int:
        return self.llm_dim // self.llm_heads
