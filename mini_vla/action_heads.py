"""Action Head variants for VLA.

Implements 4 variants:
- MLPActionHead: Simple MLP regression
- GMMActionHead: Gaussian Mixture Model
- DiffusionActionHead: DiT-based conditional diffusion
- FlowMatchingActionHead: Conditional Flow Matching (CFM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_vla.config import MiniVLAConfig


# =============================================================================
# Shared Components
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion/flow models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time steps (scalar per sample)
        Returns:
            (B, dim) sinusoidal embedding
        """
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class ConditionEncoder(nn.Module):
    """Encode LLM condition + robot state into a unified condition vector."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.llm_proj = nn.Sequential(
            nn.Linear(config.llm_dim, config.action_head_dim),
            nn.SiLU(),
            nn.Linear(config.action_head_dim, config.action_head_dim),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(config.robot_state_dim, config.action_head_dim),
            nn.SiLU(),
            nn.Linear(config.action_head_dim, config.action_head_dim),
        )

    def forward(self, llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llm_cond: (B, d_llm) LLM last token hidden state
            robot_state: (B, d_state) current robot state
        Returns:
            (B, d_act) unified condition
        """
        c = self.llm_proj(llm_cond) + self.state_proj(robot_state)
        return c


# =============================================================================
# Variant A: MLP Regression Head
# =============================================================================

class MLPActionHead(nn.Module):
    """Simple MLP regression head.

    IN:  (B, d_llm) + (B, d_state)
    OUT: (B, T_a, d_a)
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        d_act = config.action_head_dim
        out_dim = config.action_chunk_size * config.action_dim

        self.cond_encoder = ConditionEncoder(config)
        self.mlp = nn.Sequential(
            nn.Linear(d_act, d_act * 4),
            nn.SiLU(),
            nn.Linear(d_act * 4, d_act * 4),
            nn.SiLU(),
            nn.Linear(d_act * 4, out_dim),
        )

    def forward(self, llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        c = self.cond_encoder(llm_cond, robot_state)
        out = self.mlp(c)
        return out.reshape(-1, self.config.action_chunk_size, self.config.action_dim)

    def compute_loss(self, llm_cond: torch.Tensor, robot_state: torch.Tensor,
                     gt_actions: torch.Tensor) -> torch.Tensor:
        pred = self.forward(llm_cond, robot_state)
        return F.mse_loss(pred, gt_actions)

    def predict(self, llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        return self.forward(llm_cond, robot_state)


# =============================================================================
# Variant B: GMM Head
# =============================================================================

class GMMActionHead(nn.Module):
    """Gaussian Mixture Model action head.

    Predicts K Gaussian components: (mu, sigma, pi) per action dimension.
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        K = config.gmm_num_components
        T_a = config.action_chunk_size
        d_a = config.action_dim
        d_act = config.action_head_dim

        self.cond_encoder = ConditionEncoder(config)

        self.backbone = nn.Sequential(
            nn.Linear(d_act, d_act * 4),
            nn.SiLU(),
            nn.Linear(d_act * 4, d_act * 4),
            nn.SiLU(),
        )

        # Predict per-component: mu, log_sigma, and mixing logits
        self.mu_head = nn.Linear(d_act * 4, K * T_a * d_a)
        self.log_sigma_head = nn.Linear(d_act * 4, K * T_a * d_a)
        self.logits_head = nn.Linear(d_act * 4, K)

    def _get_components(self, llm_cond: torch.Tensor, robot_state: torch.Tensor):
        B = llm_cond.shape[0]
        K = self.config.gmm_num_components
        T_a = self.config.action_chunk_size
        d_a = self.config.action_dim

        c = self.cond_encoder(llm_cond, robot_state)
        h = self.backbone(c)

        mu = self.mu_head(h).reshape(B, K, T_a, d_a)
        log_sigma = self.log_sigma_head(h).reshape(B, K, T_a, d_a)
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=2.0)
        logits = self.logits_head(h)  # (B, K)

        return mu, log_sigma, logits

    def compute_loss(self, llm_cond: torch.Tensor, robot_state: torch.Tensor,
                     gt_actions: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of GMM."""
        mu, log_sigma, logits = self._get_components(llm_cond, robot_state)
        sigma = torch.exp(log_sigma)

        # gt_actions: (B, T_a, d_a) -> (B, 1, T_a, d_a)
        gt = gt_actions.unsqueeze(1)

        # Log probability of each component
        log_prob = -0.5 * (((gt - mu) / sigma) ** 2 + 2 * log_sigma + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=(-2, -1))  # (B, K) sum over T_a and d_a

        # Log mixing weights
        log_pi = F.log_softmax(logits, dim=-1)  # (B, K)

        # Log-sum-exp for mixture
        log_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)  # (B,)

        return -log_mixture.mean()

    def predict(self, llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Sample from the most likely component."""
        mu, log_sigma, logits = self._get_components(llm_cond, robot_state)
        # Pick component with highest weight
        best_k = logits.argmax(dim=-1)  # (B,)
        B = mu.shape[0]
        actions = mu[torch.arange(B, device=mu.device), best_k]  # (B, T_a, d_a)
        return actions


# =============================================================================
# Variant C: Diffusion DiT Head
# =============================================================================

class AdaLNDiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # AdaLN modulation: predict (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_a, d_act) action tokens
            cond: (B, d_act) condition (c + t_emb)
        """
        mod = self.adaLN_modulation(cond).unsqueeze(1)  # (B, 1, 6*d)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Self-Attention with AdaLN
        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + alpha1 * h

        # FFN with AdaLN
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        x = x + alpha2 * h

        return x


class DiffusionActionHead(nn.Module):
    """Diffusion DiT Action Head.

    Uses DDPM training and DDIM inference.
    Network predicts noise ε given (noisy_action, time_step, condition).
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        d_act = config.action_head_dim

        # Condition encoder
        self.cond_encoder = ConditionEncoder(config)

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(d_act),
            nn.Linear(d_act, d_act),
            nn.SiLU(),
            nn.Linear(d_act, d_act),
        )

        # Action input projection
        self.action_proj = nn.Linear(config.action_dim, d_act)

        # DiT blocks
        self.blocks = nn.ModuleList([
            AdaLNDiTBlock(d_act) for _ in range(config.action_head_depth)
        ])

        # Output projection
        self.final_norm = nn.LayerNorm(d_act, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_act, 2 * d_act))
        self.output_proj = nn.Linear(d_act, config.action_dim)

        # Noise schedule (linear beta schedule)
        self._setup_noise_schedule(config.diffusion_steps)

    def _setup_noise_schedule(self, num_steps: int):
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _denoise(self, noisy_action: torch.Tensor, t: torch.Tensor,
                 llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Predict noise ε_θ(a_t, t, c).

        Args:
            noisy_action: (B, T_a, d_a)
            t: (B,) integer time steps
            llm_cond: (B, d_llm)
            robot_state: (B, d_state)
        Returns:
            (B, T_a, d_a) predicted noise
        """
        # Condition
        c = self.cond_encoder(llm_cond, robot_state)  # (B, d_act)
        t_emb = self.time_emb(t.float())  # (B, d_act)
        cond = c + t_emb  # (B, d_act)

        # Project noisy action to d_act
        x = self.action_proj(noisy_action)  # (B, T_a, d_act)

        # DiT blocks
        for blk in self.blocks:
            x = blk(x, cond)

        # Final output
        mod = self.final_adaLN(cond).unsqueeze(1)  # (B, 1, 2*d_act)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + gamma) + beta
        noise_pred = self.output_proj(x)  # (B, T_a, d_a)

        return noise_pred

    def compute_loss(self, llm_cond: torch.Tensor, robot_state: torch.Tensor,
                     gt_actions: torch.Tensor) -> torch.Tensor:
        """DDPM training loss: MSE between predicted and actual noise.

        Args:
            llm_cond: (B, d_llm)
            robot_state: (B, d_state)
            gt_actions: (B, T_a, d_a)
        """
        B = gt_actions.shape[0]
        device = gt_actions.device

        # Sample random time steps
        t = torch.randint(0, self.config.diffusion_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(gt_actions)

        # Forward diffusion: q(a_t | a_0) = sqrt(alpha_bar) * a_0 + sqrt(1-alpha_bar) * eps
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(B, 1, 1)
        noisy_action = sqrt_alpha * gt_actions + sqrt_one_minus_alpha * noise

        # Predict noise
        noise_pred = self._denoise(noisy_action, t, llm_cond, robot_state)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict(self, llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """DDIM inference.

        Args:
            llm_cond: (B, d_llm)
            robot_state: (B, d_state)
        Returns:
            (B, T_a, d_a)
        """
        B = llm_cond.shape[0]
        device = llm_cond.device
        T_a = self.config.action_chunk_size
        d_a = self.config.action_dim

        # Start from pure noise
        a_t = torch.randn(B, T_a, d_a, device=device)

        # DDIM step indices (evenly spaced)
        total_steps = self.config.diffusion_steps
        infer_steps = self.config.ddim_inference_steps
        step_size = total_steps // infer_steps
        timesteps = list(range(total_steps - 1, -1, -step_size))[:infer_steps]

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self._denoise(a_t, t, llm_cond, robot_state)

            # DDIM update
            alpha_bar = self.alphas_cumprod[t_val]
            alpha_bar_prev = self.alphas_cumprod[max(t_val - step_size, 0)] if t_val > 0 else torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (a_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

            # DDIM deterministic step
            a_t = torch.sqrt(alpha_bar_prev) * pred_x0 + \
                  torch.sqrt(1 - alpha_bar_prev) * noise_pred

        return a_t


# =============================================================================
# Variant D: Flow Matching Head
# =============================================================================

class FlowMatchingActionHead(nn.Module):
    """Flow Matching (Conditional Flow Matching / CFM) Action Head.

    Learns a vector field v_θ that maps noise to actions along a straight path.
    Training: aₜ = (1-t)·ε + t·a₀, target = a₀ - ε
    Inference: Euler ODE solver, 10 steps
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        d_act = config.action_head_dim

        # Condition encoder
        self.cond_encoder = ConditionEncoder(config)

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(d_act),
            nn.Linear(d_act, d_act),
            nn.SiLU(),
            nn.Linear(d_act, d_act),
        )

        # Action input projection
        self.action_proj = nn.Linear(config.action_dim, d_act)

        # DiT blocks (same architecture as diffusion)
        self.blocks = nn.ModuleList([
            AdaLNDiTBlock(d_act) for _ in range(config.action_head_depth)
        ])

        # Output projection
        self.final_norm = nn.LayerNorm(d_act, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_act, 2 * d_act))
        self.output_proj = nn.Linear(d_act, config.action_dim)

    def _predict_velocity(self, a_t: torch.Tensor, t: torch.Tensor,
                          llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Predict velocity field v_θ(a_t, t, c).

        Args:
            a_t: (B, T_a, d_a) interpolated sample
            t: (B,) continuous time in [0, 1]
            llm_cond: (B, d_llm)
            robot_state: (B, d_state)
        Returns:
            (B, T_a, d_a) predicted velocity
        """
        c = self.cond_encoder(llm_cond, robot_state)
        t_emb = self.time_emb(t)
        cond = c + t_emb

        x = self.action_proj(a_t)

        for blk in self.blocks:
            x = blk(x, cond)

        mod = self.final_adaLN(cond).unsqueeze(1)
        gamma, beta = mod.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + gamma) + beta
        v = self.output_proj(x)

        return v

    def compute_loss(self, llm_cond: torch.Tensor, robot_state: torch.Tensor,
                     gt_actions: torch.Tensor) -> torch.Tensor:
        """Conditional Flow Matching loss.

        aₜ = (1-t)·ε + t·a₀
        target = a₀ - ε
        L = ||v_θ(aₜ, t, c) - target||²
        """
        B = gt_actions.shape[0]
        device = gt_actions.device

        # Sample time t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)

        # Sample noise
        eps = torch.randn_like(gt_actions)

        # Linear interpolation
        t_expand = t.reshape(B, 1, 1)
        a_t = (1 - t_expand) * eps + t_expand * gt_actions

        # Target velocity: u_t = a_0 - eps
        target = gt_actions - eps

        # Predict velocity
        v_pred = self._predict_velocity(a_t, t, llm_cond, robot_state)

        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def predict(self, llm_cond: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Euler ODE solver inference.

        Start from noise (t=0), integrate to action (t=1).
        """
        B = llm_cond.shape[0]
        device = llm_cond.device
        T_a = self.config.action_chunk_size
        d_a = self.config.action_dim
        num_steps = self.config.flow_inference_steps

        # Start from noise
        a_t = torch.randn(B, T_a, d_a, device=device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self._predict_velocity(a_t, t, llm_cond, robot_state)
            a_t = a_t + dt * v

        return a_t


# =============================================================================
# Factory
# =============================================================================

def build_action_head(config: MiniVLAConfig) -> nn.Module:
    """Build action head based on config.action_head_type."""
    heads = {
        "mlp": MLPActionHead,
        "gmm": GMMActionHead,
        "diffusion": DiffusionActionHead,
        "flow_matching": FlowMatchingActionHead,
    }
    if config.action_head_type not in heads:
        raise ValueError(f"Unknown action_head_type: {config.action_head_type}. "
                         f"Choose from {list(heads.keys())}")
    return heads[config.action_head_type](config)
