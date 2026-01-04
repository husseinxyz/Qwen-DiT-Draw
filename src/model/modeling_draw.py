"""
Qwen-DiT-Draw: Vision-Language Model with Diffusion Transformer for Trajectory Prediction

Architecture inspired by NVIDIA GR00T N1:
- Frozen Qwen2.5-VL backbone for vision-language understanding
- DiT action head with self-attention over trajectory tokens
- Flow matching training for continuous trajectory generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from transformers import Qwen2_5_VLForConditionalGeneration


@dataclass
class TrajectoryConfig:
    """Configuration for chunked trajectory prediction (GR00T-style)."""
    chunk_size: int = 16                  # Points per chunk (H=16 like GR00T)
    action_dim: int = 3                   # (x, y, state) per point - state=1 means STOP
    dit_hidden_size: int = 512           # DiT hidden dimension
    dit_num_layers: int = 6              # Number of DiT blocks
    dit_num_heads: int = 8               # Attention heads
    dit_dropout: float = 0.1             # Dropout rate
    num_inference_steps: int = 16        # Euler integration steps (K)
    backbone_dim: int = 2048             # Qwen2.5-VL hidden size


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps and positions."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        dtype = t.dtype
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        # Use same dtype as input
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb_scale)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TrajectoryEncoder(nn.Module):
    """
    Encodes noisy trajectory points + diffusion timestep into tokens.

    Each point in the trajectory becomes a separate token.
    GR00T-style: per-point MLP encoding with shared timestep embedding.
    """

    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.config = config

        # Per-point MLP encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
        )

        # Diffusion timestep embedding
        self.time_encoder = SinusoidalPositionalEncoding(config.dit_hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
        )

        # Positional encoding for trajectory position (which point in sequence)
        self.pos_encoder = SinusoidalPositionalEncoding(config.dit_hidden_size)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,  # (B, T, 2)
        t: torch.Tensor                   # (B,) diffusion timestep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_tokens: (B, T, hidden_size) - one token per trajectory point
            t_emb: (B, hidden_size) - timestep embedding for AdaLN
        """
        B, T, _ = noisy_trajectory.shape
        device = noisy_trajectory.device
        dtype = noisy_trajectory.dtype

        # Encode each point
        action_tokens = self.point_encoder(noisy_trajectory)  # (B, T, hidden)

        # Add positional encoding for sequence position (match input dtype)
        positions = torch.arange(T, device=device, dtype=dtype)
        pos_emb = self.pos_encoder(positions)  # (T, hidden)
        action_tokens = action_tokens + pos_emb.unsqueeze(0)  # (B, T, hidden)

        # Encode diffusion timestep
        t_emb = self.time_encoder(t)  # (B, hidden)
        t_emb = self.time_mlp(t_emb)  # (B, hidden)

        return action_tokens, t_emb


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep.
    Used in DiT for diffusion timestep conditioning.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.scale_shift = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden) - input tokens
            t_emb: (B, hidden) - timestep embedding
        """
        orig_dtype = x.dtype
        x = self.norm(x)
        x = x.to(orig_dtype)  # LayerNorm may output float32, convert back
        scale, shift = self.scale_shift(t_emb).chunk(2, dim=-1)
        # Broadcast scale/shift: (B, hidden) -> (B, 1, hidden)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block (GR00T-style).

    1. Self-attention over trajectory tokens (points attend to each other)
    2. Cross-attention to VLM hidden states (trajectory attends to image/text)
    3. FFN with AdaLayerNorm conditioning
    """

    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.config = config
        hidden = config.dit_hidden_size
        heads = config.dit_num_heads

        # AdaLayerNorm for timestep conditioning
        self.adaln_self = AdaLayerNorm(hidden)
        self.adaln_cross = AdaLayerNorm(hidden)
        self.adaln_ffn = AdaLayerNorm(hidden)

        # Self-attention over trajectory tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=config.dit_dropout,
            batch_first=True
        )

        # Cross-attention to VLM features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=config.dit_dropout,
            batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(config.dit_dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(config.dit_dropout),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,  # (B, T, hidden)
        cond_tokens: torch.Tensor,     # (B, S, hidden) - VLM features
        t_emb: torch.Tensor            # (B, hidden) - timestep embedding
    ) -> torch.Tensor:
        """Process trajectory tokens with self-attention and cross-attention."""

        # Self-attention (trajectory points attend to each other)
        residual = action_tokens
        x = self.adaln_self(action_tokens, t_emb)
        x, _ = self.self_attn(x, x, x)
        action_tokens = residual + x

        # Cross-attention (trajectory attends to VLM features)
        residual = action_tokens
        x = self.adaln_cross(action_tokens, t_emb)
        x, _ = self.cross_attn(x, cond_tokens, cond_tokens)
        action_tokens = residual + x

        # FFN
        residual = action_tokens
        x = self.adaln_ffn(action_tokens, t_emb)
        x = self.ffn(x)
        action_tokens = residual + x

        return action_tokens


class TrajectoryDecoder(nn.Module):
    """Decodes trajectory tokens back to (x, y) coordinates."""

    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            nn.SiLU(),
            nn.Linear(config.dit_hidden_size, config.action_dim),
        )

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_tokens: (B, T, hidden)
        Returns:
            trajectory: (B, T, 2) - predicted velocity or trajectory
        """
        return self.decoder(action_tokens)


class DiTTrajectoryHead(nn.Module):
    """
    Complete DiT head for trajectory prediction.

    Combines:
    - TrajectoryEncoder: noisy trajectory + timestep -> tokens
    - DiTBlocks: self-attention + cross-attention processing
    - TrajectoryDecoder: tokens -> predicted velocity
    """

    def __init__(self, config: TrajectoryConfig):
        super().__init__()
        self.config = config

        # Project VLM features to DiT hidden size
        self.cond_proj = nn.Linear(config.backbone_dim, config.dit_hidden_size)

        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(config)

        # DiT blocks
        self.dit_blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.dit_num_layers)
        ])

        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(config)

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.dit_hidden_size)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,  # (B, T, 2)
        t: torch.Tensor,                  # (B,) diffusion timestep [0, 1]
        cond_tokens: torch.Tensor         # (B, S, backbone_dim) VLM hidden states
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Returns:
            velocity: (B, T, 2) - predicted velocity for each trajectory point
        """
        # Discretize timestep for potential embedding lookup
        t_discretized = (t * 1000).long().clamp(0, 999)

        # Project conditioning tokens
        cond_tokens = self.cond_proj(cond_tokens)  # (B, S, hidden)

        # Encode trajectory
        action_tokens, t_emb = self.trajectory_encoder(noisy_trajectory, t)

        # Process through DiT blocks
        for block in self.dit_blocks:
            action_tokens = block(action_tokens, cond_tokens, t_emb)

        # Final norm and decode
        action_tokens = self.final_norm(action_tokens)
        velocity = self.trajectory_decoder(action_tokens)

        return velocity

    @torch.no_grad()
    def sample(
        self,
        cond_tokens: torch.Tensor,  # (B, S, backbone_dim)
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate chunk via Euler integration (inference).

        Returns:
            chunk: (B, chunk_size, 3) - predicted trajectory chunk with (x, y, state)
        """
        num_steps = num_steps or self.config.num_inference_steps
        B = cond_tokens.shape[0]
        chunk_size = self.config.chunk_size
        action_dim = self.config.action_dim
        device = cond_tokens.device
        dtype = cond_tokens.dtype

        # Start from pure noise
        chunk = torch.randn(B, chunk_size, action_dim, device=device, dtype=dtype)

        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device, dtype=dtype)
            velocity = self.forward(chunk, t, cond_tokens)
            chunk = chunk + velocity * dt

        # Clamp to valid range
        chunk = chunk.clamp(0, 1)

        return chunk


class Qwen2_5_VL_Draw(nn.Module):
    """
    Full model: Qwen2.5-VL backbone + DiT trajectory head.

    GR00T-style chunked prediction:
    - Frozen VLM extracts features from canvas image + instruction
    - Trainable DiT predicts next chunk of 16 (x, y) points
    - Visual feedback loop: model sees canvas after each chunk is drawn
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        config: Optional[TrajectoryConfig] = None,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()

        self.config = config or TrajectoryConfig()

        # Load Qwen2.5-VL backbone
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa"  # Use SDPA for efficiency
        )

        # Update backbone_dim from actual model
        self.config.backbone_dim = self.backbone.config.hidden_size

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # DiT trajectory head (same dtype as backbone for consistency)
        self.trajectory_head = DiTTrajectoryHead(self.config)
        if dtype is not None:
            self.trajectory_head = self.trajectory_head.to(dtype)

    def get_vlm_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract hidden states from frozen VLM."""
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
        # Use last hidden state as conditioning
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        target_trajectory: Optional[torch.Tensor] = None,  # (B, chunk_size, 3)
        trajectory_mask: Optional[torch.Tensor] = None,    # (B, chunk_size) - 1=real, 0=ignore
    ) -> dict:
        """
        Forward pass for training.

        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask
            pixel_values: Processed image pixels
            image_grid_thw: Image grid dimensions
            target_trajectory: Target chunk (B, chunk_size, 3) with (x, y, state)
            trajectory_mask: Loss mask (B, chunk_size) - 1=real point, 0=ignore

        Returns:
            dict with 'loss' (training) or 'chunk' (inference)
        """
        # Get VLM features
        hidden_states = self.get_vlm_features(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )

        if target_trajectory is not None:
            # Training: compute flow matching loss with mask
            loss = self.compute_flow_matching_loss(hidden_states, target_trajectory, trajectory_mask)
            return {"loss": loss}
        else:
            # Inference: sample chunk
            chunk = self.trajectory_head.sample(hidden_states)
            return {"chunk": chunk}

    def compute_flow_matching_loss(
        self,
        cond_tokens: torch.Tensor,       # (B, S, hidden)
        target_trajectory: torch.Tensor,  # (B, T, 3) with (x, y, state)
        trajectory_mask: Optional[torch.Tensor] = None  # (B, T) - 1=real, 0=ignore
    ) -> torch.Tensor:
        """
        Flow matching loss for trajectory prediction with masking.

        Same as GR00T:
        1. Sample random timestep t ~ U(0, 1)
        2. Sample noise ~ N(0, I)
        3. Interpolate: noisy = (1-t)*noise + t*target
        4. Velocity target: v = target - noise
        5. Loss: MSE(predicted_velocity, v) with mask applied

        Mask is used to ignore padded positions in final chunks.
        """
        B, T, D = target_trajectory.shape
        device = target_trajectory.device

        # Get dtype from the model (trajectory head)
        model_dtype = next(self.trajectory_head.parameters()).dtype

        # Convert target trajectory to model dtype
        target_trajectory = target_trajectory.to(model_dtype)

        # Sample random timestep for each batch element
        t = torch.rand(B, device=device, dtype=model_dtype)

        # Sample noise
        noise = torch.randn_like(target_trajectory)

        # Interpolate (flow matching forward process)
        # noisy_trajectory = (1 - t) * noise + t * target
        t_expand = t.view(B, 1, 1)  # (B, 1, 1) for broadcasting
        noisy_trajectory = (1 - t_expand) * noise + t_expand * target_trajectory

        # Target velocity
        velocity_target = target_trajectory - noise

        # Predict velocity
        velocity_pred = self.trajectory_head(noisy_trajectory, t, cond_tokens)

        # Compute per-element MSE
        mse = (velocity_pred - velocity_target) ** 2  # (B, T, D)

        if trajectory_mask is not None:
            # Apply mask: expand (B, T) -> (B, T, D) for broadcasting
            mask = trajectory_mask.unsqueeze(-1).expand_as(mse)  # (B, T, D)
            # Masked mean: only count real points
            masked_mse = mse * mask
            loss = masked_mse.sum() / (mask.sum() + 1e-8)
        else:
            # No mask: regular MSE
            loss = mse.mean()

        return loss

    @torch.no_grad()
    def predict_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate next trajectory chunk for inference.

        Returns:
            chunk: (B, chunk_size, 2) - normalized (x, y) coordinates
        """
        hidden_states = self.get_vlm_features(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )
        chunk = self.trajectory_head.sample(hidden_states, num_steps)
        return chunk


# Utility function for counting trainable parameters
def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": 100 * trainable / total if total > 0 else 0
    }
