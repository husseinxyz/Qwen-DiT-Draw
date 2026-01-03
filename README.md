# Qwen-DiT-Draw

<div align="center">

**Vision-Language Model with Diffusion Transformer for Continuous Mouse Trajectory Prediction**

[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/TESS-Computer/qwen-dit-draw)
[![Blog](https://img.shields.io/badge/Blog-Post-green)](https://husseinxyz.com/tess/qwen-dit-draw/)

</div>

---

Extension of [Qwen-DiT-Click](https://github.com/husseinxyz/Qwen-Clicking-DiT) to predict **continuous mouse trajectories** instead of single click points.

Given a screenshot and instruction like *"draw a circle"*, the model predicts a sequence of (x, y) coordinates forming the complete trajectory.

## Architecture

```
Input: Screenshot + Text instruction ("draw a smiley face")
                    |
    +-------------------------------+
    |     Qwen2.5-VL (Frozen)       |
    |  * Vision Transformer         |
    |  * Language Model (3B)        |
    +-------------------------------+
                    | hidden states (conditioning)
                    v
    +-------------------------------+
    |   DiT Action Head (Trained)   |
    |                               |
    |  [Action Encoder]             |
    |   * Per-point MLP encoding    |
    |   * Timestep embedding        |
    |   * Positional encoding       |
    |         |                     |
    |         v                     |
    |  [DiT Transformer Blocks x6]  |
    |   * Self-attention over       |
    |     trajectory tokens         |
    |   * Cross-attention to        |
    |     VLM hidden states         |
    |   * AdaLayerNorm (timestep)   |
    |         |                     |
    |         v                     |
    |  [Action Decoder]             |
    |   * Per-token MLP decoding    |
    |   * Output: (T, 2) trajectory |
    +-------------------------------+
                    |
    Output: [(x1,y1), (x2,y2), ..., (xT,yT)]
            normalized coordinates [0, 1]
```

## Method: Flow Matching on Trajectories

Same flow matching approach as GR00T, but for mouse trajectories:

### Training
```python
T = 64  # trajectory length (number of points)

# Sample trajectory from dataset
target_trajectory = [(x1,y1), ..., (x64,y64)]  # shape: (T, 2)

# Sample noise and timestep
noise = torch.randn(T, 2)
t = torch.rand(1)

# Interpolate (flow matching)
noisy_trajectory = (1 - t) * noise + t * target_trajectory
velocity_target = target_trajectory - noise

# Model predicts velocity field
velocity_pred = model(image, prompt, noisy_trajectory, t)
loss = MSE(velocity_pred, velocity_target)
```

### Inference (Euler Integration)
```python
# Start from pure noise
trajectory = torch.randn(T, 2)

# Iteratively denoise (K=16 steps)
for k in range(K):
    t = k / K
    velocity = model(image, prompt, trajectory, t)
    trajectory = trajectory + velocity * (1/K)

# Final trajectory is the predicted drawing
```

## Key Differences from Qwen-DiT-Click

| Aspect | Click Model | Draw Model |
|--------|-------------|------------|
| Output | Single (x, y) | Trajectory (T, 2) |
| Action tokens | 1 | T (e.g., 64) |
| Self-attention | N/A | Over trajectory points |
| Use case | Click buttons | Draw shapes, drag, paint |

## DiT Block Architecture (GR00T-style)

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, cond_dim):
        # Self-attention over trajectory tokens
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads)

        # Cross-attention to VLM features
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads)

        # AdaLayerNorm for timestep conditioning
        self.adaln = AdaLayerNorm(hidden_size)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, action_tokens, cond_tokens, t_emb):
        # Self-attention (trajectory points attend to each other)
        x = self.adaln(action_tokens, t_emb)
        x = x + self.self_attn(x, x, x)

        # Cross-attention (trajectory attends to VLM features)
        x = x + self.cross_attn(x, cond_tokens, cond_tokens)

        # FFN
        x = x + self.ffn(x)
        return x
```

## Installation

```bash
git clone https://github.com/husseinxyz/qwen-dit-draw.git
cd qwen-dit-draw
pip install -r requirements.txt
```

## Training

```bash
python -m src.train.train_draw \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path quickdraw_dataset \
    --output_dir outputs/dit_draw \
    --trajectory_length 64 \
    --num_train_epochs 3
```

## Inference

```python
from src.inference import load_model, predict_trajectory

model, processor = load_model("outputs/dit_draw")
trajectory = predict_trajectory(
    model, processor,
    image="canvas.png",
    prompt="draw a circle"
)

# trajectory is a list of (x, y) points
for x, y in trajectory:
    move_mouse(x * screen_width, y * screen_height)
```

## Dataset

Training on drawing/sketch datasets:
- **Quick, Draw!**: 50M sketches with stroke data
- **Synthetic**: Programmatically generated shapes (circles, squares, etc.)

## Roadmap

- [x] Single click prediction (Qwen-DiT-Click)
- [ ] Trajectory prediction (this project)
- [ ] Game controls (WASD + mouse)
- [ ] Keyboard actions
- [ ] General computer control agent

## References

- [NVIDIA GR00T N1](https://arxiv.org/abs/2503.14734) - Flow matching for robot action sequences
- [Qwen-DiT-Click](https://github.com/husseinxyz/Qwen-Clicking-DiT) - Previous single-click model
- [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset) - Sketch data

## License

MIT
