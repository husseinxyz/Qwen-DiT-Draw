# Qwen-DiT-Draw

<div align="center">

**Vision-Language Model with Diffusion Transformer for Continuous Mouse Trajectory Prediction**

[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/TESS-Computer/qwen-dit-draw)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue)](https://huggingface.co/datasets/TESS-Computer/quickdraw-circles)
[![Blog](https://img.shields.io/badge/Blog-Post-green)](https://husseinxyz.com/tess/qwen-dit-draw/)

</div>

---

> **Branch:** `main` — Uses absolute (x, y) coordinates
>
> See [`delta` branch](https://github.com/TESS-Computer/qwen-dit-draw/tree/delta) for relative (dx, dy) movements ([GR00T N1.6 style](https://arxiv.org/abs/2503.14734)).

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

## Design Decisions

Key architectural choices inspired by [NVIDIA GR00T N1](https://arxiv.org/abs/2503.14734) for building a general-purpose System 1 model:

### 1. Chunked Prediction (not full trajectory)

**Why?** A general model must handle variable-length tasks - a circle might need 32 points, a complex drawing might need 500.

```
GR00T approach (what we use):
┌─────────────────────────────────────────────────────────┐
│ Chunk 1 → Execute → Chunk 2 → Execute → ... → Done      │
│ (16 pts)            (16 pts)                            │
└─────────────────────────────────────────────────────────┘

vs. Fixed-length (what we DON'T use):
┌─────────────────────────────────────────────────────────┐
│ Predict all 64 points at once (inflexible)              │
└─────────────────────────────────────────────────────────┘
```

### 2. Chunk Size: 16 Points

Matches GR00T's H=16 action horizon. Each inference predicts the next 16 (x, y) mouse positions.

| Chunk Size | Trade-off |
|------------|-----------|
| Small (4-8) | Easier prediction, more inference calls |
| **16 (chosen)** | **Balanced, proven with GR00T** |
| Large (32+) | Fewer calls, harder prediction |

### 3. Variable Length with Stop Signal

**Why?** Don't force all drawings to same length. Instead, model learns when to stop.

```python
# Output per chunk: 16 points with state
[(x1, y1, state), (x2, y2, state), ..., (x16, y16, state)]

# state = 0: continue drawing
# state = 1: stroke complete (pen lift)
```

**Example - Drawing a circle (32 points total):**
```
Chunk 1: [(x,y,0), (x,y,0), ..., (x,y,0)]  → 16 points, keep going
Chunk 2: [(x,y,0), (x,y,0), ..., (x,y,1)]  → 16 points, last one says "done"
```

### 4. Visual Feedback Loop

Model sees the canvas AFTER each chunk is drawn, enabling:
- Course correction
- Context-aware continuation
- Natural stopping based on visual state

```
Step 1: Blank canvas + "draw circle" → Chunk 1 (16 pts)
Step 2: Canvas with partial circle   → Chunk 2 (16 pts)
Step 3: Canvas with more drawn       → Chunk 3 says "done"
```

---

## Variable Length Handling (GR00T-style)

Robotic VLAs face a key challenge: they don't know ahead of time how long a task will take.

| Task | Approximate Actions |
|------|---------------------|
| Pick up a cup | ~30 actions |
| Draw a circle | ~32 points |
| Make a sandwich | ~500 actions |
| Draw complex art | ~1000+ points |

### Our Approach: Stop Signal + Loss Masking

**Output format:**
```python
chunk = [(x1, y1, state), (x2, y2, state), ..., (x16, y16, state)]
# state = 0: continue drawing
# state = 1: stroke complete (STOP)
```

**Key insight:** Model ALWAYS predicts 16 points, but `state=1` means STOP. Points AFTER the stop signal are predicted but NOT executed.

**Example - Short circle (32 points):**
```
Chunk 1: 16 points, all state=0 → keep going
Chunk 2: 16 points, last state=1 → DONE!
```

**Example - Long squiggle (80 points):**
```
Chunk 1: 16 points, state=0 → continue
Chunk 2: 16 points, state=0 → continue
Chunk 3: 16 points, state=0 → continue
Chunk 4: 16 points, state=0 → continue
Chunk 5: 16 points, last state=1 → DONE!
```

### Training with Loss Masking

For final chunks with fewer than 16 real points:
```python
# Circle with 25 points → Chunk 2 has only 9 real points

chunk = [
    (x,y,0), (x,y,0), ..., (x,y,1),  # 9 real points
    (0,0,0), (0,0,0), ..., (0,0,0)   # 7 masked positions
]

mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
#       ↑ real points (count in loss)  ↑ ignored
```

**Why masking instead of padding?**
- Padding distorts the trajectory distribution
- Masking preserves natural drawing patterns
- Model learns WHERE to put state=1, not fixed positions

| Approach | Fixed (64 pts) | Variable + Stop Signal |
|----------|----------------|------------------------|
| Flexibility | Limited | Unlimited length |
| Realism | Distorted | Natural drawings |
| Future-proof | No | Yes - works for any task |
| Complexity | Simpler | Slightly more complex |

---

## Method: Flow Matching on Chunks

Same flow matching approach as GR00T, applied to 16-point chunks:

### Training
```python
CHUNK_SIZE = 16  # points per chunk

# Sample a chunk from trajectory
target_chunk = trajectory[start:start+16]  # shape: (16, 3) with (x, y, state)

# Sample noise and timestep
noise = torch.randn(16, 3)
t = torch.rand(1)

# Interpolate (flow matching)
noisy_chunk = (1 - t) * noise + t * target_chunk
velocity_target = target_chunk - noise

# Model predicts velocity field for this chunk
velocity_pred = model(canvas_image, prompt, noisy_chunk, t)
loss = MSE(velocity_pred, velocity_target)
```

### Inference (Chunked Euler Integration)
```python
canvas = blank_canvas()
full_trajectory = []

while True:
    # Predict next chunk
    chunk = torch.randn(16, 3)  # start from noise

    for k in range(K):  # K=16 denoising steps
        t = k / K
        velocity = model(canvas, prompt, chunk, t)
        chunk = chunk + velocity * (1/K)

    # Execute chunk
    for (x, y, state) in chunk:
        full_trajectory.append((x, y))
        draw_on_canvas(canvas, x, y)

        if state > 0.5:  # stop signal
            return full_trajectory
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
    --num_train_epochs 30
```

### Training Epochs: Key Learning from VLA Literature

**VLA models require significantly more epochs than typical LLM/VLM training.**

| Model | Epochs | Data Size | Source |
|-------|--------|-----------|--------|
| **OpenVLA** | 27 epochs | 970k trajectories | [Paper](https://arxiv.org/abs/2406.09246) |
| **GR00T N1** | 100 epochs (finetune) | 3k real + synthetic | [Whitepaper](https://arxiv.org/abs/2503.14734) |
| **pi0** | 8-12 epochs | 10k+ hours | [Paper](https://arxiv.org/abs/2410.24164) |

> *"Typical LLM or VLM training runs complete at most one or two epochs... In contrast, we found it important for VLA training to iterate through the training dataset significantly more times, with real robot performance continually improving until training action token accuracy surpasses 95%."* — OpenVLA Paper

**Recommendation**: Train for **20-30 epochs minimum** for simple shapes, potentially 50+ for complex trajectories.

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

## Dataset: Quick, Draw! Circles

### The Data Challenge

Most VLM training uses static image-text pairs, but we need **trajectory data** - sequences of (x, y) coordinates representing mouse movements. No large-scale mouse trajectory dataset exists for computer control.

### Our Solution: Quick, Draw!

[Quick, Draw!](https://quickdraw.withgoogle.com/data) contains 50M+ human-drawn sketches with **stroke data** - exactly what we need:

```python
# Quick, Draw! data format (example circle)
{
    "drawing": [
        [[x1, x2, x3, ...], [y1, y2, y3, ...], [t1, t2, t3, ...]]  # stroke 1
    ],
    "word": "circle"
}
# Coordinates in [0, 255] range, with timestamps
```

This gives us real human drawing trajectories - natural, variable-length, with the imperfections that make them realistic.

### Dataset Configuration

| Setting | Value | Reasoning |
|---------|-------|-----------|
| **Category** | Circle | Simple shape, single stroke, good for POC |
| **Samples** | 10,000 circles | Similar to click model (20k), sufficient for POC |
| **Canvas** | 512×512 white | Fits Qwen2.5-VL optimal range (~448-634px) |
| **Normalization** | [0, 1] | Consistent with click model |
| **Chunk size** | 16 points | Matches GR00T action horizon |

### Assumptions and Limitations

**Data Assumptions:**
1. **Single category only** - Circles for proof of concept. Can extend to all 345 Quick, Draw! categories.
2. **Single stroke** - Circles are typically drawn in one continuous stroke. Multi-stroke shapes would need pen-lift handling.
3. **Synthetic canvas** - White background, not real screenshots. Future work: overlay on real UI.
4. **Fixed instruction** - "draw a circle" for all samples. Future: varied prompts.

**Architecture Assumptions:**
5. **Frozen VLM backbone** - Only the DiT action head is trained. Qwen2.5-VL weights are frozen.
6. **Output format: (x, y, state)** - Each point has 3 dimensions. `state=1` means STOP.
7. **Chunk size = 16** - Model always predicts 16 points per forward pass (like GR00T's H=16).
8. **Stateless inference** - Each chunk prediction is independent. Model sees current canvas + instruction (repeated every chunk).

**Training Assumptions:**
9. **Flow matching loss** - MSE between predicted and target velocity fields.
10. **Loss masking** - Positions after the last real point in final chunks are masked (not padded).
11. **No data augmentation** - Raw trajectories used as-is. Could add rotation/scaling in future.
12. **bfloat16 training** - Uses mixed precision for memory efficiency.

### Data Pipeline

```
Quick, Draw! (streaming from Google Cloud)
    ↓
Filter: label == "circle"
    ↓
Take: 10,000 samples
    ↓
For each sample:
    1. Parse stroke data → list of (x, y) points
    2. Normalize [0, 255] → [0, 1]
    3. Split into 16-point chunks with state signal
    4. Create canvas image (blank for first chunk, partial for rest)
    5. Save: {image, instruction, trajectory, mask}
```

### Generated Dataset Format

```python
{
    "image": PIL.Image,           # 512×512 canvas
    "instruction": "draw a circle",
    "trajectory": (16, 3),        # (x, y, state) per point
    "mask": (16,),                # 1=real point, 0=ignore in loss
    "is_last": bool,              # True if final chunk
}
```

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
