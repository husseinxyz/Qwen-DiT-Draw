"""
Generate training dataset from Quick, Draw! circles.

Creates chunked training samples and uploads to HuggingFace Hub.

Usage:
    # Generate locally first
    python scripts/generate_dataset.py --num_circles 10000 --output_dir data/circles

    # Upload to HuggingFace
    python scripts/generate_dataset.py --num_circles 10000 --upload --repo_id TESS-Computer/quickdraw-circles
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.quickdraw_loader import load_circles, get_circle_stats
from src.dataset.trajectory_utils import (
    resample_trajectory,
    normalize_trajectory,
    generate_training_samples,
)


def generate_dataset(
    num_circles: int = 10000,
    output_dir: str = "data/circles",
    chunk_size: int = 16,
    canvas_size: int = 512,
    use_deltas: bool = False,
):
    """
    Generate training dataset from Quick, Draw! circles.

    VARIABLE LENGTH: Each circle keeps its natural length (no resampling).
    Model learns when to stop via is_last flag.

    Args:
        num_circles: Number of circles to process
        output_dir: Output directory for dataset
        chunk_size: Points per chunk (16 like GR00T)
        canvas_size: Canvas size in pixels
        use_deltas: If True, output delta movements instead of absolute coords

    Returns:
        Path to generated dataset
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    traj_dir = output_dir / "trajectories"

    images_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {num_circles} circles from Quick, Draw!...")
    circles = load_circles(num_circles)
    stats = get_circle_stats(circles)
    print(f"Dataset stats: {stats}")

    coord_mode = "DELTA" if use_deltas else "ABSOLUTE"
    print(f"\nGenerating training samples (VARIABLE LENGTH - no resampling)...")
    print(f"Config: chunk_size={chunk_size}, canvas={canvas_size}x{canvas_size}, coords={coord_mode}")

    all_samples = []
    sample_idx = 0

    for circle_idx, circle_points in enumerate(tqdm(circles, desc="Processing circles")):
        # NO RESAMPLING - keep natural length!
        # Just normalize from Quick, Draw! range [0, 255] to [0, 1]
        normalized = normalize_trajectory(np.array(circle_points), original_range=(0, 255))

        # Generate chunked training samples (variable number of chunks per circle)
        samples = generate_training_samples(normalized, chunk_size, canvas_size, use_deltas=use_deltas)

        for chunk_idx, sample in enumerate(samples):
            # Save image
            image_filename = f"sample_{sample_idx:06d}.png"
            image_path = images_dir / image_filename
            sample["image"].save(image_path)

            # Save trajectory
            traj_filename = f"sample_{sample_idx:06d}.npy"
            traj_path = traj_dir / traj_filename
            np.save(traj_path, sample["target_chunk"])

            # Save mask
            mask_filename = f"sample_{sample_idx:06d}_mask.npy"
            mask_path = traj_dir / mask_filename
            np.save(mask_path, sample["mask"])

            # Record metadata
            all_samples.append({
                "image": image_filename,
                "trajectory": traj_filename,
                "mask": mask_filename,
                "instruction": sample["instruction"],
                "is_last": sample["is_last"],
                "n_real_points": sample["n_real_points"],
                "circle_idx": circle_idx,
                "chunk_idx": chunk_idx,
            })

            sample_idx += 1

    # Save metadata
    metadata = {
        "num_circles": len(circles),
        "num_samples": len(all_samples),
        "variable_length": True,  # Circles keep natural length
        "chunk_size": chunk_size,
        "canvas_size": canvas_size,
        "use_deltas": use_deltas,  # True = (dx, dy, state), False = (x, y, state)
        "samples": all_samples,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset generated successfully!")
    print(f"  Circles: {len(circles)}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Avg samples per circle: {len(all_samples) / len(circles):.1f}")
    print(f"  Output directory: {output_dir.absolute()}")

    return output_dir


def create_dataset_card(metadata: dict) -> str:
    """Generate a dataset card (README.md) for HuggingFace."""
    return f'''---
license: mit
task_categories:
  - robotics
tags:
  - trajectory-prediction
  - mouse-control
  - computer-control
  - quick-draw
  - diffusion
size_categories:
  - 10K<n<100K
---

# Quick, Draw! Circles - Trajectory Dataset

Dataset for training trajectory prediction models, specifically designed for the [Qwen-DiT-Draw](https://github.com/husseinxyz/qwen-dit-draw) project.

## Dataset Description

This dataset contains chunked trajectory data from the [Quick, Draw!](https://quickdraw.withgoogle.com/data) circle category, formatted for training diffusion-based trajectory prediction models.

### Key Features

- **Variable-length trajectories** with stop signals (GR00T-style)
- **16-point chunks** with (x, y, state) format
- **Loss masking** for handling variable-length final chunks
- **512×512 canvas images** showing drawing progression

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | {metadata['num_samples']} |
| Source circles | {metadata['num_circles']} |
| Chunk size | {metadata['chunk_size']} points |
| Canvas size | {metadata['canvas_size']}×{metadata['canvas_size']} |
| Avg chunks/circle | {metadata['num_samples'] / metadata['num_circles']:.1f} |

## Data Format

Each sample contains:

```python
{{
    "image": Image,           # 512×512 canvas (white for first chunk, partial drawing for rest)
    "instruction": str,       # "draw a circle"
    "trajectory": [[x, y, state], ...],  # 16 points, normalized [0, 1]
    "mask": [1, 1, ..., 0, 0],           # 1=real point, 0=ignore in loss
    "is_last": bool,          # True if final chunk of trajectory
    "n_real_points": int,     # Number of real points in this chunk (1-16)
    "circle_idx": int,        # Source circle index
    "chunk_idx": int,         # Chunk index within circle
}}
```

### State Signal

- `state = 0`: Continue drawing
- `state = 1`: Stroke complete (STOP)

The model learns WHERE to place the stop signal, not a fixed position.

### Loss Masking

For final chunks with fewer than 16 real points:
```
mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        ↑ real points (count in loss)  ↑ ignored
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("TESS-Computer/quickdraw-circles")

# Access a sample
sample = dataset["train"][0]
image = sample["image"]           # PIL Image
trajectory = sample["trajectory"] # List of [x, y, state]
mask = sample["mask"]             # Loss mask
```

## Source

Data sourced from [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset) (circle category only).

## License

MIT License

## Citation

```bibtex
@misc{{quickdraw-circles-trajectory,
    title={{Quick, Draw! Circles Trajectory Dataset}},
    author={{TESS Computer}},
    year={{2025}},
    url={{https://huggingface.co/datasets/TESS-Computer/quickdraw-circles}}
}}
```
'''


def upload_to_huggingface(
    data_dir: str,
    repo_id: str,
    private: bool = False,
):
    """
    Upload generated dataset to HuggingFace Hub with proper datacard.

    Args:
        data_dir: Path to generated dataset
        repo_id: HuggingFace repo ID (e.g., "TESS-Computer/quickdraw-circles")
        private: Whether to make repo private
    """
    from datasets import Dataset, Features, Value, Image, Sequence
    from huggingface_hub import HfApi, create_repo, upload_file

    data_dir = Path(data_dir)

    print(f"Loading dataset from {data_dir}...")

    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    samples = metadata["samples"]
    print(f"Found {len(samples)} samples")

    # Build dataset
    print("Building HuggingFace dataset...")

    def generate_examples():
        for sample in tqdm(samples, desc="Loading samples"):
            image_path = data_dir / "images" / sample["image"]
            traj_path = data_dir / "trajectories" / sample["trajectory"]
            mask_path = data_dir / "trajectories" / sample["mask"]

            # Load trajectory (16, 3) with (x, y, state)
            trajectory = np.load(traj_path).tolist()

            # Load mask
            mask = np.load(mask_path).tolist()

            yield {
                "image": str(image_path),
                "instruction": sample["instruction"],
                "trajectory": trajectory,
                "mask": mask,
                "is_last": sample["is_last"],
                "n_real_points": sample["n_real_points"],
                "circle_idx": sample["circle_idx"],
                "chunk_idx": sample["chunk_idx"],
            }

    # Create dataset from generator
    data_list = list(generate_examples())
    dataset = Dataset.from_list(data_list)

    # Cast image column
    dataset = dataset.cast_column("image", Image())

    print(f"Dataset: {dataset}")
    print(f"Features: {dataset.features}")

    # Create repo and upload
    print(f"\nUploading to {repo_id}...")

    api = HfApi()
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload dataset
    dataset.push_to_hub(
        repo_id,
        commit_message="Upload Quick, Draw! circles dataset for trajectory prediction",
    )

    # Upload datacard
    print("Uploading dataset card...")
    datacard = create_dataset_card(metadata)
    datacard_path = data_dir / "README.md"
    with open(datacard_path, "w") as f:
        f.write(datacard)

    api.upload_file(
        path_or_fileobj=str(datacard_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    print(f"\nDataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory training dataset")
    parser.add_argument("--num_circles", type=int, default=10000, help="Number of circles")
    parser.add_argument("--output_dir", type=str, default="data/circles", help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=16, help="Points per chunk")
    parser.add_argument("--canvas_size", type=int, default=512, help="Canvas size")
    parser.add_argument("--use_deltas", action="store_true", help="Output delta movements instead of absolute coords")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--repo_id", type=str, default="TESS-Computer/quickdraw-circles", help="HF repo ID")
    parser.add_argument("--private", action="store_true", help="Make HF repo private")
    args = parser.parse_args()

    # Generate dataset (variable length - no resampling)
    output_dir = generate_dataset(
        num_circles=args.num_circles,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        canvas_size=args.canvas_size,
        use_deltas=args.use_deltas,
    )

    # Upload if requested
    if args.upload:
        upload_to_huggingface(
            data_dir=output_dir,
            repo_id=args.repo_id,
            private=args.private,
        )


if __name__ == "__main__":
    main()
