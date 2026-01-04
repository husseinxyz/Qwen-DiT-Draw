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
    target_length: int = 64,
    chunk_size: int = 16,
    canvas_size: int = 512,
):
    """
    Generate training dataset from Quick, Draw! circles.

    Args:
        num_circles: Number of circles to process
        output_dir: Output directory for dataset
        target_length: Points per trajectory after resampling
        chunk_size: Points per chunk
        canvas_size: Canvas size in pixels

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

    print(f"\nGenerating training samples...")
    print(f"Config: target_length={target_length}, chunk_size={chunk_size}, canvas={canvas_size}x{canvas_size}")

    all_samples = []
    sample_idx = 0

    for circle_idx, circle_points in enumerate(tqdm(circles, desc="Processing circles")):
        # Resample and normalize
        resampled = resample_trajectory(circle_points, target_length)
        normalized = normalize_trajectory(resampled, original_range=(0, 255))

        # Generate chunked training samples
        samples = generate_training_samples(normalized, chunk_size, canvas_size)

        for chunk_idx, sample in enumerate(samples):
            # Save image
            image_filename = f"sample_{sample_idx:06d}.png"
            image_path = images_dir / image_filename
            sample["image"].save(image_path)

            # Save trajectory
            traj_filename = f"sample_{sample_idx:06d}.npy"
            traj_path = traj_dir / traj_filename
            np.save(traj_path, sample["target_chunk"])

            # Record metadata
            all_samples.append({
                "image": image_filename,
                "trajectory": traj_filename,
                "instruction": sample["instruction"],
                "is_last": sample["is_last"],
                "circle_idx": circle_idx,
                "chunk_idx": chunk_idx,
            })

            sample_idx += 1

    # Save metadata
    metadata = {
        "num_circles": len(circles),
        "num_samples": len(all_samples),
        "target_length": target_length,
        "chunk_size": chunk_size,
        "canvas_size": canvas_size,
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


def upload_to_huggingface(
    data_dir: str,
    repo_id: str,
    private: bool = False,
):
    """
    Upload generated dataset to HuggingFace Hub.

    Args:
        data_dir: Path to generated dataset
        repo_id: HuggingFace repo ID (e.g., "TESS-Computer/quickdraw-circles")
        private: Whether to make repo private
    """
    from datasets import Dataset, Features, Value, Image, Sequence
    from huggingface_hub import HfApi, create_repo

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

            trajectory = np.load(traj_path).tolist()

            yield {
                "image": str(image_path),
                "instruction": sample["instruction"],
                "trajectory": trajectory,
                "is_last": sample["is_last"],
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

    dataset.push_to_hub(
        repo_id,
        commit_message="Upload Quick, Draw! circles dataset for trajectory prediction",
    )

    print(f"\nDataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory training dataset")
    parser.add_argument("--num_circles", type=int, default=10000, help="Number of circles")
    parser.add_argument("--output_dir", type=str, default="data/circles", help="Output directory")
    parser.add_argument("--target_length", type=int, default=64, help="Points per trajectory")
    parser.add_argument("--chunk_size", type=int, default=16, help="Points per chunk")
    parser.add_argument("--canvas_size", type=int, default=512, help="Canvas size")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")
    parser.add_argument("--repo_id", type=str, default="TESS-Computer/quickdraw-circles", help="HF repo ID")
    parser.add_argument("--private", action="store_true", help="Make HF repo private")
    args = parser.parse_args()

    # Generate dataset
    output_dir = generate_dataset(
        num_circles=args.num_circles,
        output_dir=args.output_dir,
        target_length=args.target_length,
        chunk_size=args.chunk_size,
        canvas_size=args.canvas_size,
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
