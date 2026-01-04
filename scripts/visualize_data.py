"""
Visualize Quick, Draw! circle data processing.

Run this to inspect a few examples before generating the full dataset.

Usage:
    python scripts/visualize_data.py --num_circles 5 --output_dir visualizations
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.quickdraw_loader import load_circles, get_circle_stats
from src.dataset.trajectory_utils import (
    normalize_trajectory,
    generate_training_samples,
    render_partial_trajectory,
    create_blank_canvas,
)

import numpy as np
from PIL import Image


def visualize_circle_processing(
    circle_points: list,
    circle_idx: int,
    output_dir: Path,
    chunk_size: int = 16,
    canvas_size: int = 512,
):
    """
    Visualize the full processing pipeline for a single circle.

    VARIABLE LENGTH: No resampling - circles keep natural length.

    Creates:
    1. Original circle (raw points from Quick, Draw!)
    2. Chunked training samples (variable number based on length)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Circle {circle_idx} ---")
    print(f"Original points: {len(circle_points)}")

    # Step 1: Normalize (NO RESAMPLING - keep natural length!)
    normalized = normalize_trajectory(np.array(circle_points), original_range=(0, 255))

    # Render full circle
    full_canvas = create_blank_canvas(canvas_size)
    full_rendered = render_partial_trajectory(normalized, canvas_size, existing_canvas=full_canvas)
    full_rendered.save(output_dir / f"circle_{circle_idx}_1_full_{len(circle_points)}pts.png")
    print(f"Saved: circle_{circle_idx}_1_full_{len(circle_points)}pts.png")

    # Step 2: Generate chunked training samples (VARIABLE number of chunks)
    training_samples = generate_training_samples(normalized, chunk_size, canvas_size)
    num_chunks = len(training_samples)
    print(f"Generated {num_chunks} training samples ({len(circle_points)} pts / {chunk_size} = {num_chunks} chunks)")

    # Create a combined visualization showing the progression
    n_samples = len(training_samples)
    combined_width = canvas_size * (n_samples + 1)  # +1 for final result
    combined = Image.new("RGB", (combined_width, canvas_size + 60), (255, 255, 255))

    # Add each chunk's input canvas
    for i, sample in enumerate(training_samples):
        combined.paste(sample["image"], (i * canvas_size, 30))

    # Add final rendered result
    final_canvas = create_blank_canvas(canvas_size)
    final_rendered = render_partial_trajectory(normalized, canvas_size, existing_canvas=final_canvas)
    combined.paste(final_rendered, (n_samples * canvas_size, 30))

    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    for i in range(n_samples):
        label = f"Chunk {i+1} input"
        draw.text((i * canvas_size + 10, 8), label, fill=(0, 0, 0), font=font)

    draw.text((n_samples * canvas_size + 10, 8), "Final result", fill=(0, 0, 0), font=font)

    combined.save(output_dir / f"circle_{circle_idx}_3_chunks_progression.png")
    print(f"Saved: circle_{circle_idx}_3_chunks_progression.png")

    # Save individual chunk inputs and targets
    for i, sample in enumerate(training_samples):
        sample["image"].save(output_dir / f"circle_{circle_idx}_chunk{i}_input.png")
        # Also render what the target chunk looks like
        target_canvas = sample["image"].copy()
        # Extract only (x, y) from (x, y, state) - first 2 columns
        n_real = sample["n_real_points"]
        target_xy = sample["target_chunk"][:n_real, :2]
        target_rendered = render_partial_trajectory(
            target_xy, canvas_size,
            existing_canvas=target_canvas,
            line_color=(255, 0, 0)  # Red for target
        )
        target_rendered.save(output_dir / f"circle_{circle_idx}_chunk{i}_with_target.png")

    print(f"Saved individual chunk images")

    return training_samples


def main():
    parser = argparse.ArgumentParser(description="Visualize Quick, Draw! circle processing")
    parser.add_argument("--num_circles", type=int, default=5, help="Number of circles to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=16, help="Points per chunk")
    parser.add_argument("--canvas_size", type=int, default=512, help="Canvas size in pixels")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.num_circles} circles from Quick, Draw!...")
    circles = load_circles(args.num_circles)

    if len(circles) == 0:
        print("ERROR: No circles loaded!")
        return

    stats = get_circle_stats(circles)
    print(f"\nDataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nProcessing and visualizing {len(circles)} circles (VARIABLE LENGTH - no resampling)...")
    print(f"Config: chunk_size={args.chunk_size}, canvas={args.canvas_size}x{args.canvas_size}")

    total_samples = 0
    for i, circle in enumerate(circles):
        samples = visualize_circle_processing(
            circle, i, output_dir,
            chunk_size=args.chunk_size,
            canvas_size=args.canvas_size,
        )
        total_samples += len(samples)

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Circles processed: {len(circles)}")
    print(f"  Total training samples: {total_samples}")
    print(f"  Avg samples per circle: {total_samples / len(circles):.1f}")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"\nOpen the output directory to inspect the visualizations!")


if __name__ == "__main__":
    main()
