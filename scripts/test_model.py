"""
Test and visualize the trained Qwen-DiT-Draw model.

Usage:
    # Run on Modal (uses trained checkpoint)
    python modal_train.py inference --image white_canvas.png

    # Local visualization of results
    python scripts/test_model.py --checkpoint_dir /path/to/checkpoint
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_white_canvas(size: int = 512) -> Image.Image:
    """Create a white canvas for testing."""
    return Image.new("RGB", (size, size), "white")


def visualize_trajectory(
    image: Image.Image,
    trajectory: list,
    output_path: str = "trajectory_output.png",
    show: bool = True
):
    """
    Visualize a trajectory on top of an image.

    Args:
        image: PIL Image (canvas)
        trajectory: List of [x, y, state] points, normalized [0,1]
        output_path: Where to save the visualization
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get image dimensions
    width, height = image.size

    # Convert trajectory to pixel coordinates
    points = np.array(trajectory)
    x_coords = points[:, 0] * width
    y_coords = points[:, 1] * height
    states = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))

    # Plot 1: Original image with trajectory overlay
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='x', label='End', zorder=5)
    ax1.set_title('Predicted Trajectory')
    ax1.legend()
    ax1.axis('off')

    # Plot 2: Draw the trajectory as a stroke
    ax2 = axes[1]
    canvas_with_stroke = image.copy()
    draw = ImageDraw.Draw(canvas_with_stroke)

    # Draw the stroke
    for i in range(len(x_coords) - 1):
        draw.line(
            [(x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1])],
            fill='blue',
            width=3
        )

    ax2.imshow(canvas_with_stroke)
    ax2.set_title('Drawn Result')
    ax2.axis('off')

    # Plot 3: Trajectory in normalized space
    ax3 = axes[2]
    ax3.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
    ax3.scatter(points[0, 0], points[0, 1], c='green', s=100, marker='o', label='Start')
    ax3.scatter(points[-1, 0], points[-1, 1], c='red', s=100, marker='x', label='End')

    # Color points by sequence order
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    ax3.scatter(points[:, 0], points[:, 1], c=colors, s=20, alpha=0.7)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(1, 0)  # Flip y-axis to match image coordinates
    ax3.set_xlabel('X (normalized)')
    ax3.set_ylabel('Y (normalized)')
    ax3.set_title('Trajectory in Normalized Space')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_trajectory(trajectory: list) -> dict:
    """Analyze trajectory properties."""
    points = np.array(trajectory)

    # Compute center
    center_x = points[:, 0].mean()
    center_y = points[:, 1].mean()

    # Compute bounding box
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()

    # Compute approximate radius (for circles)
    distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
    avg_radius = distances.mean()
    radius_std = distances.std()

    # Check if trajectory closes (circle detection)
    start = points[0, :2]
    end = points[-1, :2]
    closure_distance = np.linalg.norm(end - start)

    # Compute total path length
    path_length = np.sum(np.sqrt(np.sum(np.diff(points[:, :2], axis=0)**2, axis=1)))

    return {
        "num_points": len(points),
        "center": [float(center_x), float(center_y)],
        "bounding_box": {
            "min_x": float(min_x), "max_x": float(max_x),
            "min_y": float(min_y), "max_y": float(max_y),
            "width": float(max_x - min_x),
            "height": float(max_y - min_y),
        },
        "avg_radius": float(avg_radius),
        "radius_std": float(radius_std),
        "closure_distance": float(closure_distance),
        "is_closed": closure_distance < 0.1,  # Threshold for considering it closed
        "path_length": float(path_length),
    }


def run_batch_test(model_output_dir: str, num_samples: int = 5):
    """
    Run batch inference and visualize results.
    This is meant to be run on Modal after training.
    """
    print(f"Testing model from {model_output_dir}")
    print(f"Generating {num_samples} samples...")

    # This would be called from Modal
    # For now, just create placeholder for expected output format
    example_trajectory = [
        [0.3 + 0.2 * np.cos(t), 0.5 + 0.2 * np.sin(t), 0.0]
        for t in np.linspace(0, 2*np.pi, 16)
    ]

    return example_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen-DiT-Draw model")
    parser.add_argument("--trajectory_json", type=str, help="JSON file with trajectory")
    parser.add_argument("--output", type=str, default="trajectory_viz.png")
    parser.add_argument("--canvas_size", type=int, default=512)
    parser.add_argument("--no_show", action="store_true")

    args = parser.parse_args()

    # Create canvas
    canvas = create_white_canvas(args.canvas_size)

    if args.trajectory_json:
        # Load trajectory from JSON
        with open(args.trajectory_json, 'r') as f:
            data = json.load(f)
        trajectory = data.get("chunk", data.get("trajectory", []))
    else:
        # Demo with a sample circle trajectory
        print("No trajectory provided, generating demo circle...")
        trajectory = [
            [0.5 + 0.25 * np.cos(t), 0.5 + 0.25 * np.sin(t), 0.0]
            for t in np.linspace(0, 2*np.pi, 16)
        ]

    # Analyze trajectory
    print("\nTrajectory Analysis:")
    analysis = analyze_trajectory(trajectory)
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    # Visualize
    visualize_trajectory(
        canvas,
        trajectory,
        output_path=args.output,
        show=not args.no_show
    )
