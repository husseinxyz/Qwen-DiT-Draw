"""
Trajectory utilities for resampling, normalization, and rendering.

Key functions:
- resample_trajectory: Interpolate to fixed number of points
- normalize_trajectory: Scale coordinates to [0, 1]
- render_partial_trajectory: Draw points on canvas (for training data generation)
"""

import numpy as np
from typing import List, Tuple
from PIL import Image, ImageDraw


def resample_trajectory(
    points: List[Tuple[float, float]],
    target_length: int = 64
) -> np.ndarray:
    """
    Resample trajectory to fixed number of points using linear interpolation.

    Args:
        points: List of (x, y) points
        target_length: Number of points to resample to

    Returns:
        np.ndarray of shape (target_length, 2)
    """
    points = np.array(points)
    n_points = len(points)

    if n_points == target_length:
        return points

    # Compute cumulative arc length for parameterization
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        # Degenerate case: all points are the same
        return np.tile(points[0], (target_length, 1))

    # Evenly spaced points along arc length
    target_lengths = np.linspace(0, total_length, target_length)

    # Interpolate x and y separately
    resampled = np.zeros((target_length, 2))
    resampled[:, 0] = np.interp(target_lengths, cumulative_length, points[:, 0])
    resampled[:, 1] = np.interp(target_lengths, cumulative_length, points[:, 1])

    return resampled


def normalize_trajectory(
    points: np.ndarray,
    original_range: Tuple[float, float] = (0, 255)
) -> np.ndarray:
    """
    Normalize trajectory coordinates to [0, 1] range.

    Quick, Draw! uses [0, 255] range by default.

    Args:
        points: np.ndarray of shape (N, 2)
        original_range: (min, max) of original coordinates

    Returns:
        np.ndarray of shape (N, 2) with values in [0, 1]
    """
    min_val, max_val = original_range
    normalized = (points - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def denormalize_trajectory(
    points: np.ndarray,
    canvas_size: int = 512
) -> np.ndarray:
    """
    Convert normalized [0, 1] coordinates to pixel coordinates.

    Args:
        points: np.ndarray of shape (N, 2) in [0, 1]
        canvas_size: Size of canvas in pixels

    Returns:
        np.ndarray of shape (N, 2) in pixel coordinates
    """
    return points * canvas_size


def render_partial_trajectory(
    points: np.ndarray,
    canvas_size: int = 512,
    line_width: int = 3,
    line_color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Tuple[int, int, int] = (255, 255, 255),
    existing_canvas: Image.Image = None
) -> Image.Image:
    """
    Render trajectory points on a canvas.

    For training data generation:
    - Blank canvas for first chunk
    - Existing canvas with partial drawing for subsequent chunks

    Args:
        points: np.ndarray of shape (N, 2) in [0, 1] normalized coordinates
        canvas_size: Canvas dimensions (square)
        line_width: Width of drawn lines
        line_color: RGB color for trajectory
        background_color: RGB color for background
        existing_canvas: Optional existing canvas to draw on

    Returns:
        PIL Image with trajectory rendered
    """
    if existing_canvas is not None:
        canvas = existing_canvas.copy()
    else:
        canvas = Image.new("RGB", (canvas_size, canvas_size), background_color)

    draw = ImageDraw.Draw(canvas)

    # Convert normalized to pixel coordinates
    pixel_points = denormalize_trajectory(points, canvas_size)

    # Draw lines connecting consecutive points
    for i in range(len(pixel_points) - 1):
        x1, y1 = pixel_points[i]
        x2, y2 = pixel_points[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)

    return canvas


def create_blank_canvas(
    canvas_size: int = 512,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Create a blank white canvas."""
    return Image.new("RGB", (canvas_size, canvas_size), background_color)


def split_into_chunks_with_state(
    trajectory: np.ndarray,
    chunk_size: int = 16
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split trajectory into chunks with done state signal (GR00T-style).

    Key insight: Model always predicts 16 points, but state=1 means STOP.
    Points AFTER state=1 are predicted but NOT executed.

    For training, we use a MASK to ignore loss on positions after the real endpoint.

    Args:
        trajectory: np.ndarray of shape (T, 2) - just (x, y) coordinates
        chunk_size: Points per chunk

    Returns:
        List of (chunk, mask) tuples:
        - chunk: shape (chunk_size, 3) with (x, y, state)
        - mask: shape (chunk_size,) with 1=real point, 0=ignore in loss
    """
    n_points = len(trajectory)
    n_chunks = (n_points + chunk_size - 1) // chunk_size  # Ceiling division

    results = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_points)

        # Get raw (x, y) points for this chunk
        xy_points = trajectory[start:end]
        n_real_points = len(xy_points)

        # Create (x, y, state) array - zeros for positions we don't care about
        chunk = np.zeros((chunk_size, 3))

        # Create mask: 1 for real points, 0 for positions to ignore
        mask = np.zeros(chunk_size)

        # Fill in real points
        chunk[:n_real_points, :2] = xy_points
        chunk[:n_real_points, 2] = 0  # state=0: continue
        mask[:n_real_points] = 1  # These points count in loss

        # Mark the LAST real point of the ENTIRE trajectory with state=1
        is_last_chunk = (i == n_chunks - 1)
        if is_last_chunk:
            # Last real point gets state=1 (done)
            chunk[n_real_points - 1, 2] = 1

        # Positions beyond n_real_points: mask=0, so they're ignored in loss
        # We set them to zeros (arbitrary, won't affect training)

        results.append((chunk, mask))

    return results


def generate_training_samples(
    trajectory: np.ndarray,
    chunk_size: int = 16,
    canvas_size: int = 512
) -> List[dict]:
    """
    Generate training samples from a single trajectory (GR00T-style).

    Variable length with done signal:
    - Each point is (x, y, state)
    - state=0: continue, state=1: done
    - Mask indicates which positions to include in loss

    Args:
        trajectory: np.ndarray of shape (T, 2) normalized [0, 1]
        chunk_size: Points per chunk
        canvas_size: Canvas size in pixels

    Returns:
        List of training sample dictionaries with:
        - image: canvas showing previous drawing
        - instruction: text prompt
        - target_chunk: (chunk_size, 3) with (x, y, state)
        - mask: (chunk_size,) with 1=real, 0=ignore
    """
    chunk_results = split_into_chunks_with_state(trajectory, chunk_size)
    samples = []

    canvas = create_blank_canvas(canvas_size)

    for i, (chunk, mask) in enumerate(chunk_results):
        # Count real points from mask
        n_real = int(mask.sum())

        # Create training sample
        sample = {
            "image": canvas.copy(),
            "instruction": "draw a circle",
            "target_chunk": chunk,  # (chunk_size, 3) with (x, y, state)
            "mask": mask,           # (chunk_size,) for loss masking
            "chunk_index": i,
            "is_last": i == len(chunk_results) - 1,
            "n_real_points": n_real,
        }
        samples.append(sample)

        # Update canvas with ONLY real points (not masked positions)
        real_xy = chunk[:n_real, :2]
        canvas = render_partial_trajectory(real_xy, canvas_size, existing_canvas=canvas)

    return samples


if __name__ == "__main__":
    # Test trajectory utilities
    import matplotlib.pyplot as plt

    # Create a simple circle trajectory for testing
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([
        0.5 + 0.3 * np.cos(t),  # x: center at 0.5, radius 0.3
        0.5 + 0.3 * np.sin(t),  # y: center at 0.5, radius 0.3
    ], axis=1)

    # Test resampling
    resampled = resample_trajectory(circle.tolist(), target_length=64)
    print(f"Resampled shape: {resampled.shape}")

    # Test chunking
    chunks = split_into_chunks(resampled, chunk_size=16)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk shapes: {[c.shape for c in chunks]}")

    # Test training sample generation
    samples = generate_training_samples(resampled)
    print(f"Generated {len(samples)} training samples")

    # Visualize
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    for i, sample in enumerate(samples):
        axes[i].imshow(sample["image"])
        axes[i].set_title(f"Chunk {i+1} input\n(is_last={sample['is_last']})")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("/tmp/trajectory_test.png")
    print("Saved visualization to /tmp/trajectory_test.png")
