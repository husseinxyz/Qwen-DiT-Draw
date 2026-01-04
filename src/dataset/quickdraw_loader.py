"""
Quick, Draw! Dataset Loader

Downloads and parses circle drawings from Google's Quick, Draw! dataset.
Uses the simplified .ndjson format from the official repository.

Source: https://storage.googleapis.com/quickdraw_dataset/full/simplified/
Format: NDJSON with 'drawing' field containing stroke data
"""

import json
import urllib.request
import gzip
from io import BytesIO
from typing import List, Tuple, Iterator
import numpy as np
from pathlib import Path


# Google Cloud Storage URL for Quick, Draw! simplified data
QUICKDRAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"


def parse_stroke_data(drawing: List[List[List[int]]]) -> List[Tuple[float, float]]:
    """
    Parse Quick, Draw! stroke format into list of (x, y) points.

    Quick, Draw! format:
        drawing = [stroke1, stroke2, ...]
        stroke = [[x1, x2, ...], [y1, y2, ...]]

    We concatenate all strokes into a single trajectory.
    For circles, there's typically just one continuous stroke.

    Args:
        drawing: List of strokes, each stroke is [[xs], [ys]]

    Returns:
        List of (x, y) tuples representing the trajectory
    """
    points = []
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for x, y in zip(xs, ys):
            points.append((float(x), float(y)))
    return points


def download_category(category: str = "circle", cache_dir: str = None) -> Path:
    """
    Download Quick, Draw! category data from Google Cloud Storage.

    Args:
        category: Category name (e.g., 'circle', 'square')
        cache_dir: Directory to cache downloaded file

    Returns:
        Path to downloaded .ndjson file
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "quickdraw"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{category}.ndjson"

    if cache_file.exists():
        print(f"Using cached data: {cache_file}")
        return cache_file

    url = QUICKDRAW_URL.format(category=category)
    print(f"Downloading {category} from {url}...")

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            with open(cache_file, "wb") as f:
                f.write(data)
        print(f"Downloaded and cached: {cache_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {category}: {e}")

    return cache_file


def stream_circles(
    num_samples: int = 10000,
    category: str = "circle",
    cache_dir: str = None
) -> Iterator[List[Tuple[float, float]]]:
    """
    Stream circle drawings from Quick, Draw! dataset.

    Downloads from Google Cloud Storage if not cached.
    Quick, Draw! coordinates are in range [0, 255].

    Args:
        num_samples: Number of samples to yield
        category: Category name
        cache_dir: Cache directory for downloaded data

    Yields:
        List of (x, y) points for each circle trajectory
    """
    cache_file = download_category(category, cache_dir)

    count = 0
    with open(cache_file, "r") as f:
        for line in f:
            if count >= num_samples:
                break

            try:
                sample = json.loads(line)
                drawing = sample["drawing"]
                points = parse_stroke_data(drawing)

                # Skip if too few points (degenerate drawing)
                if len(points) < 10:
                    continue

                yield points
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue


def load_circles(
    num_samples: int = 10000,
    category: str = "circle",
    cache_dir: str = None
) -> List[List[Tuple[float, float]]]:
    """
    Load circle drawings into memory.

    Args:
        num_samples: Number of circles to load
        category: Category name
        cache_dir: Cache directory

    Returns:
        List of circle trajectories, each trajectory is list of (x, y) points
    """
    circles = list(stream_circles(num_samples, category, cache_dir))
    print(f"Loaded {len(circles)} circles from Quick, Draw!")
    return circles


def get_circle_stats(circles: List[List[Tuple[float, float]]]) -> dict:
    """Get statistics about loaded circles."""
    lengths = [len(c) for c in circles]
    return {
        "num_circles": len(circles),
        "min_points": min(lengths),
        "max_points": max(lengths),
        "avg_points": np.mean(lengths),
        "median_points": np.median(lengths),
    }


if __name__ == "__main__":
    # Test loading
    print("Loading 100 circles for testing...")
    circles = load_circles(100)
    stats = get_circle_stats(circles)
    print(f"Stats: {stats}")
    print(f"First circle has {len(circles[0])} points")
    print(f"First 5 points: {circles[0][:5]}")
