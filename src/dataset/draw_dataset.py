"""
PyTorch Dataset for trajectory prediction training.

Loads pre-generated training samples (canvas images + target chunks)
and prepares them for the Qwen2.5-VL + DiT model.

Supports two coordinate modes:
- absolute: (x, y) are normalized screen coordinates [0, 1]
- delta: (dx, dy) are relative movements from previous point
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np
from PIL import Image


def absolute_to_deltas(trajectory: np.ndarray) -> np.ndarray:
    """
    Convert absolute coordinates to delta movements.

    Args:
        trajectory: (N, 3) array with (x, y, state)

    Returns:
        (N, 3) array with (dx, dy, state)
        - First point: (x, y, state) - absolute starting position
        - Rest: (dx, dy, state) - delta from previous point
    """
    deltas = trajectory.copy()
    # Keep first point as absolute (starting position)
    # Convert rest to deltas: current - previous
    deltas[1:, :2] = trajectory[1:, :2] - trajectory[:-1, :2]
    return deltas


def deltas_to_absolute(deltas: np.ndarray, start_pos: np.ndarray = None) -> np.ndarray:
    """
    Convert delta movements back to absolute coordinates.

    For chunk 0 (start_pos=None): first element is absolute, rest are deltas
        deltas = [abs_start, Δ1, Δ2, ...]
        cumsum gives correct absolute positions

    For chunk 1+ (start_pos=prev_last_point): ALL elements are deltas
        deltas = [Δ0, Δ1, Δ2, ...] where Δ0 is delta from prev chunk's last
        cumsum + start_pos gives correct absolute positions

    Args:
        deltas: (N, 3) array with (dx, dy, state)
        start_pos: (2,) last position from previous chunk. If None, chunk 0 format.

    Returns:
        (N, 3) array with (x, y, state) absolute coordinates
    """
    absolute = deltas.copy()
    # Cumulative sum of deltas
    absolute[:, :2] = np.cumsum(deltas[:, :2], axis=0)
    # For chunk 1+, add the previous chunk's last position
    if start_pos is not None:
        absolute[:, :2] += start_pos
    return absolute


class DrawDataset(Dataset):
    """
    Dataset for trajectory prediction training.

    Each sample contains:
    - image: Canvas showing previously drawn points
    - instruction: Text prompt (e.g., "draw a circle")
    - target_chunk: (chunk_size, 3) normalized coordinates with (x, y, state)
    - mask: (chunk_size,) - 1=real point, 0=ignore in loss
    - is_last: Whether this is the final chunk (for stop signal)

    The dataset is loaded from a pre-generated directory with structure:
        data_dir/
            metadata.json      # List of all samples
            images/
                sample_0.png
                sample_1.png
                ...
            trajectories/
                sample_0.npy       # trajectory
                sample_0_mask.npy  # mask
                ...
    """

    def __init__(
        self,
        data_dir: str,
        processor=None,
        chunk_size: int = 16,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to pre-generated dataset
            processor: Qwen2.5-VL processor for image/text encoding
            chunk_size: Expected chunk size (for validation)
            max_samples: Limit number of samples (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.chunk_size = chunk_size

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {data_dir}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.samples = self.metadata["samples"]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]

        # Load image
        image_path = self.data_dir / "images" / sample_info["image"]
        image = Image.open(image_path).convert("RGB")

        # Load trajectory (chunk_size, 3) with (x, y, state)
        traj_path = self.data_dir / "trajectories" / sample_info["trajectory"]
        target_chunk = np.load(traj_path).astype(np.float32)

        # Load mask (chunk_size,) - 1=real, 0=ignore
        mask_path = self.data_dir / "trajectories" / sample_info["mask"]
        mask = np.load(mask_path).astype(np.float32)

        # Get instruction and metadata
        instruction = sample_info["instruction"]
        is_last = sample_info["is_last"]

        return {
            "image": image,
            "instruction": instruction,
            "target_chunk": torch.from_numpy(target_chunk),  # (chunk_size, 3)
            "mask": torch.from_numpy(mask),                   # (chunk_size,)
            "is_last": torch.tensor(is_last, dtype=torch.float32),
        }


class DrawDatasetHF(Dataset):
    """
    Dataset that loads from HuggingFace datasets format.

    Expects dataset with columns:
    - image: PIL Image
    - instruction: str
    - trajectory: List[List[float]] - (chunk_size, 3) with (x, y, state)
    - mask: List[float] - (chunk_size,) - 1=real, 0=ignore
    - is_last: bool

    Supports delta mode for relative movement prediction (GR00T N1.6 style).
    """

    def __init__(
        self,
        dataset,
        processor=None,
        chunk_size: int = 16,
        use_deltas: bool = False,
    ):
        """
        Args:
            dataset: HuggingFace Dataset object
            processor: Qwen2.5-VL processor
            chunk_size: Expected chunk size
            use_deltas: If True, convert (x,y) to (dx,dy) delta movements
        """
        self.dataset = dataset
        self.processor = processor
        self.chunk_size = chunk_size
        self.use_deltas = use_deltas
        mode = "delta" if use_deltas else "absolute"
        print(f"Loaded {len(dataset)} samples from HuggingFace dataset (mode: {mode})")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]

        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        instruction = sample["instruction"]
        trajectory = np.array(sample["trajectory"], dtype=np.float32)
        mask = np.array(sample["mask"], dtype=np.float32)
        is_last = sample["is_last"]

        # Convert to deltas if enabled
        if self.use_deltas:
            trajectory = absolute_to_deltas(trajectory)

        return {
            "image": image,
            "instruction": instruction,
            "target_chunk": torch.from_numpy(trajectory),  # (chunk_size, 3)
            "mask": torch.from_numpy(mask),                 # (chunk_size,)
            "is_last": torch.tensor(is_last, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict[str, Any]], processor) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Processes images and text through Qwen2.5-VL processor.

    Args:
        batch: List of samples from dataset
        processor: Qwen2.5-VL processor

    Returns:
        Batched tensors ready for model
    """
    from qwen_vl_utils import process_vision_info

    images = [sample["image"] for sample in batch]
    instructions = [sample["instruction"] for sample in batch]
    target_chunks = torch.stack([sample["target_chunk"] for sample in batch])
    masks = torch.stack([sample["mask"] for sample in batch])
    is_last = torch.stack([sample["is_last"] for sample in batch])

    # Prepare messages for Qwen2.5-VL
    messages_batch = []
    for image, instruction in zip(images, instructions):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 401408},
                {"type": "text", "text": instruction},
            ],
        }]
        messages_batch.append(messages)

    # Process through Qwen processor
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    # Process vision info
    all_image_inputs = []
    for msgs in messages_batch:
        image_inputs, _, _ = process_vision_info(msgs, return_video_kwargs=True)
        all_image_inputs.extend(image_inputs if image_inputs else [])

    # Tokenize and prepare inputs
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        return_tensors="pt",
        padding=True,
    )

    # Add trajectory targets and mask
    inputs["target_trajectory"] = target_chunks   # (B, chunk_size, 3)
    inputs["trajectory_mask"] = masks             # (B, chunk_size)
    inputs["is_last"] = is_last                   # (B,)

    return inputs


def create_dataloader(
    dataset: Dataset,
    processor,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """Create DataLoader with proper collate function."""
    from functools import partial

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, processor=processor),
        pin_memory=True,
    )
