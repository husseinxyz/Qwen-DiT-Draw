"""
Inference utilities for trajectory prediction.

Usage:
    from src.inference import load_model, predict_chunk

    model, processor = load_model("outputs/dit_draw/best")
    chunk = predict_chunk(model, processor, "canvas.png", "draw a circle")
"""

import torch
import json
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Optional

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from src.model import Qwen2_5_VL_Draw, TrajectoryConfig
from src.dataset import render_partial_trajectory, create_blank_canvas


def load_model(
    model_path: str,
    device: str = "cuda",
) -> Tuple:
    """
    Load trained model and processor.

    Args:
        model_path: Path to model directory (with config.json and trajectory_head.pt)
        device: Device to load model on

    Returns:
        (model, processor) tuple
    """
    model_path = Path(model_path)

    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Create config
    config = TrajectoryConfig(
        chunk_size=config_dict.get("chunk_size", 16),
        dit_hidden_size=config_dict.get("dit_hidden_size", 512),
        dit_num_layers=config_dict.get("dit_num_layers", 6),
    )

    # Load model
    model = Qwen2_5_VL_Draw(
        model_id=config_dict["model_id"],
        config=config,
        freeze_backbone=True,
        dtype=torch.bfloat16,
    )

    # Load trained weights
    weights_path = model_path / "trajectory_head.pt"
    if weights_path.exists():
        model.trajectory_head.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: No trained weights found at {weights_path}")

    model = model.to(device).eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(config_dict["model_id"])

    return model, processor


def predict_chunk(
    model,
    processor,
    image: str | Image.Image,
    instruction: str = "draw a circle",
    device: str = "cuda",
    num_steps: int = None,
) -> torch.Tensor:
    """
    Predict next trajectory chunk.

    Args:
        model: Trained Qwen2_5_VL_Draw model
        processor: Qwen processor
        image: Path to image or PIL Image
        instruction: Text instruction
        device: Device
        num_steps: Inference steps (default from config)

    Returns:
        chunk: (chunk_size, 3) tensor of (x, y, state) where state=1 means done
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Prepare input
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 401408},
            {"type": "text", "text": instruction},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _, _ = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        chunk = model.predict_chunk(**inputs, num_steps=num_steps)

    return chunk[0].cpu()


def predict_full_trajectory(
    model,
    processor,
    instruction: str = "draw a circle",
    canvas_size: int = 512,
    max_chunks: int = 10,
    device: str = "cuda",
    stop_threshold: float = 0.5,
) -> Tuple[List[Tuple[float, float]], Image.Image]:
    """
    Predict full trajectory with visual feedback loop.

    Iteratively predicts chunks and updates canvas until model signals done.

    GR00T-style inference:
    1. Predict chunk of 16 points with (x, y, state)
    2. Execute points until state > threshold (STOP signal)
    3. If no stop signal, update canvas and predict next chunk
    4. Repeat until done or max_chunks reached

    Args:
        model: Trained model
        processor: Processor
        instruction: Drawing instruction
        canvas_size: Canvas size in pixels
        max_chunks: Maximum number of chunks to predict
        device: Device
        stop_threshold: State value above which to stop (default 0.5)

    Returns:
        (trajectory, final_canvas): List of (x, y) points and final canvas image
    """
    import numpy as np

    # Start with blank canvas
    canvas = create_blank_canvas(canvas_size)
    full_trajectory = []
    done = False

    for chunk_idx in range(max_chunks):
        # Predict next chunk: (chunk_size, 3) with (x, y, state)
        chunk = predict_chunk(model, processor, canvas, instruction, device)
        chunk_np = chunk.numpy()

        # Process points, checking for stop signal
        points_to_draw = []
        for point in chunk_np:
            x, y, state = point[0], point[1], point[2]
            full_trajectory.append((float(x), float(y)))
            points_to_draw.append([x, y])

            # Check stop signal
            if state > stop_threshold:
                done = True
                break

        # Update canvas with executed points
        if len(points_to_draw) > 0:
            canvas = render_partial_trajectory(
                np.array(points_to_draw),
                canvas_size=canvas_size,
                existing_canvas=canvas
            )

        if done:
            break

    return full_trajectory, canvas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run trajectory inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--instruction", type=str, default="draw a circle", help="Instruction")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output image")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path)

    print(f"Predicting chunk for: {args.instruction}")
    chunk = predict_chunk(model, processor, args.image, args.instruction)

    print(f"Predicted {len(chunk)} points:")
    for i, (x, y) in enumerate(chunk.numpy()):
        print(f"  Point {i+1}: ({x:.4f}, {y:.4f})")

    # Visualize
    from PIL import Image
    image = Image.open(args.image).convert("RGB")
    canvas = render_partial_trajectory(
        chunk.numpy(),
        canvas_size=image.width,
        line_color=(255, 0, 0),
        existing_canvas=image,
    )
    canvas.save(args.output)
    print(f"Saved visualization to {args.output}")
