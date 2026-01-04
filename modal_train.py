"""
Modal.com training script for Qwen-DiT-Draw.

Usage:
    modal run modal_train.py                      # Train
    modal run modal_train.py --action generate    # Generate dataset
    modal run modal_train.py --action all         # Generate + Train
"""

import modal

# Define the image with all dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("git")
    .pip_install("wheel", "setuptools", "pip", extra_options="--upgrade")
    .pip_install(
        "torch>=2.1.0",
        "torchvision",
        "transformers>=4.45.0",
        "accelerate>=0.26.0",
        "datasets",
        "pillow",
        "qwen-vl-utils",
        "peft",
        "bitsandbytes",
        "scipy",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "einops",
        "tqdm",
        "packaging",
        "wandb",
        "matplotlib",
        "huggingface_hub",
        "numpy",
    )
    .run_commands("mkdir -p /app")
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .add_local_dir("scripts", remote_path="/app/scripts", copy=True)
)

# Create Modal app
app = modal.App("qwen-dit-draw", image=image)

# Volume to persist model checkpoints and data
volume = modal.Volume.from_name("qwen-dit-draw-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    gpu="H100",
    timeout=3600 * 6,  # 6 hours max
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
)
def train(
    dataset_id: str = "TESS-Computer/quickdraw-circles",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    max_samples: int = None,
):
    """Run training on Modal GPU."""
    import os
    import subprocess
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")

    os.environ["PYTHONPATH"] = "/app:/app/src:" + os.environ.get("PYTHONPATH", "")
    os.environ["HF_HOME"] = f"{VOLUME_PATH}/hf_cache"

    # HF login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=False)

    # W&B setup
    os.environ["WANDB_PROJECT"] = "qwen-dit-draw"

    output_dir = f"{VOLUME_PATH}/outputs/dit_draw"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "-m", "src.train.train_draw",
        "--dataset_id", dataset_id,
        "--output_dir", output_dir,
        "--num_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--use_wandb",
        "--run_name", "qwen-dit-draw-h100",
    ]

    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    print("=" * 60)
    print("Starting training...")
    print(f"Dataset: {dataset_id}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd="/app")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    volume.commit()

    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

    return output_dir


@app.function(
    cpu=4,
    memory=16384,
    timeout=3600 * 2,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
)
def generate_dataset(
    num_circles: int = 10000,
    upload: bool = True,
    repo_id: str = "TESS-Computer/quickdraw-circles",
):
    """Generate Quick, Draw! circles dataset and upload to HuggingFace."""
    import os
    import subprocess
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")

    os.environ["PYTHONPATH"] = "/app:/app/src:" + os.environ.get("PYTHONPATH", "")
    os.environ["HF_HOME"] = f"{VOLUME_PATH}/hf_cache"

    # HF login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=False)

    data_dir = f"{VOLUME_PATH}/data/circles"

    cmd = [
        "python", "scripts/generate_dataset.py",
        "--num_circles", str(num_circles),
        "--output_dir", data_dir,
    ]

    if upload:
        cmd.extend(["--upload", "--repo_id", repo_id])

    print("=" * 60)
    print(f"Generating {num_circles} circles...")
    print(f"Output: {data_dir}")
    if upload:
        print(f"Will upload to: {repo_id}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd="/app")

    if result.returncode != 0:
        raise RuntimeError(f"Dataset generation failed with code {result.returncode}")

    volume.commit()

    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    return data_dir


@app.function(
    gpu="A10G",
    timeout=300,
    volumes={VOLUME_PATH: volume},
)
def inference(image_path: str, instruction: str = "draw a circle"):
    """Run inference with trained model."""
    import os
    import sys
    import torch
    import json
    from PIL import Image

    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")

    from src.model import Qwen2_5_VL_Draw, TrajectoryConfig
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Load model config
    model_dir = f"{VOLUME_PATH}/outputs/dit_draw/best"
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config_dict = json.load(f)

    config = TrajectoryConfig(
        chunk_size=config_dict["chunk_size"],
        dit_hidden_size=config_dict["dit_hidden_size"],
        dit_num_layers=config_dict["dit_num_layers"],
    )

    # Load model
    model = Qwen2_5_VL_Draw(
        model_id=config_dict["model_id"],
        config=config,
        freeze_backbone=True,
        dtype=torch.bfloat16,
    )

    # Load trained weights
    model.trajectory_head.load_state_dict(
        torch.load(os.path.join(model_dir, "trajectory_head.pt"))
    )
    model = model.to("cuda").eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(config_dict["model_id"])

    # Load image
    image = Image.open(image_path).convert("RGB")

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
    inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        chunk = model.predict_chunk(**inputs)

    chunk = chunk[0].cpu().numpy().tolist()

    return {
        "instruction": instruction,
        "chunk": chunk,
        "chunk_size": len(chunk),
    }


@app.function(
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def test_inference(instruction: str = "draw a circle", checkpoint: str = "checkpoint-3500"):
    """
    Test inference with a white canvas and visualize results.

    Args:
        instruction: What to tell the model (default: "draw a circle")
        checkpoint: Which checkpoint to use ("best", "final", or "checkpoint-XXXX")
    """
    import os
    import sys
    import torch
    import json
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    os.chdir("/app")
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")

    from src.model import Qwen2_5_VL_Draw, TrajectoryConfig
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Determine model directory
    base_dir = f"{VOLUME_PATH}/outputs/dit_draw"
    if checkpoint in ["best", "final"]:
        model_dir = os.path.join(base_dir, checkpoint)
    else:
        model_dir = os.path.join(base_dir, checkpoint)

    print(f"Loading model from {model_dir}...")

    # Check if this is an accelerator checkpoint or our custom format
    config_path = os.path.join(model_dir, "config.json")
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    trajectory_head_path = os.path.join(model_dir, "trajectory_head.pt")

    # Default config (used for accelerator checkpoints)
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    if os.path.exists(config_path):
        # Our custom format with config.json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        model_id = config_dict.get("model_id", model_id)
        config = TrajectoryConfig(
            chunk_size=config_dict.get("chunk_size", 16),
            dit_hidden_size=config_dict.get("dit_hidden_size", 512),
            dit_num_layers=config_dict.get("dit_num_layers", 6),
        )
    else:
        # Accelerator checkpoint - use defaults
        print("Using default config (accelerator checkpoint)")
        config = TrajectoryConfig()

    # Load model
    model = Qwen2_5_VL_Draw(
        model_id=model_id,
        config=config,
        freeze_backbone=True,
        dtype=torch.bfloat16,
    )

    # Load trained weights
    if os.path.exists(trajectory_head_path):
        # Our custom format
        model.trajectory_head.load_state_dict(torch.load(trajectory_head_path, weights_only=True))
        print("Loaded trajectory_head.pt")
    elif os.path.exists(safetensors_path):
        # Accelerator checkpoint - extract trajectory head weights from full model
        from safetensors.torch import load_file
        full_state = load_file(safetensors_path)

        # Filter for trajectory_head keys
        head_state = {}
        for k, v in full_state.items():
            if k.startswith("trajectory_head."):
                new_key = k.replace("trajectory_head.", "")
                head_state[new_key] = v

        if head_state:
            model.trajectory_head.load_state_dict(head_state)
            print(f"Loaded {len(head_state)} tensors from model.safetensors")
        else:
            raise ValueError("No trajectory_head weights found in safetensors!")
    else:
        raise FileNotFoundError(f"No weights found in {model_dir}. Available: {os.listdir(model_dir)}")

    model = model.to("cuda").eval()
    print("Model loaded!")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Create white canvas
    canvas_size = 512
    image = Image.new("RGB", (canvas_size, canvas_size), "white")

    # Multi-chunk inference with visual feedback loop
    print(f"Running inference: '{instruction}'...")
    all_points = []
    max_chunks = 10  # Safety limit
    chunk_count = 0

    while chunk_count < max_chunks:
        chunk_count += 1

        # Prepare inputs for current canvas state
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
        inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Predict chunk
        with torch.no_grad():
            chunk = model.predict_chunk(**inputs)

        chunk = chunk[0].float().cpu().numpy()  # bf16 -> float32 -> numpy
        print(f"Chunk {chunk_count}: {len(chunk)} points")

        # Process points and draw on canvas
        draw = ImageDraw.Draw(image)
        stop_detected = False

        for i, point in enumerate(chunk):
            x, y, state = point[0], point[1], point[2]
            px, py = int(x * canvas_size), int(y * canvas_size)

            all_points.append((x, y, state))

            # Draw point on canvas (visual feedback)
            if i > 0:
                prev_x, prev_y = chunk[i-1][0], chunk[i-1][1]
                prev_px, prev_py = int(prev_x * canvas_size), int(prev_y * canvas_size)
                draw.line([(prev_px, prev_py), (px, py)], fill='blue', width=3)

            # Check for stop signal
            if state > 0.5:
                print(f"  Stop signal detected at point {i+1}")
                stop_detected = True
                break

        if stop_detected:
            break

    print(f"Total: {len(all_points)} points across {chunk_count} chunks")

    # Convert to numpy for visualization
    chunk = np.array(all_points)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert to pixel coords
    x_coords = chunk[:, 0] * canvas_size
    y_coords = chunk[:, 1] * canvas_size

    # Plot 1: Trajectory overlay
    axes[0].imshow(image)
    axes[0].plot(x_coords, y_coords, 'b-', linewidth=2)
    axes[0].scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start')
    axes[0].scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='x', label='End')
    axes[0].set_title(f'Predicted Trajectory ({len(chunk)} pts, {chunk_count} chunks)\n"{instruction}"')
    axes[0].legend()
    axes[0].axis('off')

    # Plot 2: Draw the stroke
    canvas_with_stroke = image.copy()
    draw = ImageDraw.Draw(canvas_with_stroke)
    for i in range(len(x_coords) - 1):
        draw.line(
            [(x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1])],
            fill='blue', width=3
        )
    axes[1].imshow(canvas_with_stroke)
    axes[1].set_title('Drawn Result')
    axes[1].axis('off')

    # Plot 3: Normalized space
    axes[2].plot(chunk[:, 0], chunk[:, 1], 'b-', linewidth=2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(chunk)))
    axes[2].scatter(chunk[:, 0], chunk[:, 1], c=colors, s=30)
    axes[2].scatter(chunk[0, 0], chunk[0, 1], c='green', s=100, marker='o', zorder=5)
    axes[2].scatter(chunk[-1, 0], chunk[-1, 1], c='red', s=100, marker='x', zorder=5)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(1, 0)
    axes[2].set_aspect('equal')
    axes[2].set_title('Trajectory (normalized)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = f"{VOLUME_PATH}/outputs/dit_draw/test_result.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {viz_path}")

    # Commit to volume
    volume.commit()

    return {
        "instruction": instruction,
        "chunk": chunk.tolist(),
        "chunk_size": len(chunk),
        "num_chunks": chunk_count,
        "viz_path": viz_path,
        "checkpoint_used": checkpoint,
    }


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def list_checkpoints():
    """List files on volume."""
    import subprocess
    result = subprocess.run(
        ["find", "/vol", "-type", "f", "-name", "*.pt"],
        capture_output=True, text=True
    )
    print("Model files:")
    print(result.stdout or "None found")

    result2 = subprocess.run(
        ["ls", "-laR", "/vol/outputs/"],
        capture_output=True, text=True
    )
    print("\nDirectory listing:")
    print(result2.stdout[:3000] if result2.stdout else "Empty")


@app.local_entrypoint()
def main(
    action: str = "train",
    num_circles: int = 10000,
    epochs: int = 3,
    max_samples: int = None,
    repo_id: str = "TESS-Computer/quickdraw-circles",
):
    """
    Entry point for modal commands.

    Usage:
        modal run modal_train.py                              # Train
        modal run modal_train.py --action generate            # Generate dataset
        modal run modal_train.py --action all                 # Generate + Train
        modal run modal_train.py --action list                # List checkpoints
    """
    if action == "train":
        print("Starting training on Modal H100...")
        output_dir = train.remote(
            epochs=epochs,
            max_samples=max_samples,
        )
        print(f"Done! Model saved to: {output_dir}")

    elif action == "generate":
        print(f"Generating {num_circles} circles...")
        data_dir = generate_dataset.remote(
            num_circles=num_circles,
            upload=True,
            repo_id=repo_id,
        )
        print(f"Done! Data saved to: {data_dir}")

    elif action == "all":
        print("Running full pipeline: generate → train")
        print("\nStep 1: Generate dataset...")
        generate_dataset.remote(
            num_circles=num_circles,
            upload=True,
            repo_id=repo_id,
        )
        print("\nStep 2: Train model...")
        train.remote(
            dataset_id=repo_id,
            epochs=epochs,
            max_samples=max_samples,
        )
        print("\nAll done!")

    elif action == "list":
        print("Listing checkpoints on volume...")
        list_checkpoints.remote()

    elif action == "test":
        print("Running inference test with white canvas...")
        result = test_inference.remote()
        print("\n" + "="*60)
        print("INFERENCE RESULT:")
        print("="*60)
        print(f"Instruction: {result['instruction']}")
        print(f"Predicted {result['chunk_size']} points")
        print("\nTrajectory (x, y, state):")
        for i, point in enumerate(result['chunk']):
            print(f"  {i:2d}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
        print(f"\nVisualization saved to: {result.get('viz_path', 'N/A')}")
        print("="*60)

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: train, generate, all, list, test")
