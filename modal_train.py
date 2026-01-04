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

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: train, generate, all, list")
