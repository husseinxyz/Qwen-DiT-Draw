"""
Training script for Qwen-DiT-Draw trajectory prediction.

Trains only the DiT action head while keeping the VLM backbone frozen.
Uses flow matching loss for continuous trajectory generation.

Usage:
    python -m src.train.train_draw \
        --dataset_id TESS-Computer/quickdraw-circles \
        --output_dir outputs/dit_draw \
        --num_epochs 3
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoProcessor
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import Qwen2_5_VL_Draw, TrajectoryConfig, count_parameters
from src.dataset import DrawDatasetHF


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen-DiT-Draw")

    # Data
    parser.add_argument("--dataset_id", type=str, default="TESS-Computer/quickdraw-circles",
                        help="HuggingFace dataset ID")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (for debugging)")

    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base VLM model")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Points per chunk")
    parser.add_argument("--dit_hidden_size", type=int, default=512,
                        help="DiT hidden dimension")
    parser.add_argument("--dit_num_layers", type=int, default=6,
                        help="Number of DiT blocks")
    parser.add_argument("--use_deltas", action="store_true",
                        help="Use delta (relative) movements instead of absolute coordinates")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/dit_draw",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save every N steps")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use W&B logging")
    parser.add_argument("--wandb_project", type=str, default="qwen-dit-draw",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")

    return parser.parse_args()


def collate_fn_with_processor(batch, processor):
    """Collate batch and process through Qwen processor."""
    from qwen_vl_utils import process_vision_info

    images = [sample["image"] for sample in batch]
    instructions = [sample["instruction"] for sample in batch]
    target_chunks = torch.stack([sample["target_chunk"] for sample in batch])
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

    # Tokenize
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        return_tensors="pt",
        padding=True,
    )

    inputs["target_trajectory"] = target_chunks
    inputs["is_last"] = is_last

    return inputs


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B
    if args.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    # Load processor
    accelerator.print(f"Loading processor from {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Load dataset
    accelerator.print(f"Loading dataset from {args.dataset_id}...")
    dataset = load_dataset(args.dataset_id, split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    accelerator.print(f"Dataset size: {len(dataset)} samples")

    # Create PyTorch dataset
    train_dataset = DrawDatasetHF(
        dataset,
        processor,
        chunk_size=args.chunk_size,
        use_deltas=args.use_deltas,
    )

    # Create dataloader with collate function
    from functools import partial
    collate = partial(collate_fn_with_processor, processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    # Create model
    accelerator.print(f"Loading model {args.model_id}...")
    config = TrajectoryConfig(
        chunk_size=args.chunk_size,
        dit_hidden_size=args.dit_hidden_size,
        dit_num_layers=args.dit_num_layers,
    )

    model = Qwen2_5_VL_Draw(
        model_id=args.model_id,
        config=config,
        freeze_backbone=True,
        dtype=torch.bfloat16,
    )

    # Print parameter counts
    param_info = count_parameters(model)
    accelerator.print(f"Total params: {param_info['total']:,}")
    accelerator.print(f"Trainable params: {param_info['trainable']:,} ({param_info['trainable_percent']:.2f}%)")

    # Optimizer (only for trainable params)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.num_epochs // args.grad_accum_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)

    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    accelerator.print("Starting training...")
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                # Forward pass with trajectory mask for loss masking
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    target_trajectory=batch["target_trajectory"],
                    trajectory_mask=batch.get("trajectory_mask"),
                )

                loss = outputs["loss"]

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

            # Logging
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                if args.use_wandb:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                    }, step=global_step)

            # Save checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(checkpoint_dir)
                accelerator.print(f"Saved checkpoint to {checkpoint_dir}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        accelerator.print(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")

        if args.use_wandb and accelerator.is_main_process:
            accelerator.log({"train/epoch_loss": avg_epoch_loss}, step=global_step)

        # Save best model
        if avg_epoch_loss < best_loss and accelerator.is_main_process:
            best_loss = avg_epoch_loss
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)

            # Save model
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                unwrapped_model.trajectory_head.state_dict(),
                os.path.join(best_dir, "trajectory_head.pt"),
            )

            # Save config
            with open(os.path.join(best_dir, "config.json"), "w") as f:
                json.dump({
                    "model_id": args.model_id,
                    "chunk_size": args.chunk_size,
                    "dit_hidden_size": args.dit_hidden_size,
                    "dit_num_layers": args.dit_num_layers,
                    "use_deltas": args.use_deltas,
                    "best_loss": best_loss,
                }, f, indent=2)

            accelerator.print(f"Saved best model (loss: {best_loss:.4f})")

    # Final save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.trajectory_head.state_dict(),
            os.path.join(final_dir, "trajectory_head.pt"),
        )

        with open(os.path.join(final_dir, "config.json"), "w") as f:
            json.dump({
                "model_id": args.model_id,
                "chunk_size": args.chunk_size,
                "dit_hidden_size": args.dit_hidden_size,
                "dit_num_layers": args.dit_num_layers,
                "final_loss": avg_epoch_loss,
            }, f, indent=2)

        accelerator.print(f"Training complete! Model saved to {final_dir}")

    if args.use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
