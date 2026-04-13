"""Training script for Transformer Language Model."""

import argparse
import json
import time
import wandb
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from adamw import AdamW
from lr_schedule import get_lr_cosine_scheduler
from gradient import clip_grad_norm
from dataloader import get_batch
from checkpoint import save_checkpoint, load_checkpoint
from generation import generate

from student.tokenizer import BPETokenizer
from student.transformer_lm import TransformerLM
from student.modules.cross_entropy import cross_entropy


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # Data arguments
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer directory")

    # Model architecture
    parser.add_argument("--context-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1344, help="Feed-forward hidden dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta parameter")
    parser.add_argument("--use-rope", action="store_true", help="Use rotary position embeddings")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon for RMSNorm")
    parser.add_argument("--use-norm", action="store_true", help="Use RMSNorm layers")
    parser.add_argument(
        "--norm-position",
        type=str,
        default="pre",
        choices=["pre", "post"],
        help="Normalization position: pre-norm or post-norm",
    )
    parser.add_argument(
        "--ffn-type", type=str, default="swiglu", choices=["swiglu", "silu"], help="Feed-forward type: swiglu or silu"
    )
    parser.add_argument("--tied-weights", action="store_true", help="Tie LM head weights with token embeddings")

    # Training hyperparameters
    parser.add_argument("--total-steps", type=int, default=10_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--eval-batches", type=int, default=100, help="Number of batches to evaluate on")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Maximum learning rate")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Ratio of warmup steps to total steps")

    # Optimizer
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay coefficient")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--optimizer-eps", type=float, default=1e-8, help="Adam epsilon")

    # Gradient clipping
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every N steps")

    # Checkpointing
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--save-best", action="store_true", help="Save best model based on validation loss")
    parser.add_argument("--save-latest", action="store_true", help="Overwrite latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="transformer-lm-training", help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default="csciga3033-llm-reasoners", help="Wandb entity")

    # Text generation
    parser.add_argument("--generate-samples", action="store_true", help="Generate samples during training")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Generation prompt")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    # System
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validation
    assert args.d_model % args.num_heads == 0, "d_model must be divisible by num_heads"

    return args


def load_data(path, dtype="uint16"):
    """Load memory-mapped numpy array."""
    return np.load(path, mmap_mode="r")


def load_tokenizer(tokenizer_path):
    """Load BPE tokenizer from directory."""
    tokenizer_path = Path(tokenizer_path)
    return BPETokenizer.from_files(
        vocab_filepath=str(tokenizer_path / "vocab.json"), merges_filepath=str(tokenizer_path / "merges.json")
    )


def evaluate(model, val_data, args, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(args.eval_batches):
            inputs, targets = get_batch(val_data, args.batch_size, args.context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / args.eval_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()
    return {"loss": avg_loss, "perplexity": perplexity}


def train(args):
    """Main training function."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    print(f"Loaded train data: {len(train_data):,} tokens")
    print(f"Loaded val data: {len(val_data):,} tokens")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = len(tokenizer.vocab)
    eos_token_id = tokenizer.inverse_vocab.get("<|endoftext|>".encode("utf-8"))
    print(f"Vocab size: {vocab_size:,}")

    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rope=args.use_rope,
        eps=args.eps,
        use_norm=args.use_norm,
        norm_position=args.norm_position,
        ffn_type=args.ffn_type,
        tied_weights=args.tied_weights,
        device=device,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params // 10**6}M")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
    )

    # Setup checkpointing
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        start_step = load_checkpoint(args.checkpoint, model, optimizer)
        print(f"Loaded checkpoint from step {start_step}")
    else:
        start_step = 0

    # Initialize wandb
    wandb_writer = None
    if args.wandb:
        wandb_writer = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=dict(vars(args)),
            allow_val_change=True,
        )
        wandb_table = wandb.Table(
            columns=["step", "top_p", "temperature", "prompt", "completion"],
            log_mode="INCREMENTAL",
            optional=True,
            allow_mixed_types=True,
        )
        # wandb.watch(model, log="gradients", log_freq=args.log_interval)

    # Training loop
    model.train()
    print(f"\nStarting training from step {start_step} to {args.total_steps}")

    # Running averages for logging
    running_loss = 0.0
    running_step_count = 0
    running_tokens_seen = 0
    global_tokens_seen = 0

    # Start times for measuring tokens/sec
    global_start_time = time.time()
    running_start_time = global_start_time

    # Initialize validation tracking
    best_val_loss = float("inf")

    warmup_steps = int(args.total_steps * args.warmup_ratio)
    progress_bar = tqdm(total=args.total_steps - start_step, initial=0)
    for step in range(start_step, args.total_steps):
        # Zero gradients
        model.train()
        optimizer.zero_grad()

        # Get learning rate for this step
        lr = get_lr_cosine_scheduler(
            step,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=args.total_steps,
        )

        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Create logging dict
        log_dict = {
            "step": step + 1,
            "lr": lr,
        }

        # Sample batch
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            clip_grad_norm(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        # Update metrics
        log_dict["train_loss"] = loss.item()
        log_dict["train_perplexity"] = torch.exp(loss).item()
        running_loss += loss.item()
        running_step_count += 1
        num_tokens_in_step = args.batch_size * args.context_length
        global_tokens_seen += num_tokens_in_step
        running_tokens_seen += num_tokens_in_step

        # Logging
        if (step + 1) % args.log_interval == 0:
            # Compute averages
            avg_loss = running_loss / running_step_count
            avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
            running_elapsed = time.time() - running_start_time
            global_elapsed = time.time() - global_start_time
            running_tokens_per_sec = running_tokens_seen / running_elapsed if running_elapsed > 0 else 0
            global_tokens_per_sec = global_tokens_seen / global_elapsed if global_elapsed > 0 else 0

            # Update log dict with averages and speed
            log_dict["train_loss"] = avg_loss
            log_dict["train_perplexity"] = avg_perplexity
            log_dict["train_running_tokens_seen"] = running_tokens_seen
            log_dict["train_global_tokens_seen"] = global_tokens_seen
            log_dict["train_running_tokens_per_sec"] = running_tokens_per_sec
            log_dict["train_global_tokens_per_sec"] = global_tokens_per_sec

            # Print log dict
            log_str = json.dumps(
                {
                    "step": step + 1,
                    "lr": lr,
                    "train_loss": round(avg_loss, 4),
                    "train_perplexity": round(avg_perplexity, 4),
                    "running_tokens_seen": running_tokens_seen,
                    "global_tokens_seen": global_tokens_seen,
                    "running_tokens_per_sec": round(running_tokens_per_sec, 3),
                    "global_tokens_per_sec": round(global_tokens_per_sec, 3),
                }
            )
            progress_bar.write(log_str)

            # Wandb logging
            if wandb_writer is not None:
                wandb_writer.log(log_dict, step=step + 1)

            # Reset running averages
            running_loss = 0.0
            running_step_count = 0
            running_tokens_seen = 0
            running_start_time = time.time()

        # Validation
        if (step + 1) % args.eval_interval == 0:
            val_metrics = evaluate(model, val_data, args, device)
            current_val_loss = val_metrics["loss"]
            current_val_ppl = val_metrics["perplexity"]

            # Print log dict
            log_str = json.dumps(
                {
                    "step": step + 1,
                    "val_loss": round(current_val_loss, 4),
                    "val_perplexity": round(current_val_ppl, 4),
                }
            )
            progress_bar.write(log_str)

            # Wandb logging
            if wandb_writer is not None:
                wandb_writer.log(
                    {
                        "val_loss": current_val_loss,
                        "val_perplexity": current_val_ppl,
                    },
                    step=step + 1,
                )

            # Track the best model
            if args.save_best and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_path = output_dir / "best_model.pt"
                save_checkpoint(model, optimizer, step + 1, best_path)
                progress_bar.write(f"New best model saved (loss: {best_val_loss:.4f})")

                if wandb_writer is not None:
                    wandb_writer.summary["best_val_loss"] = best_val_loss

        # Text generation
        if args.generate_samples and (step + 1) % args.eval_interval == 0:
            # Use a specific prompt or the default from args
            with torch.no_grad():
                greedy_sample = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    device=device,
                    eos_token_id=eos_token_id,
                )
                nucleus_sample = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    eos_token_id=eos_token_id,
                )

                # Wandb logging for samples
                if wandb_writer is not None:
                    wandb_table.add_data(step + 1, 1.0, 0.0, args.prompt, greedy_sample)
                    wandb_table.add_data(step + 1, args.top_p, args.temperature, args.prompt, nucleus_sample)
                    wandb_writer.log({"generated_samples": wandb_table}, step=step + 1)

                progress_bar.write(f"\n[Step {step + 1}] Greedy: {greedy_sample}")
                progress_bar.write(f"[Step {step + 1}] Nucleus: {nucleus_sample}\n")

        # Checkpointing
        if (step + 1) % args.save_interval == 0:
            # Set checkpoint path based on whether we're saving the latest or step-specific checkpoint
            if args.save_latest:
                checkpoint_path = output_dir / "latest.pt"
            else:
                checkpoint_path = output_dir / f"step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            progress_bar.write(f"Checkpoint saved: {checkpoint_path}")

        # Update progress bar
        progress_bar.set_postfix(
            {
                "step": step + 1,
                "lr": f"{lr:.3e}",
                "loss": f"{log_dict['train_loss']:.4f}",
            }
        )
        progress_bar.update(1)

    # Save final model
    final_path = output_dir / "final_model.pt"
    save_checkpoint(model, optimizer, args.total_steps, final_path)
    print(f"\nSaved final model: {final_path}")

    # # Final evaluation
    # val_metrics = evaluate(model, val_data, args, device)
    # print(f"\nFinal validation: loss={val_metrics['loss']:.4f}, perplexity={val_metrics['perplexity']:.2f}")

    if wandb_writer is not None:
        wandb_writer.finish()

    progress_bar.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
