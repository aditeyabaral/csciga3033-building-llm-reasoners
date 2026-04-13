import argparse
import json
import torch
from pathlib import Path
from tqdm.auto import tqdm

from student.tokenizer import BPETokenizer
from student.transformer_lm import TransformerLM
from generation import generate

# Defined model architecture
MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000.0,
    "use_rope": True,
    "eps": 1e-5,
    "use_norm": True,
    "norm_position": "pre",
    "ffn_type": "swiglu",
    "tied_weights": True,
}

# Defined prompts to generate from
PROMPTS = [
    "Once upon a time",
    "In a faraway land",
    "There was a little",
    "One day, a brave",
    "The magical forest",
    "A long time ago",
    "It was a sunny day when",
    "The little girl loved to",
]


def load_model(checkpoint_path: str, tokenizer_path: str, device: str):
    """Load model and tokenizer."""
    # Load tokenizer
    tokenizer_path = Path(tokenizer_path)
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=str(tokenizer_path / "vocab.json"),
        merges_filepath=str(tokenizer_path / "merges.json"),
    )

    # Initialize model
    model = TransformerLM(**MODEL_CONFIG)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Training iteration: {checkpoint.get('iteration', 'unknown')}")
    return model, tokenizer


def generate_samples(model, tokenizer, prompts, max_tokens, temperature, top_p, eos_token_id, device):
    """Generate text for all prompts."""
    results = []
    total_prompts = len(prompts) * 2  # Each prompt generates 2 outputs (greedy + nucleus)
    progress_bar = tqdm(total=total_prompts, desc="Generating samples")
    for i, prompt in enumerate(prompts):
        # Greedy decoding
        greedy = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            device=device,
            eos_token_id=eos_token_id,
        )
        progress_bar.update(1)

        # Nucleus sampling
        nucleus = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            eos_token_id=eos_token_id,
        )
        progress_bar.update(1)

        results.append(
            {
                "prompt": prompt,
                "greedy_output": greedy,
                "nucleus_output": nucleus,
            }
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained model")

    # Required model and tokenizer paths
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer directory")

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")

    # Output
    parser.add_argument("--output", default="generated_samples.json", help="Output JSON file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(args)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.tokenizer_path, args.device)
    eos_token_id = tokenizer.inverse_vocab.get("<|endoftext|>".encode("utf-8"))

    # Generate
    results = generate_samples(
        model, tokenizer, PROMPTS, args.max_tokens, args.temperature, args.top_p, eos_token_id, args.device
    )

    # Save to JSON
    output = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "model_config": MODEL_CONFIG,
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        },
        "samples": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved {len(results)} samples to {args.output}")
