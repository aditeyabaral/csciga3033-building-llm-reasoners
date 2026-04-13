"""Train BPE tokenizer and optionally tokenize datasets."""

import argparse
import json
from pathlib import Path

from student.bpe import train_bpe
from student.tokenizer import BPETokenizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")

    # Required arguments
    parser.add_argument("--input", type=str, required=True, help="Path to training text file")
    parser.add_argument("--vocab-size", type=int, required=True, help="Target vocabulary size")

    # Output
    parser.add_argument("--output-dir", type=str, default="./tokenizer", help="Directory to save tokenizer files")

    # Tokenizer configuration
    parser.add_argument(
        "--special-tokens", nargs="*", default=["<|endoftext|>"], help="Special tokens to add to vocabulary"
    )
    return parser.parse_args()


def save_tokenizer(tokenizer, output_dir):
    """Save tokenizer vocabulary and merges to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving tokenizer to {output_dir}...")

    # Save vocabulary as JSON
    vocab_dict = {str(k): list(v) for k, v in tokenizer.vocab.items()}
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"  ✓ Saved vocabulary: {vocab_path}")

    # Save merges as JSON
    merges_list = [[list(m[0]), list(m[1])] for m in tokenizer.merges]
    merges_path = output_dir / "merges.json"
    with open(merges_path, "w") as f:
        json.dump(merges_list, f, indent=2)
    print(f"  ✓ Saved merges: {merges_path}")

    # Save configuration
    config = {
        "vocab_size": len(tokenizer.vocab),
        "num_merges": len(tokenizer.merges),
        "special_tokens": tokenizer.special_tokens,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config: {config_path}")


if __name__ == "__main__":
    """Main tokenizer training pipeline."""
    args = parse_args()
    print(args)

    # Train BPE tokenizer
    print(f"\nTraining BPE tokenizer on: {args.input}")
    print(f"Target vocabulary size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")

    vocab, merges = train_bpe(input_path=args.input, vocab_size=args.vocab_size, special_tokens=args.special_tokens)

    print("\n✓ Training complete!")
    print(f"  Final vocabulary size: {len(vocab)}")
    print(f"  Number of merges: {len(merges)}")

    # Create tokenizer instance
    tokenizer = BPETokenizer(vocab, merges, special_tokens=args.special_tokens)

    # Save tokenizer
    save_tokenizer(tokenizer, args.output_dir)

    # Verify if tokenizer can be loaded correctly
    print("\nVerifying tokenizer loading...")
    loaded_tokenizer = BPETokenizer.from_files(
        vocab_filepath=Path(args.output_dir) / "vocab.json",
        merges_filepath=Path(args.output_dir) / "merges.json",
        special_tokens=args.special_tokens,
    )
    assert loaded_tokenizer.vocab == tokenizer.vocab, "Loaded vocab does not match original vocab!"
    assert loaded_tokenizer.merges == tokenizer.merges, "Loaded merges do not match original merges!"
    assert loaded_tokenizer.special_tokens == tokenizer.special_tokens, (
        "Loaded special tokens do not match original special tokens!"
    )
    print("✓ Tokenizer loaded successfully and matches original tokenizer.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTokenizer saved to: {args.output_dir}/")
    print(f"\t- vocab.json ({len(vocab)} tokens)")
    print(f"\t- merges.json ({len(merges)} merges)")
    print("\t- config.json")
    print("\n✓ Done!")
    print("=" * 80)
