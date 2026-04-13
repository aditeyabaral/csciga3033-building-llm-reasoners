"""Encode text data to tokenized .npy files."""

import argparse
from pathlib import Path
import numpy as np
from student.tokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Encode text data with BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer directory")
    parser.add_argument(
        "--dtype", type=str, default="uint16", choices=["uint16", "uint32", "int32", "int64"], help="Output data type"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=f"{args.tokenizer_path}/vocab.json", merges_filepath=f"{args.tokenizer_path}/merges.json"
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Read text
    print(f"Reading {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    # Encode
    print("Encoding...")
    tokens = tokenizer.encode(text)
    print(f"Encoded to {len(tokens):,} tokens")

    # Convert to numpy
    tokens_array = np.array(tokens, dtype=getattr(np, args.dtype))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tokens_array)
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"✓ Saved {len(tokens_array):,} tokens to {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Data type: {tokens_array.dtype}")
