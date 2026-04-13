import regex as re
from collections import defaultdict
from tqdm.auto import tqdm
import multiprocessing as mp

from student.pretokenization_example import find_chunk_boundaries


# The pre-tokenization pattern
PAT_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Precompute single-byte objects (avoids millions of allocations)
BYTE_TABLE = {i: bytes([i]) for i in range(256)}


def get_token_pair_freqs(token_freqs: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Given a dictionary of token frequencies, compute the frequencies of adjacent token pairs.

    Args:
        token_freqs: Dict mapping token tuples to their frequencies
    Returns:
        pair_freqs: Dict mapping token pairs (token1, token2) to their frequencies
    """
    pair_freqs = defaultdict(int)
    for token_tuple, freq in token_freqs.items():
        if len(token_tuple) < 2:
            continue  # No pairs in single-byte tokens
        # Split token into individual bytes
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize a chunk of the input file and count pre-token frequencies.

    Args:
        args: Tuple containing (input_path, start, end, special_tokens)
            input_path: Path to input text file
            start: Start byte position of the chunk
            end: End byte position of the chunk
            special_tokens: List of special tokens to consider during pre-tokenization

    Returns:
        token_freqs: Dict mapping pre-token tuples to their frequencies in the chunk
    """
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    # Split on special tokens to avoid breaking them
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if special_pattern:
        text_parts = re.split(f"({special_pattern})", chunk_text)
        # Filter out special tokens and empty strings
        text_parts = [part for part in text_parts if part and part not in special_tokens]
    else:
        text_parts = [chunk_text]

    # Pre-tokenize each part
    token_freqs = defaultdict(int)
    for part in tqdm(text_parts, desc="Pre-tokenizing chunk", leave=False):
        for match in PAT_RE.finditer(part):
            token = match.group()
            token_bytes = token.encode("utf-8")
            token_tuple = tuple(BYTE_TABLE[b] for b in token_bytes)
            token_freqs[token_tuple] += 1

    return token_freqs


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str] = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to training text file
        vocab_size: Maximum vocabulary size (including base 256 + special tokens)
        special_tokens: List of special tokens to add to vocabulary

    Returns:
        vocab: Dict mapping token ID to bytes
        merges: List of merge operations (token1, token2)
    """

    # Initialize special tokens
    if special_tokens is None:
        special_tokens = []

    vocab = {i: BYTE_TABLE[i] for i in range(256)}
    next_token_id = 256

    # Add special tokens to vocabulary
    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    # Pre-tokenize corpus in parallel
    print("Pre-tokenizing corpus...")
    num_proc = mp.cpu_count()

    # Find chunk boundaries
    split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
    split_token_bytes = split_token.encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_proc, split_token_bytes)

    # Prepare args for parallel processing
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    token_freqs = defaultdict(int)

    if len(chunk_args) > 1:
        with mp.Pool(processes=num_proc) as pool:
            for chunk_freqs in pool.imap_unordered(pretokenize_chunk, chunk_args):
                for token_tuple, freq in chunk_freqs.items():
                    token_freqs[token_tuple] += freq
    else:
        result = pretokenize_chunk(chunk_args[0])
        token_freqs.update(result)

    print(f"Pre-tokenization complete. Found {len(token_freqs)} unique pre-tokens.")

    # Compute initial token pair frequencies
    pair_freqs = get_token_pair_freqs(token_freqs)

    # BPE Merging
    merges = []
    num_merges = vocab_size - len(vocab)
    print(f"Computing {num_merges} BPE merges...")

    for _ in tqdm(range(num_merges), desc="BPE merges"):
        if not pair_freqs:
            break  # No more pairs to merge

        # Find the most frequent pair
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        # Create new token by merging the best pair
        merged_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = merged_token
        next_token_id += 1

        # Track changes to pair frequencies
        pair_freq_changes = defaultdict(int)

        # Update token frequencies by rebuilding the dict
        new_token_freqs = defaultdict(int)
        for token_tuple, freq in token_freqs.items():
            if len(token_tuple) < 2:
                new_token_freqs[token_tuple] = freq
                continue

            # Apply merge to token bytes
            i = 0
            new_tokens = []
            while i < len(token_tuple):
                # Check if the best pair matches at position i
                if i < len(token_tuple) - 1 and token_tuple[i] == best_pair[0] and token_tuple[i + 1] == best_pair[1]:
                    # Track old pairs for frequency update
                    if i > 0:
                        old_pair = (new_tokens[-1], best_pair[0])
                        pair_freq_changes[old_pair] -= freq
                    if i + 2 < len(token_tuple):
                        old_pair = (best_pair[1], token_tuple[i + 2])
                        pair_freq_changes[old_pair] -= freq

                    # Merge the pair into a single token
                    new_tokens.append(merged_token)

                    # Track new pairs for frequency update
                    if len(new_tokens) > 1:
                        new_pair = (new_tokens[-2], new_tokens[-1])
                        pair_freq_changes[new_pair] += freq
                    if i + 2 < len(token_tuple):
                        new_pair = (merged_token, token_tuple[i + 2])
                        pair_freq_changes[new_pair] += freq

                    i += 2
                else:
                    # No match, keep original token
                    new_tokens.append(token_tuple[i])
                    i += 1

            new_token_freqs[tuple(new_tokens)] = freq

        token_freqs = new_token_freqs

        # Update pair frequencies based on changes
        del pair_freqs[best_pair]  # Remove merged pair
        for pair, change in pair_freq_changes.items():
            pair_freqs[pair] = pair_freqs.get(pair, 0) + change
            if pair_freqs[pair] <= 0:
                del pair_freqs[pair]

    print("BPE training complete.")
    return vocab, merges
