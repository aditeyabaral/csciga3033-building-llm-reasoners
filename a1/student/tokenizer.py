import json
from pathlib import Path
import regex as re
from tqdm.auto import tqdm
from student.bpe import PAT_RE
from collections.abc import Iterable, Iterator


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Initialize a BPE tokenizer.

        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of BPE merges in order of creation
            special_tokens: Optional list of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Create inverse vocab for decoding (bytes mapped to token ID)
        self.inverse_vocab = {v: k for k, v in vocab.items()}

        # Create merge map for encoding (pair mapped to new token ID)
        self.merge_map = {pair: i for i, pair in enumerate(merges)}

        # Precompute special token byte representations
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Load a tokenizer from JSON files."""
        # Load vocabulary from JSON
        with open(vocab_filepath) as f:
            vocab_dict = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_dict.items()}

        # Load merges from JSON
        with open(merges_filepath) as f:
            merges_list = json.load(f)
        merges = [(bytes(m[0]), bytes(m[1])) for m in merges_list]

        # Load special tokens from config.json if not provided
        if special_tokens is None:
            # Infer config path from vocab path
            config_path = Path(vocab_filepath).parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    special_tokens = config.get("special_tokens", [])

        return cls(vocab, merges, special_tokens)

    def _apply_merge(self, token_tuple: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """
        Apply BPE merges to a tuple of tokens.

        Args:
            token_tuple: Tuple of tokens (each token is bytes)

        Returns:
            Tuple of tokens after applying merges
        """
        if len(token_tuple) < 2:
            return token_tuple  # No merges possible

        # Keep merging until no more merges can be applied
        while True:
            # Find pair with highest priority (lowest index in merges)
            min_rank = float("inf")
            min_pair_idx = -1
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                rank = self.merge_map.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    min_pair_idx = i

            # If no more merges can be applied, break
            if min_pair_idx == -1:
                break

            # Merge the best pair
            new_tokens = []
            i = 0
            while i < len(token_tuple):
                if i == min_pair_idx:
                    # Merge this pair
                    merged_token = token_tuple[i] + token_tuple[i + 1]
                    new_tokens.append(merged_token)
                    i += 2  # Skip the next token as it's merged
                else:
                    new_tokens.append(token_tuple[i])
                    i += 1

            token_tuple = tuple(new_tokens)

            if len(token_tuple) < 2:
                break  # No more merges possible

        return token_tuple

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        if not text:
            return []

        token_ids = []

        # Handle special tokens if defined
        if self.special_pattern:
            text_parts = re.split(f"({self.special_pattern})", text)
        else:
            text_parts = [text]

        for part in text_parts:
            if not part:
                continue

            # Check if part is a special token
            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                token_id = self.inverse_vocab.get(token_bytes)
                if token_id is not None:
                    token_ids.append(token_id)
                continue

            # Pre-tokenize part into byte-level tokens
            for match in PAT_RE.finditer(part):
                token = match.group()
                # Convert token to tuple of byte values
                token_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                # Apply BPE merges
                token_tuple = self._apply_merge(token_tuple)
                # Map to token IDs
                for token in token_tuple:
                    if token in self.inverse_vocab:
                        token_ids.append(self.inverse_vocab[token])
                    else:
                        for byte in token:
                            token_ids.append(byte)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding from an iterable of strings.

        Args:
            iterable: Iterable of strings (e.g., file handle)

        Returns:
            Iterator yielding token IDs
        """
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            # Find last occurence of special token to split
            last_idx = 0
            for i in range(len(buffer) - 1, -1, -1):
                if buffer[i] in (" ", "\n", "\t"):
                    last_idx = i + 1
                    break

            # If buffer is large enough, encode up to last_idx
            if len(buffer) > 1_000 and last_idx > 0:
                to_encode = buffer[:last_idx]
                buffer = buffer[last_idx:]

                # Encode and yield tokens
                for token_id in self.encode(to_encode):
                    yield token_id

        # Encode any remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Collect all token bytes
        token_bytes = []
        for token_id in ids:
            token = self.vocab.get(token_id)
            if token is not None:
                token_bytes.append(self.vocab[token_id])

        # Concatenate and decode
        byte_sequence = b"".join(token_bytes)
        # Decode to UTF-8 string
        text = byte_sequence.decode("utf-8", errors="replace")
        return text
