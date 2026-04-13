from student.tokenizer import BPETokenizer

# Load tokenizer
tokenizer = BPETokenizer.from_files(
    vocab_filepath="tokenizer/tokenizer_10k/vocab.json",
    merges_filepath="tokenizer/tokenizer_10k/merges.json",
    special_tokens=["<|endoftext|>"],
)

# Sample 10 documents efficiently without loading entire file
sampled_docs = []
current_doc = []

with open("data/TinyStoriesV2-GPT4-train.txt") as f:
    for line in f:
        if "<|endoftext|>" in line:
            # Split on special token
            parts = line.split("<|endoftext|>")

            # Add first part to current document
            current_doc.append(parts[0])
            doc_text = "".join(current_doc).strip()

            if doc_text:  # Only add non-empty documents
                sampled_docs.append(doc_text)
                if len(sampled_docs) == 10:
                    break

            # Start new document with remaining parts
            current_doc = [parts[-1]] if len(parts) > 1 and parts[-1] else []
        else:
            current_doc.append(line)

print(f"Collected {len(sampled_docs)} documents")
print(sampled_docs[0])

# Calculate compression ratio
total_bytes = sum(len(doc.encode("utf-8")) for doc in sampled_docs)
total_tokens = sum(len(tokenizer.encode(doc)) for doc in sampled_docs)

compression_ratio = total_bytes / total_tokens
print(f"Compression ratio: {compression_ratio:.2f} bytes/token")
print(f"Total bytes: {total_bytes:,}")
print(f"Total tokens: {total_tokens:,}")
