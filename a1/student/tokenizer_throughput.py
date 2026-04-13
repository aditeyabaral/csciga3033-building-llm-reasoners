from student.tokenizer import BPETokenizer
import time

# Load tokenizer
tokenizer = BPETokenizer.from_files(
    vocab_filepath="tokenizer/tokenizer_10k/vocab.json",
    merges_filepath="tokenizer/tokenizer_10k/merges.json",
    special_tokens=["<|endoftext|>"],
)

# Read first 100 documents for throughput test
docs = []
current_doc = []

with open("data/TinyStoriesV2-GPT4-train.txt", "r") as f:
    for line in f:
        if "<|endoftext|>" in line:
            parts = line.split("<|endoftext|>")
            current_doc.append(parts[0])
            doc_text = "".join(current_doc).strip()

            if doc_text:
                docs.append(doc_text)
                if len(docs) == 100:
                    break

            current_doc = [parts[-1]] if len(parts) > 1 and parts[-1] else []
        else:
            current_doc.append(line)

# Combine documents
sample_text = "<|endoftext|>".join(docs)
sample_bytes = len(sample_text.encode("utf-8"))

print(f"Sample size: {sample_bytes / 1e6:.2f} MB ({len(docs)} documents)")

# Benchmark encoding (3 runs)
times = []
for i in range(3):
    start = time.time()
    tokens = tokenizer.encode(sample_text)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {i + 1}: {elapsed:.3f}s")

# Calculate throughput
avg_time = sum(times) / len(times)
throughput = sample_bytes / avg_time  # bytes/second

print(f"\nThroughput: {throughput / 1e6:.2f} MB/s")

# Time for Pile (825 GB)
pile_bytes = 825e9
time_hours = pile_bytes / throughput / 3600
time_days = time_hours / 24

print(f"Time to tokenize Pile (825 GB): {time_hours:.1f} hours ({time_days:.1f} days)")
