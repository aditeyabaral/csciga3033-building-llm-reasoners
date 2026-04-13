import json

with open("tokenizer/tokenizer_10k/vocab.json") as f:
    vocab = json.load(f)

longest_token = None
longest_length = 0
longest_id = None

for token_id, token_bytes in vocab.items():
    if int(token_id) < 256:  # Skip base bytes
        continue
    token_len = len(token_bytes)
    if token_len > longest_length:
        longest_length = token_len
        longest_token = bytes(token_bytes)
        longest_id = token_id

decoded = longest_token.decode("utf-8", errors="replace")
print(f"Longest token ID: {longest_id}")
print(f"Length: {longest_length} bytes")
print(f"Bytes: {longest_token}")
print(f"Decoded: '{decoded}'")
