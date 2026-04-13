"""Text generation with temperature and nucleus sampling."""

import torch
from student.modules.softmax import softmax


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device | str = "cuda",
    eos_token_id: int | None = None,
) -> str:
    """
    Generate text autoregressively from a prompt.

    Args:
        model: TransformerLM model
        tokenizer: BPETokenizer for encoding/decoding
        prompt: Initial text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Softmax temperature for controlling randomness
        top_p: Nucleus sampling threshold (0.0 to 1.0)
        device: Device to run generation on
        eos_token_id: Optional end-of-sequence token ID to stop generation early

    Returns:
        Generated text as a string (prompt + generated tokens)
    """
    was_training = model.training
    model.eval()

    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]

    # Get model's context length
    context_length = model.context_length

    # Generate tokens autoregressively
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context window if necessary
            # Keep the most recent tokens that fit in context
            if input_ids.size(1) > context_length:
                input_ids = input_ids[:, -context_length:]

            # Forward pass - get logits for next token
            logits = model(input_ids)  # [1, seq_len, vocab_size]

            # Get logits for the last position (next token prediction)
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Convert to probabilities using our custom softmax
            probs = softmax(next_token_logits, dim=-1)  # [1, vocab_size]

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                probs = top_p_filtering(probs, top_p)

            # Sample next token
            if temperature == 0:
                # Greedy decoding
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [1, 1]
            else:
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    # Decode the generated sequence
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    # Restore model's training state
    if was_training:
        model.train()

    return generated_text


def top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to a probability distribution.

    Keeps the smallest set of tokens whose cumulative probability >= top_p,
    and sets all other probabilities to 0.

    Args:
        probs: Probability distribution [batch_size, vocab_size]
        top_p: Cumulative probability threshold (0.0 to 1.0)

    Returns:
        Filtered and renormalized probability distribution
    """
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff: tokens to remove (those with cumulative prob > top_p)
    # We keep tokens where cumulative_probs <= top_p
    # Shift by 1 to keep at least the first token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask in original order
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    # indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
    # indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    # Set removed indices to 0
    probs = probs.clone()
    probs[indices_to_remove] = 0.0

    # Renormalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs
