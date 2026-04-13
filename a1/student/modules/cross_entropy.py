"""
Cross-entropy loss implementation with numerical stability.
"""

import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with numerical stability.

    Given logits and targets, computes:
    ℓ = -log(softmax(inputs)[targets])

    Uses the log-sum-exp trick for numerical stability.

    Args:
        inputs: Logits of shape (batch_size, vocab_size)
        targets: Target class indices of shape (batch_size,)

    Returns:
        Scalar tensor with the average cross-entropy loss across all examples
    """
    # Subtract max for numerical stability (log-sum-exp trick)
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values  # (..., 1)
    shifted_logits = inputs - max_logits  # (..., vocab_size)

    # Compute log(sum(exp(shifted_logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # Get the logit corresponding to the target class
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Compute cross-entropy for each example
    max_logits_squeezed = max_logits.squeeze(-1)
    loss_per_example = -target_logits + max_logits_squeezed + log_sum_exp

    # Return the average loss across all batch dimensions
    return loss_per_example.mean()
