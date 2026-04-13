"""
Scaled dot-product attention implementation.
"""

import torch
from .softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k)
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Optional boolean mask of shape (..., n_queries, n_keys)
              True means attend, False means mask out (set to -inf)

    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    # Get d_k for scaling
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K.T / sqrt(d_k)
    scores = Q @ K.transpose(-2, -1) / (d_k**0.5)

    # Apply mask if provided
    if mask is not None:
        # Where mask is False, set scores to -inf
        scores = scores.masked_fill(~mask, float("-inf"))

    # Apply softmax to get attention weights
    attn_weights = softmax(scores, dim=-1)

    # Handle case where entire row is masked (all -inf -> NaN after softmax)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Compute weighted sum of values
    output = attn_weights @ V
    return output
