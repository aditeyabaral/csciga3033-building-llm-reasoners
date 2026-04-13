"""
Softmax implementation with numerical stability.
"""

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to a tensor along a specified dimension.

    Uses the numerically stable version: softmax(x) = softmax(x - max(x))
    This prevents overflow when computing exp(x) for large values.

    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Args:
        x: Input tensor of any shape
        dim: Dimension along which to apply softmax

    Returns:
        Output tensor of same shape as input, with values in (0, 1)
        that sum to 1 along the specified dimension
    """
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    # Compute exp(x - max)
    exp_x = torch.exp(x_shifted)

    # Compute sum along the dimension
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    # Normalize
    return exp_x / sum_exp
