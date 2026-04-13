"""
Gradient utilities for training.
"""

import torch
from collections.abc import Iterable


def clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clip gradients by global L2 norm.

    If the global L2 norm of gradients exceeds max_norm, scale all gradients
    down proportionally so the total norm equals max_norm.

    Args:
        parameters: Iterable of parameters whose gradients to clip
        max_norm: Maximum L2 norm for gradients
        eps: Small constant for numerical stability (default: 1e-6)
    """
    # Collect all parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]

    if len(params_with_grad) == 0:
        return

    # Compute the global L2 norm of all gradients
    total_norm = torch.sqrt(sum(torch.sum(p.grad.data**2) for p in params_with_grad))

    # Compute the clipping coefficient
    clip_coef = max_norm / (total_norm + eps)

    # Only clip if the norm exceeds max_norm
    if clip_coef < 1.0:
        # Scale all gradients by clip_coef
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)
