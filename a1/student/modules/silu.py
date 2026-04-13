"""
SiLU (Sigmoid Linear Unit) or Swish activation function.
"""

import torch
import torch.nn as nn

from .linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply SiLU activation function.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor of same shape as input
    """
    return x * torch.sigmoid(x)


class SiLUFeedForward(nn.Module):
    """
    SiLU-only feed-forward network without gating mechanism.

    Architecture:
        FFN(x) = W2(SiLU(W1(x)))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize SiLU-only FFN.

        Args:
            d_model: Model dimension (input/output size)
            d_ff: Hidden dimension (typically 4 * d_model)
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Up-projection: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # Down-projection: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SiLU-only feed-forward.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # W1(x)
        hidden = self.w1(x)

        # SiLU(W1(x))
        activated = silu(hidden)

        # W2(SiLU(W1(x)))
        output = self.w2(activated)

        return output
