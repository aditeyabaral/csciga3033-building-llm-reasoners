"""
SwiGLU feed-forward network.
Combines SiLU activation with Gated Linear Unit (GLU).
"""

import torch
import torch.nn as nn
from .linear import Linear
from .silu import silu


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    FFN(x) = W2(SiLU(W1(x)) ⊙ W3(x))

    where ⊙ is element-wise multiplication to apply the "gating" mechanism.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Initialize SwiGLU feed-forward network.

        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (typically 8/3 * d_model, rounded to multiple of 64)
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear transformations
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # W1(x)
        w1_out = self.w1(x)

        # SiLU(W1(x))
        silu_out = silu(w1_out)

        # W3(x)
        w3_out = self.w3(x)

        # Element-wise multiplication (gating)
        gated = silu_out * w3_out

        # W2(gated)
        return self.w2(gated)
