"""
Root Mean Square Layer Normalization (RMSNorm).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes activations using RMS.
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Initialize RMSNorm layer.

        Args:
            d_model: Hidden dimension of the model
            eps: Small constant for numerical stability
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter (one per dimension) initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., d_model)
               where ... represents any number of batch dimensions

        Returns:
            Normalized tensor of same shape as input
        """
        # Save original dtype to restore later
        in_dtype = x.dtype

        # Upcast to float32 for numerical stability
        x = x.to(torch.float32)

        # Compute RMS: sqrt(mean(x^2) + eps)
        # Shape: (..., d_model) -> (..., 1) after mean
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # Normalize: x / RMS
        # Shape: (..., d_model) / (..., 1) -> (..., d_model)
        x_normalized = x / rms

        # Apply learnable gain parameter
        # Shape: (..., d_model) * (d_model,) -> (..., d_model)
        output = x_normalized * self.weight

        # Convert back to original dtype
        return output.to(in_dtype)
