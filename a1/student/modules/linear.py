"""
Linear transformation module without bias.
"""

import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear transformation: y = Wx without bias term.
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Initialize the linear layer.

        Args:
            in_features: Size of input features (d_in)
            out_features: Size of output features (d_out)
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Create weight parameter: shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with truncated normal distribution.

        Using: N(μ=0, σ²=2/(d_in + d_out)) truncated at [-3σ, 3σ]
        """
        # Compute standard deviation
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute y = x @ W.T

        Args:
            x: Input tensor of shape (..., in_features)
               where ... represents any number of batch dimensions

        Returns:
            Output tensor of shape (..., out_features)
        """
        return x @ self.weight.T
