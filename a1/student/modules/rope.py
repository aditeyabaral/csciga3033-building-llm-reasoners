"""
Rotary Position Embeddings (RoPE).
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Applies rotation to query and key vectors based on their position in the sequence.
    This allows the model to incorporate relative position information.
    """

    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: torch.device | None = None):
        """
        Initialize RoPE module.

        Args:
            d_k: Dimension of query/key vectors
            theta: Base for computing rotation angles (typically 10000.0)
            max_seq_len: Maximum sequence length
            device: Device to place buffers on
        """
        super().__init__()

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Precompute cos and sin values for all positions
        cos, sin = self._precompute_cos_sin(max_seq_len, d_k, theta, device)

        # Register as buffers
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_cos_sin(self, max_seq_len: int, d_k: int, theta: float, device: torch.device | None = None):
        """
        Precompute cos and sin values for all positions and dimensions.

        Returns:
            cos: Tensor of shape (max_seq_len, d_k)
            sin: Tensor of shape (max_seq_len, d_k)
        """
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # Dimension indices: [0, 2, 4, ..., d_k-2]
        # We process pairs of dimensions, so we have d_k/2 pairs
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

        # Compute frequencies: theta^(-2i/d_k) for i in [0, d_k/2)
        freqs = theta ** (-dim_indices / d_k)

        # Compute angles: position * frequency
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Each angle applies to a pair of dimensions
        # We need to repeat each angle twice: [angle_0, angle_0, angle_1, angle_1, ...]
        angles = torch.repeat_interleave(angles, 2, dim=1)

        # Compute cos and sin
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to input tensor.

        For each pair of dimensions (x[..., 2i], x[..., 2i+1]), apply rotation:
        [x_2i', x_2i+1'] = [cos  -sin] [x_2i  ]
                           [sin   cos] [x_2i+1]

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            cos: Cosine values of shape (seq_len, d_k)
            sin: Sine values of shape (seq_len, d_k)

        Returns:
            Rotated tensor of same shape as x
        """
        # Split into even and odd dimensions
        # x[..., ::2] gets dimensions [0, 2, 4, ...] (even)
        # x[..., 1::2] gets dimensions [1, 3, 5, ...] (odd)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Get cos and sin for even and odd positions
        cos_even = cos[..., ::2]
        cos_odd = cos[..., 1::2]
        sin_even = sin[..., ::2]
        sin_odd = sin[..., 1::2]

        # Apply rotation
        # New even: x_even * cos - x_odd * sin
        # New odd:  x_even * sin + x_odd * cos
        rotated_even = x_even * cos_even - x_odd * sin_even
        rotated_odd = x_even * sin_odd + x_odd * cos_odd

        # Interleave back: [even_0, odd_0, even_1, odd_1, ...]
        # Stack along new dimension: (..., seq_len, d_k // 2, 2)
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)

        # Flatten last two dimensions: (..., seq_len, d_k)
        return rotated.flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)
                            Values are indices into the precomputed cos/sin buffers

        Returns:
            Rotated tensor of same shape as x
        """
        # Index into precomputed cos and sin using token positions
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # Apply rotation
        return self._apply_rotation(x, cos, sin)
