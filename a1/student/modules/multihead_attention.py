"""
Multi-head self-attention implementation.
"""

import torch
import torch.nn as nn
from .linear import Linear
from .attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Splits the input into multiple heads, applies attention to each head
    independently, then concatenates and projects the results.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int = None,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize multi-head self-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            use_rope: Whether to use RoPE
            max_seq_len: Maximum sequence length (required if use_rope=True)
            theta: RoPE theta parameter (default 10000.0)
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # Query, Key, Value projections (all heads combined)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Output projection
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE (if enabled)
        self.rope = None
        if use_rope:
            assert max_seq_len is not None, "max_seq_len must be provided when use_rope=True"
            self.rope = RotaryPositionalEmbedding(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len, device=device)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (..., seq_len, d_model)

        Returns:
            Tensor of shape (..., num_heads, seq_len, d_k)
        """
        *batch_dims, seq_len, d_model = x.shape
        # Reshape to (..., seq_len, num_heads, d_k)
        x = x.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        # Transpose to (..., num_heads, seq_len, d_k)
        # This puts num_heads before seq_len so attention is computed per head
        x = x.transpose(-3, -2)

        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back into a single dimension.

        Args:
            x: Tensor of shape (..., num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        *batch_dims, num_heads, seq_len, d_k = x.shape
        # Transpose to (..., seq_len, num_heads, d_k)
        x = x.transpose(-3, -2)
        # Reshape to (..., seq_len, d_model)
        x = x.reshape(*batch_dims, seq_len, self.d_model)
        return x

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future positions.

        Args:
            seq_len: Sequence length
            device: Device to place mask on

        Returns:
            Boolean mask of shape (seq_len, seq_len)
            True means attend, False means mask out
        """
        # Create a lower triangular matrix
        # mask[i, j] = True if i >= j (can attend to current and past)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Token positions for RoPE, shape (..., seq_len)
                           Required if use_rope=True

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        *batch_dims, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into multiple heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Apply RoPE if enabled
        if self.use_rope:
            assert token_positions is not None, "token_positions required when use_rope=True"
            # RoPE is applied per head, so we need to handle the num_heads dimension
            orig_shape = Q.shape
            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)

            # Expand token_positions to match: (..., seq_len) -> (...*num_heads, seq_len)
            token_positions_expanded = token_positions.unsqueeze(-2).expand(*batch_dims, self.num_heads, seq_len)
            token_positions_flat = token_positions_expanded.reshape(-1, seq_len)

            # Apply RoPE
            Q_flat = self.rope(Q_flat, token_positions_flat)
            K_flat = self.rope(K_flat, token_positions_flat)

            # Reshape back
            Q = Q_flat.reshape(orig_shape)
            K = K_flat.reshape(orig_shape)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # Apply attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # Combine heads
        combined = self._combine_heads(attn_output)

        # Output projection
        output = self.output_proj(combined)
        return output
