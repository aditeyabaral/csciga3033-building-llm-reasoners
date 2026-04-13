"""
Pre-norm Transformer block.
"""

import torch
import torch.nn as nn
from .rms_norm import RMSNorm
from .multihead_attention import MultiHeadSelfAttention
from .swiglu import SwiGLUFeedForward
from .silu import SiLUFeedForward


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.

    Architecture:
        x = x + MultiHeadSelfAttention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = None,
        theta: float = 10000.0,
        use_rope: bool = True,
        eps: float = 1e-5,
        use_norm: bool = True,
        norm_position: str = "pre",
        ffn_type: str = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize Transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length (required if use_rope=True)
            theta: RoPE theta parameter
            use_rope: Whether to use rotary position embeddings
            eps: Epsilon for RMSNorm
            use_norm: Whether to use normalization layers
            norm_position: 'pre' for pre-norm, 'post' for post-norm
            ffn_type: 'swiglu' for SwiGLU, 'silu' for SiLU-only FFN
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.use_norm = use_norm
        self.norm_position = norm_position
        self.ffn_type = ffn_type

        # Validate arguments
        if use_rope:
            assert max_seq_len is not None, "max_seq_len required when use_rope=True"

        if norm_position not in ["pre", "post"]:
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position}")

        if ffn_type not in ["swiglu", "silu"]:
            raise ValueError(f"ffn_type must be 'swiglu' or 'silu', got {ffn_type}")

        # Normalization layers (conditional on use_norm)
        if use_norm:
            self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        else:
            self.ln1 = None
            self.ln2 = None

        # Multi-head self-attention (with or without RoPE)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len if use_rope else None,
            theta=theta if use_rope else None,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network (SwiGLU or SiLU-only)
        if ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            # For SiLU-only, use 4*d_model to match SwiGLU parameter count
            silu_d_ff = 4 * d_model
            # Round to multiple of 64 for efficiency
            silu_d_ff = ((silu_d_ff + 63) // 64) * 64
            self.ffn = SiLUFeedForward(d_model, silu_d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Token positions for RoPE, shape (batch, seq_len)
                           Required if use_rope=True

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        if self.use_rope:
            assert token_positions is not None, "token_positions required when use_rope=True"

        # Apply attention sublayer based on norm position
        if self.norm_position == "pre":
            # Pre-norm: norm → sublayer → residual

            # First sublayer: Multi-head self-attention
            if self.use_norm:
                normalized = self.ln1(x)
            else:
                normalized = x
            attn_output = self.attn(normalized, token_positions=token_positions)
            x = x + attn_output

            # Second sublayer: Feed-forward
            if self.use_norm:
                normalized = self.ln2(x)
            else:
                normalized = x
            ffn_output = self.ffn(normalized)
            x = x + ffn_output

        elif self.norm_position == "post":
            # Post-norm: sublayer → residual → norm

            # First sublayer: Multi-head self-attention
            attn_output = self.attn(x, token_positions=token_positions)
            if self.use_norm:
                x = self.ln1(x + attn_output)
            else:
                x = x + attn_output

            # Second sublayer: Feed-forward
            ffn_output = self.ffn(x)
            if self.use_norm:
                x = self.ln2(x + ffn_output)
            else:
                x = x + ffn_output

        return x
