"""
Transformer Language Model with support for multiple architectural variants.
"""

import torch
import torch.nn as nn
from student.modules import Embedding, TransformerBlock, RMSNorm, Linear


class TransformerLM(nn.Module):
    """
    Transformer Language Model with configurable architecture variants.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        eps: float = 1e-5,
        use_norm: bool = True,
        norm_position: str = "pre",
        ffn_type: str = "swiglu",
        tied_weights: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize Transformer Language Model.

        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum sequence length
            d_model: Model dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            rope_theta: RoPE theta parameter
            use_rope: Whether to use rotary position embeddings
            eps: Epsilon for RMSNorm
            use_norm: Whether to use normalization layers
            norm_position: 'pre' for pre-norm, 'post' for post-norm
            ffn_type: 'swiglu' for SwiGLU, 'silu' for SiLU-only FFN
            tied_weights: Whether to tie LM head weights with token embeddings
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        self.use_norm = use_norm
        self.norm_position = norm_position
        self.ffn_type = ffn_type
        self.tied_weights = tied_weights

        # Validate arguments
        if norm_position not in ["pre", "post"]:
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position}")

        if ffn_type not in ["swiglu", "silu"]:
            raise ValueError(f"ffn_type must be 'swiglu' or 'silu', got {ffn_type}")

        # Token embeddings
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length if use_rope else None,
                    theta=rope_theta if use_rope else None,
                    use_rope=use_rope,
                    eps=eps,
                    use_norm=use_norm,
                    norm_position=norm_position,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm (only for pre-norm architecture with norm enabled)
        # Post-norm doesn't need final norm since last block already normalizes
        if use_norm and norm_position == "pre":
            self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        else:
            self.ln_final = None

        # Output projection (LM head)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

        if tied_weights:
            # Tie weights of LM head and token embeddings
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass of the language model.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            token_positions: Token positions for RoPE of shape (batch_size, sequence_length)
                           If None and use_rope=True, will be auto-generated as [0, 1, 2, ...]

        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Generate token positions if needed
        if self.use_rope and token_positions is None:
            token_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Token embeddings
        x = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # Final layer norm (only for pre-norm with norm enabled)
        if self.ln_final is not None:
            x = self.ln_final(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits
