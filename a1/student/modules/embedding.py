"""
Token embedding module.
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Embedding layer that maps token IDs to dense vectors.

    Similar to nn.Embedding but with custom initialization.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initialize the embedding layer.

        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of embedding vectors (d_model)
            device: Device to place parameters on
            dtype: Data type of parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding weight matrix (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize embedding weights with truncated normal distribution.

        Using: N(μ=0, σ²=1) truncated at [-3, 3]
        """
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Tensor of token IDs, shape (..., sequence_length)
                      where ... represents any number of batch dimensions

        Returns:
            Embeddings tensor of shape (..., sequence_length, embedding_dim)
        """
        # Index into the embedding matrix
        return self.weight[token_ids]
