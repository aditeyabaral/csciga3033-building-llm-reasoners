"""
Data loading utilities.
"""

import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and their corresponding labels.

    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device (e.g., torch.device('cpu'), torch.device('cuda:0'), torch.device('mps'))

    Returns:
        Tuple of (inputs, targets), each of shape (batch_size, context_length)
        - inputs[i]: sequence of context_length tokens
        - targets[i]: next tokens for each position (shifted by 1)
    """
    # Calculate valid starting positions
    num_valid_starts = len(dataset) - context_length

    # Sample random starting indices
    start_indices = np.random.randint(0, num_valid_starts, size=batch_size)

    # Extract sequences
    inputs = np.stack([dataset[start : start + context_length] for start in start_indices])

    # Extract targets
    targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in start_indices])

    # Convert to PyTorch tensors and move to device
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs_tensor, targets_tensor
