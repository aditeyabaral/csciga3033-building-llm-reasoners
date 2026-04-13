"""
Checkpointing utilities for saving and loading model state.
"""

import torch
from typing import BinaryIO, IO
import os


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model, optimizer state, and iteration number to a checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        iteration: Current training iteration number
        out: Path or file-like object to save checkpoint to
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint.

    Args:
        src: Path or file-like object to load checkpoint from
        model: Model to restore state into
        optimizer: Optimizer to restore state into

    Returns:
        The iteration number from the checkpoint
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
