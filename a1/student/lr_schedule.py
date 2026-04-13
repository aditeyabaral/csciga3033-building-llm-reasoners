"""
Learning rate scheduling functions.
"""

import math


def get_lr_cosine_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning rate scheduler with linear warmup.

    Args:
        it: Current iteration number
        max_learning_rate: α_max, maximum learning rate
        min_learning_rate: α_min, minimum/final learning rate
        warmup_iters: T_w, number of warmup iterations
        cosine_cycle_iters: T_c, number of cosine annealing iterations

    Returns:
        Learning rate for the given iteration
    """
    # Warm-up phase: linearly increase from 0 to max_learning_rate
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # Cosine annealing phase
    elif it <= cosine_cycle_iters:
        # Progress through cosine cycle (0 to 1)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Cosine annealing formula
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay

    # Post-annealing: constant at minimum learning rate
    else:
        return min_learning_rate
