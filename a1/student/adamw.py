"""
AdamW optimizer implementation.
"""

import torch
import math
from collections.abc import Callable


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Implements Algorithm 2 from Loshchilov & Hutter (2019).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (α)
            betas: Coefficients (β1, β2) for computing running averages
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient (λ)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss

        Returns:
            Optional loss value if closure is provided
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Get state for this parameter
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    # First moment estimate
                    state["m"] = torch.zeros_like(p.data)
                    # Second moment estimate
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                state["t"] += 1
                t = state["t"]

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply weight decay (decoupled from gradient update)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
