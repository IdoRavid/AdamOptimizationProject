"""Adam with L2 regularization (baseline A1)."""
from .adam import Adam


class AdamL2(Adam):
    """
    Adam with L2 regularization (baseline A1).
    L2 is applied through the gradient, which couples lr and weight_decay.
    """
    
    def step(self) -> None:
        # L2 regularization: g_t = g_t + λ * θ_t (added to gradient)
        if self.config.weight_decay > 0:
            for param in self.params:
                if param.grad is not None:
                    param.grad.data.add_(param.data, alpha=self.config.weight_decay)
        super().step()
