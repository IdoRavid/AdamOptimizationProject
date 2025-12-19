"""Standard Adam optimizer."""
from typing import Dict
from torch import Tensor

from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Standard Adam optimizer.
    
    For L2 regularization, add weight_decay to gradient before calling step().
    This is the "wrong" way per AdamW paper - used as baseline (A1).
    """
    
    def compute_first_moment(self, grad: Tensor, state: Dict) -> Tensor:
        # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        state['m'].mul_(self.config.beta1).add_(grad, alpha=1 - self.config.beta1)
        return state['m']
    
    def compute_second_moment(self, grad: Tensor, state: Dict, param: Tensor) -> Tensor:
        # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        state['v'].mul_(self.config.beta2).addcmul_(grad, grad, value=1 - self.config.beta2)
        return state['v']
    
    def compute_update(self, m: Tensor, v: Tensor, state: Dict) -> Tensor:
        # Bias correction: m̂_t = m_t / (1 - β₁ᵗ), v̂_t = v_t / (1 - β₂ᵗ)
        bc1 = 1 - self.config.beta1 ** self.step_count
        bc2 = 1 - self.config.beta2 ** self.step_count
        m_hat = m / bc1
        v_hat = v / bc2
        # Update: m̂_t / (√v̂_t + ε)
        return m_hat / (v_hat.sqrt() + self.config.eps)
