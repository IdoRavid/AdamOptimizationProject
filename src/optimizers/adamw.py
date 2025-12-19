"""AdamW optimizer with decoupled weight decay."""
from dataclasses import dataclass
from typing import Iterator, Optional
from torch import Tensor
import torch.nn as nn

from .base import OptimizerConfig
from .adam import Adam


@dataclass
class AdamWConfig(OptimizerConfig):
    """AdamW-specific configuration."""
    normalized_decay: bool = False
    batch_size: int = 128
    total_batches: Optional[int] = None


class AdamW(Adam):
    """
    AdamW: Adam with decoupled weight decay.
    
    Key insight: Weight decay is applied directly to weights, NOT through gradient.
    This decouples the learning rate and weight decay hyperparameters.
    """
    
    def __init__(self, params: Iterator[nn.Parameter], config: AdamWConfig,
                 lr_scheduler=None, total_steps: Optional[int] = None):
        super().__init__(params, config, lr_scheduler, total_steps)
        self.config: AdamWConfig = config
    
    def weight_decay_step(self, param: Tensor, lr: float) -> None:
        if self.config.weight_decay <= 0:
            return
        
        wd = self.config.weight_decay
        
        # Normalized weight decay: λ_norm = λ * √(batch_size / total_batches)
        if self.config.normalized_decay and self.config.total_batches:
            wd *= (self.config.batch_size / self.config.total_batches) ** 0.5
        
        # Decoupled weight decay: θ_t = θ_t * (1 - α * λ)
        # Applied directly to weights, not through gradient
        param.data.mul_(1 - lr * wd)
