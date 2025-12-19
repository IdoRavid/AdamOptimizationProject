"""Base optimizer classes and configurations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator
import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class OptimizerConfig:
    """Base configuration for optimizers."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


class BaseOptimizer(ABC):
    """
    Abstract base optimizer with hooks for moment computation.
    
    Designed for easy extension:
    - Override compute_first_moment() for momentum variations
    - Override compute_second_moment() for Adafactor's factored version
    - Override weight_decay_step() for AdamW's decoupled decay
    """
    
    def __init__(self, params: Iterator[nn.Parameter], config: OptimizerConfig, 
                 lr_scheduler=None, total_steps: Optional[int] = None):
        self.params = list(params)
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.state: Dict[int, Dict] = {}
        self.step_count = 0
        self.total_steps = total_steps
    
    def _get_state(self, param: Tensor) -> Dict:
        """Get or initialize state for a parameter."""
        pid = id(param)
        if pid not in self.state:
            self.state[pid] = self._init_state(param)
        return self.state[pid]
    
    def _init_state(self, param: Tensor) -> Dict:
        """Initialize optimizer state for a parameter."""
        return {
            'm': torch.zeros_like(param.data),
            'v': torch.zeros_like(param.data),
        }
    
    def get_current_lr(self) -> float:
        """Get current learning rate from scheduler or config."""
        if self.lr_scheduler is None:
            return self.config.lr
        return self.lr_scheduler.get_lr(
            self.config.lr, self.step_count, self.total_steps or 1
        )
    
    @abstractmethod
    def compute_first_moment(self, grad: Tensor, state: Dict) -> Tensor:
        """Compute/update first moment estimate."""
        pass
    
    @abstractmethod
    def compute_second_moment(self, grad: Tensor, state: Dict, param: Tensor) -> Tensor:
        """Compute/update second moment estimate."""
        pass
    
    @abstractmethod
    def compute_update(self, m: Tensor, v: Tensor, state: Dict) -> Tensor:
        """Compute parameter update from moments."""
        pass
    
    def weight_decay_step(self, param: Tensor, lr: float) -> None:
        """Apply weight decay. Override for decoupled decay (AdamW)."""
        pass
    
    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self) -> None:
        """Perform one optimization step."""
        self.step_count += 1
        lr = self.get_current_lr()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            state = self._get_state(param)
            
            # Update moments
            m = self.compute_first_moment(grad, state)
            v = self.compute_second_moment(grad, state, param)
            
            # Compute and apply update
            update = self.compute_update(m, v, state)
            param.data.add_(update, alpha=-lr)
            
            # Weight decay (decoupled for AdamW)
            self.weight_decay_step(param, lr)
