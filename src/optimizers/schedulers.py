"""Learning rate schedulers."""
from abc import ABC, abstractmethod
import math


class LRScheduler(ABC):
    """Base learning rate scheduler."""
    
    @abstractmethod
    def get_lr(self, base_lr: float, step: int, total_steps: int) -> float:
        pass


class FixedLR(LRScheduler):
    """Constant learning rate."""
    
    def get_lr(self, base_lr: float, step: int, total_steps: int) -> float:
        return base_lr


class CosineLR(LRScheduler):
    """Cosine annealing from base_lr to 0."""
    
    def get_lr(self, base_lr: float, step: int, total_steps: int) -> float:
        return base_lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))


class StepDropLR(LRScheduler):
    """Step-drop schedule: drop LR by factor at specified epochs."""
    
    def __init__(self, drop_epochs=(30, 60, 80), drop_factor=0.1):
        self.drop_epochs = drop_epochs
        self.drop_factor = drop_factor
    
    def get_lr(self, base_lr: float, epoch: int, total_epochs: int) -> float:
        lr = base_lr
        for drop_epoch in self.drop_epochs:
            if epoch >= drop_epoch:
                lr *= self.drop_factor
        return lr


class WarmRestartsLR(LRScheduler):
    """Cosine annealing with warm restarts (SGDR)."""
    
    def __init__(self, T_0: int = 10, T_mult: int = 2, lr_min: float = 0.0):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.lr_min = lr_min
    
    def get_lr(self, base_lr: float, step: int, total_steps: int) -> float:
        # Find current restart period
        T_cur = step
        T_i = self.T_0
        
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        return self.lr_min + 0.5 * (base_lr - self.lr_min) * (1 + math.cos(math.pi * T_cur / T_i))
