"""Optimizer implementations."""
from .base import BaseOptimizer, OptimizerConfig
from .schedulers import LRScheduler, FixedLR, CosineLR, StepDropLR, WarmRestartsLR
from .adam import Adam
from .adam_l2 import AdamL2
from .adamw import AdamW, AdamWConfig
from .adafactor import Adafactor, AdafactorConfig, AdafactorC1, AdafactorC2
from .combined import CombinedAdamWAdafactor, CombinedConfig, CombinedD1, CombinedD2, CombinedD3

__all__ = [
    'BaseOptimizer', 'OptimizerConfig',
    'LRScheduler', 'FixedLR', 'CosineLR', 'StepDropLR', 'WarmRestartsLR',
    'Adam', 'AdamL2', 'AdamW', 'AdamWConfig','Adafactor', 'AdafactorConfig', 'AdafactorC1', 'AdafactorC2',
    'CombinedAdamWAdafactor', 'CombinedConfig', 'CombinedD1', 'CombinedD2', 'CombinedD3',
]
