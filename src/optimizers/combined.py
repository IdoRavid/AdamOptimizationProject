"""
Combined (D) variants: AdamW-style decoupled/normalized weight decay
+ Adafactor-style second-moment factorization, clipping, and relative step scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from torch import Tensor
import torch.nn as nn

from .adamw import AdamWConfig  # normalized_decay, batch_size, total_batches
from .adafactor import Adafactor, AdafactorConfig


@dataclass
class CombinedConfig(AdamWConfig, AdafactorConfig):
    """
    Combined config: union of AdamWConfig and AdafactorConfig.

    Resolution notes:
    - OptimizerConfig fields (lr, beta1, eps, weight_decay) are shared and compatible.
    - AdamWConfig provides normalized_decay/batch_size/total_batches used in weight decay.
    - AdafactorConfig provides factored/clipping/relative_step fields used in update.
    """
    pass


class CombinedAdamWAdafactor(Adafactor):
    """
    D-group optimizer: Adafactor updates + AdamW normalized decoupled weight decay.

    We inherit Adafactor for moment/update logic, and override weight_decay_step
    to match AdamW exactly (including normalized decay).
    """

    def __init__(
        self,
        params: Iterator[nn.Parameter],
        config: CombinedConfig,
        lr_scheduler=None,
        total_steps: Optional[int] = None,
    ):
        super().__init__(params, config, lr_scheduler, total_steps)
        self.config: CombinedConfig = config

    def weight_decay_step(self, param: Tensor, lr: float) -> None:
        """
        Copy of AdamW.weight_decay_step behavior (with param_rms caching preserved).
        """
        state = self._get_state(param)
        try:
            state["param_rms"] = float(self._rms(param.data))
        except Exception:
            state["param_rms"] = 1.0

        if self.config.weight_decay <= 0:
            return

        wd = float(self.config.weight_decay)

        # Normalized weight decay: λ_norm = λ * √(batch_size / total_batches)
        if getattr(self.config, "normalized_decay", False) and getattr(self.config, "total_batches", None):
            wd *= (self.config.batch_size / self.config.total_batches) ** 0.5

        # Decoupled weight decay: θ_t = θ_t * (1 - α * λ)
        param.data.mul_(1.0 - lr * wd)


# Optional aliases for D IDs (purely convenience)
class CombinedD1(CombinedAdamWAdafactor):
    pass


class CombinedD2(CombinedAdamWAdafactor):
    pass


class CombinedD3(CombinedAdamWAdafactor):
    pass
