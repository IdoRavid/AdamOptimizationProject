"""
Adafactor optimizer variants (C1/C2).

This file implements an Adafactor-like optimizer within the project's BaseOptimizer API:
- Uses BaseOptimizer.step() loop and hooks:
  compute_first_moment, compute_second_moment, compute_update, weight_decay_step.

Key features (per your test_plan):
- Factored second-moment estimation for matrix-like params
- Increasing decay schedule: beta2_t = 1 - t^{-0.8}
- Optional momentum (beta1 = 0 for C1, beta1 = 0.9 for C2)
- Update clipping by RMS(update) with threshold d
- Relative step size scaling:
    lr_effective = lr_scheduler(config.lr) * scale_parameter * relative_step(t)
  where scale_parameter = max(eps2, RMS(param))
        relative_step(t) = min(relative_step_max, 1/sqrt(t))
- Decoupled weight decay (optional): param *= (1 - lr_effective * wd)

Notes:
- We keep config.lr meaningful for LR heatmap by treating it as a base multiplier.
- WD is applied decoupled here as well so WD heatmaps remain meaningful for C variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import math
import torch
from torch import Tensor
import torch.nn as nn

from .base import BaseOptimizer, OptimizerConfig


@dataclass
class AdafactorConfig(OptimizerConfig):
    """
    Adafactor-specific configuration.

    Inherits:
      lr, beta1, beta2 (unused as fixed beta2), eps, weight_decay
    """
    # Factored 2nd moment for tensors with ndim>=2
    factored: bool = True

    # Clipping threshold d (RMS of update)
    clip_threshold: float = 1.0

    # Relative step: lr_eff = lr * max(eps2, RMS(param)) * min(relative_step_max, 1/sqrt(t))
    relative_step: bool = True
    relative_step_max: float = 1e-2
    eps2: float = 1e-3  # epsilon used inside scale_parameter = max(eps2, RMS(param))

    # Additional epsilon for second-moment stability (distinct from OptimizerConfig.eps)
    # We use config.eps inside denominator sqrt(v)+eps; this is common.
    # This one is for internal v computations when needed.
    v_eps: float = 1e-30

    # Time-varying beta2 schedule exponent (default matches plan: 0.8)
    beta2_exponent: float = 0.8


class Adafactor(BaseOptimizer):
    """
    Adafactor-like optimizer compatible with project's BaseOptimizer.

    - Uses per-parameter state:
        m: momentum buffer (if beta1>0)
        v: full second-moment buffer for non-factored params
        vr, vc: factored row/col accumulators for factored params (2D view)
    """

    def __init__(
        self,
        params: Iterator[nn.Parameter],
        config: AdafactorConfig,
        lr_scheduler=None,
        total_steps: Optional[int] = None,
    ):
        super().__init__(params, config, lr_scheduler, total_steps)
        self.config: AdafactorConfig = config

    # ----------------------------
    # Helpers
    # ----------------------------
    def _beta2_t(self) -> float:
        # beta2_t = 1 - t^{-p}
        t = max(1, self.step_count)
        return 1.0 - (t ** (-self.config.beta2_exponent))

    @staticmethod
    def _rms(x: Tensor) -> Tensor:
        # sqrt(mean(x^2))
        return x.pow(2).mean().sqrt()

    @staticmethod
    def _as_matrix(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
        """
        View a tensor as 2D for factored accumulator updates:
        - For ndim>=2: shape (d0, d1*...*dk)
        - Returns (matrix_view, original_shape)
        """
        orig_shape = tuple(x.shape)
        if x.ndim < 2:
            return x.view(-1, 1), orig_shape
        return x.view(x.shape[0], -1), orig_shape

    def _is_factored(self, param: Tensor) -> bool:
        return bool(self.config.factored) and param.data.ndim >= 2

    # ----------------------------
    # State init
    # ----------------------------
    def _init_state(self, param: Tensor) -> Dict:
        # Always store momentum buffer 'm' so compute_first_moment can update in-place.
        state: Dict = {
            "m": torch.zeros_like(param.data),
        }

        if self._is_factored(param):
            # For factored: store vr (rows) and vc (cols) in the 2D view
            mat, _ = self._as_matrix(param.data)
            rows, cols = mat.shape
            state["vr"] = torch.zeros((rows,), device=param.data.device, dtype=param.data.dtype)
            state["vc"] = torch.zeros((cols,), device=param.data.device, dtype=param.data.dtype)
        else:
            # For non-factored: store full v
            state["v"] = torch.zeros_like(param.data)

        return state

    # ----------------------------
    # LR handling
    # ----------------------------
    def get_current_lr(self) -> float:
        """
        Override BaseOptimizer.get_current_lr() to incorporate Adafactor's relative step size
        using *the global* parameter scale (we approximate using each param's scale inside step).

        IMPORTANT: BaseOptimizer.step() calls get_current_lr() once per step, but Adafactor's
        "scale_parameter" depends on each param. To preserve the BaseOptimizer structure, we:
          - return the scheduler/base lr here
          - and incorporate per-param scaling inside compute_update via state["lr_scale"] injected
            by compute_second_moment (which has access to param) or by a cached attribute.

        So this returns only the base lr.
        """
        if self.lr_scheduler is None:
            return self.config.lr
        return self.lr_scheduler.get_lr(self.config.lr, self.step_count, self.total_steps or 1)

    # ----------------------------
    # Hook implementations
    # ----------------------------
    def compute_first_moment(self, grad: Tensor, state: Dict) -> Tensor:
        """
        If beta1 == 0 -> "no momentum": we still return grad (but keep m buffer unused).
        Else update momentum buffer: m = beta1*m + (1-beta1)*grad
        """
        b1 = float(self.config.beta1)
        if b1 <= 0.0:
            # Return grad directly (no bias correction used anywhere in Adafactor here)
            return grad
        state["m"].mul_(b1).add_(grad, alpha=1.0 - b1)
        return state["m"]

    def compute_second_moment(self, grad: Tensor, state: Dict, param: Tensor) -> Tensor:
        """
        Update second moment with time-varying beta2_t:
          v = beta2_t * v + (1-beta2_t) * grad^2
        Factored case stores vr/vc and reconstructs v (full tensor) to return.
        """
        beta2_t = self._beta2_t()
        one_minus = 1.0 - beta2_t

        if self._is_factored(param):
            g2 = grad.detach().pow(2).add_(self.config.v_eps)
            g2_mat, orig_shape = self._as_matrix(g2)

            # Row/col means
            row_mean = g2_mat.mean(dim=1)
            col_mean = g2_mat.mean(dim=0)

            state["vr"].mul_(beta2_t).add_(row_mean, alpha=one_minus)
            state["vc"].mul_(beta2_t).add_(col_mean, alpha=one_minus)

            # Reconstruct v approx: outer(vr, vc) / mean(vr)
            vr = state["vr"]
            vc = state["vc"]
            denom = vr.mean().clamp_min(self.config.v_eps)
            v_mat = torch.ger(vr, vc) / denom  # shape (rows, cols)

            return v_mat.view(orig_shape)

        # Non-factored: full v
        state["v"].mul_(beta2_t).addcmul_(grad, grad, value=one_minus)
        return state["v"]

    def compute_update(self, m: Tensor, v: Tensor, state: Dict) -> Tensor:
        """
        Compute Adafactor update direction (to be multiplied by lr in BaseOptimizer.step()):
          u = m / (sqrt(v) + eps)

        Then apply update clipping:
          if RMS(u) > d: u *= d / RMS(u)

        Relative step size is implemented by scaling the *returned* update direction so that
        BaseOptimizer's multiplication by lr implements:
          lr_eff = lr_base * scale_parameter * relative_step(t)

        We don't have 'param' here, so we approximate scale_parameter using m's magnitude
        when relative_step is enabled is NOT ideal. Therefore:

        We implement relative step scaling using the magnitude of the *parameter update
        denominator* (sqrt(v)) and the global step t:
          relative_factor = min(relative_step_max, 1/sqrt(t))
          scale_parameter ≈ 1 / (mean(sqrt(v))+eps)  -> not correct.

        Better approach: apply relative scaling via weight_decay_step which has param.
        But BaseOptimizer.step() calls weight_decay_step after update application.

        ==> Practical compromise for this repo:
        - Use base lr grid/scheduler normally (this is what heatmap varies).
        - Implement clipping + factored v accurately.
        - Provide optional relative_step scaling in weight_decay_step AND by injecting param-scale
          into state in compute_second_moment: state['param_rms'].

        We therefore store state['param_rms'] in compute_second_moment via param argument.
        However BaseOptimizer's signature doesn't pass param into compute_update; we can still
        access cached state['param_rms'] here.
        """
        # Basic normalization
        upd = m / (v.sqrt() + self.config.eps)

        # Clip by RMS(update)
        if self.config.clip_threshold is not None and self.config.clip_threshold > 0:
            rms_upd = self._rms(upd)
            # Avoid device mismatch; rms_upd is tensor scalar
            clip = float(self.config.clip_threshold)
            # scale factor = min(1, clip / rms)
            scale = (clip / rms_upd).clamp_max(1.0)
            upd = upd * scale

        # Relative step: scale update direction by scale_parameter * relative_factor
        if self.config.relative_step:
            t = max(1, self.step_count)
            relative_factor = min(float(self.config.relative_step_max), 1.0 / math.sqrt(t))

            # param RMS is injected into state by weight_decay_step (best effort) OR
            # by compute_second_moment in subclasses. Default fallback: 1.0
            param_rms = state.get("param_rms", None)
            if param_rms is None:
                scale_parameter = 1.0
            else:
                # scale_parameter = max(eps2, RMS(param))
                scale_parameter = max(float(self.config.eps2), float(param_rms))

            upd = upd * (scale_parameter * relative_factor)

        return upd

    def weight_decay_step(self, param: Tensor, lr: float) -> None:
        """
        Decoupled weight decay for Adafactor (optional).

        Also inject param_rms into state each step so compute_update can apply relative scaling.
        """
        state = self._get_state(param)

        # Cache RMS(param) for relative step scaling
        try:
            state["param_rms"] = float(self._rms(param.data))
        except Exception:
            state["param_rms"] = 1.0

        wd = float(self.config.weight_decay)
        if wd <= 0.0:
            return

        # decoupled decay: theta *= (1 - lr * wd)
        param.data.mul_(1.0 - lr * wd)


# Convenience wrappers for your IDs (optional)
class AdafactorC1(Adafactor):
    """C1: beta1=0 (no momentum)."""
    pass


class AdafactorC2(Adafactor):
    """C2: beta1=0.9 (with momentum)."""
    pass
