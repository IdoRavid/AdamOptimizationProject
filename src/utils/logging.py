"""Training logging utilities."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class TrainingLog:
    """Stores all training metrics for analysis."""
    
    # Per-epoch metrics
    epoch_train_losses: List[float] = field(default_factory=list)
    epoch_test_losses: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # Per-step metrics (optional)
    step_losses: List[float] = field(default_factory=list)
    step_grad_norms: List[float] = field(default_factory=list)
    step_lrs: List[float] = field(default_factory=list)
    
    # Reconstruction samples
    reconstruction_samples: Dict[int, Tuple] = field(default_factory=dict)
    
    # Summary
    best_test_loss: float = float('inf')
    best_epoch: int = 0
    total_runtime: float = 0.0
    
    def log_step(self, loss: float, grad_norm: Optional[float] = None, lr: Optional[float] = None):
        self.step_losses.append(loss)
        if grad_norm is not None:
            self.step_grad_norms.append(grad_norm)
        if lr is not None:
            self.step_lrs.append(lr)
    
    def log_epoch(self, train_loss: float, test_loss: float, epoch_time: float):
        self.epoch_train_losses.append(train_loss)
        self.epoch_test_losses.append(test_loss)
        self.epoch_times.append(epoch_time)
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_epoch = len(self.epoch_test_losses)
    
    def save_reconstructions(self, epoch: int, originals: torch.Tensor, reconstructed: torch.Tensor):
        self.reconstruction_samples[epoch] = (originals.detach().cpu(), reconstructed.detach().cpu())
    
    def to_dict(self) -> dict:
        return {
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_test_losses': self.epoch_test_losses,
            'epoch_times': self.epoch_times,
            'best_test_loss': self.best_test_loss,
            'best_epoch': self.best_epoch,
            'total_runtime': self.total_runtime,
        }
