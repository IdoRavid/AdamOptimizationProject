"""Experiment and grid search utilities."""
from dataclasses import dataclass, field
from typing import List, Optional
import json
import numpy as np

from .logging import TrainingLog


@dataclass
class GridSearchResult:
    """Result for one hyperparameter combination."""
    lr_multiplier: float
    wd_multiplier: float
    learning_rate: float
    weight_decay: float
    final_train_loss: float
    final_test_loss: float
    best_test_loss: float
    best_epoch: int
    training_log: Optional[TrainingLog] = None


@dataclass
class OptimizerExperiment:
    """Full experiment results for one optimizer variant."""
    optimizer_id: str
    optimizer_name: str
    config: dict
    results: List[GridSearchResult] = field(default_factory=list)
    
    def get_heatmap_data(self):
        """Extract data for heatmap visualization."""
        lr_mults = sorted(set(r.lr_multiplier for r in self.results))
        wd_mults = sorted(set(r.wd_multiplier for r in self.results))
        
        heatmap = np.full((len(lr_mults), len(wd_mults)), np.nan)
        
        for r in self.results:
            i = lr_mults.index(r.lr_multiplier)
            j = wd_mults.index(r.wd_multiplier)
            heatmap[i, j] = r.final_test_loss
        
        return lr_mults, wd_mults, heatmap
    
    def plot_heatmap(self, ax=None, title=None, vmin=None, vmax=None):
        """Plot heatmap like Figure 2 in AdamW paper."""
        import matplotlib.pyplot as plt
        
        lr_mults, wd_mults, heatmap = self.get_heatmap_data()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(heatmap, cmap='jet', aspect='auto', origin='lower',
                       vmin=vmin, vmax=vmax)
        
        ax.set_xticks(range(len(wd_mults)))
        ax.set_xticklabels([f'{w}' for w in wd_mults], rotation=45)
        ax.set_yticks(range(len(lr_mults)))
        ax.set_yticklabels([f'1/{int(1/lr)}' if lr < 1 else str(int(lr)) for lr in lr_mults])
        
        ax.set_xlabel('Weight decay multiplier')
        ax.set_ylabel('Learning rate multiplier')
        ax.set_title(title or self.optimizer_name)
        
        plt.colorbar(im, ax=ax)
        
        # Mark top-10 best configs
        sorted_results = sorted(self.results, key=lambda r: r.final_test_loss)[:10]
        for r in sorted_results:
            i = lr_mults.index(r.lr_multiplier)
            j = wd_mults.index(r.wd_multiplier)
            ax.plot(j, i, 'ko', markersize=8, markerfacecolor='none', markeredgewidth=2)
        
        return ax
    
    def save(self, filepath: str):
        data = {
            'optimizer_id': self.optimizer_id,
            'optimizer_name': self.optimizer_name,
            'config': self.config,
            'results': [
                {
                    'lr_multiplier': r.lr_multiplier,
                    'wd_multiplier': r.wd_multiplier,
                    'learning_rate': r.learning_rate,
                    'weight_decay': r.weight_decay,
                    'final_train_loss': r.final_train_loss,
                    'final_test_loss': r.final_test_loss,
                    'best_test_loss': r.best_test_loss,
                    'best_epoch': r.best_epoch,
                }
                for r in self.results
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizerExperiment':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        exp = cls(
            optimizer_id=data['optimizer_id'],
            optimizer_name=data['optimizer_name'],
            config=data['config'],
        )
        exp.results = [
            GridSearchResult(**r) for r in data['results']
        ]
        return exp
