#!/usr/bin/env python3
"""Analyze Phase 2 results."""
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms

plt.rcParams['figure.facecolor'] = '#eee9e0'
plt.rcParams['axes.facecolor'] = '#eee9e0'
plt.rcParams['savefig.facecolor'] = '#eee9e0'

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase2'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

IMAGE_SIZE = 28 * 28

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, IMAGE_SIZE), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_phase2_results():
    """Load phase2_results.json and phase2_E1_2_results.json."""
    results = {}
    for fname in ['phase2_results.json', 'phase2_E1_2_results.json']:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                results.update(json.load(f))
    return results

def plot_phase2_curves(results, metric='test_loss', output=None):
    """Plot training curves for all Phase 2 runs."""
    output = output or ANALYSIS_DIR / f'phase2_{metric}.png'
    variants = sorted(set(name.split('_')[0] for name in results.keys()))
    
    fig, axes = plt.subplots(1, len(variants), figsize=(5*len(variants), 4))
    if len(variants) == 1:
        axes = [axes]
    
    for ax, variant in zip(axes, variants):
        for name, hist in results.items():
            if name.startswith(variant + '_'):
                label = f"lr={hist['config']['lr_mult']}, wd={hist['config']['wd_mult']}"
                ax.plot(hist[metric], label=label, alpha=0.8)
        ax.set_title(f'{variant} - {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")

def print_phase2_summary(results):
    """Print summary of Phase 2 results."""
    print("\n" + "="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    print(f"{'Run':<30} {'Final Train':<12} {'Final Test':<12} {'Best Test':<12}")
    print("-"*80)
    
    for name, hist in sorted(results.items()):
        final_train = hist['train_loss'][-1]
        final_test = hist['test_loss'][-1]
        best_test = min(hist['test_loss'])
        print(f"{name:<30} {final_train:<12.6f} {final_test:<12.6f} {best_test:<12.6f}")


def plot_reconstructions(results, variants, img_indices=[4, 0], output=None):
    """Plot reconstruction progress across epochs for best config of each variant."""
    output = output or ANALYSIS_DIR / 'phase2_reconstructions.png'
    epochs = [25, 50, 75, 100]
    
    # Find best config for each variant (by final test loss)
    best_configs = {}
    for name, hist in results.items():
        v = name.split('_')[0]
        if v in variants:
            final_test = hist['test_loss'][-1]
            if v not in best_configs or final_test < best_configs[v][1]:
                # Store the full run name (e.g., "A1_lr2_wd0")
                best_configs[v] = (name, final_test)
    
    print("Best configs selected:")
    for v, (name, loss) in best_configs.items():
        print(f"  {v}: {name} (test_loss={loss:.6f})")
    
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True,
                                       transform=transforms.ToTensor())
    
    for img_idx in img_indices:
        img = test_data[img_idx][0].view(-1)
        out = output.parent / f'{output.stem}_{img_idx}{output.suffix}'
        
        fig, axes = plt.subplots(len(variants), len(epochs) + 1, figsize=(12, 2.5 * len(variants)))
        if len(variants) == 1:
            axes = [axes]
        
        for row, variant in enumerate(variants):
            axes[row][0].imshow(img.view(28, 28).numpy(), cmap='gray')
            axes[row][0].set_title('Original' if row == 0 else '')
            axes[row][0].set_ylabel(variant, fontsize=12, fontweight='bold')
            axes[row][0].set_xticks([]); axes[row][0].set_yticks([])
            
            if variant not in best_configs:
                for col in range(1, len(epochs) + 1):
                    axes[row][col].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
                continue
            
            base = best_configs[variant][0]
            
            for col, ep in enumerate(epochs, 1):
                ckpt_path = MODELS_DIR / f'{base}_epoch{ep}.pt'
                if not ckpt_path.exists():
                    axes[row][col].text(0.5, 0.5, 'Missing', ha='center', va='center')
                    axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
                    continue
                
                model = Autoencoder()
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
                model.eval()
                with torch.no_grad():
                    recon = model(img.unsqueeze(0)).squeeze().view(28, 28).numpy()
                
                axes[row][col].imshow(recon, cmap='gray')
                axes[row][col].set_title(f'Ep{ep}' if row == 0 else '')
                axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")

if __name__ == '__main__':
    ANALYSIS_DIR.mkdir(exist_ok=True)
    results = load_phase2_results()
    if results:
        plot_phase2_curves(results, 'test_loss')
        plot_phase2_curves(results, 'train_loss')
        plot_phase2_curves(results, 'grad_norm')
        plot_phase2_curves(results, 'weight_norm')
        print_phase2_summary(results)
    
    # Plot reconstructions for key variants
    plot_reconstructions(results, ['A1', 'B1', 'B3', 'E1', 'E2'])
