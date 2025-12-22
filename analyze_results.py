#!/usr/bin/env python3
"""Analyze grid search results and generate visualizations."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('results')

def load_results():
    """Load all JSON result files."""
    data = {}
    for f in sorted(RESULTS_DIR.glob('*.json')):
        name = f.stem.split('_')[0]  # e.g., 'A1' from 'A1_Adam+L2.json'
        with open(f) as fp:
            data[name] = json.load(fp)
        print(f"Loaded {name}: {len(data[name]['results'])}/144 configs")
    return data

def plot_heatmaps(data, output='results/heatmaps.png'):
    """Generate heatmap grid for all optimizers."""
    n = len(data)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    vmin, vmax = 0.01, 0.15
    
    for idx, (name, d) in enumerate(data.items()):
        results = d['results']
        lr_mults = sorted(set(r['lr_multiplier'] for r in results))
        wd_mults = sorted(set(r['wd_multiplier'] for r in results))
        
        heatmap = np.full((len(lr_mults), len(wd_mults)), np.nan)
        for r in results:
            i = lr_mults.index(r['lr_multiplier'])
            j = wd_mults.index(r['wd_multiplier'])
            heatmap[i, j] = r['best_test_loss']
        
        ax = axes[idx]
        im = ax.imshow(heatmap, cmap='RdYlGn_r', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}: {d['optimizer_name']}", fontsize=10)
        ax.set_xlabel('WD mult')
        ax.set_ylabel('LR mult')
        ax.set_xticks([0, len(wd_mults)//2, len(wd_mults)-1])
        ax.set_xticklabels(['1/1024', '1/32', '2'])
        ax.set_yticks([0, len(lr_mults)//2, len(lr_mults)-1])
        ax.set_yticklabels(['1/1024', '1/32', '2'])
    
    for idx in range(len(data), len(axes)):
        axes[idx].axis('off')
    
    plt.colorbar(im, ax=axes, shrink=0.6, label='Test Loss')
    plt.suptitle('Hyperparameter Sensitivity Heatmaps', fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")

def plot_boxplot(data, output='results/boxplot.png'):
    """Generate box & whiskers plot."""
    fig, ax = plt.subplots(figsize=(max(8, len(data)*1.5), 6))
    
    box_data = [[r['best_test_loss'] for r in d['results']] for d in data.values()]
    bp = ax.boxplot(box_data, tick_labels=list(data.keys()), patch_artist=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Best Test Loss')
    ax.set_xlabel('Optimizer Variant')
    ax.set_title('Loss Distribution Across Hyperparameter Grid')
    ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='Good (0.02)')
    ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Bad (0.10)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90)
    print(f"{'Variant':<8} {'Min':<10} {'Median':<10} {'Max':<10} {'<0.02':<8} {'<0.10':<8} {'Best LR':<10} {'Best WD'}")
    print("-"*90)
    
    for name, d in data.items():
        results = d['results']
        losses = [r['best_test_loss'] for r in results]
        best = min(results, key=lambda r: r['best_test_loss'])
        print(f"{name:<8} {min(losses):<10.4f} {np.median(losses):<10.4f} {max(losses):<10.4f} "
              f"{sum(1 for l in losses if l < 0.02):<8} {sum(1 for l in losses if l < 0.10):<8} "
              f"{best['lr_multiplier']:<10} {best['wd_multiplier']}")
    
    # Decoupling analysis
    print("\n" + "="*90)
    print("DECOUPLING ANALYSIS: Good configs (<0.02) per LR row")
    print("="*90)
    
    for name, d in data.items():
        results = d['results']
        lr_mults = sorted(set(r['lr_multiplier'] for r in results))
        wd_mults = sorted(set(r['wd_multiplier'] for r in results))
        
        heatmap = {(r['lr_multiplier'], r['wd_multiplier']): r['best_test_loss'] for r in results}
        good_per_lr = [sum(1 for wd in wd_mults if heatmap.get((lr, wd), 1) < 0.02) for lr in lr_mults]
        
        total = sum(good_per_lr)
        max_row = max(good_per_lr)
        pattern = 'WIDE (decoupled)' if max_row >= 10 else 'NARROW (coupled)'
        print(f"{name}: total={total}/144, max_per_row={max_row}/12, pattern={pattern}")
    
    # Ranking
    print("\n" + "="*90)
    print("RANKING (by median loss)")
    print("="*90)
    
    ranking = sorted([(n, np.median([r['best_test_loss'] for r in d['results']])) for n, d in data.items()], key=lambda x: x[1])
    for i, (name, median) in enumerate(ranking):
        print(f"{i+1}. {name}: median={median:.4f}")

if __name__ == '__main__':
    data = load_results()
    if data:
        plot_heatmaps(data)
        plot_boxplot(data)
        print_summary(data)
