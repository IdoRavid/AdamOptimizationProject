#!/usr/bin/env python3
"""Analyze grid search results and generate visualizations."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.facecolor'] = '#eee9e0'
plt.rcParams['axes.facecolor'] = '#eee9e0'
plt.rcParams['savefig.facecolor'] = '#eee9e0'

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase1'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

def load_results():
    """Load all JSON result files (grid search format only)."""
    data = {}
    for f in sorted(RESULTS_DIR.glob('*.json')):
        name = f.stem.split('_')[0]  # e.g., 'A1' from 'A1_Adam+L2.json'
        with open(f) as fp:
            content = json.load(fp)
        # Skip non-grid-search files (E variants have 'history' instead of 'results')
        if 'results' not in content:
            continue
        data[name] = content
        print(f"Loaded {name}: {len(data[name]['results'])}/144 configs")
    return data



def _plot_heatmap_group(data, layout, title, output, vmin=0.01, vmax=0.15):
    """Helper to plot a heatmap group with shared color scale."""
    nrows = len(layout)
    fig, axes = plt.subplots(nrows, 4, figsize=(24, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    im = None
    
    for row_idx, (group_title, variants) in enumerate(layout):
        for col_idx in range(4):
            ax = axes[row_idx][col_idx]
            
            if col_idx >= len(variants):
                ax.axis('off')
                continue
            
            name = variants[col_idx]
            
            if col_idx == 0:
                ax.set_ylabel(group_title + '\n\nLR mult', fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel('LR mult', fontsize=12)
            
            if name not in data:
                ax.text(0.5, 0.5, f'{name}\n(pending)', ha='center', va='center', fontsize=16)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            
            d = data[name]
            results = d['results']
            lr_mults = sorted(set(r['lr_multiplier'] for r in results))
            wd_mults = sorted(set(r['wd_multiplier'] for r in results))
            
            heatmap = np.full((len(lr_mults), len(wd_mults)), np.nan)
            for r in results:
                i = lr_mults.index(r['lr_multiplier'])
                j = wd_mults.index(r['wd_multiplier'])
                heatmap[i, j] = r['best_test_loss']
            
            im = ax.imshow(heatmap, cmap='Spectral', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}: {d['optimizer_name']}", fontsize=14, fontweight='bold')
            ax.set_xlabel('WD mult', fontsize=12)
            ax.set_xticks([0, len(wd_mults)//2, len(wd_mults)-1])
            ax.set_xticklabels(['1/1024', '1/32', '2'], fontsize=10)
            ax.set_yticks([0, len(lr_mults)//2, len(lr_mults)-1])
            ax.set_yticklabels(['1/1024', '1/32', '2'], fontsize=10)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Test Loss', fontsize=14)
    
    plt.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output}")


def plot_heatmaps(data, output=None):
    """Generate two heatmap plots: A-B and C-E with shared color scale."""
    vmin, vmax = 0.01, 0.15
    
    layout_ab = [
        ('A: Baseline', ['A1']),
        ('B1-B4: AdamW + LR Schedules', ['B1', 'B2', 'B3', 'B4']),
        ('B5-B8: AdamW + Normalized', ['B5', 'B6', 'B7', 'B8']),
    ]
    _plot_heatmap_group(data, layout_ab, 'Hyperparameter Sensitivity: A-B Variants', 
                        ANALYSIS_DIR / 'heatmaps_AB.png', vmin, vmax)
    
    layout_cde = [
        ('C: Adafactor', ['C1', 'C2']),
        ('D: Combined', ['D1', 'D2', 'D3']),
        ('E: Projection', ['E1', 'E2']),
    ]
    _plot_heatmap_group(data, layout_cde, 'Hyperparameter Sensitivity: C-E Variants',
                        ANALYSIS_DIR / 'heatmaps_CDE.png', vmin, vmax)

def plot_boxplot(data, output=None):
    output = output or ANALYSIS_DIR / 'boxplot.png'
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
    ANALYSIS_DIR.mkdir(exist_ok=True)
    data = load_results()
    if data:
        plot_heatmaps(data)
        plot_boxplot(data)
        print_summary(data)
