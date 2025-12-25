#!/usr/bin/env python3
"""Analyze Phase 2 results."""
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'phase2'
MODELS_DIR = PROJECT_ROOT / 'results' / 'models'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'

def load_phase2_results():
    """Load phase2_results.json."""
    results_file = RESULTS_DIR / 'phase2_results.json'
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return {}
    with open(results_file) as f:
        return json.load(f)

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

if __name__ == '__main__':
    ANALYSIS_DIR.mkdir(exist_ok=True)
    results = load_phase2_results()
    if results:
        plot_phase2_curves(results, 'test_loss')
        plot_phase2_curves(results, 'train_loss')
        plot_phase2_curves(results, 'grad_norm')
        plot_phase2_curves(results, 'weight_norm')
        print_phase2_summary(results)
