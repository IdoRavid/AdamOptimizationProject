# AdamW Optimizer Comparison on Fashion-MNIST

Optimization course project comparing Adam variants on an Autoencoder reconstruction task.

## Quick Start

### Recommended: Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IdoRavid/AdamOptimizationProject/blob/main/Omer%26Ido_Optimizing_Fashion.ipynb)

The notebook automatically clones the repository and loads all dependencies.

### Fallback: Manual Setup
If the automatic setup fails:
1. Download the repository as ZIP from GitHub
2. Upload to Colab: `Omer&Ido_Optimizing_Fashion.ipynb`
3. Upload the `src/` folder to `/content/src/`
4. Upload final model weights: `results/models/B3_lr2_wd0.25_epoch100.pt` to `/content/`

## Papers Implemented
- **AdamW** (Loshchilov & Hutter, 2019) - Decoupled Weight Decay Regularization
- **Adafactor** (Shazeer & Stern, 2018) - Adaptive Learning Rates with Sublinear Memory Cost

## Goal
1. **Hyperparameter Decoupling**: Generate sensitivity heatmaps (learning rate vs weight decay) showing AdamW decouples these hyperparameters (rectangular optimal region) vs Adam+L2 (diagonal optimal region)
2. **LR Schedule Comparison**: Compare Fixed, StepDrop, Cosine, and WarmRestarts schedules
3. **Normalized Weight Decay**: Test batch-size-independent weight decay normalization
4. **Adafactor Variants**: Explore memory-efficient optimization with/without momentum
5. **Projection Experiments**: Project Adafactor's factored second-moment vectors onto convex sets (L2 ball, box constraints) to study regularization effects

## Final Model
**B3 (AdamW + Cosine LR)** with LR=0.002, WD=0.00025 — best balance of convergence and generalization.

## Optimizer Variants

| ID    | Optimizer                   | Description                                                    |
|-------|-----------------------------|----------------------------------------------------------------|
| A1    | Adam+L2                     | Baseline - L2 through gradient (coupled)                       |
| B1-B4 | AdamW                       | Decoupled weight decay + Fixed/StepDrop/Cosine/WarmRestarts LR |
| B5-B8 | AdamW+Norm                  | Normalized weight decay + LR schedules                         |
| C1-C2 | Adafactor                   | Factored second moments (C1: β1=0, C2: β1=0.9)                 |
| D1-D3 | Adafactor+Norm              | Adafactor + normalized decay + LR schedules                    |
| E1-E2 | Adafactor+L2Ball            | Adafactor with L2 ball projection on vr/vc                     |
| E3-E4 | Adafactor+Box               | Adafactor with box constraint projection on vr/vc              |

## Project Structure
```
├── Omer&Ido_Optimizing_Fashion.ipynb  # Main notebook
├── src/
│   ├── optimizers/
│   │   ├── base.py          # BaseOptimizer ABC
│   │   ├── schedulers.py    # LR schedulers (Fixed, Cosine, StepDrop, WarmRestarts)
│   │   ├── adam.py          # Base Adam
│   │   ├── adam_l2.py       # Adam+L2 (A1)
│   │   ├── adamw.py         # AdamW (B1-B8)
│   │   ├── adafactor.py     # Adafactor with projection support (C1-E4)
│   │   └── combined.py      # Combined AdamW+Adafactor (D1-D3)
│   ├── analysis/
│   │   ├── analyze_phase_1.py  # Heatmap generation
│   │   └── analyze_phase_2.py  # Training curves analysis
│   └── utils/
│       ├── logging.py       # TrainingLog
│       └── experiment.py    # GridSearchResult, OptimizerExperiment
├── results/
│   ├── phase1/              # Grid search results (144 configs × 16 optimizers)
│   ├── phase2/              # Deep training results (100 epochs)
│   ├── projection_test/     # Adafactor projection experiments
│   └── models/              # Saved model checkpoints
├── analysis/                # Generated plots (heatmaps, loss curves, reconstructions)
└── documents/               # Reference papers and notes
```

## Authors
Ido Ravid & Omer Sutovsky
