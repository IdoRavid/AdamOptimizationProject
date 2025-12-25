# Test Plan: Optimizer Comparison on Fashion-MNIST Autoencoder

## Overview

This project compares optimizer variants from two papers:
1. **AdamW** (Loshchilov & Hutter, 2019) - "Decoupled Weight Decay Regularization"
2. **Adafactor** (Shazeer & Stern, 2018) - "Adaptive Learning Rates with Sublinear Memory Cost"

We also test a **combined approach** (AdamW + Adafactor's factored estimation).

**Dataset:** Fashion-MNIST (60,000 train / 10,000 test, 28×28 grayscale)
**Model:** Convolutional Autoencoder (reconstruction task, MSE loss)
**Team:** Ido (AdamW + baselines), Omer (Adafactor + combined)

---

## Background

### AdamW Key Insight
Standard Adam with L2 regularization couples lr and weight decay (optimal settings lie on a diagonal). AdamW **decouples** weight decay from the gradient update (optimal settings form a rectangle).

### Adafactor Key Insight
Reduces memory by factoring second-moment accumulator into row/column sums. Also adds update clipping, increasing decay schedule (β̂₂ₜ = 1 - t⁻⁰·⁸), and relative step sizes.

### Combined Approach
Apply Adafactor's factored estimation to AdamW - untested in literature.

---

## Optimizer Variants

### Group A: Baseline
| ID | Optimizer | Description |
|----|-----------|-------------|
| A1 | Adam + L2 | Standard Adam with L2 (coupled) |

### Group B: AdamW (8 variants)

| ID | Normalized Decay | LR Schedule |
|----|------------------|-------------|
| B1 | No | Fixed |
| B2 | No | Step-drop (30, 60, 80) |
| B3 | No | Cosine |
| B4 | No | Warm Restarts (SGDR) |
| B5 | Yes | Fixed |
| B6 | Yes | Step-drop |
| B7 | Yes | Cosine |
| B8 | Yes | Warm Restarts |

### Group C: Adafactor
| ID | Momentum (β₁) |
|----|---------------|
| C1 | 0 (none) |
| C2 | 0.9 |

### Group D: Combined
| ID | LR Schedule | Momentum |
|----|-------------|----------|
| D1 | Cosine | No |
| D2 | Warm Restarts | No |
| D3 | Warm Restarts | Yes (0.9) |

---

## Hyperparameter Grid

**Learning Rate:** 12 multipliers × base_lr (0.001)
```
[1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
```

**Weight Decay:** 12 multipliers × base_wd (0.001)
```
[0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
```

**Grid size:** 12 × 12 = 144 configs per optimizer

**Fixed params:** batch_size=128, β₁=0.9, β₂=0.999, ε=1e-8, seed=42

---

## Execution Plan

### Phase 1: Grid Search (30 epochs)

Run 144 configs for each optimizer to generate heatmaps.

**Output per optimizer:** JSON with all 144 results (lr_mult, wd_mult, best_test_loss, epoch_losses)

**Analysis:**
- Heatmaps showing lr/wd sensitivity
- Decoupling pattern: horizontal bands (decoupled) vs diagonal (coupled)
- Ranking by median loss and good config count (<0.02)

### Phase 2: Deep Training (100 epochs)

**Selected Optimizers:**

| ID | Optimizer          | LR mult | WD mult | Why selected                      |
|----|--------------------|---------|---------|-----------------------------------|
| A1 | Adam+L2            | 2       | 0       | Control (coupled baseline)        |
| B3 | AdamW Cosine       | 2       | 0.5     | Standard AdamW, no normalization  |
| B4 | AdamW WarmRestarts | 2       | 0.03125 | Shows SGDR benefit                |
| B5 | AdamW Fixed+Norm   | 2       | 32      | Best overall, shows normalization |

**Logging (per epoch):**
- Train/test loss
- Learning rate
- Gradient norm, weight norm

**Checkpoints:** Save model at epochs 25, 50, 75, 100

**Reconstruction samples:** Every 25 epochs, same 8 test images

**Analysis:**
- Loss curves (all 4 optimizers on same plot)
- LR schedule visualization
- Reconstruction quality progression
- Final comparison table

### Phase 3: Final Report

**Presentation (8-10 min):**
1. Theoretical background + update formulas
2. Phase 1 heatmaps (decoupling patterns)
3. Phase 2 loss curves + reconstructions
4. Conclusions

---

## LR Schedule Implementations

**Cosine:** `lr = initial_lr * 0.5 * (1 + cos(π * epoch / total_epochs))`

**Warm Restarts:** Cosine with periodic restarts (T₀=10, T_mult=2)

**Step-Drop:** ×0.1 at epochs 30, 60, 80

---

## File Structure

```
Project/
├── Omer&Ido_Optimizing_Fashion.ipynb
├── test_plan.md
├── src/
│   ├── optimizers/
│   │   ├── base.py, adam.py, adam_l2.py, adamw.py, adafactor.py
│   │   └── schedulers.py
│   ├── utils/
│   │   ├── logging.py, experiment.py
│   └── analysis/
│       └── analyze_phase_1.py
├── results/           # Phase 1 JSON outputs
├── analysis/          # Generated plots
└── documents/         # Papers
```

---

## References

1. Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. arXiv:1711.05101
2. Shazeer & Stern (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. arXiv:1804.04235
3. Loshchilov & Hutter (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv:1608.03983
