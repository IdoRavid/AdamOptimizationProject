# Test Plan: Optimizer Comparison on Fashion-MNIST Autoencoder

## Status

| Component | Status | Owner |
|-----------|--------|-------|
| A1: Adam+L2 | ✅ Done | Ido |
| B1-B8: AdamW variants | ✅ Done | Ido |
| C1-C2: Adafactor | ⏳ TODO | Omer |
| D1-D3: Combined | ⏳ TODO | Omer |
| Grid search infrastructure | ✅ Done | Ido |
| Notebook integration | ✅ Done | Ido |

**Next steps:**
1. Ido: Test run on Colab to verify everything works
2. Omer: Implement Adafactor (C1-C2) and Combined (D1-D3) in `src/optimizers/adafactor.py`
3. Both: Run full grid search

---

## Overview

This project compares optimizer variants from two papers:
1. **AdamW** (Loshchilov & Hutter, 2019) - "Decoupled Weight Decay Regularization"
2. **Adafactor** (Shazeer & Stern, 2018) - "Adaptive Learning Rates with Sublinear Memory Cost"

We also test a novel **combined approach** (AdamW + Adafactor's factored estimation).

**Goal:** Generate hyperparameter sensitivity heatmaps (learning rate vs weight decay) for each optimizer variant, similar to Figure 2 in the AdamW paper.

**Dataset:** Fashion-MNIST (60,000 train / 10,000 test images, 28×28 grayscale)

**Model:** Autoencoder (architecture TBD, reconstruction task)

**Metric:** Test reconstruction loss (MSE)

**Team Division:**
- **Ido**: AdamW variants (Group B) + baselines (Group A)
- **Omer**: Adafactor variants (Group C) + combined approach (Group D)

---

## Background

### AdamW Key Insight
Standard Adam with L2 regularization couples the learning rate and weight decay hyperparameters (optimal settings lie on a diagonal). AdamW **decouples** weight decay from the gradient update, making hyperparameters more independent (optimal settings form a rectangle).

### Adafactor Key Insight
Reduces memory by storing only row/column sums of the second-moment accumulator instead of the full matrix. Also introduces:
- Update clipping for stability
- Increasing decay schedule (β̂₂ₜ = 1 - t⁻⁰·⁸)
- Relative step sizes (scale updates by parameter magnitude)

### Combined Approach (Experimental)
Apply Adafactor's factored estimation and clipping to AdamW. This is untested in literature - we investigate whether it works on small models where memory savings aren't needed.

---

## Optimizer Variants (15 total)

### Group A: Baseline (1)

| ID | Optimizer | Description |
|----|-----------|-------------|
| A1 | Adam + L2 | Standard Adam with L2 regularization (the "wrong" way per AdamW paper) |

### Group B: AdamW Variants (8)

AdamW = Adam with **decoupled** weight decay (weight decay applied directly to weights, not through gradient).

| ID | Normalized Decay | LR Schedule | Description |
|----|------------------|-------------|-------------|
| B1 | No | Fixed | Basic AdamW, constant LR |
| B2 | No | Step-drop | LR drops at epochs 30, 60, 80 |
| B3 | No | Cosine | Cosine annealing to 0 |
| B4 | No | Warm Restarts | Cosine with periodic restarts (SGDR-style) |
| B5 | Yes | Fixed | Normalized decay: λ = λ_norm × √(batch_size / (total_batches × epochs)) |
| B6 | Yes | Step-drop | Normalized decay + step-drop |
| B7 | Yes | Cosine | Normalized decay + cosine |
| B8 | Yes | Warm Restarts | **AdamWR** - paper's best variant |

**Normalized weight decay** scales the decay factor based on training length, making optimal λ more consistent across different epoch counts.

### Group C: Adafactor Variants (2)

| ID | Momentum (β₁) | Decay Schedule (β̂₂ₜ) | Update Clipping | Step Size |
|----|---------------|----------------------|-----------------|-----------|
| C1 | 0 (none) | 1 - t⁻⁰·⁸ | d=1 | Relative |
| C2 | 0.9 | 1 - t⁻⁰·⁸ | d=1 | Relative |

**Adafactor components:**
- **Factored second moments:** Store row sums R and column sums C, approximate V ≈ RC^T / sum(R)
- **Update clipping:** Scale down updates when RMS(update) > d
- **Relative step size:** α_t = max(ε₂, RMS(params)) × ρ_t
- **Increasing decay:** β̂₂ₜ = 1 - t⁻⁰·⁸ (starts at 0, approaches 1)

### Group D: Combined AdamW-Factored (3)

Experimental: AdamW's decoupled weight decay + Adafactor's memory-saving techniques.

| ID | Normalized Decay | LR Schedule | Factored | Momentum | Clipping | Step Size |
|----|------------------|-------------|----------|----------|----------|-----------|
| D1 | Yes | Cosine | Yes | No | d=1 | Relative |
| D2 | Yes | Warm Restarts | Yes | No | d=1 | Relative |
| D3 | Yes | Warm Restarts | Yes | Yes (0.9) | d=1 | Relative |

---

## Hyperparameter Grid

### Learning Rate (12 values)

Multiply base learning rate by these factors:
```
lr_multipliers = [1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
```

Base learning rates:
- Adam/AdamW variants: `base_lr = 0.001`
- SGD: `base_lr = 0.1`
- Adafactor (relative): `base_rho = 0.01` (or use paper's schedule)

### Weight Decay (12 values)

Multiply base weight decay by these factors:
```
wd_multipliers = [0, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32]
```

Base weight decay: `base_wd = 0.001`

### Total Runs

- Grid size: 12 × 12 = **144 runs per optimizer**
- 15 optimizers × 144 runs = **2,160 total runs**

---

## Fixed Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 100 | Adjust if runs are too slow |
| Batch size | 128 | Standard for Fashion-MNIST |
| Loss function | MSE | Reconstruction loss |
| β₁ (Adam/AdamW) | 0.9 | Default Adam first moment decay |
| β₂ (Adam/AdamW) | 0.999 | Default Adam second moment decay |
| ε (Adam) | 1e-8 | Numerical stability |
| ε₁ (Adafactor) | 1e-30 | For factored estimation |
| ε₂ (Adafactor) | 1e-3 | For relative step size |
| Random seed | 42 | Single seed (add more if time permits) |

---

## Learning Rate Schedule Implementations

### Fixed
```python
lr = initial_lr  # constant throughout training
```

### Step-Drop
```python
# Drop LR by 10x at epochs 30, 60, 80
if epoch >= 80:
    lr = initial_lr * 0.001
elif epoch >= 60:
    lr = initial_lr * 0.01
elif epoch >= 30:
    lr = initial_lr * 0.1
else:
    lr = initial_lr
```

### Cosine Annealing
```python
# Decay from initial_lr to 0 over total_epochs
lr = initial_lr * 0.5 * (1 + cos(pi * current_epoch / total_epochs))
```

### Warm Restarts (SGDR)
```python
# Cosine annealing with periodic restarts
# T_0 = 10 epochs (first restart period)
# T_mult = 2 (double period after each restart)

T_cur = epochs_since_last_restart
T_i = current_restart_period  # 10, 20, 40, 80, ...

lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * T_cur / T_i))

# When T_cur == T_i: restart (reset T_cur=0, T_i *= T_mult)
```

---

## Autoencoder Architecture

Suggested architecture (adjust as needed):

```python
# Encoder
Conv2d(1, 32, 3, stride=2, padding=1)   # 28x28 -> 14x14
ReLU()
Conv2d(32, 64, 3, stride=2, padding=1)  # 14x14 -> 7x7
ReLU()
Flatten()
Linear(64*7*7, 128)  # Latent dimension = 128

# Decoder
Linear(128, 64*7*7)
Unflatten(64, 7, 7)
ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 -> 14x14
ReLU()
ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)   # 14x14 -> 28x28
Sigmoid()
```

---

## Execution Plan

### Phase 1: Validation (~1-2 hours)

1. Implement all optimizer variants
2. Run 1 config from each group (A1, B3, C1, D1) with mid-range hyperparameters:
   - lr_multiplier = 1 (base learning rate)
   - wd_multiplier = 1 (base weight decay)
3. Verify:
   - Training converges
   - Logging captures all required metrics
   - Results are saved correctly
4. Measure actual time per run to refine estimates

### Phase 2: Full Grid Execution

Run all 2,160 experiments. Suggested batching:

```
Batch 1: A1, A2, B1, B2, B3     (5 optimizers × 144 = 720 runs)
Batch 2: B4, B5, B6, B7, B8     (5 optimizers × 144 = 720 runs)
Batch 3: C1, C2, D1, D2, D3     (5 optimizers × 144 = 720 runs)
```

### Phase 3: Analysis

1. Generate heatmaps for each optimizer
2. Compare optimal regions (diagonal vs rectangular)
3. Identify best hyperparameters per optimizer
4. Statistical comparison of best results

---

## Output Format

Save results to JSON, one file per optimizer:

```json
{
  "optimizer_id": "B3",
  "optimizer_name": "AdamW_Cosine",
  "config": {
    "normalized_decay": false,
    "lr_schedule": "cosine",
    "factored": false,
    "momentum": 0.9,
    "clipping": null
  },
  "results": [
    {
      "lr_multiplier": 0.125,
      "wd_multiplier": 0.25,
      "learning_rate": 0.000125,
      "weight_decay": 0.00025,
      "final_train_loss": 0.0234,
      "final_test_loss": 0.0251,
      "best_test_loss": 0.0248,
      "best_epoch": 87,
      "epoch_train_losses": [0.15, 0.08, ...],
      "epoch_test_losses": [0.14, 0.07, ...],
      "runtime_seconds": 185
    },
    // ... 143 more entries
  ]
}
```

---

## Visualization: Heatmaps

For each optimizer, create a heatmap:
- **X-axis:** Weight decay (log scale)
- **Y-axis:** Learning rate (log scale)
- **Color:** Final test loss (lower = better, use colormap like in AdamW paper)
- **Markers:** Circle the top-10 best configurations

**Key comparison:** Adam+L2 (A1) vs AdamW+Cosine (B3)
- A1 should show diagonal optimal region (coupled hyperparameters)
- B3 should show rectangular optimal region (decoupled hyperparameters)

---

## Compute Resources

| Platform | GPU | Time/run | Total time | Cost |
|----------|-----|----------|------------|------|
| Google Colab | T4 (16GB) | ~3 min | ~108 hrs | Free |
| AWS ml.g6.xlarge | L4 (24GB) | ~1.5 min | ~54 hrs | ~$45 |

**Recommendation:** Start with Colab for validation. Use ml.g6.xlarge for full grid if time-constrained.

---

## Expected Findings

1. **AdamW vs Adam+L2:** AdamW should show better test loss and more decoupled hyperparameters (rectangular vs diagonal heatmap pattern)

2. **LR schedules:** Cosine and Warm Restarts should outperform Fixed and Step-drop

3. **Normalized decay:** Should make optimal weight decay more consistent

4. **Adafactor:** May underperform on this small model since factored estimation is an approximation (designed for memory savings on large models)

5. **Combined approach:** Uncertain - interesting to see if factored estimation hurts when memory isn't a constraint

---

## References

1. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR 2019. arXiv:1711.05101

2. Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. arXiv:1804.04235

3. Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv:1608.03983


---

## Software Architecture

### Modular Optimizer API

The optimizer implementation follows an OOP design with clear separation of concerns, allowing Ido and Omer to work independently while sharing a common interface.

```
BaseOptimizer (ABC)
├── compute_first_moment()    # Override for momentum variations
├── compute_second_moment()   # Omer overrides for Adafactor's factored version
├── compute_update()          # Compute param update from moments
├── weight_decay_step()       # Ido overrides for AdamW's decoupled decay
└── step()                    # Main loop (calls above methods)

Adam(BaseOptimizer)
└── Standard Adam implementation

AdamW(Adam)                   # Ido's responsibility
└── Overrides weight_decay_step() for decoupled decay

Adafactor(BaseOptimizer)      # Omer's responsibility
└── Overrides compute_second_moment() for factored estimation
```

### Learning Rate Schedulers

```
LRScheduler (ABC)
├── FixedLR
├── StepDropLR(drop_epochs=[30,60,80])
├── CosineLR
└── WarmRestartsLR(T_0=10, T_mult=2)
```

### Configuration Classes

```python
@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

@dataclass
class AdamWConfig(OptimizerConfig):
    normalized_decay: bool = False
    total_steps: int = None

@dataclass
class AdafactorConfig(OptimizerConfig):
    factored: bool = True
    use_first_moment: bool = True
    decay_rate: float = 0.8
    clipping_threshold: float = 1.0
```

---

## Logging & Metrics Collection

### TrainingLog Class

Captures all metrics needed for analysis and presentation:

```python
@dataclass
class TrainingLog:
    # Per-epoch (always collected)
    epoch_train_losses: list
    epoch_test_losses: list
    epoch_times: list
    
    # Per-step (optional, for detailed analysis)
    step_losses: list
    step_grad_norms: list
    step_weight_norms: list
    step_lrs: list
    
    # Reconstruction samples at checkpoints
    reconstruction_samples: Dict[int, tuple]  # epoch -> (original, reconstructed)
    
    # Summary
    best_test_loss: float
    best_epoch: int
    total_runtime: float
```

### Metrics to Collect

| Metric | Frequency | Purpose |
|--------|-----------|---------|
| Train loss | Every epoch | Loss curves |
| Test loss | Every epoch | Generalization, heatmaps |
| Gradient norm | Every N steps | Stability analysis |
| Weight norm | Every N steps | Weight decay effect |
| Learning rate | Every step | Schedule verification |
| Reconstructions | Every 10 epochs | Visual quality check |

---

## Heatmap Data Collection

### GridSearchResult

One result per (lr_multiplier, wd_multiplier) pair:

```python
@dataclass
class GridSearchResult:
    lr_multiplier: float
    wd_multiplier: float
    learning_rate: float      # actual lr = base_lr * multiplier
    weight_decay: float       # actual wd = base_wd * multiplier
    final_train_loss: float
    final_test_loss: float
    best_test_loss: float
    best_epoch: int
    training_log: TrainingLog
```

### OptimizerExperiment

Full results for one optimizer variant (144 grid points):

```python
@dataclass
class OptimizerExperiment:
    optimizer_id: str         # e.g., "B3"
    optimizer_name: str       # e.g., "AdamW_Cosine"
    config: dict
    results: List[GridSearchResult]
    
    def get_heatmap_data() -> (lr_mults, wd_mults, 2D_array)
    def plot_heatmap(ax, title) -> matplotlib axis
    def save(filepath)        # JSON serialization
```

### Heatmap Visualization

Replicate Figure 2 from AdamW paper:
- X-axis: Weight decay multiplier (log scale)
- Y-axis: Learning rate multiplier (log scale)
- Color: Final test loss (jet colormap, lower=better)
- Markers: Circle top-10 best configurations

---

## Infrastructure & Execution

### Development Workflow

| Phase | Environment | Purpose |
|-------|-------------|---------|
| Development | Local (Mac) | Code, debug, single runs |
| Grid Search | Google Colab (T4 GPU) | Run 144 configs per optimizer |
| Analysis | Local | Generate plots, prepare presentation |

### Colab Setup

```python
# Mount Google Drive for persistence
from google.colab import drive
drive.mount('/content/drive')

# Results directory
RESULTS_DIR = '/content/drive/MyDrive/opt_project/results/'

# Save after each optimizer completes
experiment.save(f'{RESULTS_DIR}{optimizer_id}_{optimizer_name}.json')
```

### Estimated Runtime

| Platform | Time/run | Total (2160 runs) | Cost |
|----------|----------|-------------------|------|
| Local Mac (CPU) | ~5 min | ~180 hrs | Free |
| Colab Free (T4) | ~1.5 min | ~54 hrs | Free |
| Colab Pro (V100) | ~1 min | ~36 hrs | ~$10 |

**Strategy**: Run in batches on Colab Free, save to Drive between sessions.

### Batch Execution Plan

```
Session 1: A1, A2 (baselines)           - 288 runs, ~7 hrs
Session 2: B1, B2, B3, B4               - 576 runs, ~14 hrs  
Session 3: B5, B6, B7, B8               - 576 runs, ~14 hrs
Session 4: C1, C2, D1, D2, D3           - 720 runs, ~18 hrs
```

---

## File Structure

```
Project/
├── project_opt_25.ipynb      # Main notebook (run on Colab)
├── test_plan.md              # This document
├── src/
│   ├── optimizers/
│   │   ├── base.py           # BaseOptimizer, OptimizerConfig ✅
│   │   ├── schedulers.py     # LR schedulers (Fixed, Cosine, StepDrop, WarmRestarts) ✅
│   │   ├── adam.py           # Base Adam ✅
│   │   ├── adam_l2.py        # Adam+L2 (A1) ✅
│   │   ├── adamw.py          # AdamW (B1-B8) ✅
│   │   └── adafactor.py      # Adafactor (C1-C2, D1-D3) ⏳ Omer
│   └── utils/
│       ├── logging.py        # TrainingLog ✅
│       └── experiment.py     # GridSearchResult, OptimizerExperiment ✅
└── results/                  # JSON outputs from grid search
```

---

## For Omer: Implementing Adafactor

1. Create `src/optimizers/adafactor.py` - extend `BaseOptimizer` from `base.py`
2. Add C1, C2, D1-D3 to `create_optimizer()` in notebook
3. Test with: `python3 -c "from src.optimizers.adafactor import Adafactor; print('OK')"`

---
