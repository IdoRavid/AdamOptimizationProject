# AdamW Optimizer Comparison on Fashion-MNIST

Optimization course project comparing Adam variants on an Autoencoder reconstruction task.

## Papers Implemented
- **AdamW** (Loshchilov & Hutter, 2019) - Decoupled Weight Decay Regularization
- **Adafactor** (Shazeer & Stern, 2018) - Adaptive Learning Rates with Sublinear Memory Cost

## Goal
Generate hyperparameter sensitivity heatmaps (learning rate vs weight decay) to show that AdamW decouples these hyperparameters (rectangular optimal region) vs Adam+L2 (diagonal optimal region).

## Optimizer Variants

| ID    | Optimizer                   | Description                                                    |
|-------|-----------------------------|----------------------------------------------------------------|
| A1    | Adam+L2                     | Baseline - L2 through gradient (coupled)                       |
| B1-B4 | AdamW                       | Decoupled weight decay + Fixed/StepDrop/Cosine/WarmRestarts LR |
| B5-B8 | AdamW+Norm                  | Normalized weight decay + LR schedules                         |
| C1-C2 | Adafactor                   | Factored second moments (C1: β1=0, C2: β1=0.9)                 |
| D1    | Adafactor+Norm Cosine       | Adafactor + normalized decay + Cosine LR (β1=0)                |
| D2    | Adafactor+Norm WarmRestarts | Adafactor + normalized decay + WarmRestarts LR (β1=0)          |
| D3    | Adafactor+Norm WarmRestarts | Adafactor + normalized decay + WarmRestarts LR (β1=0.9)        |
| E1-E4 | Adafactor+Projection        | Adafactor with projected second-moment vectors                 |

## Phase 1 Progress (30 epochs, 144 configs each)

| ID | Variant                              | Owner | User      | Progress    |
|----|--------------------------------------|-------|-----------|-------------|
| A1 | Adam+L2                              | Ido   | dodor25   | 144/144 ✅   |
| B1 | AdamW Fixed                          | Omer  |           | 144/144 ✅   |
| B2 | AdamW StepDrop                       | Ido   | ido@huji  | 144/144 ✅   |
| B3 | AdamW Cosine                         | Ido   | idoravid6 | 144/144 ✅   |
| B4 | AdamW WarmRestarts                   | Ido   | dodor25   | 144/144 ✅   |
| B5 | AdamW+Norm Fixed                     | Ido   | ido@huji  | 144/144 ✅   |
| B6 | AdamW+Norm StepDrop                  | Ido   | idoravid6 | 144/144 ✅   |
| B7 | AdamW+Norm Cosine                    | Omer  | normal    | 144/144 ✅   |
| B8 | AdamW+Norm WarmRestarts              | Omer  | huji      | 144/144 ✅   |
| C1 | Adafactor (no momentum)              | Ido   | dodor25   | 144/144 ✅   |
| C2 | Adafactor (momentum)                 | Ido   | idoravid6 | 144/144 ✅   |
| D1 | Adafactor+Norm Cosine (β1=0)         | Omer  | normal    | 144/144 ✅   |
| D2 | Adafactor+Norm WarmRestarts (β1=0)   | Omer  | huji      | 144/144 ✅   |
| D3 | Adafactor+Norm WarmRestarts (β1=0.9) | Ido   | idoravid6 | 144/144 ✅   |
| E1 | Adafactor+L2Ball (β1=0)              | Omer  |           | 30 epochs ✅ |
| E2 | Adafactor+L2Ball (β1=0.9)            | Omer  |           | 30 epochs ✅ |
| E3 | Adafactor+Box (β1=0)                 | Omer  |           | 30 epochs ✅ |
| E4 | Adafactor+Box (β1=0.9)               | Omer  |           | 30 epochs ✅ |

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IdoRavid/AdamOptimizationProject/blob/main/Omer%26Ido_Optimizing_Fashion.ipynb)

## Project Structure
```
├── Omer&Ido_Optimizing_Fashion.ipynb  # Main notebook
├── src/
│   ├── optimizers/
│   │   ├── base.py          # BaseOptimizer ABC
│   │   ├── schedulers.py    # LR schedulers
│   │   ├── adam.py          # Base Adam
│   │   ├── adam_l2.py       # Adam+L2 (A1)
│   │   └── adamw.py         # AdamW (B1-B8)
│   └── utils/
│       ├── logging.py       # TrainingLog
│       └── experiment.py    # GridSearchResult, heatmap plotting
├── results/                 # JSON outputs from experiments
└── test_plan.md            # Detailed experiment plan
```

## Authors
Ido Ravid & Omer Sutovsky
