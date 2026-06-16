# Membership Inference Attacks on Differentially Private Image Classifiers

A course project studying how privacy budget ε, attack type, and model architecture
affect membership leakage in differentially private image classification.

## What is included

- `src/dp_mia_experiment.py` — main experiment script. Trains image classifiers under
  standard (Adam) and DP-SGD (Opacus) regimes, then evaluates black-box and white-box
  membership inference attacks across architectures and datasets.
- `results/metrics.csv` — generated experiment metrics.
- `results/plots/` — generated figures (per-dataset subdirectories).
- `report/` — final written report.
- `slides/` — final presentation.

## Datasets

| Key | Dataset | Role |
|-----|---------|------|
| `fmnist` | Fashion-MNIST 28×28 | Sanity check |
| `cifar10` | CIFAR-10 32×32 | Main experiment |
| `digits` | sklearn digits 8×8 | Fast local check (no download) |

## Models

| Key | Architecture | Params |
|-----|-------------|--------|
| `small_cnn` | 2 conv blocks, GroupNorm | ~14k |
| `deep_cnn` | 4 conv blocks, GroupNorm | ~375k |

Both use `GroupNorm` throughout (required for Opacus per-sample gradient computation).

## Attacks evaluated

- **Black-box**: loss-based, confidence-based
- **White-box**: gradient-norm-based
- **Combined**: 5-fold cross-validated logistic regression over all three signals

## Metrics

`test_accuracy`, `train_test_gap`, `mia_{loss,confidence,gradient,combined}_auc`,
`mia_combined_tpr_at_1pct_fpr`, `mia_combined_tpr_at_10pct_fpr`, `runtime_seconds`

## Reproduce

```bash
pip install torch torchvision opacus scikit-learn pandas matplotlib

# Full experiment: Fashion-MNIST + CIFAR-10, both architectures
python src/dp_mia_experiment.py --datasets fmnist cifar10 --models small_cnn deep_cnn

# Fast sanity check (no download needed)
python src/dp_mia_experiment.py --datasets digits --models small_cnn --epochs 10
```

Key arguments:

```
--datasets      fmnist cifar10 digits   (default: fmnist cifar10)
--models        small_cnn deep_cnn      (default: both)
--epsilons      1.0 2.0 4.0 8.0        (default: all four)
--epochs        int                     (default: 15)
--target-train-size  int               (default: 800)
--output-dir    path                   (default: results/)
--data-dir      path                   (default: data/)
```

## DP accounting

Privacy accounting uses Opacus 1.x RDP accountant via `make_private_with_epsilon`.
`delta` is set to `1 / target_train_size`. The `actual_epsilon` column in `metrics.csv`
records the ε achieved by the RDP accountant, which may differ slightly from the target.
