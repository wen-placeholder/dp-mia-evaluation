# Membership Inference Attacks on Differentially Private Image Classifiers

A course project evaluating how differential privacy mitigates membership inference attacks (MIA), and comparing two DP training mechanisms — **DP-SGD** and **DP-Adam** — across privacy budgets, model architectures, and datasets.

## Key Findings

- Both DP mechanisms effectively reduce MIA success (AUC drops from ~0.61 to ~0.50 on CIFAR-10)
- **DP-SGD outperforms DP-Adam** on simpler tasks (Fashion-MNIST), with accuracy gaps up to 32% at the same ε
- **DP-Adam outperforms DP-SGD** on harder tasks (CIFAR-10) with deeper models, where adaptive learning rates better handle noisy gradients
- Optimizer choice affects utility significantly, but has no consistent effect on privacy protection

## Experiment Design

| Dimension | Values |
|-----------|--------|
| Training modes | Non-private (Adam), DP-SGD (Opacus), DP-Adam (Opacus) |
| Privacy budget ε | 1.0, 2.0, 4.0, 8.0 |
| Datasets | Fashion-MNIST, CIFAR-10 |
| Architectures | SmallCNN (~14k params), DeepCNN (~375k params) |
| Random seeds | 3 (results reported as mean across seeds) |

## Attacks Evaluated

- **Black-box**: loss-based, confidence-based
- **White-box**: gradient-norm-based
- **Combined**: 5-fold cross-validated logistic regression over all three signals

## Results

Results are in `results/metrics.csv`. Key columns: `test_accuracy`, `train_test_gap`, `mia_combined_auc`, `mia_combined_tpr_at_1pct_fpr`.

Plots are in `results/plots/{fmnist,cifar10}/`:
- `epsilon_tradeoff.png` — accuracy and MIA AUC vs ε, comparing DP-SGD and DP-Adam
- `utility_vs_leakage.png` — test accuracy vs MIA AUC per configuration
- `overfitting_vs_attack.png` — train-test gap vs MIA AUC scatter

## Reproduce

```bash
pip install -r requirements.txt

# Full experiment (3 seeds, ~34 min)
python src/dp_mia_experiment.py

# Fast sanity check (no download needed, ~2 min)
python src/dp_mia_experiment.py --datasets digits --models small_cnn --epochs 5 --seeds 7
```

Key arguments:

```
--datasets      fmnist cifar10 digits     (default: fmnist cifar10)
--models        small_cnn deep_cnn        (default: both)
--epsilons      1.0 2.0 4.0 8.0          (default: all four)
--seeds         7 42 123                  (default: all three)
--epochs        int                       (default: 15)
--clip-norm     float                     (default: 1.0)
--output-dir    path                      (default: results/)
```

## Models

Both architectures use `GroupNorm` (required for Opacus per-sample gradient computation).

| Key | Architecture | Params |
|-----|-------------|--------|
| `small_cnn` | 2 conv blocks + GroupNorm | ~14k |
| `deep_cnn` | 4 conv blocks + GroupNorm | ~375k |

## DP Accounting

Privacy accounting uses Opacus 1.x RDP accountant via `make_private_with_epsilon`. `delta` is set to `1 / target_train_size`. The `actual_epsilon` column in `metrics.csv` records the ε achieved by the accountant.
