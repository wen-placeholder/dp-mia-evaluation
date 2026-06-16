"""Membership inference attacks on differentially private image classifiers.

Entry point: orchestrates the experiment sweep and saves results.

Datasets : digits (fast, no download) | fmnist | cifar10
Models   : small_cnn | deep_cnn
Training : non-private (Adam) | DP-SGD via Opacus
Attacks  : loss-based, confidence-based, gradient-norm, combined (5-fold CV LR)

Usage
-----
    python src/dp_mia_experiment.py --datasets fmnist cifar10 --models small_cnn deep_cnn
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import TensorDataset

from attacks import evaluate_run
from data import load_splits, set_seed
from models import DATASET_META, make_model
from plotting import plot_results
from training import train_dp_opacus, train_non_private


@dataclass
class RunConfig:
    mode:           str
    dataset:        str
    model_arch:     str
    epsilon:        float | None   # target ε (None for non-private)
    actual_epsilon: float | None   # achieved ε from Opacus RDP accountant
    clip_norm:      float
    epochs:         int
    batch_size:     int
    learning_rate:  float
    seed:           int


def run_one(
    config: RunConfig,
    target_train_size: int,
    audit_size: int,
    delta: float,
    data_dir: Path,
) -> dict:
    set_seed(config.seed)
    train_pair, member_audit, nonmember_audit, test_pair = load_splits(
        config.dataset, config.seed, target_train_size, audit_size, data_dir,
    )
    train_ds = TensorDataset(*train_pair)
    test_ds  = TensorDataset(*test_pair)
    model    = make_model(config.model_arch, config.dataset)

    t0 = time.time()
    if config.mode == "non_private":
        train_non_private(model, train_ds, config.epochs, config.batch_size, config.learning_rate)
    else:
        optimizer_type = "adam" if config.mode == "dp_adam" else "sgd"
        model, config.actual_epsilon = train_dp_opacus(
            model, train_ds, config.epochs, config.batch_size, config.learning_rate,
            config.clip_norm, config.epsilon, delta, optimizer_type,
        )

    metrics = evaluate_run(model, train_ds, member_audit, nonmember_audit, test_ds)
    metrics.update(asdict(config))
    metrics["runtime_seconds"] = round(time.time() - t0, 2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir",        type=Path,  default=Path("results"))
    parser.add_argument("--data-dir",          type=Path,  default=Path("data"))
    parser.add_argument("--datasets",  nargs="+", default=["fmnist", "cifar10"],
                        choices=list(DATASET_META))
    parser.add_argument("--models",    nargs="+", default=["small_cnn", "deep_cnn"],
                        choices=["small_cnn", "deep_cnn"])
    parser.add_argument("--epsilons",  nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--epochs",            type=int,   default=15)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--target-train-size", type=int,   default=800)
    parser.add_argument("--audit-size",        type=int,   default=200)
    parser.add_argument("--clip-norm",         type=float, default=1.0)
    parser.add_argument("--seeds",   nargs="+", type=int,   default=[7, 42, 123])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    delta = 1.0 / args.target_train_size

    configs: list[RunConfig] = []
    for seed in args.seeds:
        for ds in args.datasets:
            for arch in args.models:
                configs.append(RunConfig(
                    mode="non_private", dataset=ds, model_arch=arch,
                    epsilon=None, actual_epsilon=None, clip_norm=args.clip_norm,
                    epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=1e-3, seed=seed,
                ))
                for eps in args.epsilons:
                    configs.append(RunConfig(
                        mode="dp_sgd", dataset=ds, model_arch=arch,
                        epsilon=eps, actual_epsilon=None, clip_norm=args.clip_norm,
                        epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=0.1, seed=seed,
                    ))
                    configs.append(RunConfig(
                        mode="dp_adam", dataset=ds, model_arch=arch,
                        epsilon=eps, actual_epsilon=None, clip_norm=args.clip_norm,
                        epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=1e-3, seed=seed,
                    ))

    rows: list[dict] = []
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{total}] {cfg.dataset}/{cfg.model_arch}  mode={cfg.mode}  ε={cfg.epsilon}")
        try:
            rows.append(run_one(cfg, args.target_train_size, args.audit_size, delta, args.data_dir))
            last = rows[-1]
            print(
                f"       test_acc={last['test_accuracy']:.3f}  "
                f"mia_auc={last['mia_combined_auc']:.3f}  "
                f"actual_ε={last.get('actual_epsilon', 'N/A')}  "
                f"t={last['runtime_seconds']}s"
            )
        except Exception as exc:
            print(f"       FAILED: {exc}")

    if not rows:
        print("No results — nothing to save.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "datasets": args.datasets,
        "models": args.models,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "target_train_size": args.target_train_size,
        "audit_size": args.audit_size,
        "delta": delta,
        "epsilons": args.epsilons,
        "seeds": args.seeds,
        "dp_accounting": "Opacus 1.x RDP accountant via make_private_with_epsilon",
    }, indent=2))

    plot_results(df, args.output_dir / "plots")

    summary_cols = [
        "dataset", "model_arch", "mode", "epsilon", "actual_epsilon",
        "test_accuracy", "train_test_gap",
        "mia_loss_auc", "mia_confidence_auc", "mia_gradient_auc",
        "mia_combined_auc", "mia_combined_tpr_at_1pct_fpr", "mia_combined_tpr_at_10pct_fpr",
        "runtime_seconds",
    ]
    print("\n" + df[[c for c in summary_cols if c in df.columns]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
