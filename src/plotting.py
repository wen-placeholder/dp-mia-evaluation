"""Visualization for experiment results.

Produces three plots per dataset (saved under plots/<dataset>/):
  utility_vs_leakage.png   — test accuracy vs MIA AUC, grouped by architecture
  epsilon_tradeoff.png     — accuracy and MIA AUC as ε varies (DP runs only)
  overfitting_vs_attack.png — train-test gap vs MIA AUC scatter
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _run_label(row) -> str:
    return "Non-private" if pd.isna(row.epsilon) else f"ε={row.epsilon:g}"


def _plot_dataset(df: pd.DataFrame, title: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    archs = sorted(df["model_arch"].unique())

    # ── Utility vs leakage (one panel per architecture) ──────────────────
    fig, axes = plt.subplots(
        1, len(archs), figsize=(6 * len(archs), 5), sharey=True, squeeze=False
    )
    for ax, arch in zip(axes[0], archs):
        sub = df[df["model_arch"] == arch].sort_values(
            ["mode", "epsilon"], na_position="first"
        )
        xs     = np.arange(len(sub))
        colors = ["#2F4858" if pd.isna(e) else "#3E885B" for e in sub["epsilon"]]
        ax.bar(xs - 0.18, sub["test_accuracy"],    width=0.36, color=colors,    label="Test acc")
        ax.bar(xs + 0.18, sub["mia_combined_auc"], width=0.36, color="#C65D3A", label="MIA AUC")
        ax.set_xticks(xs)
        ax.set_xticklabels([_run_label(r) for r in sub.itertuples()], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(arch)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title} — utility and membership leakage", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "utility_vs_leakage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Privacy-utility trade-off across ε (one line per arch × mode) ──────
    dp = df[df["mode"].isin(["dp_sgd", "dp_adam"])].sort_values("epsilon")
    if not dp.empty:
        style = {
            ("small_cnn", "dp_sgd"):  {"marker": "o", "linestyle": "-",  "color": "#2F4858"},
            ("small_cnn", "dp_adam"): {"marker": "o", "linestyle": "--", "color": "#2F4858"},
            ("deep_cnn",  "dp_sgd"):  {"marker": "s", "linestyle": "-",  "color": "#C65D3A"},
            ("deep_cnn",  "dp_adam"): {"marker": "s", "linestyle": "--", "color": "#C65D3A"},
        }
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        for (arch, mode), grp in dp.groupby(["model_arch", "mode"]):
            kw = style.get((arch, mode), {})
            label = f"{arch} / {mode}"
            ax1.plot(grp["epsilon"], grp["test_accuracy"],    label=label, **kw)
            ax2.plot(grp["epsilon"], grp["mia_combined_auc"], label=label, **kw)
        ax1.set(xlabel="Target ε", ylabel="Test accuracy",    title="Accuracy vs ε")
        ax2.set(xlabel="Target ε", ylabel="MIA combined AUC", title="Privacy leakage vs ε")
        for ax in (ax1, ax2):
            ax.legend(fontsize=8)
        fig.suptitle(f"{title} — DP-SGD vs DP-Adam: privacy-utility trade-off")
        fig.tight_layout()
        fig.savefig(out / "epsilon_tradeoff.png", dpi=200)
        plt.close(fig)

    # ── Overfitting vs attack success ─────────────────────────────────────
    markers = {"small_cnn": "o", "deep_cnn": "s"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in df.itertuples():
        ax.scatter(
            row.train_test_gap, row.mia_combined_auc,
            marker=markers.get(row.model_arch, "^"), s=80,
        )
        ax.annotate(
            f"{row.model_arch[:5]} {_run_label(row)}",
            (row.train_test_gap, row.mia_combined_auc),
            xytext=(5, 3), textcoords="offset points", fontsize=7,
        )
    ax.set(
        xlabel="Train-test gap", ylabel="MIA AUC",
        title=f"{title} — overfitting vs attack",
    )
    fig.tight_layout()
    fig.savefig(out / "overfitting_vs_attack.png", dpi=200)
    plt.close(fig)


def _aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average numeric columns across seeds; keep one row per (dataset, model_arch, mode, epsilon)."""
    group_keys = ["dataset", "model_arch", "mode", "epsilon"]
    numeric = df.select_dtypes("number").columns.difference(["seed"])
    agg = df.groupby(group_keys, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(c).rstrip("_") if c[1] else c[0] for c in agg.columns
    ]
    # promote mean columns back to bare names so _plot_dataset still works
    for col in numeric:
        if f"{col}_mean" in agg.columns:
            agg[col] = agg[f"{col}_mean"]
    return agg


def plot_results(df: pd.DataFrame, plot_dir: Path) -> None:
    """Generate all plots, one subdirectory per dataset."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    df_agg = _aggregate_seeds(df)
    for ds in df_agg["dataset"].unique():
        _plot_dataset(df_agg[df_agg["dataset"] == ds].copy(), ds, plot_dir / ds)
