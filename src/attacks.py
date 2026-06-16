"""Membership inference attack implementations and evaluation.

Three attack signals are computed:
  - loss-based   (black-box): members have lower cross-entropy loss
  - confidence   (black-box): members have higher max-softmax confidence
  - gradient norm (white-box): members have smaller per-sample gradient norms

A logistic regression meta-classifier combining all three signals is evaluated
with 5-fold cross-validation to avoid in-sample AUC inflation.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset


@torch.no_grad()
def predict_stats(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> dict[str, np.ndarray]:
    """Compute per-sample loss, max confidence, and accuracy."""
    model.eval()
    logits = model(x)
    probs  = F.softmax(logits, dim=1)
    return {
        "loss":       F.cross_entropy(logits, y, reduction="none").cpu().numpy(),
        "confidence": probs.max(dim=1).values.cpu().numpy(),
        "correct":    (probs.argmax(dim=1) == y).float().cpu().numpy(),
    }


def gradient_norms(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> np.ndarray:
    """Per-sample gradient ℓ₂ norm (white-box signal).

    Members tend to have smaller gradient norms because the model has already
    fitted them; non-members produce larger loss gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    norms: list[float] = []
    model.eval()
    for xi, yi in zip(x, y):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xi.unsqueeze(0)), yi.unsqueeze(0)).backward()
        norm = torch.sqrt(sum(
            (p.grad.detach() ** 2).sum() for p in params if p.grad is not None
        ))
        norms.append(norm.item())
    model.zero_grad(set_to_none=True)
    return np.array(norms)


def tpr_at_fpr(
    labels: np.ndarray, scores: np.ndarray, max_fpr: float
) -> float:
    """TPR achieved at the highest threshold where FPR ≤ max_fpr."""
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= max_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def evaluate_run(
    model: nn.Module,
    train_ds: TensorDataset,
    member_audit: tuple[torch.Tensor, torch.Tensor],
    nonmember_audit: tuple[torch.Tensor, torch.Tensor],
    test_ds: TensorDataset,
) -> dict[str, float]:
    """Compute utility metrics and all MIA attack scores for one trained model."""
    train_x, train_y = train_ds.tensors
    test_x,  test_y  = test_ds.tensors
    train_s = predict_stats(model, train_x, train_y)
    test_s  = predict_stats(model, test_x,  test_y)

    mem_x, mem_y = member_audit
    non_x, non_y = nonmember_audit
    mem_s = predict_stats(model, mem_x, mem_y)
    non_s = predict_stats(model, non_x, non_y)
    mem_g = gradient_norms(model, mem_x, mem_y)
    non_g = gradient_norms(model, non_x, non_y)

    labels      = np.concatenate([np.ones(len(mem_y)),   np.zeros(len(non_y))])
    loss_scores = -np.concatenate([mem_s["loss"],         non_s["loss"]])   # lower loss → member
    conf_scores =  np.concatenate([mem_s["confidence"],   non_s["confidence"]])
    grad_scores = -np.concatenate([mem_g,                 non_g])           # lower norm → member

    # Combined attack: 5-fold CV prevents in-sample AUC inflation
    feats = StandardScaler().fit_transform(np.column_stack([loss_scores, conf_scores, grad_scores]))
    combo = cross_val_predict(
        LogisticRegression(random_state=0, max_iter=1000),
        feats, labels, cv=5, method="predict_proba",
    )[:, 1]

    return {
        "train_accuracy":                float(train_s["correct"].mean()),
        "test_accuracy":                 float(test_s["correct"].mean()),
        "train_test_gap":                float(train_s["correct"].mean() - test_s["correct"].mean()),
        "member_mean_loss":              float(mem_s["loss"].mean()),
        "nonmember_mean_loss":           float(non_s["loss"].mean()),
        "mia_loss_auc":                  float(roc_auc_score(labels, loss_scores)),
        "mia_confidence_auc":            float(roc_auc_score(labels, conf_scores)),
        "mia_gradient_auc":              float(roc_auc_score(labels, grad_scores)),
        "mia_combined_auc":              float(roc_auc_score(labels, combo)),
        "mia_combined_tpr_at_1pct_fpr":  tpr_at_fpr(labels, combo, 0.01),
        "mia_combined_tpr_at_10pct_fpr": tpr_at_fpr(labels, combo, 0.10),
    }
