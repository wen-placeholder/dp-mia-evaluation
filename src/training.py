"""Training routines: standard (Adam) and differentially private (Opacus DP-SGD)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_non_private(
    model: nn.Module,
    train_ds: TensorDataset,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Standard training with Adam."""
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


def train_dp_opacus(
    model: nn.Module,
    train_ds: TensorDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    epsilon: float,
    delta: float,
    optimizer_type: str = "sgd",
) -> tuple[nn.Module, float]:
    """DP training via Opacus with RDP accounting (DP-SGD or DP-Adam).

    Returns the unwrapped model and the ε actually achieved by the accountant.
    `drop_last=True` is required so every batch has the same size, which is
    assumed by Opacus when computing the noise multiplier from the target ε.
    """
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    model = ModuleValidator.fix(model)  # no-op for GroupNorm models; guards against BatchNorm
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    if optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    engine = PrivacyEngine()
    model, opt, loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=clip_norm,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

    achieved_eps = engine.get_epsilon(delta)
    model.remove_hooks()  # detach per-sample gradient hooks before returning
    underlying = getattr(model, "_module", model)  # unwrap GradSampleModule
    return underlying, achieved_eps
