"""Dataset loading and train/audit split construction.

All loaders return four tensor pairs:
    (train, member_audit, nonmember_audit, test)

Member samples are a subset of the training set; non-member samples come from
the same distribution but were never seen by the model.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_pair(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def _make_member_splits(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    target_train_size: int,
    audit_size: int,
) -> tuple:
    """Carve pool → train + nonmember-audit; then train → train + member-audit."""
    x_train, x_nonmember, y_train, y_nonmember = train_test_split(
        x_pool, y_pool,
        train_size=target_train_size,
        test_size=audit_size,
        random_state=seed + 1,
        stratify=y_pool,
    )
    _, x_member, _, y_member = train_test_split(
        x_train, y_train,
        train_size=target_train_size - audit_size,
        test_size=audit_size,
        random_state=seed + 2,
        stratify=y_train,
    )
    return (
        tensor_pair(x_train, y_train),
        tensor_pair(x_member, y_member),
        tensor_pair(x_nonmember, y_nonmember),
        tensor_pair(x_test, y_test),
    )


def load_digits_splits(seed: int, target_train_size: int, audit_size: int) -> tuple:
    """sklearn digits 8×8 — no download required."""
    d = load_digits()
    x = (d.images.astype("float32") / 16.0)[:, None, :, :]
    y = d.target.astype("int64")
    x_pool, x_test, y_pool, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    return _make_member_splits(x_pool, y_pool, x_test, y_test, seed, target_train_size, audit_size)


def load_torchvision_splits(
    dataset_name: str,
    seed: int,
    target_train_size: int,
    audit_size: int,
    data_dir: Path,
) -> tuple:
    """Fashion-MNIST or CIFAR-10, downloaded via torchvision."""
    import torchvision
    import torchvision.transforms as T

    if dataset_name == "fmnist":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        train_tv = torchvision.datasets.FashionMNIST(data_dir, train=True,  download=True, transform=tf)
        test_tv  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    elif dataset_name == "cifar10":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_tv = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf)
        test_tv  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def to_numpy(ds, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        ldr = DataLoader(ds, batch_size=512, shuffle=False)
        xs, ys = [], []
        for xb, yb in ldr:
            xs.append(xb.numpy())
            ys.append(yb.numpy())
        x_all, y_all = np.concatenate(xs), np.concatenate(ys)
        if max_samples and len(x_all) > max_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(x_all), max_samples, replace=False)
            x_all, y_all = x_all[idx], y_all[idx]
        return x_all, y_all

    pool_needed = target_train_size + audit_size * 2 + 500
    x_pool, y_pool = to_numpy(train_tv, max_samples=pool_needed)
    x_test, y_test = to_numpy(test_tv,  max_samples=2000)
    return _make_member_splits(x_pool, y_pool, x_test, y_test, seed, target_train_size, audit_size)


def load_splits(
    dataset_name: str,
    seed: int,
    target_train_size: int,
    audit_size: int,
    data_dir: Path,
) -> tuple:
    """Dispatch to the correct loader by dataset name."""
    if dataset_name == "digits":
        return load_digits_splits(seed, target_train_size, audit_size)
    return load_torchvision_splits(dataset_name, seed, target_train_size, audit_size, data_dir)
