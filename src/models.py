"""CNN architectures for the DP-MIA experiment.

All models use GroupNorm instead of BatchNorm — a requirement for Opacus
per-sample gradient computation. AdaptiveAvgPool2d makes each architecture
input-size agnostic, so the same class works for digits (8px), FMNIST (28px),
and CIFAR-10 (32px).
"""

from __future__ import annotations

from torch import nn
import torch


DATASET_META: dict[str, dict] = {
    "digits":  {"in_channels": 1, "num_classes": 10},
    "fmnist":  {"in_channels": 1, "num_classes": 10},
    "cifar10": {"in_channels": 3, "num_classes": 10},
}


class SmallCNN(nn.Module):
    """Shallow two-block CNN (~14k parameters)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """Deeper four-block CNN with higher channel capacity (~375k parameters)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def make_model(arch: str, dataset: str) -> nn.Module:
    """Instantiate a model by architecture name and dataset key."""
    meta = DATASET_META[dataset]
    cls = {"small_cnn": SmallCNN, "deep_cnn": DeepCNN}[arch]
    return cls(in_channels=meta["in_channels"], num_classes=meta["num_classes"])
