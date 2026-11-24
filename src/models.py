"""Network architectures for murmur classification from log-mel windows.

Every model receives a batch of windows shaped (batch, 64 bands, 400 frames),
passed through the model's ``prepare`` function, and returns 3 logits in the
order of ``data.CLASSES`` (Present, Unknown, Absent).

Architectures:

- ``CNN2D``: the window as a 1-channel image; 3x3 filters slide over
  frequency and time.
- ``CNN1D``: the 64 bands as channels; filters slide over time only.
- ``CRNN``: the CNN1D blocks followed by a bidirectional GRU that reads the
  sequence of CNN features in order.
- ``resnet18``: ResNet18 pretrained on ImageNet, with a new 3-class output
  layer and optionally frozen early layers.

Use ``build_model(name)`` to get a model together with its ``prepare``
function.
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Input formats
# ---------------------------------------------------------------------------

def as_image(windows: torch.Tensor) -> torch.Tensor:
    """(batch, 64, 400) -> (batch, 1, 64, 400): one channel, like a grey image."""
    return windows.unsqueeze(1)


def as_rgb_image(windows: torch.Tensor) -> torch.Tensor:
    """(batch, 64, 400) -> (batch, 3, 64, 400): the channel repeated 3 times,
    because ResNet18 was pretrained on colour images."""
    return windows.unsqueeze(1).repeat(1, 3, 1, 1)


def as_sequence(windows: torch.Tensor) -> torch.Tensor:
    """(batch, 64, 400) unchanged: 64 band channels over 400 time steps."""
    return windows


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class CNN2D(nn.Module):
    """4 blocks of Conv2d 3x3 + BatchNorm + ReLU + MaxPool (16, 32, 64, 128
    filters), global average pooling, dropout and a linear layer.
    98,019 parameters."""

    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        def block(n_in, n_out):
            return nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(n_out),
                nn.ReLU(),
                nn.MaxPool2d(2),  # halves frequency and time
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)  # one number per filter
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (batch, 1, 64, 400)
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


def _conv1d_block(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(n_in, n_out, kernel_size=5, padding=2),  # 5 frames (50 ms), all bands
        nn.BatchNorm1d(n_out),
        nn.ReLU(),
        nn.MaxPool1d(2),  # halves time
    )


class CNN1D(nn.Module):
    """4 blocks of Conv1d (kernel 5) + BatchNorm + ReLU + MaxPool over time
    (64, 128, 128, 256 filters), average over time, dropout and a linear layer.
    309,699 parameters."""

    def __init__(self, n_bands: int = 64, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv1d_block(n_bands, 64),
            _conv1d_block(64, 128),
            _conv1d_block(128, 128),
            _conv1d_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # average over time
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):  # x: (batch, 64 bands, 400 frames)
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


class CRNN(nn.Module):
    """3 Conv1d blocks (time 400 -> 50 steps of 80 ms), then a bidirectional
    GRU over the 50 steps. The final state of each direction goes to dropout
    and a linear layer. 219,203 parameters."""

    def __init__(self, n_bands: int = 64, n_classes: int = 3, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv1d_block(n_bands, 64),
            _conv1d_block(64, 128),
            _conv1d_block(128, 128),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, n_classes)  # 2 directions

    def forward(self, x):  # x: (batch, 64 bands, 400 frames)
        x = self.cnn(x)  # (batch, 128, 50)
        x = x.transpose(1, 2)  # (batch, 50, 128): the GRU expects time in the middle
        _, h = self.gru(x)  # h: (2 directions, batch, hidden)
        x = torch.cat([h[0], h[1]], dim=1)  # (batch, 2 * hidden)
        return self.classifier(self.dropout(x))


def resnet18(n_classes: int = 3, dropout: float = 0.3, freeze: int = 0, pretrained: bool = True) -> nn.Module:
    """ResNet18 with a new output layer for 3 classes. 11,178,051 parameters.

    Args:
        freeze: how many of the 4 layer groups (layer1..layer4) stay fixed,
            counting from the input. 0 trains everything; 2 fixes the stem,
            layer1 and layer2.
        pretrained: start from the ImageNet weights (downloaded once by
            torchvision). Use False when loading our own saved weights.
    """
    from torchvision.models import ResNet18_Weights
    from torchvision.models import resnet18 as tv_resnet18

    net = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, n_classes))
    if freeze > 0:
        frozen = [net.conv1, net.bn1] + [getattr(net, f"layer{i}") for i in range(1, freeze + 1)]
        for layer in frozen:
            for p in layer.parameters():
                p.requires_grad = False  # the optimizer leaves these weights unchanged
    return net


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "cnn2d": (CNN2D, as_image),
    "cnn1d": (CNN1D, as_sequence),
    "crnn": (CRNN, as_sequence),
    "resnet18": (resnet18, as_rgb_image),
}


def build_model(name: str, **kwargs):
    """Return (model, prepare) for an architecture in ``ARCHITECTURES``.

    ``kwargs`` go to the model constructor (e.g. ``freeze=2`` for resnet18).
    """
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; choose from {list(ARCHITECTURES)}")
    constructor, prepare = ARCHITECTURES[name]
    return constructor(**kwargs), prepare


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
