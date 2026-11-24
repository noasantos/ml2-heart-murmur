"""Training: data loading, SpecAugment, training loop with early stopping,
checkpoints and the results table.

The loop optimizes a class-weighted cross-entropy on windows, but selects the
best epoch by the patient-level weighted accuracy on validation: window
probabilities are averaged per recording, and the hierarchical thresholds of
``evaluate.tune_thresholds`` turn them into one label per patient.

Typical use (see ``src/experiments.py`` for the full experiments)::

    train_ds, val_ds = make_datasets("train", "val")
    model, prepare = build_model("crnn")
    model, best, history = train(model, prepare, train_ds, val_ds,
                                 augment=spec_augment, clip=1.0)
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import evaluate
from .data import CACHE_DIR, REPO_ROOT, WindowDataset, get_split_manifest, load_split
from .models import build_model

MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
WINDOW_S, HOP_S = 4.0, 2.0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_datasets(*names: str, window_s: float = WINDOW_S, hop_s: float = HOP_S):
    """WindowDatasets of the requested splits ("train", "val", "test"), in order.

    Uses the cache built by ``python -m src.preprocess`` and the saved split.
    """
    manifest = pd.read_csv(CACHE_DIR / "manifest.csv")
    split = load_split()
    datasets = [WindowDataset(get_split_manifest(manifest, split, n), window_s=window_s, hop_s=hop_s)
                for n in names]
    return datasets[0] if len(datasets) == 1 else datasets


def patient_labels() -> pd.Series:
    """True Murmur label of every patient, indexed by patient_id."""
    return load_split().set_index("patient_id")["murmur"]


def class_weights(dataset: WindowDataset) -> np.ndarray:
    """Inverse-frequency weights of the window labels: n / (3 * count per class).
    About [2.22, 7.44, 0.41] (Present, Unknown, Absent) on our training split."""
    labels = np.array([dataset.labels[r] for r in dataset.window_recordings()])
    counts = np.bincount(labels, minlength=3)
    return len(labels) / (3 * counts)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def spec_augment(windows: torch.Tensor, n_freq: int = 2, max_freq: int = 8,
                 n_time: int = 2, max_time: int = 40) -> torch.Tensor:
    """SpecAugment: masks random frequency bands and time steps of each window.

    windows: batch (n, 64 bands, 400 frames), already standardized, so the
    masked value 0 is the training mean. ``max_freq=8`` bands is 1/8 of the
    height; ``max_time=40`` frames is 0.4 s. Applied to training batches only.
    """
    n, F, T = windows.shape
    dev = windows.device

    for _ in range(n_freq):
        width = torch.randint(0, max_freq + 1, (n, 1), device=dev)
        start = (torch.rand(n, 1, device=dev) * (F - width + 1)).long()
        bands = torch.arange(F, device=dev).unsqueeze(0)
        mask = (bands >= start) & (bands < start + width)  # (n, F)
        windows = windows.masked_fill(mask[:, :, None], 0.0)

    for _ in range(n_time):
        width = torch.randint(0, max_time + 1, (n, 1), device=dev)
        start = (torch.rand(n, 1, device=dev) * (T - width + 1)).long()
        steps = torch.arange(T, device=dev).unsqueeze(0)
        mask = (steps >= start) & (steps < start + width)  # (n, T)
        windows = windows.masked_fill(mask[:, None, :], 0.0)

    return windows


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, loss_fn, optimizer, prepare, augment=None, clip=None, device=None) -> float:
    """One pass over the training windows; returns the mean training loss."""
    device = device or get_device()
    model.train()
    total, n = 0.0, 0
    for windows, labels, _ in loader:
        windows, labels = windows.to(device), labels.to(device)
        if augment is not None:  # augmentation: training only
            windows = augment(windows)
        optimizer.zero_grad()
        loss = loss_fn(model(prepare(windows)), labels)
        loss.backward()
        if clip is not None:  # gradient clipping: caps large gradients (useful for RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / n


@torch.no_grad()
def recording_probabilities(model, dataset, prepare) -> np.ndarray:
    """(n_recordings, 3) probabilities: window probabilities averaged per recording."""
    probs, rec = evaluate.window_probabilities(model, dataset, transform=prepare)
    return evaluate.recording_probabilities(probs, rec, len(dataset.recordings))


def validate(model, dataset, prepare, labels: pd.Series | None = None):
    """Patient-level weighted accuracy on ``dataset`` with the thresholds tuned
    on it. Returns (wa, t_present, t_unknown)."""
    labels = patient_labels() if labels is None else labels
    rec_probs = recording_probabilities(model, dataset, prepare)
    t_p, t_u, wa = evaluate.tune_thresholds(rec_probs, dataset.recordings["patient_id"], labels)
    return wa, t_p, t_u


def train(model, prepare, train_ds, val_ds, augment=None, epochs: int = 30, patience: int = 6,
          lr: float = 1e-3, weight_decay: float = 1e-4, clip: float | None = None,
          batch_size: int = 64, weights: np.ndarray | None = None, labels: pd.Series | None = None,
          device=None, verbose: bool = True):
    """Train with Adam and class-weighted cross-entropy; early stopping on the
    patient-level validation WA.

    Returns (model with the best weights, best, history), where ``best`` holds
    wa, epoch, t_p, t_u and the weights of the best epoch, and ``history`` is a
    DataFrame with one row per epoch.
    """
    device = device or get_device()
    # on Windows, worker processes would copy all spectrograms: load in the main process
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0 if os.name == "nt" else 2)
    weights = class_weights(train_ds) if weights is None else weights
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    labels = patient_labels() if labels is None else labels

    history, best, bad_epochs = [], {"wa": -1.0}, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, loader, loss_fn, optimizer, prepare, augment, clip, device)
        wa, t_p, t_u = validate(model, val_ds, prepare, labels)
        seconds = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_wa": wa, "seconds": seconds})
        if verbose:
            print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | val WA {wa:.4f} "
                  f"(thresholds {t_p}, {t_u}) | {seconds:.0f} s")

        if wa > best["wa"]:  # new record: keep a copy
            best = {"wa": wa, "epoch": epoch, "t_p": t_p, "t_u": t_u,
                    "weights": copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stopping: {patience} epochs without improvement "
                          f"(best epoch: {best['epoch']})")
                break

    model.load_state_dict(best["weights"])  # back to the best epoch
    return model, best, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: str | Path, model, best: dict, config: dict, history: pd.DataFrame | None = None):
    """Save the weights, the validation thresholds and the configuration.

    ``config`` must contain "architecture" (a key of ``models.ARCHITECTURES``)
    and may contain "model_kwargs" (e.g. {"freeze": 2}), so that
    ``load_checkpoint`` can rebuild the network.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "t_present": best["t_p"], "t_unknown": best["t_u"],
                "val_wa": best["wa"], "epoch": best["epoch"], "config": config}, path)
    if history is not None:
        history.to_csv(path.with_name(path.stem + "_history.csv"), index=False)


def load_checkpoint(path: str | Path, architecture: str | None = None, device=None, **model_kwargs):
    """Rebuild a saved model. Returns (model in eval mode, prepare, checkpoint).

    Also reads the checkpoints saved by the first (Colab, Portuguese) version
    of this code: keys "pesos", "wa_val", "epoca" and the layer name
    "classificador". For those, pass ``architecture`` (and e.g. ``freeze=2``).
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    if "pesos" in ckpt:  # first version
        ckpt = {"weights": ckpt["pesos"], "t_present": ckpt["t_present"], "t_unknown": ckpt["t_unknown"],
                "val_wa": ckpt["wa_val"], "epoch": ckpt["epoca"], "config": ckpt.get("config", {})}
    config = ckpt["config"]
    architecture = architecture or config.get("architecture")
    if architecture is None:
        raise ValueError(f"{path}: architecture unknown, pass architecture=...")
    kwargs = {**config.get("model_kwargs", {}), **model_kwargs}
    if architecture == "resnet18":
        kwargs["pretrained"] = False  # the saved weights replace the ImageNet ones
    model, prepare = build_model(architecture, **kwargs)
    weights = {k.replace("classificador.", "classifier."): v for k, v in ckpt["weights"].items()}
    model.load_state_dict(weights)
    return model.to(device).eval(), prepare, ckpt


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def record_result(row: dict, path: str | Path = RESULTS_DIR / "experiments.csv") -> pd.DataFrame:
    """Add one run to the results table, replacing an older row of the same
    experiment and seed. Returns the updated table."""
    path = Path(path)
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        same = (table["experiment"] == row["experiment"]) & (table["seed"] == row["seed"])
        table = table[~same]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table
