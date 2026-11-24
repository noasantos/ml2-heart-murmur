"""Figures of the CNN experiments.

- ``plot_training``: training loss and validation WA per epoch of one run.
- ``plot_results``: validation WA of every experiment (mean of 3 seeds) and
  of every ensemble, as horizontal bars.
- ``plot_spec_augment``: one window before and after SpecAugment.

The patient-level confusion matrix is ``evaluate.plot_confusion_matrix``.

Command line, to redraw the figures from the saved tables::

    python -m src.plots

reads ``results/experiments.csv``, ``results/ensembles.csv`` and the per-epoch
histories in ``results/histories/``, and writes to ``results/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from .train import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
ALWAYS_PRESENT_VAL_WA = 0.493  # trivial classifier on our validation split
BLUE, ORANGE = "#2a78d6", "#eb6834"

EXPERIMENT_LABELS = {
    "cnn_base": "2D CNN (baseline)",
    "cnn_specaug": "2D CNN + augmentation",
    "resnet18_specaug": "ResNet18 + augmentation",
    "resnet18_specaug_lr3e-5": "ResNet18 + augmentation, lower lr",
    "resnet18_freeze2_specaug": "ResNet18 + augmentation, 2 layer groups frozen",
    "resnet18_freeze3_specaug": "ResNet18 + augmentation, 3 layer groups frozen",
    "cnn1d_specaug": "1D CNN + augmentation",
    "crnn_specaug": "CRNN (1D CNN + GRU) + augmentation",
}
ENSEMBLE_ORDER = ["3 ResNet18", "CRNN + ResNet18", "CRNN + ResNet18 + 1D CNN", "all 4 models", "3 CRNN"]


def _save(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)


def plot_training(history: pd.DataFrame, best_epoch: int, title: str, path=None):
    """Training loss (left) and patient-level validation WA (right) per epoch."""
    fig, (ax_loss, ax_wa) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax_loss.plot(history["epoch"], history["train_loss"], "o-")
    ax_loss.set_title("training loss")
    ax_wa.plot(history["epoch"], history["val_wa"], "o-", color="#d6452f")
    ax_wa.axvline(best_epoch, color="green", ls=":", label=f"best epoch ({best_epoch})")
    ax_wa.axhline(ALWAYS_PRESENT_VAL_WA, color="gray", ls="--",
                  label=f"always Present ({ALWAYS_PRESENT_VAL_WA})")
    ax_wa.set_title("validation weighted accuracy")
    ax_wa.legend(fontsize=8)
    for ax in (ax_loss, ax_wa):
        ax.set_xlabel("epoch")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_results(experiments: pd.DataFrame, ensembles: pd.DataFrame, path=None):
    """Validation WA: single models (mean of the seeds, blue) then ensembles (orange).

    ``experiments`` has one row per run (columns experiment, best_val_wa);
    ``ensembles`` one row per ensemble (columns ensemble, val_wa).
    """
    means = experiments.groupby("experiment")["best_val_wa"].mean()
    singles = [e for e in EXPERIMENT_LABELS if e in means.index]
    ens = ensembles.set_index("ensemble")["val_wa"]
    ens_names = [e for e in ENSEMBLE_ORDER if e in ens.index]

    names = [EXPERIMENT_LABELS[e] for e in singles] + [f"Ensemble: {e}" for e in ens_names]
    values = [means[e] for e in singles] + [ens[e] for e in ens_names]
    colors = [BLUE] * len(singles) + [ORANGE] * len(ens_names)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names, values, color=colors, height=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
    ax.invert_yaxis()  # first experiment on top
    ax.set_xlim(0, 0.95)
    ax.set_xlabel("weighted accuracy (validation)")
    ax.set_title("Validation results", loc="left", fontsize=14, fontweight="bold", pad=28)
    ax.legend(handles=[Patch(color=BLUE, label="single model (mean of 3 seeds)"),
                       Patch(color=ORANGE, label="ensemble")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_spec_augment(window, augmented, path=None):
    """One window (64 bands x 400 frames) before and after SpecAugment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    for ax, x, title in zip(axes, (window, augmented), ("original window", "after SpecAugment")):
        ax.imshow(x, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("frame (10 ms)")
    axes[0].set_ylabel("mel band")
    fig.tight_layout()
    _save(fig, path)
    return fig


def main():
    experiments = pd.read_csv(RESULTS_DIR / "experiments.csv")
    ensembles = pd.read_csv(RESULTS_DIR / "ensembles.csv")
    plot_results(experiments, ensembles, FIGURES_DIR / "validation_results.png")
    n = 0
    for row in experiments.itertuples():
        run = f"{row.experiment}_seed{row.seed}"
        hist_path = RESULTS_DIR / "histories" / f"{run}.csv"
        if hist_path.exists():
            fig = plot_training(pd.read_csv(hist_path), row.best_epoch,
                                f"{EXPERIMENT_LABELS.get(row.experiment, row.experiment)}, seed {row.seed}",
                                FIGURES_DIR / "training" / f"{run}.png")
            plt.close(fig)
            n += 1
    print(f"figures written to {FIGURES_DIR} ({n} training curves)")


if __name__ == "__main__":
    main()
