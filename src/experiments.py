"""All CNN experiments in one place.

- ``EXPERIMENTS``: 8 training experiments. Each one is trained with the 3
  seeds in ``SEEDS`` and compared on validation by mean and std.
- ``ENSEMBLES``: 5 ensembles, which average the predictions of already
  trained models (see ``src/final_evaluation.py``).

Every experiment except the baseline uses SpecAugment. All share the same
pre-processing (log-mel, 64 bands, 4 s windows with a 2 s step), class-weighted
cross-entropy, Adam with weight decay 1e-4, batch 64, at most 30 epochs and
early stopping with patience 6 on the patient-level validation WA.

Command line::

    python -m src.experiments --list
    python -m src.experiments crnn_specaug                  # seeds 42, 1, 2
    python -m src.experiments crnn_specaug --seeds 42

Each run saves ``models/<experiment>_seed<seed>.pt`` (weights and validation
thresholds) with its per-epoch history, and adds a row to
``results/experiments.csv``.
"""

from __future__ import annotations

import argparse

import torch

from .models import build_model, count_parameters
from .train import (MODELS_DIR, RESULTS_DIR, make_datasets, record_result, save_checkpoint,
                    spec_augment, train)

SEEDS = [42, 1, 2]

EXPERIMENTS = {
    "cnn_base": {
        "architecture": "cnn2d", "augment": False, "lr": 1e-3,
        "description": "2D CNN baseline, no augmentation",
    },
    "cnn_specaug": {
        "architecture": "cnn2d", "augment": True, "lr": 1e-3,
        "description": "2D CNN with SpecAugment",
    },
    "resnet18_specaug": {
        "architecture": "resnet18", "augment": True, "lr": 1e-4,
        "description": "ResNet18 pretrained on ImageNet, all layers trained",
    },
    "resnet18_specaug_lr3e-5": {
        "architecture": "resnet18", "augment": True, "lr": 3e-5,
        "description": "ResNet18 pretrained, lower learning rate",
    },
    "resnet18_freeze2_specaug": {
        "architecture": "resnet18", "model_kwargs": {"freeze": 2}, "augment": True, "lr": 1e-4,
        "description": "ResNet18 pretrained, stem + layer1-2 frozen",
    },
    "resnet18_freeze3_specaug": {
        "architecture": "resnet18", "model_kwargs": {"freeze": 3}, "augment": True, "lr": 1e-4,
        "description": "ResNet18 pretrained, stem + layer1-3 frozen",
    },
    "cnn1d_specaug": {
        "architecture": "cnn1d", "augment": True, "lr": 1e-3,
        "description": "1D CNN (bands as channels, filters over time)",
    },
    "crnn_specaug": {
        "architecture": "crnn", "augment": True, "lr": 1e-3, "clip": 1.0,
        "description": "CRNN: 3 Conv1d blocks + bidirectional GRU",
    },
}

# ensemble name -> experiments whose 3 seeds are averaged
ENSEMBLES = {
    "3 CRNN": ["crnn_specaug"],
    "3 ResNet18": ["resnet18_specaug"],
    "CRNN + ResNet18": ["crnn_specaug", "resnet18_specaug"],
    "CRNN + ResNet18 + 1D CNN": ["crnn_specaug", "resnet18_specaug", "cnn1d_specaug"],
    "all 4 models": ["crnn_specaug", "resnet18_specaug", "cnn1d_specaug", "cnn_specaug"],
}

COMMON = {"weight_decay": 1e-4, "epochs": 30, "patience": 6, "batch_size": 64}


def run_experiment(name: str, seed: int, train_ds=None, val_ds=None, device=None, verbose: bool = True):
    """Train experiment ``name`` with one seed, save the model and record the result.

    Pass ``train_ds`` and ``val_ds`` to reuse datasets across runs (loading
    them takes a while). Returns (model, best, history).
    """
    exp = EXPERIMENTS[name]
    if train_ds is None or val_ds is None:
        train_ds, val_ds = make_datasets("train", "val")

    torch.manual_seed(seed)
    model, prepare = build_model(exp["architecture"], **exp.get("model_kwargs", {}))
    model, best, history = train(
        model, prepare, train_ds, val_ds,
        augment=spec_augment if exp["augment"] else None,
        lr=exp["lr"], clip=exp.get("clip"), device=device, verbose=verbose, **COMMON)

    run = f"{name}_seed{seed}"
    config = {"experiment": name, "seed": seed, "architecture": exp["architecture"],
              "model_kwargs": exp.get("model_kwargs", {}), "augment": exp["augment"],
              "lr": exp["lr"], "clip": exp.get("clip"), **COMMON}
    save_checkpoint(MODELS_DIR / f"{run}.pt", model, best, config, history)
    record_result({
        "experiment": name, "seed": seed, "description": exp["description"],
        "architecture": exp["architecture"], "freeze": exp.get("model_kwargs", {}).get("freeze", 0),
        "augment": exp["augment"], "lr": exp["lr"], "clip": exp.get("clip"),
        "parameters": count_parameters(model), "trainable_parameters": count_parameters(model, True),
        "best_val_wa": best["wa"], "best_epoch": best["epoch"],
        "t_present": best["t_p"], "t_unknown": best["t_u"],
        "seconds_per_epoch": round(history["seconds"].mean(), 1),
    }, RESULTS_DIR / "experiments.csv")
    if verbose:
        print(f"{run}: best val WA {best['wa']:.4f} at epoch {best['epoch']}")
    return model, best, history


def summary(path=RESULTS_DIR / "experiments.csv"):
    """Mean and std of the best validation WA over seeds, per experiment."""
    import pandas as pd

    table = pd.read_csv(path)
    return (table.groupby("experiment")["best_val_wa"].agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False).round(4))


def main():
    parser = argparse.ArgumentParser(description="Train the CNN experiments.")
    parser.add_argument("experiments", nargs="*", help="names from EXPERIMENTS, or 'all'")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--list", action="store_true", help="list the experiments and exit")
    args = parser.parse_args()

    if args.list or not args.experiments:
        for name, exp in EXPERIMENTS.items():
            print(f"{name:26s} {exp['description']}")
        return
    names = list(EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiments: {sorted(unknown)}")

    train_ds, val_ds = make_datasets("train", "val")
    for name in names:
        for seed in args.seeds:
            run_experiment(name, seed, train_ds, val_ds)
    print(summary())


if __name__ == "__main__":
    main()
