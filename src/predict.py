"""Predictions of the final model on new, unlabelled recordings (the hidden data).

Two steps, from the repository root (the folder needs the patient .txt, .hea
and .wav files, in the same format as the CirCor training data):

    python -m src.preprocess --recordings-dir data/hidden --no-labels --cache-dir data/cache_hidden
    python -m src.predict --recordings-dir data/hidden --cache-dir data/cache_hidden

The spectrograms are standardized with the training statistics
(``data/cache``), so ``python -m src.preprocess`` must have been run on the
training data first. Writes ``submission/predictions.csv`` with one row per
patient: ``patient_id,murmur``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import evaluate
from .data import CACHE_DIR, REPO_ROOT, WindowDataset, build_manifest
from .final_evaluation import load_final_ensemble, predict_patients
from .train import HOP_S, WINDOW_S, get_device


def main():
    parser = argparse.ArgumentParser(description="Predict the murmur label of unlabelled patients.")
    parser.add_argument("--recordings-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="cache written by src.preprocess --no-labels for these recordings")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "submission" / "predictions.csv")
    args = parser.parse_args()

    manifest = build_manifest(args.recordings_dir, labels_csv=None)
    dataset = WindowDataset(manifest, window_s=WINDOW_S, hop_s=HOP_S,
                            cache_dir=args.cache_dir, stats_dir=CACHE_DIR)
    final = load_final_ensemble()
    predictions = predict_patients(final, dataset, get_device())
    evaluate.save_submission(predictions, args.out)
    print(f"{len(predictions)} patients -> {args.out}")
    print(predictions["murmur"].value_counts().to_string())


if __name__ == "__main__":
    main()
