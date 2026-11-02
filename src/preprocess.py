"""Build the cached inputs used by both models.

Run from the repository root:

    python -m src.preprocess                    # default band, 25-800 Hz
    python -m src.preprocess --band 25 400      # another variant, cached apart

It writes (all under ``data/``, not versioned):

- ``data/cache/manifest.csv``: one row per recording (see ``build_manifest``);
- ``data/cache/<variant>/logmel/<recording_id>.npy``: float16 log-mel of each
  recording, ``<variant>`` being ``band25-800`` etc.;
- ``data/cache/<variant>/logmel_stats.npz`` and ``mfcc_stats.npz``: per-row
  mean/std over the training split, used to standardize the features;

and, only if it does not exist yet, the shared split ``splits/patient_split.csv``.
Existing spectrograms are kept unless ``--force`` is given.

For the hidden data (no labels), point it to that folder and a separate cache
(the statistics then come from the training cache of the same variant):

    python -m src.preprocess --recordings-dir data/hidden --no-labels \\
        --cache-dir data/cache_hidden
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .data import (CACHE_DIR, LABELS_CSV, RECORDINGS_DIR, SPLIT_PATH, build_manifest,
                   get_split_manifest, load_patients, load_split, make_patient_split)
from .features import BAND, FEATURES, compute_feature, recording_logmel, variant_name


def _cache_one(wav: str, out: Path, band, force: bool) -> None:
    if out.exists() and not force:
        return
    np.save(out, recording_logmel(wav, band).astype(np.float16))


def train_stats(recording_ids, logmel_dir: Path,
                feature: str = "logmel") -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean and std of ``feature`` over every frame of the given recordings."""
    total = total_sq = None
    n = 0
    for rec_id in recording_ids:
        x = compute_feature(np.load(logmel_dir / f"{rec_id}.npy"), feature).astype(np.float64)
        if total is None:
            total, total_sq = np.zeros(len(x)), np.zeros(len(x))
        total += x.sum(axis=1)
        total_sq += (x ** 2).sum(axis=1)
        n += x.shape[1]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"), default=list(BAND),
                   help="band-pass cut-offs in Hz (default: %(default)s)")
    p.add_argument("--recordings-dir", type=Path, default=RECORDINGS_DIR)
    p.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    p.add_argument("--no-labels", action="store_true",
                   help="unlabelled data: no split and no statistics")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--force", action="store_true", help="recompute existing spectrograms")
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    args = p.parse_args()
    band = tuple(args.band)
    variant = variant_name(band)

    labels_csv = None if args.no_labels else args.labels_csv
    manifest = build_manifest(args.recordings_dir, labels_csv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.cache_dir / "manifest.csv", index=False)
    print(f"{len(manifest)} recordings from {manifest['patient_id'].nunique()} patients")

    variant_dir = args.cache_dir / variant
    logmel_dir = variant_dir / "logmel"
    logmel_dir.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=args.jobs)(
        delayed(_cache_one)(wav, logmel_dir / f"{rec_id}.npy", band, args.force)
        for wav, rec_id in tqdm(list(zip(manifest["wav"], manifest["recording_id"])),
                                desc=f"log-mel {variant}"))

    if labels_csv is None:
        return

    if SPLIT_PATH.exists():
        split = load_split()
        print(f"Using existing split {SPLIT_PATH.relative_to(SPLIT_PATH.parents[1])}")
    else:
        split = make_patient_split(load_patients(labels_csv))
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(SPLIT_PATH, index=False)
        print(f"Created split {SPLIT_PATH.relative_to(SPLIT_PATH.parents[1])}")
    missing = set(manifest["patient_id"]) - set(split["patient_id"])
    if missing:
        raise SystemExit(f"{len(missing)} patients are not in the split file")

    print(split.groupby(["split", "murmur"]).size().unstack(fill_value=0)
          .reindex(["train", "val", "test"])[["Present", "Unknown", "Absent"]])

    train = get_split_manifest(manifest, split, "train")
    for feature in FEATURES:
        mean, std = train_stats(train["recording_id"], logmel_dir, feature)
        np.savez(variant_dir / f"{feature}_stats.npz", mean=mean, std=std)
    print(f"Saved {', '.join(FEATURES)} training statistics of {variant} "
          f"from {len(train)} recordings")


if __name__ == "__main__":
    main()
