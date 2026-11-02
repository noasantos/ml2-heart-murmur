"""Loading recordings and labels, patient-level splits, windowed dataset.

Main entry points:

- ``build_manifest``: one row per recording, with the patient label and the
  recording-level training label.
- ``make_patient_split`` / ``load_split``: stratified train/val/test split by
  patient, saved to ``splits/patient_split.csv`` so the whole group uses the
  same split.
- ``WindowDataset``: fixed-length windows of the cached log-mel spectrograms
  (see ``src/preprocess.py``), ready for a PyTorch ``DataLoader``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "circor"
RECORDINGS_DIR = DATA_DIR / "training_data"
LABELS_CSV = DATA_DIR / "training_data.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
SPLIT_PATH = REPO_ROOT / "splits" / "patient_split.csv"

# Order used for class indices everywhere (models, metrics, confusion matrix).
CLASSES = ("Present", "Unknown", "Absent")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Segmentation labels in the .tsv files.
SEG_LABELS = {0: "unannotated", 1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


# ---------------------------------------------------------------------------
# Patients and recordings
# ---------------------------------------------------------------------------

def load_patients(labels_csv: Path = LABELS_CSV) -> pd.DataFrame:
    """Patient table with the columns we use, one row per patient."""
    df = pd.read_csv(labels_csv)
    out = pd.DataFrame({
        "patient_id": df["Patient ID"].astype(int),
        "murmur": df["Murmur"],
        "murmur_locations": df["Murmur locations"].fillna(""),
        "outcome": df["Outcome"],
    })
    return out


def _read_patient_txt(txt_path: Path) -> tuple[int, list[tuple[str, str]]]:
    """Return (patient_id, [(location, recording_id), ...]) from a patient .txt.

    The first line is ``<patient_id> <n_recordings> <fs>`` and the next
    ``n_recordings`` lines are ``<location> <hea> <wav> <tsv>``.
    """
    lines = txt_path.read_text().splitlines()
    pid, n_rec, _fs = lines[0].split()
    recs = []
    for line in lines[1:1 + int(n_rec)]:
        location, _hea, wav, *_ = line.split()
        recs.append((location, Path(wav).stem))
    return int(pid), recs


def _read_header(hea_path: Path) -> tuple[int, int]:
    """Return (sampling rate, number of samples) from a WFDB .hea file."""
    _name, _n_sig, fs, n_samples = hea_path.read_text().split("\n", 1)[0].split()[:4]
    return int(fs), int(n_samples)


def load_segmentation(tsv_path: Path) -> np.ndarray:
    """Read a .tsv segmentation file as an array of (start_s, end_s, label).

    Tolerates Windows line endings and replaces labels outside 0-4 (one file
    has a stray ``28``) with 0 (unannotated).
    """
    rows = []
    for line in Path(tsv_path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start, end, label = float(parts[0]), float(parts[1]), int(float(parts[2]))
        if label not in SEG_LABELS:
            label = 0
        rows.append((start, end, label))
    return np.array(rows, dtype=float).reshape(-1, 3)


def annotated_span(seg: np.ndarray) -> tuple[float, float]:
    """First and last second covered by an S1/systole/S2/diastole annotation.

    Returns (nan, nan) when the file has no annotation at all.
    """
    ann = seg[seg[:, 2] > 0] if len(seg) else seg
    if len(ann) == 0:
        return float("nan"), float("nan")
    return float(ann[:, 0].min()), float(ann[:, 1].max())


def recording_label(patient_murmur: str, murmur_locations: str, location: str) -> str:
    """Training label for one recording.

    A Present patient often has a murmur audible only at some locations. A
    recording from a location where it was not heard is labelled Absent, so the
    model is not taught that a normal-sounding recording is a murmur. Unknown
    and Absent patients pass their label to every recording. Evaluation always
    uses the patient label.
    """
    if patient_murmur != "Present":
        return patient_murmur
    audible = set(filter(None, murmur_locations.split("+")))
    return "Present" if location in audible else "Absent"


def build_manifest(recordings_dir: Path = RECORDINGS_DIR,
                   labels_csv: Path | None = LABELS_CSV) -> pd.DataFrame:
    """One row per recording.

    Columns: patient_id, recording_id, location, wav, tsv, sr, duration_s,
    ann_start_s, ann_end_s and, when ``labels_csv`` is given, murmur (patient
    label), murmur_audible and label (recording-level training label).

    With ``labels_csv=None`` it works on unlabelled data (e.g. the hidden set),
    which only needs the patient .txt, .hea and .wav files.
    """
    recordings_dir = Path(recordings_dir)
    rows = []
    for txt in sorted(recordings_dir.glob("*.txt")):
        pid, recs = _read_patient_txt(txt)
        for location, rec_id in recs:
            sr, n_samples = _read_header(recordings_dir / f"{rec_id}.hea")
            tsv = recordings_dir / f"{rec_id}.tsv"
            if tsv.exists():
                ann_start, ann_end = annotated_span(load_segmentation(tsv))
            else:
                tsv, ann_start, ann_end = None, float("nan"), float("nan")
            rows.append({
                "patient_id": pid,
                "recording_id": rec_id,
                "location": location,
                "wav": str(recordings_dir / f"{rec_id}.wav"),
                "tsv": str(tsv) if tsv else None,
                "sr": sr,
                "duration_s": n_samples / sr,
                "ann_start_s": ann_start,
                "ann_end_s": ann_end,
            })
    manifest = pd.DataFrame(rows)

    if labels_csv is not None:
        patients = load_patients(labels_csv)
        manifest = manifest.merge(patients[["patient_id", "murmur", "murmur_locations"]],
                                  on="patient_id", how="left", validate="many_to_one")
        manifest["murmur_audible"] = [
            loc in set(ml.split("+")) for loc, ml in
            zip(manifest["location"], manifest["murmur_locations"])
        ]
        manifest["label"] = [
            recording_label(m, ml, loc) for m, ml, loc in
            zip(manifest["murmur"], manifest["murmur_locations"], manifest["location"])
        ]
        manifest = manifest.drop(columns="murmur_locations")
    return manifest


# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------

def make_patient_split(patients: pd.DataFrame, val_size: float = 0.15,
                       test_size: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Stratified train/val/test split of patients by the Murmur label.

    Returns a DataFrame with columns patient_id, murmur, split.
    """
    from sklearn.model_selection import train_test_split

    ids, y = patients["patient_id"].to_numpy(), patients["murmur"].to_numpy()
    trainval_ids, test_ids, trainval_y, _ = train_test_split(
        ids, y, test_size=test_size, stratify=y, random_state=seed)
    train_ids, val_ids = train_test_split(
        trainval_ids, test_size=val_size / (1 - test_size), stratify=trainval_y,
        random_state=seed)

    split = pd.Series("train", index=ids)
    split[val_ids] = "val"
    split[test_ids] = "test"
    out = patients[["patient_id", "murmur"]].copy()
    out["split"] = split.loc[out["patient_id"]].to_numpy()
    return out.sort_values("patient_id").reset_index(drop=True)


def load_split(path: Path = SPLIT_PATH) -> pd.DataFrame:
    """Load the saved patient split (patient_id, murmur, split)."""
    return pd.read_csv(path)


def get_split_manifest(manifest: pd.DataFrame, split: pd.DataFrame,
                       name: str) -> pd.DataFrame:
    """Recordings of the patients in split ``name`` ("train", "val" or "test")."""
    ids = split.loc[split["split"] == name, "patient_id"]
    return manifest[manifest["patient_id"].isin(ids)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Windowed dataset over cached log-mel spectrograms
# ---------------------------------------------------------------------------

def window_starts(n_frames: int, win: int, hop: int) -> list[int]:
    """Start frames of fixed-length windows covering the whole recording.

    Windows advance by ``hop``; a last window aligned to the end is added so
    the tail of the recording is not lost. A recording shorter than ``win``
    gives a single window starting at 0 (padded later).
    """
    if n_frames <= win:
        return [0]
    starts = list(range(0, n_frames - win + 1, hop))
    if starts[-1] != n_frames - win:
        starts.append(n_frames - win)
    return starts


def load_feature_stats(feature: str = "logmel", band=None,
                       cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean and std of a feature, computed on the training split only.

    ``band`` selects the pre-processing variant (default 25-800 Hz). Always
    read from the training cache, also when the features come from the hidden
    data.
    """
    from .features import BAND, n_features, variant_name

    n_features(feature)  # validates the name
    path = Path(cache_dir) / variant_name(band or BAND) / f"{feature}_stats.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run python -m src.preprocess"
            + (f" --band {band[0]:g} {band[1]:g}" if band else ""))
    stats = np.load(path)
    return stats["mean"], stats["std"]


try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # data loading without torch still works
    torch, Dataset = None, object


class WindowDataset(Dataset):
    """Fixed-length windows of log-mel spectrograms or MFCCs.

    Each item is ``(x, y, rec_idx)``:

    - ``x``: float32 tensor of shape (``self.n_features``, n_frames):
      64 x 400 for log-mel, 40 x 400 for MFCC (20 coefficients + 20 deltas)
      with the default 4 s window. Each row is standardized with the training
      statistics of the same feature and band. For a 2D CNN use
      ``x.unsqueeze(0)`` (channel dimension); for an RNN use ``x.T`` (time
      steps first, ``input_size=self.n_features``).
    - ``y``: class index (see ``CLASSES``) of the recording label, or -1 when
      the manifest has no labels.
    - ``rec_idx``: row of ``self.recordings``, used to group window
      predictions back into recordings and patients.

    Pre-processing variants (compare them on validation with the same model):

    - ``feature``: "logmel" (default) or "mfcc";
    - ``band``: band-pass of the cached variant, e.g. ``(25, 400)``; it must
      have been built with ``python -m src.preprocess --band 25 400``;
    - ``window_s``, ``hop_s``: window length and step in seconds;
    - ``trim_to_annotation``: keep only the part of each recording covered by
      the .tsv segmentation (removes noisy unannotated edges). Only use it for
      training: the hidden data may not have .tsv files.

    Args:
        recordings: manifest rows for one split (``get_split_manifest``).
        label_col: "label" (recording-level) or "murmur" (patient label).
        cache_dir: cache holding the features (``data/cache_hidden`` for the
            hidden data); the statistics always come from ``stats_dir``.
        stats_dir: training cache with the statistics (default ``data/cache``).
    """

    def __init__(self, recordings: pd.DataFrame, feature: str = "logmel", band=None,
                 window_s: float = 4.0, hop_s: float = 2.0,
                 trim_to_annotation: bool = False, label_col: str = "label",
                 cache_dir: Path = CACHE_DIR, stats_dir: Path = CACHE_DIR):
        from .features import BAND, FRAMES_PER_SECOND, compute_feature, n_features, variant_name

        band = tuple(band) if band is not None else BAND
        self.feature, self.band = feature, band
        self.n_features = n_features(feature)
        mean, std = load_feature_stats(feature, band, stats_dir)
        self.mean = mean.astype(np.float32)[:, None]
        self.std = std.astype(np.float32)[:, None]
        self.recordings = recordings.reset_index(drop=True)
        self.win = int(round(window_s * FRAMES_PER_SECOND))
        hop = int(round(hop_s * FRAMES_PER_SECOND))
        logmel_dir = Path(cache_dir) / variant_name(band) / "logmel"

        has_labels = label_col in self.recordings.columns
        self.specs, self.labels, self.index = [], [], []
        for i, rec in self.recordings.iterrows():
            path = logmel_dir / f"{rec['recording_id']}.npy"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run python -m src.preprocess "
                                        f"--band {band[0]:g} {band[1]:g}")
            spec = np.load(path)
            if feature != "logmel":  # computed on the whole recording, like the statistics
                spec = compute_feature(spec, feature).astype(np.float16)
            if trim_to_annotation and not np.isnan(rec["ann_start_s"]):
                a = int(rec["ann_start_s"] * FRAMES_PER_SECOND)
                b = int(np.ceil(rec["ann_end_s"] * FRAMES_PER_SECOND))
                if b - a >= self.win:
                    spec = spec[:, a:b]
            self.specs.append(spec)
            self.labels.append(CLASS_TO_IDX[rec[label_col]] if has_labels else -1)
            self.index.extend((i, s) for s in window_starts(spec.shape[1], self.win, hop))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, start = self.index[k]
        x = self.specs[i][:, start:start + self.win].astype(np.float32)
        x = (x - self.mean) / self.std
        if x.shape[1] < self.win:  # pad short recordings with the mean (0)
            x = np.pad(x, ((0, 0), (0, self.win - x.shape[1])))
        return torch.from_numpy(x), self.labels[i], i

    def window_recordings(self) -> np.ndarray:
        """Recording index of every window (same order as the dataset)."""
        return np.array([i for i, _ in self.index])
