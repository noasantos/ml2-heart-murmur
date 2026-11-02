"""Exploration figures and dataset summary for the slides.

Run from the repository root, after ``python -m src.preprocess``:

    python -m src.eda

Writes to ``results/figures/``:

- ``class_distribution.png``: patients per Murmur class and recordings per
  recording-level label;
- ``recording_durations.png``: histogram of recording durations;
- ``example_recordings.png``: raw signal, filtered signal with the .tsv
  segmentation, and log-mel spectrogram for an Absent and a Present recording;

and ``results/data_summary.csv`` (patients, recordings and 4 s windows per
split and class).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import (CACHE_DIR, CLASSES, LABELS_CSV, REPO_ROOT, get_split_manifest,  # noqa: E402
                   load_feature_stats, load_segmentation, load_split, window_starts)
from .features import FMAX, FMIN, FRAMES_PER_SECOND, SR, load_audio, logmel, preprocess_signal  # noqa: E402

FIG_DIR = REPO_ROOT / "results" / "figures"

# Validated categorical slots 1-3 (light mode) of the dataviz reference palette.
CLASS_COLORS = {"Present": "#2a78d6", "Unknown": "#eb6834", "Absent": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#9cc3ef", "#2a78d6", "#0d2f5c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _bar_panel(ax, counts: pd.Series, title: str, unit: str) -> None:
    total = counts.sum()
    bars = ax.bar(counts.index, counts.values, width=0.6,
                  color=[CLASS_COLORS[c] for c in counts.index],
                  edgecolor=SURFACE, linewidth=2)
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value}  ({value / total:.0%})", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.tick_params(axis="x", length=0)


def class_distribution(manifest: pd.DataFrame, patients: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar_panel(axes[0], patients["murmur"].value_counts().reindex(CLASSES),
               "Patients per Murmur label", "patients")
    _bar_panel(axes[1], manifest["label"].value_counts().reindex(CLASSES),
               "Recordings per training label", "recordings")
    fig.text(0.01, 0.02, "Recordings of Present patients taken at a location where the "
             "murmur was not heard are labelled Absent for training.", color=INK_2,
             fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def recording_durations(manifest: pd.DataFrame) -> None:
    d = manifest["duration_s"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(d, bins=np.arange(0, d.max() + 2, 2), color=CLASS_COLORS["Present"],
            edgecolor=SURFACE, linewidth=1.5)
    med = d.median()
    ax.axvline(med, color=INK_2, linewidth=1.2, linestyle="--")
    ax.annotate(f"median {med:.1f} s", (med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=INK_2)
    ax.axvline(4, color=INK, linewidth=1.2)
    ax.annotate("4 s window", (4, ax.get_ylim()[1] * 0.80), xytext=(6, 0),
                textcoords="offset points", color=INK)
    ax.set_title(f"Recording duration ({len(d)} recordings, {d.min():.1f}-{d.max():.1f} s)")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("recordings")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recording_durations.png", dpi=200)
    plt.close(fig)


def _heart_sound_contrast(rec: pd.Series) -> float:
    """Mean power inside S1/S2 over mean power elsewhere (filtered signal)."""
    x = preprocess_signal(load_audio(rec["wav"])) ** 2
    inside = np.zeros(len(x), dtype=bool)
    annotated = np.zeros(len(x), dtype=bool)
    for s, e, lab in load_segmentation(rec["tsv"]):
        a, b = int(s * SR), int(e * SR)
        annotated[a:b] = lab > 0
        inside[a:b] = lab in (1, 3)
    rest = annotated & ~inside
    return float(x[inside].mean() / x[rest].mean()) if inside.any() and rest.any() else 0.0


def _pick_examples(manifest: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """A loud Present recording at its most audible location, and the Absent
    recording (among fully annotated ones) where S1/S2 stand out the most."""
    raw = pd.read_csv(LABELS_CSV)
    loud = raw[(raw["Murmur"] == "Present") & (raw["Systolic murmur grading"] == "III/VI")]
    present = manifest[manifest["patient_id"].isin(loud["Patient ID"])].merge(
        loud[["Patient ID", "Most audible location"]], left_on="patient_id",
        right_on="Patient ID")
    present = present[present["location"] == present["Most audible location"]]
    absent = manifest[(manifest["murmur"] == "Absent")]
    cov = (absent["ann_end_s"] - absent["ann_start_s"]) / absent["duration_s"]
    absent = absent[cov > 0.9].sort_values("recording_id")
    contrast = absent.apply(_heart_sound_contrast, axis=1)
    return [("Absent", absent.loc[contrast.idxmax()]),
            ("Present", present.sort_values("recording_id").iloc[0])]


def example_recordings(manifest: pd.DataFrame, seconds: float = 4.0) -> None:
    examples = _pick_examples(manifest)
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.4]})
    for col, (label, rec) in enumerate(examples):
        start = rec["ann_start_s"] if not np.isnan(rec["ann_start_s"]) else 0.0
        a, b = int(start * SR), int((start + seconds) * SR)
        raw = load_audio(rec["wav"])
        clean = preprocess_signal(raw)
        t = np.arange(b - a) / SR

        ax = axes[0, col]
        ax.plot(t, raw[a:b], color=INK_2, linewidth=0.6)
        ax.set_title(f"{label}: patient {rec['patient_id']}, {rec['location']}  "
                     f"({rec['recording_id']})")
        ax.set_ylabel("raw")
        ax.grid(False)

        ax = axes[1, col]
        for s, e, lab in load_segmentation(rec["tsv"]):
            if lab in (1, 3) and e > start and s < start + seconds:
                ax.axvspan(max(s, start) - start, min(e, start + seconds) - start,
                           color=GRID, linewidth=0)
                ax.text((max(s, start) + min(e, start + seconds)) / 2 - start, 1.02,
                        "S1" if lab == 1 else "S2", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.plot(t, clean[a:b], color=CLASS_COLORS[label], linewidth=0.7)
        ax.set_ylabel("25-800 Hz,\nnormalized")
        ax.grid(False)

        ax = axes[2, col]
        spec = logmel(clean)[:, int(start * FRAMES_PER_SECOND):int((start + seconds) * FRAMES_PER_SECOND)]
        ax.imshow(spec, origin="lower", aspect="auto", cmap=SEQ_CMAP,
                  extent=(0, seconds, 0, spec.shape[0]), vmin=np.percentile(spec, 5),
                  vmax=spec.max())
        ax.set_yticks([0, spec.shape[0]], [f"{FMIN:.0f} Hz", f"{FMAX:.0f} Hz"])
        ax.set_ylabel("log-mel (64 bands)")
        ax.set_xlabel("time (s)")
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_recordings.png", dpi=200)
    plt.close(fig)


def data_summary(manifest: pd.DataFrame, split: pd.DataFrame, window_s: float = 4.0,
                 hop_s: float = 2.0) -> pd.DataFrame:
    win, hop = int(window_s * FRAMES_PER_SECOND), int(hop_s * FRAMES_PER_SECOND)
    manifest = manifest.assign(windows=[
        len(window_starts(int(d * FRAMES_PER_SECOND) + 1, win, hop)) for d in manifest["duration_s"]])
    rows = []
    for name in ["train", "val", "test"]:
        part = get_split_manifest(manifest, split, name)
        pats = split[split["split"] == name]
        for c in CLASSES:
            rows.append({"split": name, "class": c,
                         "patients": int((pats["murmur"] == c).sum()),
                         "recordings": int((part["label"] == c).sum()),
                         "windows": int(part.loc[part["label"] == c, "windows"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(CACHE_DIR / "manifest.csv")
    split = load_split()
    load_feature_stats()  # fails early if preprocess was not run

    class_distribution(manifest, split)
    recording_durations(manifest)
    example_recordings(manifest)
    summary = data_summary(manifest, split)
    summary.to_csv(REPO_ROOT / "results" / "data_summary.csv", index=False)
    print(summary.pivot(index="split", columns="class").reindex(["train", "val", "test"]))
    print(f"Figures saved in {FIG_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
