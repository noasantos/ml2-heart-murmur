"""Weighted accuracy, confusion matrix and patient-level aggregation.

Predictions go window -> recording -> patient:

1. ``window_probabilities``: softmax output of a model for every window of a
   ``WindowDataset`` (class order ``CLASSES`` = Present, Unknown, Absent);
2. ``recording_probabilities``: mean over the windows of each recording;
3. ``patient_predictions``: one label per patient, with one of two rules:

   - ``"hierarchical"`` (default): Present if some recording has
     P(Present) >= ``t_present``; otherwise Unknown if some recording has
     P(Unknown) >= ``t_unknown``; otherwise Absent. A murmur can be audible at
     a single location, so one confident recording is enough. The thresholds
     are chosen on the validation set with ``tune_thresholds``.
   - ``"mean"``: average the recording probabilities and take the argmax.

``evaluate_patients`` then gives the weighted accuracy and the patient-level
confusion matrix (mandatory in the report) plus additional metrics.

The weighted accuracy matches ``compute_weighted_accuracy`` of the official
Challenge code (https://github.com/physionetchallenges/evaluation-2022): the
confusion matrix is indexed ``[predicted, true]`` and each true Present,
Unknown and Absent patient weighs 5, 3 and 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASS_TO_IDX, CLASSES

CLASS_WEIGHTS = np.array([5.0, 3.0, 1.0])  # Present, Unknown, Absent


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _to_idx(labels) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return labels.astype(int)
    return np.array([CLASS_TO_IDX[c] for c in labels])


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """3x3 counts, ``cm[i, j]`` = patients predicted as class i whose true class is j.

    Labels can be class names or indices (order ``CLASSES``).
    """
    t, p = _to_idx(y_true), _to_idx(y_pred)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (p, t), 1)
    return cm


def weighted_accuracy(y_true, y_pred) -> float:
    """PhysioNet Challenge 2022 murmur weighted accuracy."""
    weighted = confusion_matrix(y_true, y_pred) * CLASS_WEIGHTS[None, :]
    total = weighted.sum()
    return float(np.trace(weighted) / total) if total > 0 else float("nan")


def classification_metrics(y_true, y_pred) -> dict:
    """Weighted accuracy, confusion matrix and additional metrics."""
    cm = confusion_matrix(y_true, y_pred)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    precision = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    out = {
        "weighted_accuracy": weighted_accuracy(y_true, y_pred),
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm,
    }
    for c, r, pr, f in zip(CLASSES, recall, precision, f1):
        out[f"recall_{c}"] = float(r)
        out[f"precision_{c}"] = float(pr)
        out[f"f1_{c}"] = float(f)
    return out


# ---------------------------------------------------------------------------
# Aggregation window -> recording -> patient
# ---------------------------------------------------------------------------

def window_probabilities(model, dataset, batch_size: int = 256, device=None,
                         transform=None) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities of ``model`` for every window of ``dataset``.

    ``transform`` adapts a batch to the model input, e.g.
    ``lambda x: x.unsqueeze(1)`` for a 2D CNN or ``lambda x: x.transpose(1, 2)``
    for an RNN. Returns (probabilities (n_windows, 3), recording index per window).
    """
    import torch
    from torch.utils.data import DataLoader

    device = device or next(model.parameters()).device
    model.eval()
    probs, recs = [], []
    with torch.no_grad():
        for x, _y, rec_idx in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = x.to(device)
            if transform is not None:
                x = transform(x)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
            recs.append(rec_idx.numpy())
    return np.concatenate(probs), np.concatenate(recs)


def recording_probabilities(window_probs: np.ndarray, window_rec_idx: np.ndarray,
                            n_recordings: int) -> np.ndarray:
    """Mean window probability of each recording, shape (n_recordings, 3)."""
    sums = np.zeros((n_recordings, window_probs.shape[1]))
    np.add.at(sums, window_rec_idx, window_probs)
    counts = np.bincount(window_rec_idx, minlength=n_recordings)[:, None]
    if (counts == 0).any():
        raise ValueError("some recordings have no window")
    return sums / counts


def _patient_table(rec_probs: np.ndarray, patient_ids) -> pd.DataFrame:
    df = pd.DataFrame(rec_probs, columns=[f"p_{c}" for c in CLASSES])
    df["patient_id"] = np.asarray(patient_ids)
    g = df.groupby("patient_id")
    out = g.mean().add_prefix("mean_")
    out = out.join(g.max().add_prefix("max_"))
    return out


def patient_predictions(rec_probs: np.ndarray, patient_ids, rule: str = "hierarchical",
                        t_present: float = 0.5, t_unknown: float = 0.5) -> pd.DataFrame:
    """One prediction per patient.

    Args:
        rec_probs: (n_recordings, 3) probabilities, order ``CLASSES``.
        patient_ids: patient of each recording (e.g. ``dataset.recordings["patient_id"]``).
        rule: "hierarchical" or "mean" (see module docstring).
        t_present, t_unknown: thresholds of the hierarchical rule.

    Returns a DataFrame with columns patient_id, murmur and the aggregated
    probabilities (mean_p_* and max_p_* over the patient's recordings).
    """
    table = _patient_table(rec_probs, patient_ids)
    if rule == "hierarchical":
        pred = np.where(table["max_p_Present"] >= t_present, "Present",
                        np.where(table["max_p_Unknown"] >= t_unknown, "Unknown", "Absent"))
    elif rule == "mean":
        mean = table[[f"mean_p_{c}" for c in CLASSES]].to_numpy()
        pred = np.array(CLASSES)[mean.argmax(axis=1)]
    else:
        raise ValueError(f"unknown rule {rule!r}")
    table.insert(0, "murmur", pred)
    return table.reset_index()


def tune_thresholds(rec_probs: np.ndarray, patient_ids, patient_labels: pd.Series,
                    grid=np.round(np.arange(0.05, 0.96, 0.05), 2)) -> tuple[float, float, float]:
    """Thresholds of the hierarchical rule that maximize the weighted accuracy.

    Use the **validation** recordings only. ``patient_labels`` maps patient_id
    to the true Murmur label. Ties go to the threshold pair closest to 0.5.
    Returns (t_present, t_unknown, validation weighted accuracy).
    """
    table = _patient_table(rec_probs, patient_ids)
    y_true = patient_labels.loc[table.index].to_numpy()
    p_max, u_max = table["max_p_Present"].to_numpy(), table["max_p_Unknown"].to_numpy()
    best = (-1.0, 0.0, 0.5, 0.5)
    for tp in grid:
        for tu in grid:
            pred = np.where(p_max >= tp, "Present", np.where(u_max >= tu, "Unknown", "Absent"))
            wa = weighted_accuracy(y_true, pred)
            dist = abs(tp - 0.5) + abs(tu - 0.5)
            if (wa, -dist) > (best[0], -best[1]):
                best = (wa, dist, float(tp), float(tu))
    return best[2], best[3], best[0]


def evaluate_patients(predictions: pd.DataFrame, patient_labels: pd.Series) -> dict:
    """Metrics of a ``patient_predictions`` table against the true labels.

    ``patient_labels`` maps patient_id to the Murmur label (e.g.
    ``split.set_index("patient_id")["murmur"]``).
    """
    y_true = patient_labels.loc[predictions["patient_id"]].to_numpy()
    return classification_metrics(y_true, predictions["murmur"].to_numpy())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_metrics(metrics: dict) -> str:
    """Readable text block with the confusion matrix and the main metrics."""
    cm = pd.DataFrame(metrics["confusion_matrix"],
                      index=[f"pred {c}" for c in CLASSES],
                      columns=[f"true {c}" for c in CLASSES])
    lines = [f"Weighted accuracy: {metrics['weighted_accuracy']:.3f}",
             f"Accuracy: {metrics['accuracy']:.3f}   macro F1: {metrics['macro_f1']:.3f}",
             "Recall: " + "  ".join(f"{c} {metrics[f'recall_{c}']:.2f}" for c in CLASSES),
             cm.to_string()]
    return "\n".join(lines)


def plot_confusion_matrix(cm: np.ndarray, path: str | Path, title: str = "") -> None:
    """Save the patient-level confusion matrix (rows predicted, columns true)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#9cc3ef", "#2a78d6", "#0d2f5c"])
    norm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)  # share of each true class
    fig, ax = plt.subplots(figsize=(4.6, 4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="#fcfcfb" if norm[i, j] > 0.55 else "#0b0b0b", fontsize=10)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("true label")
    ax.set_ylabel("predicted label")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_submission(predictions: pd.DataFrame, path: str | Path) -> None:
    """Write the bonus-competition csv: one row per patient, ``patient_id,murmur``."""
    out = predictions[["patient_id", "murmur"]].sort_values("patient_id")
    if out["patient_id"].duplicated().any():
        raise ValueError("duplicated patient_id")
    if not set(out["murmur"]) <= set(CLASSES):
        raise ValueError(f"labels must be in {CLASSES}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
