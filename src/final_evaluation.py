"""Final model: ensemble choice on validation and one evaluation on the test set.

The final model is an ensemble: the recording probabilities of several saved
models are averaged, and the hierarchical thresholds (``evaluate``) chosen on
validation turn them into one label per patient. Its description is stored in
``models/final_ensemble.json`` (members and thresholds).

Command line::

    python -m src.final_evaluation                       # test results of the final model
    python -m src.final_evaluation --compare-ensembles   # first redo the ensemble choice

The default only needs the three CRNN checkpoints shipped in ``models/``.
``--compare-ensembles`` compares the 5 ensembles of ``experiments.ENSEMBLES``
on validation and rewrites ``models/final_ensemble.json`` with the best one;
it needs all 12 member checkpoints (train them with ``src.experiments``).

Outputs, in ``results/``: ``test_metrics.json``, ``test_predictions.csv``,
``test_per_model.csv``, ``figures/confusion_matrix_test.png`` and, with
``--compare-ensembles``, ``ensembles.csv``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import evaluate
from .data import CLASSES
from .experiments import ENSEMBLES, SEEDS
from .train import (MODELS_DIR, RESULTS_DIR, get_device, load_checkpoint, make_datasets,
                    patient_labels, recording_probabilities)

FINAL_ENSEMBLE = MODELS_DIR / "final_ensemble.json"


def run_names(experiments: list[str], seeds=SEEDS) -> list[str]:
    return [f"{e}_seed{s}" for e in experiments for s in seeds]


def member_probabilities(runs: list[str], dataset, device=None) -> dict[str, np.ndarray]:
    """Recording probabilities of each saved run on ``dataset``."""
    out = {}
    for run in runs:
        model, prepare, _ = load_checkpoint(MODELS_DIR / f"{run}.pt", device=device)
        out[run] = recording_probabilities(model, dataset, prepare)
    return out


def compare_ensembles(val_ds, device=None) -> pd.DataFrame:
    """Validation WA of every ensemble in ``ENSEMBLES``, with thresholds tuned
    on validation. Saves the table and the best ensemble as the final model."""
    labels = patient_labels()
    all_runs = sorted({r for exps in ENSEMBLES.values() for r in run_names(exps)})
    missing = [r for r in all_runs if not (MODELS_DIR / f"{r}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints in {MODELS_DIR}: {missing}. "
                                "Train them with python -m src.experiments")
    probs = member_probabilities(all_runs, val_ds, device)

    rows = []
    for name, exps in ENSEMBLES.items():
        members = run_names(exps)
        mean = np.mean([probs[m] for m in members], axis=0)
        t_p, t_u, wa = evaluate.tune_thresholds(mean, val_ds.recordings["patient_id"], labels)
        rows.append({"ensemble": name, "models": len(members), "val_wa": round(wa, 4),
                     "t_present": t_p, "t_unknown": t_u, "members": members})
    table = pd.DataFrame(rows).sort_values("val_wa", ascending=False).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.drop(columns="members").to_csv(RESULTS_DIR / "ensembles.csv", index=False)

    best = table.iloc[0]
    save_final_ensemble(best["ensemble"], best["members"], best["t_present"], best["t_unknown"], best["val_wa"])
    return table


def save_final_ensemble(name, members, t_present, t_unknown, val_wa):
    FINAL_ENSEMBLE.parent.mkdir(parents=True, exist_ok=True)
    FINAL_ENSEMBLE.write_text(json.dumps({
        "ensemble": name, "members": list(members), "t_present": float(t_present),
        "t_unknown": float(t_unknown), "val_wa": float(val_wa)}, indent=2))


def load_final_ensemble() -> dict:
    return json.loads(FINAL_ENSEMBLE.read_text())


def predict_patients(final: dict, dataset, device=None, member_probs: dict | None = None) -> pd.DataFrame:
    """One prediction per patient of ``dataset`` with the final ensemble."""
    member_probs = member_probs or member_probabilities(final["members"], dataset, device)
    mean = np.mean([member_probs[m] for m in final["members"]], axis=0)
    return evaluate.patient_predictions(mean, dataset.recordings["patient_id"],
                                        t_present=final["t_present"], t_unknown=final["t_unknown"])


def evaluate_on_test(test_ds, device=None) -> dict:
    """Test metrics of the final ensemble, plus each saved model on its own.

    Use once, after every choice has been made on validation.
    """
    labels = patient_labels()
    final = load_final_ensemble()
    patients = test_ds.recordings["patient_id"]

    # each single model that is available, with its own validation thresholds
    singles = sorted(p.stem for p in MODELS_DIR.glob("*_seed*.pt"))
    probs = member_probabilities(sorted(set(singles) | set(final["members"])), test_ds, device)
    rows = []
    for run in singles:
        _, _, ckpt = load_checkpoint(MODELS_DIR / f"{run}.pt", device="cpu")
        pred = evaluate.patient_predictions(probs[run], patients,
                                            t_present=ckpt["t_present"], t_unknown=ckpt["t_unknown"])
        m = evaluate.evaluate_patients(pred, labels)
        experiment, seed = run.rsplit("_seed", 1)
        rows.append({"experiment": experiment, "seed": int(seed), "val_wa": ckpt["val_wa"],
                     "test_wa": m["weighted_accuracy"], "test_macro_f1": m["macro_f1"]})
    per_model = pd.DataFrame(rows)
    # keep the rows of models not available here (e.g. trained elsewhere), replace the others
    per_model_path = RESULTS_DIR / "test_per_model.csv"
    if per_model_path.exists() and len(per_model):
        old = pd.read_csv(per_model_path)
        keep = ~old.set_index(["experiment", "seed"]).index.isin(per_model.set_index(["experiment", "seed"]).index)
        per_model = pd.concat([old[keep], per_model], ignore_index=True)

    predictions = predict_patients(final, test_ds, member_probs=probs)
    metrics = evaluate.evaluate_patients(predictions, labels)
    y_test = labels.loc[patients.unique()].to_numpy()
    always_present = evaluate.weighted_accuracy(y_test, np.full(len(y_test), "Present"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    per_model.to_csv(per_model_path, index=False)
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(
        {"final_model": final, "always_present_wa": always_present, **serializable}, indent=2))
    evaluate.plot_confusion_matrix(
        metrics["confusion_matrix"], RESULTS_DIR / "figures" / "confusion_matrix_test.png",
        title=f"Test: {final['ensemble']}, WA {metrics['weighted_accuracy']:.3f}")
    return {"final": final, "metrics": metrics, "per_model": per_model, "always_present": always_present,
            "predictions": predictions}


def print_report(result: dict) -> None:
    final, metrics = result["final"], result["metrics"]
    print(f"FINAL MODEL: ensemble {final['ensemble']} "
          f"(thresholds {final['t_present']}, {final['t_unknown']}), validation WA {final['val_wa']:.3f}")
    print(evaluate.format_metrics(metrics))
    print("Precision: " + "  ".join(f"{c} {metrics[f'precision_{c}']:.2f}" for c in CLASSES))
    print(f"\nreference 'always Present' on test: {result['always_present']:.3f}")
    if len(result["per_model"]):
        print("\nEach model on its own (mean and std over seeds):")
        print(result["per_model"].groupby("experiment")[["val_wa", "test_wa", "test_macro_f1"]]
              .agg(["mean", "std"]).round(3))


def main():
    parser = argparse.ArgumentParser(description="Final ensemble and test-set evaluation.")
    parser.add_argument("--compare-ensembles", action="store_true",
                        help="redo the ensemble choice on validation (needs all 12 checkpoints)")
    args = parser.parse_args()
    device = get_device()

    if args.compare_ensembles:
        table = compare_ensembles(make_datasets("val"), device)
        print(table.drop(columns="members").to_string(index=False), "\n")
    print_report(evaluate_on_test(make_datasets("test"), device))


if __name__ == "__main__":
    main()
