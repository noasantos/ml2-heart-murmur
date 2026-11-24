# Heart Murmur Detection from Phonocardiograms

Machine Learning II (CC3043), DCC/FCUP, 2026/2027.

Group project: deep learning classifiers that detect heart murmurs in
cardiac auscultation recordings from the CirCor DigiScope dataset. The
full project statement is in [`docs/project_statement.pdf`](docs/project_statement.pdf).

## Group

| Name | Student number |
|------|----------------|
| Noa Santos | up202201392 |
| Telma Freitas | up202301622 |
| Yunnie Zita | up202309709 |

Professor: Francesco Renna

## Results in short

| Model | Validation WA | Test WA |
|---|---|---|
| Final model: average of 3 CRNNs (1D CNN + bidirectional GRU) | 0.824 | **0.859** |
| Always predict Present (trivial) | 0.493 | 0.525 |

The test set was used once, after every choice had been made on validation.
The slides discuss the experiments; `results/` holds every number and figure
(see [What is in `results/`](#what-is-in-results)).

## Reproduce the results

All commands run from the repository root, in this order.

### 1. Install (Python 3.10 or newer)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the data (not included)

Either of:

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/circor https://physionet.org/files/circor-heart-sound/1.0.3/
aws s3 sync --no-sign-request s3://physionet-open/circor-heart-sound/1.0.3/ data/circor   # faster, needs awscli
```

The patient table must end up at `data/circor/training_data.csv` and the
recordings (`.wav`, `.hea`, `.tsv`, `.txt`) in `data/circor/training_data/`.

### 3. Pre-process (about 2 minutes on 2 CPU cores, faster with more)

```bash
python -m src.preprocess
```

Computes one log-mel spectrogram per recording and the normalization
statistics of the training split, in `data/cache/` (about 900 MB). The patient
split is read from `splits/patient_split.csv`, which is included.

### 4. Final results on the test set (about 30 s, no training, no GPU needed)

```bash
python -m src.final_evaluation
```

Loads the final model shipped in `models/` (3 CRNNs and their thresholds) and
prints the test weighted accuracy, the patient-level confusion matrix and the
additional metrics. A GPU is used if available, but it is not needed. It writes
`results/test_metrics.json`, `results/test_predictions.csv` and
`results/figures/confusion_matrix_test.png`. Expected output: weighted
accuracy 0.859.

### 5. Train the CNN experiments (optional)

```bash
python -m src.experiments --list                    # the 8 experiments
python -m src.experiments crnn_specaug              # one experiment, seeds 42, 1, 2
python -m src.experiments crnn_specaug --seeds 42   # one run
python -m src.experiments all                       # all 24 runs
```

Each run saves `models/<experiment>_seed<seed>.pt` and adds a row to
`results/experiments.csv`. On a Colab T4 GPU an epoch takes about 9 s for the
CRNN, 8 s for the 1D CNN, 19 s for the 2D CNN and 53 s for ResNet18; with
early stopping a run lasts 7 to 21 epochs, and all 24 runs about 2 hours. A
CPU is much slower, mostly for ResNet18. Results can differ slightly between
runs on a GPU, because some GPU operations are not deterministic.

After training the 12 ensemble members (`crnn_specaug`, `resnet18_specaug`,
`cnn1d_specaug`, `cnn_specaug`), the ensemble choice can be redone:

```bash
python -m src.final_evaluation --compare-ensembles
```

### 6. Redraw the figures (optional)

```bash
python -m src.plots     # validation chart and the 24 training curves
python -m src.eda       # exploration figures and results/data_summary.csv
```

### 7. Predictions on the hidden data

With the hidden recordings in a folder such as `data/hidden/` (patient `.txt`,
`.hea` and `.wav` files; no labels or `.tsv` needed), after step 3:

```bash
python -m src.preprocess --recordings-dir data/hidden --no-labels --cache-dir data/cache_hidden
python -m src.predict --recordings-dir data/hidden --cache-dir data/cache_hidden
```

Writes `submission/predictions.csv`, one row per patient (`patient_id,murmur`).

### Notebook (optional)

`notebooks/ml2_heart_murmur.ipynb` walks through the same steps with
explanations: pre-processing, the CNN experiments, the RNN, and the final
model. It imports everything from `src/`, needs steps 1 to 3, and does not
train a CNN unless `RUN_TRAINING = True` is set in section 2.1. The RNN
section (2.2) trains its models when run.

To run it on Google Colab, upload the repository folder (or clone it), open the
notebook, run steps 1 to 3 in a cell with `!` in front of each command, and
choose a GPU runtime if training.

## Task

Each patient has to be classified into one of three classes according to
the `Murmur` label:

- **Present**: a murmur was heard
- **Absent**: no murmur was heard
- **Unknown**: the expert could not tell (noise, poor recording quality)

Things to keep in mind:

- The label belongs to the patient, not to the recording. A patient usually
  has several recordings (one per auscultation location: AV, PV, TV, MV) and
  the final system must give **one prediction per patient**.
- The classes are imbalanced. Most patients are Absent and Unknown is the
  smallest class.
- Splits must be done **by patient**, and the test set cannot be used for any
  decision (architecture, hyperparameters, thresholds, early stopping...).

## Evaluation metric

Weighted accuracy from the murmur task of the PhysioNet Challenge 2022,
computed at patient level. Present patients count 5 times, Unknown 3 times
and Absent once:

```
WA = (5*m_PP + 3*m_UU + m_AA) /
     (5*(m_PP + m_UP + m_AP) + 3*(m_PU + m_UU + m_AU) + (m_PA + m_UA + m_AA))
```

where `m_ij` is the number of patients predicted as class `i` whose true class
is `j`. For each classifier we report the weighted accuracy and the
patient-level confusion matrix (both mandatory), plus additional metrics
(recall, precision and F1 per class, macro F1, accuracy).

Reference implementation:
https://github.com/physionetchallenges/evaluation-2022

## Method

**Data.** CirCor DigiScope Phonocardiogram Dataset, version 1.0.3 (public
part: 3163 recordings from 942 patients).
[Dataset page](https://physionet.org/content/circor-heart-sound/1.0.3/),
[reference paper](https://ieeexplore.ieee.org/abstract/document/9658215).

**Pre-processing** of each recording (`src/features.py`):

1. band-pass filter 25-800 Hz (4th-order Butterworth, zero phase);
2. amplitude normalization (zero mean, unit standard deviation);
3. log-mel spectrogram: 64 bands (25-800 Hz), 50 ms window, 10 ms step
   (100 frames per second);
4. standardization of each row with the mean/std of the **training** split
   only;
5. windows of 4 s (64 x 400) with a 2 s step.

**Split** (`src/data.py`): 70/15/15 by patient, stratified by `Murmur`, seed
42 (train 654, val 146, test 142 patients). 70 children attended both
screening campaigns and have two patient IDs, linked by the `Additional ID`
column of `training_data.csv`. The split groups both IDs of a child, so they
always fall in the same subset (`subject_groups`).

**Labels.** Training labels are per recording: a Present patient's recording
taken at a location where the murmur was not heard (`Murmur locations`) is
labelled Absent. Evaluation always uses the patient label.

**From windows to patients** (`src/evaluate.py`). The probability of a
recording is the mean over its windows. The patient is Present if some
recording has P(Present) >= `t_present`, otherwise Unknown if some recording
has P(Unknown) >= `t_unknown`, otherwise Absent (a murmur may be audible at a
single location). Both thresholds are chosen on validation.

**CNN training** (`src/train.py`, `src/experiments.py`). Cross-entropy
weighted by the inverse class frequency of the windows, Adam (weight decay
1e-4), batch 64, at most 30 epochs, early stopping with patience 6 on the
patient-level validation WA. SpecAugment (2 frequency masks of up to 8 bands,
2 time masks of up to 0.4 s) in every experiment except the baseline. Each
experiment uses 3 seeds (42, 1, 2).

| Experiment | Architecture | Validation WA (mean ± std) |
|---|---|---|
| `cnn_base` | 2D CNN, no augmentation | 0.749 ± 0.017 |
| `cnn_specaug` | 2D CNN | 0.759 ± 0.008 |
| `resnet18_specaug` | ResNet18 pretrained on ImageNet | 0.797 ± 0.011 |
| `resnet18_specaug_lr3e-5` | ResNet18, learning rate 3e-5 | 0.776 ± 0.011 |
| `resnet18_freeze2_specaug` | ResNet18, first 2 layer groups frozen | 0.784 ± 0.027 |
| `resnet18_freeze3_specaug` | ResNet18, first 3 layer groups frozen | 0.754 ± 0.009 |
| `cnn1d_specaug` | 1D CNN (bands as channels, filters over time) | 0.782 ± 0.004 |
| `crnn_specaug` | CRNN: 1D CNN + bidirectional GRU | 0.808 ± 0.011 |

Five ensembles (averaged recording probabilities, thresholds tuned on
validation) were compared on validation; the average of the 3 CRNNs was best
(0.824) and is the final model.

**RNN** (Yunnie Zita): notebook section 2.2.

## Repository structure

```
.
├── data/          dataset and cache (not versioned)
├── docs/          project statement
├── models/        final model: 3 CRNN checkpoints and final_ensemble.json
├── notebooks/     ml2_heart_murmur.ipynb (walkthrough of the whole project)
├── results/       tables and figures (see below)
├── slides/        presentation (PDF)
├── splits/        patient-level train/val/test split (shared by the group)
├── src/
│   ├── data.py              recordings and labels, patient split, windowed dataset
│   ├── features.py          filtering, log-mel and MFCC
│   ├── preprocess.py        feature cache, split and training statistics
│   ├── eda.py               exploration figures and data summary
│   ├── evaluate.py          weighted accuracy, confusion matrix, window -> patient
│   ├── models.py            2D CNN, 1D CNN, CRNN, ResNet18
│   ├── train.py             SpecAugment, training loop, checkpoints
│   ├── experiments.py       the 8 CNN experiments and the 5 ensembles
│   ├── final_evaluation.py  ensemble choice and test-set evaluation
│   ├── plots.py             training curves and results chart
│   └── predict.py           predictions on the hidden data
└── submission/    predictions on the hidden data
```

### What is in `results/`

| File | Content |
|---|---|
| `experiments.csv` | the 24 CNN runs: settings, best validation WA, best epoch, thresholds |
| `histories/` | training loss and validation WA per epoch of each run |
| `ensembles.csv` | validation WA of the 5 ensembles |
| `test_metrics.json`, `test_predictions.csv` | final model on the test set |
| `test_per_model.csv` | each single model on the test set (reference only) |
| `figures/validation_results.png` | validation WA of every experiment and ensemble |
| `figures/training/` | training curves of the 24 runs |
| `figures/confusion_matrix_test.png` | confusion matrix of the final model on the test set |
| `data_summary.csv`, other figures | data exploration (`src/eda.py`) |

## Using the code from Python

### Loading the data in a model

```python
import pandas as pd
from torch.utils.data import DataLoader
from src.data import CACHE_DIR, WindowDataset, get_split_manifest, load_split

manifest = pd.read_csv(CACHE_DIR / "manifest.csv")
split = load_split()

train_ds = WindowDataset(get_split_manifest(manifest, split, "train"),
                         window_s=4.0, hop_s=2.0)
loader = DataLoader(train_ds, batch_size=64, shuffle=True)
x, y, rec_idx = next(iter(loader))   # x: (64, 64 mel bands, 400 frames)

x_cnn = x.unsqueeze(1)      # (batch, 1, 64, 400)  2D CNN
x_rnn = x.transpose(1, 2)   # (batch, 400, 64)     RNN, time steps first
```

`rec_idx` indexes `train_ds.recordings`, which has the `patient_id` of each
window, to aggregate window predictions into one prediction per patient.
`train_ds.n_features` is the number of rows of `x` (the RNN `input_size`).
The dataset loads the training statistics of the chosen variant by itself.
`src.train.make_datasets("train", "val")` builds the same datasets in one call.

### Training and evaluating a model

```python
from src.models import build_model
from src.train import make_datasets, spec_augment, train, validate

train_ds, val_ds = make_datasets("train", "val")
model, prepare = build_model("crnn")              # also "cnn2d", "cnn1d", "resnet18"
model, best, history = train(model, prepare, train_ds, val_ds, augment=spec_augment, clip=1.0)
print(best["wa"], best["t_p"], best["t_u"])        # validation WA and thresholds
```

### Evaluation functions

`src/evaluate.py` turns window predictions into one label per patient and
computes the metrics. Its weighted accuracy gives exactly the same values as
the official Challenge code (checked on 3000 random cases).

```python
from src.evaluate import (evaluate_patients, format_metrics, patient_predictions,
                          recording_probabilities, tune_thresholds, window_probabilities)

labels = split.set_index("patient_id")["murmur"]

def rec_probs(ds):
    probs, rec = window_probabilities(model, ds, transform=prepare)
    return recording_probabilities(probs, rec, len(ds.recordings))

t_p, t_u, _ = tune_thresholds(rec_probs(val_ds), val_ds.recordings["patient_id"], labels)
pred = patient_predictions(rec_probs(test_ds), test_ds.recordings["patient_id"],
                           t_present=t_p, t_unknown=t_u)
print(format_metrics(evaluate_patients(pred, labels)))
```

Reference values (patient level):

| Trivial classifier | Val | Test |
|---|---|---|
| always Present | 0.493 | 0.525 |
| always Absent | 0.370 | 0.377 |
| always Unknown | 0.137 | 0.098 |

The validation WA after tuning the thresholds on validation is optimistic; it
is only used to compare models. Because the metric rewards predicting Present,
even an untrained network with tuned thresholds scored 0.53 (val) and 0.57
(test) in our checks on the first version of the split, so a model is only
useful if it clearly beats about 0.55-0.60.

### Pre-processing variants

Every pre-processing choice is a `WindowDataset` argument, so variants can be
compared on validation, changing one setting at a time:

| Choice | Default | Alternatives | How |
|---|---|---|---|
| Window length | 4 s, 2 s hop | 2 s, 6 s ... | `window_s=2.0, hop_s=1.0` |
| Input feature | log-mel, 64 rows | MFCC: 20 coefficients + 20 deltas, 40 rows | `feature="mfcc"` |
| Band-pass | 25-800 Hz | 25-400 Hz (or any band) | `band=(25, 400)` after `python -m src.preprocess --band 25 400` |
| Unannotated edges | kept | cut, training set only | `trim_to_annotation=True` |

Each band is cached in its own folder (`data/cache/band25-400/` ...). Validation
and test always use `trim_to_annotation=False`: the hidden data may not have
the .tsv files.

## Deliverables

Submitted on Moodle:

1. Code to train the models and reproduce the results, with short
   instructions on how to run it (without the dataset files)
2. Slides (PDF) for a 10-minute presentation, extra material in an appendix
3. Predictions on the hidden data (`.csv`)
4. Self-assessment form with each member's contribution

## Important dates

| Date | What |
|------|------|
| 5 November 2026 | Checkpoint during the practical class (10% of the grade) |
| 30 November 2026, 23:59 | Submission deadline on Moodle |
| 3 and 10 December 2026 | Presentations (all members must attend) |
