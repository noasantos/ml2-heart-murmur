# Heart Murmur Detection from Phonocardiograms

Machine Learning II (CC3043), DCC/FCUP, 2026/2027.

Group project: deep learning classifiers that detect heart murmurs in
cardiac auscultation recordings from the CirCor DigiScope dataset. The
full project statement is in [`docs/project_statement.pdf`](docs/project_statement.pdf).

## Group

| Name | Student number |
|------|----------------|
|  Noa Santos    |                |
|  Telma Freitas    |  202301622   |
|  Yunnie Zita    |                |


Professor: Francesco Renna

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

We have to implement two of the three architectures: MLP, CNN, RNN.
For each one we cover pre-processing, architecture, training strategy and
evaluation.

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
patient-level confusion matrix (both mandatory), plus any extra metrics.

Reference implementation:
https://github.com/physionetchallenges/evaluation-2022

## Dataset

CirCor DigiScope Phonocardiogram Dataset, version 1.0.3 (public part: 3163
recordings from 942 patients).

- Dataset page: https://physionet.org/content/circor-heart-sound/1.0.3/
- Reference paper: https://ieeexplore.ieee.org/abstract/document/9658215

The dataset is **not** included in the repository. Download it into `data/`:

```bash
wget -r -N -c -np -nH --cut-dirs=3 -P data/circor https://physionet.org/files/circor-heart-sound/1.0.3/
```

After the download the patient table should be at
`data/circor/training_data.csv` and the recordings (`.wav`, `.hea`, `.tsv`,
`.txt`) inside `data/circor/training_data/`.

## Repository structure

```
.
├── data/          dataset (not versioned)
├── docs/          project statement and other reference material
├── models/        saved model weights (not versioned)
├── notebooks/     exploration and experiments
├── results/       metrics, confusion matrices
│   └── figures/   plots used in the slides
├── slides/        presentation (PDF)
├── splits/        patient-level train/val/test split (shared by the group)
├── src/           reusable code
│   ├── data.py        loading recordings and labels, patient-level splits, windowed dataset
│   ├── features.py    pre-processing and feature extraction
│   ├── preprocess.py  builds the feature cache, the split and the training statistics
│   ├── eda.py         exploration figures and data summary
│   ├── models.py      MLP / CNN / RNN definitions
│   ├── train.py       training loop
│   └── evaluate.py    weighted accuracy, confusion matrix, patient aggregation
└── submission/    predictions on the hidden data
```

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

All commands run from the repository root with the virtual environment active.

### 1. Pre-processing (about 20 s)

```bash
python -m src.preprocess
```

This reads every recording, applies the pre-processing below and caches the
result in `data/cache/band25-800/` (not versioned, about 900 MB). It also
creates the patient split `splits/patient_split.csv` if it does not exist yet.
The split is versioned, so the whole group trains and evaluates on the same
patients.

Pre-processing of each recording:

1. band-pass filter 25-800 Hz (4th-order Butterworth, zero phase);
2. amplitude normalization (zero mean, unit standard deviation);
3. log-mel spectrogram: 64 bands (25-800 Hz), 50 ms window, 10 ms step
   (100 frames per second);
4. standardization of each row with the mean/std of the **training** split
   only (`data/cache/band25-800/logmel_stats.npz`, `mfcc_stats.npz`).

Split: 70/15/15 by patient, stratified by `Murmur`, seed 42
(train 654, val 146, test 142 patients). 70 children attended both
screening campaigns and have two patient IDs, linked by the `Additional ID`
column of `training_data.csv`. The split groups both IDs of a child, so they
always fall in the same subset (`subject_groups` in `src/data.py`).

Training labels are per recording: a Present patient's recording taken at a
location where the murmur was not heard (`Murmur locations`) is labelled
Absent. Evaluation always uses the patient label.

### 2. Loading the data in a model

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

### Pre-processing variants

The pre-processing choices are compared on **validation**, training the same
model with the same hyperparameters on each variant and changing only one
setting at a time. Every variant is a `WindowDataset` argument:

| Choice | Default | Alternatives | How |
|---|---|---|---|
| Window length | 4 s, 2 s hop | 2 s, 6 s ... | `window_s=2.0, hop_s=1.0` |
| Input feature | log-mel, 64 rows | MFCC: 20 coefficients + 20 deltas, 40 rows | `feature="mfcc"` |
| Band-pass | 25-800 Hz | 25-400 Hz (or any band) | `band=(25, 400)` after `python -m src.preprocess --band 25 400` |
| Unannotated edges | kept | cut, training set only | `trim_to_annotation=True` |

Each band is cached in its own folder (`data/cache/band25-400/` ...), so the
variants coexist; building one takes about 20 s and 900 MB. Validation and
test always use `trim_to_annotation=False`: the hidden data may not have the
.tsv files.

```python
train_ds = WindowDataset(train_rec, feature="mfcc", band=(25, 400))
val_ds = WindowDataset(val_rec, feature="mfcc", band=(25, 400))  # same variant
```

### 3. Evaluation: one prediction per patient

`src/evaluate.py` turns window predictions into one label per patient and
computes the metrics. Its weighted accuracy gives exactly the same values as
the official Challenge code (checked on 3000 random cases).

```python
from src.evaluate import (evaluate_patients, format_metrics, patient_predictions,
                          plot_confusion_matrix, recording_probabilities,
                          tune_thresholds, window_probabilities)

labels = split.set_index("patient_id")["murmur"]
val_ds = WindowDataset(get_split_manifest(manifest, split, "val"))
test_ds = WindowDataset(get_split_manifest(manifest, split, "test"))

def rec_probs(ds):
    probs, rec = window_probabilities(model, ds, transform=lambda x: x.unsqueeze(1))  # RNN: x.transpose(1, 2)
    return recording_probabilities(probs, rec, len(ds.recordings))

# thresholds chosen on validation only
t_p, t_u, _ = tune_thresholds(rec_probs(val_ds), val_ds.recordings["patient_id"], labels)

# final numbers on test, with the thresholds fixed
pred = patient_predictions(rec_probs(test_ds), test_ds.recordings["patient_id"],
                           t_present=t_p, t_unknown=t_u)
metrics = evaluate_patients(pred, labels)
print(format_metrics(metrics))
plot_confusion_matrix(metrics["confusion_matrix"], "results/figures/cm_cnn_test.png", "CNN, test")
```

Aggregation: the probability of a recording is the mean over its windows. The
patient is Present if some recording has P(Present) >= `t_present`, otherwise
Unknown if some recording has P(Unknown) >= `t_unknown`, otherwise Absent (a
murmur may be audible at a single location). `rule="mean"` averages the
recordings instead.

Reference values (patient level):

| Trivial classifier | Val | Test |
|---|---|---|
| always Present | 0.493 | 0.525 |
| always Absent | 0.370 | 0.377 |
| always Unknown | 0.137 | 0.098 |

Do not report the validation weighted accuracy obtained after tuning the
thresholds on validation: it is optimistic. The reported numbers come from the
test set. Because the metric rewards predicting Present, even an untrained
network with tuned thresholds scored 0.53 (val) and 0.57 (test) in our checks on
the first version of the split,
so a model is only useful if it clearly beats about 0.55-0.60.

`save_submission(pred, "submission/predictions.csv")` writes the csv of the
bonus competition.

### 4. Exploration figures

```bash
python -m src.eda
```

Writes the figures in `results/figures/` and `results/data_summary.csv`
(patients, recordings and 4 s windows per split and class).

## Hidden data predictions

The bonus competition expects a `.csv` with one row per patient:

```
patient_id,murmur
12345,Absent
12346,Present
```

The file goes in `submission/`.

When the hidden recordings are released (folder with the patient `.txt`,
`.hea` and `.wav` files, e.g. `data/hidden/`), cache them with the same
variant as the final model; no labels or `.tsv` files are needed:

```bash
python -m src.preprocess --recordings-dir data/hidden --no-labels --cache-dir data/cache_hidden
```

```python
from src.evaluate import save_submission
hidden = pd.read_csv("data/cache_hidden/manifest.csv")
hidden_ds = WindowDataset(hidden, cache_dir="data/cache_hidden")  # statistics from the training cache
pred = patient_predictions(rec_probs(hidden_ds), hidden_ds.recordings["patient_id"],
                           t_present=t_p, t_unknown=t_u)
save_submission(pred, "submission/predictions.csv")
```

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
