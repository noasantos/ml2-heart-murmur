# Heart Murmur Detection from Phonocardiograms

Machine Learning II (CC3043), DCC/FCUP, 2026/2027.

Group project: deep learning classifiers that detect heart murmurs in
cardiac auscultation recordings from the CirCor DigiScope dataset. The
full project statement is in [`docs/project_statement.pdf`](docs/project_statement.pdf).

## Group

| Name | Student number |
|------|----------------|
|      |                |
|      |                |
|      |                |

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
├── src/           reusable code
│   ├── data.py        loading recordings and labels, patient-level splits
│   ├── features.py    pre-processing and feature extraction
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

To be completed as the work progresses.

## Hidden data predictions

The bonus competition expects a `.csv` with one row per patient:

```
patient_id,murmur
12345,Absent
12346,Present
```

The file goes in `submission/`.

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
