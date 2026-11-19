# Network Intrusion Detection with GAN-based Data Augmentation (CSE-CIC-IDS2018)

Practical mini-project, module AI1 (4th year Cybersecurity Engineering).

**Question studied:** can synthetic samples produced by a Vanilla GAN improve the
detection of *rare* attack families by a network IDS, compared with no
rebalancing and with a classical oversampler (SMOTE)?

## Project structure

```
ids-project/
├── src/                 reusable code (imported by the notebooks)
│   ├── config.py        paths, seeds, dataset metadata
│   ├── data_io.py       step 0: download + CSV -> Parquet conversion
│   ├── eda.py           step 1: streamed audit, duplicates, time checks
│   ├── preprocess.py    step 2: cleaning, dedup, time-aware split
│   └── features.py      step 3: redundancy filter, ranking, choice of k
├── notebooks/           one notebook per pipeline step, run in order
│   ├── 00_setup_and_data_acquisition.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_feature_selection.ipynb
├── docs/                proposal and design notes
├── reports/step_reports one short write-up per step (goal / method / results / decisions)
├── results/             small tables and figures used in the report (committed)
├── data/                datasets and models (NOT committed, created by the notebooks)
└── requirements.txt
```

## Pipeline

| Step | Notebook | Status |
|---|---|---|
| 0. Setup and data acquisition | `00_setup_and_data_acquisition.ipynb` | done |
| 1. Exploratory data analysis | `01_eda.ipynb` | done |
| 2. Preprocessing and time-aware split | `02_preprocessing.ipynb` | done |
| 3. Feature selection | `03_feature_selection.ipynb` | in progress |
| 4. Vanilla GAN augmentation | `04_gan.ipynb` | to do |
| 5. IDS models and tuning | `05_models.ipynb` | to do |
| 6. Evaluation, security analysis, explainability | `06_evaluation.ipynb` | to do |

## How to run (Google Colab, no GPU required)

1. Put this repository on GitHub.
2. Open `notebooks/00_setup_and_data_acquisition.ipynb` in Colab
   (*File → Open notebook → GitHub*) and
   run all cells. The notebook:
   - mounts Google Drive and clones the repository into `MyDrive/ids-project`;
   - downloads the official CSVs (~6.9 GB) to Colab's temporary disk;
   - converts them to Parquet in `MyDrive/ids-project/data/interim/` (persistent).
3. Run the next notebooks in order.

To run locally instead: `pip install -r requirements.txt`, then run the notebooks
from the `notebooks/` folder.

## Data

CSE-CIC-IDS2018, Communications Security Establishment (CSE) & Canadian Institute
for Cybersecurity (CIC). Public bucket, anonymous access:
`s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/`.
Dataset page: https://www.unb.ca/cic/datasets/ids-2018.html

> I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion
> Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.

## Reproducibility

- Global seed `SEED = 42` in `src/config.py`, applied with `set_seed()` in every notebook.
- Raw data is never modified; every transformation is code in this repository.

## AI usage declaration

This project was built with the assistance of Claude (Anthropic) as a coding and
explanation assistant. Details on what was generated, and what was verified and how,
are given in the report.
