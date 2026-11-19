# Step 1: Exploratory data analysis

## Goal
Understand the data before modifying it, and collect the evidence that justifies each
preprocessing decision (step 2).

## Method
- **Streamed audit** of 100% of the 16.2 M flows (`src/eda.py::audit_dataset`), one Parquet
  batch of 500k rows at a time: NaN/±inf/negative counts, min/max per feature, invalid
  timestamps, flows per hour, and a 64-bit hash per flow for duplicate detection.
- **Stratified sample** (≤ 30,000 flows per class, seed 42) for distributions, Spearman
  correlations and port analysis.
- **Hypothesis test** of the 12-hour clock: observed attack windows vs. the official schedule.

## Tools
pandas, PyArrow, NumPy, matplotlib, seaborn.

## Results
_Fill in after running notebook 01 on Colab._

| # | Finding | Evidence |
|---|---|---|
| 1 | Excel truncation | |
| 2 | Invalid timestamps | |
| 3 | 12-hour clock | |
| 4 | NaN / inf | |
| 5 | Negative values | |
| 6 | Constant features | |
| 7 | Exact duplicates | |
| 8 | Conflicting labels | |
| 9 | Very rare classes | |
| 10 | Redundant features | |
| 11 | `Dst Port` concentration | |

### Early observations (local test on 15 MB slices of each day, to be confirmed on the full data)
- **12-hour clock confirmed on afternoon attacks:** DDoS-HOIC is documented at 14:05 and observed at
  02:11; Brute Force-XSS (22/02) is documented at 13:50 and observed at 01:51. Morning attacks match
  the documented times.
- **DoS-SlowHTTPTest and FTP-BruteForce** both target port 21, are ~99.9% exact duplicates, and share
  identical feature vectors. These "attacks" may be mostly failed connection attempts that are
  indistinguishable from each other (a label-quality issue to discuss).
- 8 constant features (`Bwd PSH Flags`, `Bwd URG Flags`, and the 6 `*Byts/b Avg`, `*Pkts/b Avg`,
  `*Blk Rate Avg` features).
- Many perfect redundancies (ρ = 1): `Tot Fwd Pkts` ↔ `Subflow Fwd Pkts`, `Pkt Len Std` ↔ `Pkt Len Var`, etc.
- NaN/inf appear only in `Flow Byts/s` and `Flow Pkts/s` (rates divided by a zero duration).
- `-1` in `Init Fwd/Bwd Win Byts` is a sentinel for "not observed", not an error.

## Decisions and justification
_To be decided together before step 2._
