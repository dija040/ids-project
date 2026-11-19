# Step 0: Setup and data acquisition

## Goal
Obtain CSE-CIC-IDS2018 from its official source and store it in a typed, fast, lossless
format (Parquet), without taking any cleaning decision yet. Produce a first inventory.

## Method
- Download the 10 daily CSVs ("Processed Traffic Data for ML Algorithms") over HTTPS from the
  public S3 bucket, with a size check against the bucket listing.
- Convert each CSV to Parquet in chunks of 500k rows (`src/data_io.py`):
  - remove repeated header lines (file artefacts);
  - drop identifier columns `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, which exist only in the
    20/02 file (schema consistency, privacy, shortcut learning);
  - cast the 78 features to float64 and `Timestamp` to datetime; keep `inf`/`NaN` as they are;
  - add `day` and `row_in_file` for the time-aware split.

## Tools
Google Colab (CPU), Google Drive, pandas, PyArrow (Parquet, zstd compression).

## Results
_Fill in after running notebook 00 on Colab._
- Total flows: ___ · Benign share: ___ % · Rarest class: ___ (___ flows)
- Header lines removed per file: ___
- Figure: `results/figures/00_class_distribution.png` · Tables: `results/tables/inventory_*.csv`

### Observations from the local test (01/03 file, verified)
| Observation | Consequence for later steps |
|---|---|
| 331,100 flows: 238,037 Benign and 93,063 `Infilteration` (sic) | Official labels contain spelling errors, so we need a label-normalisation table in preprocessing. |
| 25 repeated header lines inside the file | Removed during conversion (counted). |
| 4,004 `inf` and 1,834 `NaN` values in the features | To be analysed in EDA and handled in preprocessing. |
| Rows are **not in chronological order** in the CSV | We must sort by timestamp before the time-aware split. |
| Timestamps run from 01:00:00 to 12:59:59 | This looks like a **12-hour clock with no AM/PM**: afternoon traffic (13:00–17:00) appears as 01:00–05:00. To verify in EDA, because it directly affects a time-based split. |

## Decisions and justification
1. **No cleaning at this stage.** Every data-altering choice is deferred to preprocessing, where it is justified with EDA evidence.
2. **Identifier columns removed.** A model must judge flow behaviour, not memorise testbed IP addresses.
3. **float64 rather than float32.** Some counters (e.g. `Flow Duration`, in µs) exceed 10⁸ and would be rounded by float32.
