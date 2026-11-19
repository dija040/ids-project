# Step 2: Preprocessing and time-aware split

## Goal
Build clean, leakage-free train / validation / test sets from the 16.2 M raw flows, applying only
decisions justified by the EDA (step 1).

## Method
Three streamed passes (`src/preprocess.py`), so that the 8 M-flow day fits in Colab's RAM:
1. **Clean each day:** drop invalid timestamps, NaN/inf rows and impossible times; normalise labels
   and add the family label; relabel LOIC-HTTP flows over UDP; correct the 12-hour clock; drop the 8
   constant features; cast to float32; fingerprint each flow (64-bit hash of the 70 features, computed
   on the exact float64 values); sort chronologically.
2. **Plan:** remove conflicting feature vectors, remove exact duplicates (keeping the first in real
   time), assign a chronological 70/15/15 split inside every (day, class) group, and downsample Benign
   to 1 M flows in train only (seed 42).
3. **Write** `data/processed/{train,val,test}.parquet`.

## Decisions and justification
| # | Decision | Justification |
|---|---|---|
| D1 | Hours 01–07 → +12 h; drop the 14 flows dated 1970 | The capture ran 08:00–17:59 and no traffic exists at 06–07 h, so the rule is label-free. |
| D2 | Remove exact duplicates before splitting, keeping the first occurrence | Prevents identical flows from appearing in both train and test (memorisation). |
| D3 | Remove feature vectors carrying several labels; keep Infiltration flagged as unreliable | Irreducible error; relabelling would be a guess. |
| D4 | Drop NaN/inf rows (0.59%) and impossible times; drop 8 constant features; keep the `-1` sentinel | Nearly neutral (almost only Benign); no invented values. |
| — | LOIC-HTTP flows using UDP → LOIC-UDP | HTTP runs over TCP: a protocol fact, not a schedule-based guess. Same family either way. |
| D6 | Chronological 70/15/15 within each (day, class) | Respects time and keeps every class in every split. |
| D7 | Benign capped at 1 M in **train only** | Feasible CPU training; val/test keep the real proportions, so precision and FAR are honest. |
| Labels | Main target = 7 families; 15 fine labels kept for per-attack analysis | Stable estimates for rare attacks (SQLi: 84 unique flows). |
| D5 | `Dst Port` kept in the data, decided in step 3 | Will be tested as a shortcut feature. |

## Verification (built into the notebook)
- No shared feature vector between any two splits (leakage check).
- Train < validation < test in time, inside every (day, class) group.
- After the clock fix, observed attack windows match the official schedule.
  (Local test on slices: Bot 10:17–15:54 vs. documented 10:11–15:55; DDoS-HOIC 14:11 vs. 14:05;
  Web-XSS 13:51 vs. 13:50.)

## Results
_Fill in after running notebook 02 on Colab._
- Flows after cleaning: ___ · Rows removed per rule: see `results/tables/02_cleaning_log.csv`
- LOIC-UDP relabelling triggered? ___ (___ flows)
- Kept % per family (after dedup + conflicts): ___
- Final sizes: train ___ · val ___ · test ___
- Web-family flows in train (the GAN's training data): ___

## Tools
pandas, NumPy, PyArrow.
