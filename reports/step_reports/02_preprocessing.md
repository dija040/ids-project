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

## Results (full run on Colab)

### Pass 1: cleaning (16,232,943 → 16,137,168 flows)
| Rule | Benign | Infiltration | BruteForce-FTP | DDoS-LOIC-HTTP |
|---|---|---|---|---|
| Removed: NaN or inf | 94,459 | 1,295 | 6 | – |
| Removed: impossible time value | 1 | – | – | – |
| Removed: invalid timestamp (1970) | 14 | – | – | – |
| **Relabelled: LOIC-HTTP over UDP → LOIC-UDP** | – | – | – | **797** |

**The LOIC-UDP hypothesis is confirmed.** 797 flows labelled "LOIC-HTTP" on 20/02 used UDP. With the
corrected clock, they occur at **13:14–13:29**, inside the documented LOIC-UDP window (13:13–13:32).

### Pass 2: deduplication and conflicts
| Family | After cleaning | Conflicting label | Exact duplicate | **Kept** |
|---|---|---|---|---|
| Benign | 13,390,234 | 128,283 | 2,718,943 | 10,543,008 (78.7%) |
| DoS | 654,300 | 77,277 | 380,467 | 196,556 (30.0%) |
| DDoS | 1,263,933 | 0 | 487,978 | 775,955 (61.4%) |
| BruteForce | 380,943 | 175,221 | 111,642 | 94,080 (24.7%) |
| Web | 928 | 31 | 47 | 850 (91.6%) |
| Bot | 286,191 | 0 | 141,656 | 144,535 (50.5%) |
| Infiltration | 160,639 | 40,894 | 2,147 | 117,598 (73.2%) |

The most striking cases are at the fine-label level. **BruteForce-FTP keeps 39 of 193,354 flows (0.02%)**
and **DoS-SlowHTTPTest keeps 43 of 139,890 (0.03%)**: 175,191 FTP flows and 77,277 SlowHTTPTest flows
share identical feature vectors with each other. In practice the two "attacks" are the same handful of
port-21 connection patterns repeated hundreds of thousands of times. Any published result trained on
the raw data effectively scores these near-copies many times over. A quarter of Infiltration (40,894
flows) was indistinguishable from Benign.

### Final datasets
| Family | Train | Val | Test | Train ratio vs Benign |
|---|---|---|---|---|
| Benign | 1,000,000 | 1,581,452 | 1,581,447 | 1 |
| DoS | 137,592 | 29,483 | 29,481 | 7.3 |
| DDoS | 543,169 | 116,394 | 116,392 | 1.8 |
| BruteForce | 65,857 | 14,112 | 14,111 | 15.2 |
| **Web** | **597** | **128** | **125** | **1,675** |
| Bot | 101,175 | 21,680 | 21,680 | 9.9 |
| Infiltration | 82,319 | 17,640 | 17,639 | 12.1 |
| **Total** | **1,930,709** | **1,780,889** | **1,780,875** | |

**Web is now the only severely rare family** (ratio 1,675:1 in train), which makes it the natural GAN
target. Train contains 384 Brute Force-Web, 159 XSS and 54 SQL Injection flows. BruteForce is now
almost entirely SSH (65,829 of 65,857 flows).

### Verification (all passed ✅)
- 0 shared feature vectors between train/val, train/test and val/test.
- Chronological order respected in **29 / 29** (day, class) groups.
- Clock fix: all attacks now match the official schedule, e.g. SSH 14:01–15:32 (doc 14:01–15:31),
  Hulk 13:45 (doc 13:45), Infiltration 01/03 09:57–15:36 (doc 09:57–15:37), Bot 10:13–15:54 (doc
  10:11–15:55). **Remaining anomaly:** some Web-SQLi flows occur hours before the documented window
  (from 11:09 on 22/02, from 10:22 on 23/02). This is consistent with labelling by attacker/victim IP
  rather than by time, and is discussed as label noise.

## Tools
pandas, NumPy, PyArrow.
