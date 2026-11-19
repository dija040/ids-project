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

## Results (full run on Colab, 16,232,943 flows)

| # | Finding | Evidence |
|---|---|---|
| 1 | Excel truncation | **7 / 10 files** have exactly 1,048,576 CSV lines (Excel's row limit): 14/02, 15/02, 16/02, 21/02, 22/02, 23/02, 02/03. |
| 2 | Invalid timestamps | **14 flows** dated 1970 (5 on 14/02, 9 on 22/02), all Benign. |
| 3 | **12-hour clock: confirmed** | Every afternoon attack appears exactly 12 h earlier: SSH-Bruteforce documented 14:01–15:31, observed 02:01–03:32; DoS-Hulk 13:45 → 01:45; DDoS-HOIC 14:05 → 02:11; XSS 13:50 → 01:51. Morning attacks match (FTP 10:32 → 10:33, GoldenEye 09:26 → 09:27). The per-hour heatmap (`01_flows_per_hour.png`) shows traffic at hours 01–05 and 08–12 **and none at 06–07 on any day**, i.e. the capture runs 08:00–17:59 and hours 01–05 are 13:00–17:59. A label-free correction is therefore *hour 01–07 → +12 h*. |
| 4 | NaN / inf | `Flow Byts/s`: 59,721 NaN + 36,039 +inf; `Flow Pkts/s`: 95,760 +inf, i.e. the same **95,760 flows (0.59%)**, all with zero duration. Present almost only in Benign (0.72%) and Infiltration (0.80%); ~0% in the other attacks. |
| 5 | Negative / impossible values | ~15 flows with durations or inter-arrival times around **−9 × 10¹¹ µs**, and `Flow/Fwd IAT Max` up to **+9.8 × 10¹¹ µs** (much longer than the 120 s flow timeout): timer bugs. `Init Fwd/Bwd Win Byts = −1` in 4.4 M / 8.3 M flows is a sentinel ("not observed"), not an error. |
| 6 | Constant features | 8: `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd/Bwd Byts/b Avg`, `Fwd/Bwd Pkts/b Avg`, `Fwd/Bwd Blk Rate Avg`. |
| 7 | **Exact duplicates** | **4,243,403 flows (26.1%)**. FTP-BruteForce: 99.97% (193,360 flows, only ~54 distinct). DoS-SlowHTTPTest: 99.96% (~55 distinct). HOIC 71%, Hulk 69%, SSH 50%, Bot 50%, Benign 21%. Unique flows left: ~11.99 M. |
| 8 | **Conflicting labels** | **22,348 feature vectors are labelled both Benign and Infiltration.** Infiltration's top destination ports (53 / 443 / 3389) are the same as Benign's. Smaller conflicts: FTP ↔ SlowHTTPTest ↔ SSH (14 vectors), Web ↔ SQLi (5). |
| 9 | Very rare classes | SQL Injection 87 (84 unique), XSS 230 (228), Brute Force-Web 611 (555), LOIC-UDP 1,730. |
| 10 | Redundant features | Many pairs with ρ = 1 (`Tot Fwd Pkts` ↔ `Subflow Fwd Pkts`, `Pkt Len Std` ↔ `Pkt Len Var`, ...); table `01_highly_correlated_pairs.csv`. |
| 11 | `Dst Port` concentration | Almost a class identifier: FTP & SlowHTTPTest → 21 (100%), SSH → 22 (100%), Bot → 8080 (98%), all DoS/DDoS/web attacks → 80 (75–100%). Benign → 53 / 443 / 3389. |

### Additional anomalies found by comparing with the official schedule
- **LOIC-UDP on 20/02 is probably labelled as LOIC-HTTP.** LOIC-HTTP is documented 10:12–11:17,
  but labelled flows start at 01:14, i.e. 13:14 on a 24 h clock, inside the documented LOIC-UDP
  window (13:13–13:32). To verify in step 2 (protocol of these flows: UDP = 17).
- **Web attacks and Bot are labelled outside their documented windows.** SQL Injection on 22/02 is
  documented 16:15–16:29 (04:15 on the 12 h clock) but observed from 01:52 to 11:09. Bot is observed
  over the whole capture (01:00–12:59). This is consistent with labels assigned by attacker/victim IP
  rather than by time (for Bot, infected hosts may also beacon all day).
- **Infiltration labels describe hosts, not malicious flows.** Once a victim is compromised, its
  ordinary traffic (DNS, HTTPS, RDP) is labelled Infiltration, so it is identical to benign traffic.
  Expect a hard ceiling on Infiltration recall.

## Decisions and justification
_To be decided together before step 2._
