# Project proposal: GAN-Augmented Network Intrusion Detection on CSE-CIC-IDS2018

**Module:** AI1, Practical mini-project · **Topic:** 1, Network Intrusion Detection System (NIDS)
**Author:** _your name_ · **Date:** _dd/mm/2026_ · **Status:** draft v1

## 1. Problem and motivation
Flow-based NIDS learned from labelled traffic detect frequent attacks (DoS, DDoS) well but often
miss **rare attack families** (web attacks, infiltration). Those families have very few training
examples. Missing them is costly: they are typically the stealthiest intrusions.
**Research question:** does augmenting the minority attack classes with synthetic flows from a
**Vanilla GAN** improve their detection, compared with (a) no rebalancing and (b) SMOTE?

## 2. Threat model
- **Asset:** an organisation's internal network (the AWS testbed of CIC-IDS2018: ~400 hosts).
- **Attacker:** external or already-inside, able to run brute force (FTP/SSH/Web), DoS/DDoS,
  web exploitation (XSS, SQL injection), botnet C&C and internal infiltration.
- **Defender's view:** a passive sensor that sees **flow statistics only** (CICFlowMeter features).
  It does not see payloads or IP reputation. The IDS raises alerts for a SOC analyst.
- **Out of scope:** encrypted payload inspection, an attacker who actively evades the model
  (discussed only as a robustness limitation), and real-time deployment.

## 3. Dataset
CSE-CIC-IDS2018 (public, CSE & CIC, 2018): 10 capture days with ~16 M labelled flows, 78
numerical flow features, 1 benign class and 14 attack labels. The dataset is highly imbalanced:
benign traffic is more than 80% of flows, and some attack classes have fewer than 1,000 flows.
**Known limitations** (to be discussed): label noise, duplicated flows and feature artefacts
reported in the literature. Attacks are concentrated on specific days.

## 4. ML formulation
- **Main task:** multi-class classification of flows (benign + attack families), plus a derived
  binary view (benign vs. attack) for alert-level metrics.
- **Split:** time-aware. Within each day, the earliest flows go to train, then validation, then
  test (70/15/15). This respects chronology while keeping every attack family in every split.
  The test set is held out and never used for tuning.
- **Leakage rules:** scaling, feature selection, SMOTE and GAN training are fitted on the
  training split only. Host identifiers (IPs, Flow ID) are removed.

## 5. Imbalance strategy (the core of the project)
Three training-set variants, each evaluated on the same untouched test set:
1. **Original** (no rebalancing; class weights only where relevant)
2. **SMOTE** oversampling of minority classes
3. **Vanilla GAN**: one generator/discriminator pair per minority class. The quality of the
   synthetic data is checked (distribution comparison, t-SNE, train-on-synthetic/test-on-real).

## 6. Models
| Role | Model | Family |
|---|---|---|
| Baseline | Logistic Regression | linear |
| Model 2 | Random Forest | bagging ensemble of trees |
| Model 3 | LightGBM (or XGBoost) | gradient boosting |
| Model 4 | Multi-Layer Perceptron (PyTorch) | deep learning |

Hyperparameters are tuned on the validation split.

## 7. Evaluation
- Precision, Recall, F1 (macro and per class), **PR-AUC**, ROC-AUC, confusion matrix.
- **Detection rate at a fixed false-alarm rate** (e.g. 1%): the operating point a SOC can afford.
- **Security-specific analysis:** per-attack-family recall (original vs. SMOTE vs. GAN) and
  false-positive cost analysis. Optional: cross-dataset test on CIC-IDS2017.
- **Explainability:** SHAP / feature importance on the best model, with a security reading
  (which flow characteristics reveal each attack, and which may be dataset artefacts).

## 8. Tools and planning
Python, pandas, scikit-learn, imbalanced-learn, LightGBM, PyTorch, SHAP. Runs on Google Colab
(CPU is sufficient) with Git/GitHub for version control.
Steps: acquisition and EDA → preprocessing and split → feature selection → GAN → models →
evaluation and analysis → report and defense.

## 9. Risks
GAN instability or mode collapse on tabular data (mitigation: per-class GANs, loss monitoring,
quality checks). Dataset size versus Colab RAM (mitigation: Parquet and chunked processing,
documented sampling). Overly optimistic scores because of dataset artefacts (mitigation: critical
analysis, cross-dataset test).
