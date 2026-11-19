"""
Central configuration for the IDS + GAN project.

Every notebook and script imports its paths, seeds and dataset metadata from
here, so that a single change (e.g. moving the data folder) propagates
everywhere. This is part of the reproducibility requirement of the project.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is only needed from the GAN step onwards
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT = the folder that contains src/, notebooks/, ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/ lives inside the project but is git-ignored (datasets must not be
# committed). RAW_DIR can be redirected with an environment variable: on
# Colab we point it to the fast, temporary local disk (/content/raw) because
# the raw CSVs (~7 GB) are only needed once, during conversion.
DATA_DIR = Path(os.environ.get("IDS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = Path(os.environ.get("IDS_RAW_DIR", DATA_DIR / "raw"))
INTERIM_DIR = DATA_DIR / "interim"        # lossless Parquet copy of the raw CSVs
PROCESSED_DIR = DATA_DIR / "processed"    # cleaned, split, feature-selected data
AUGMENTED_DIR = DATA_DIR / "augmented"    # training sets with synthetic samples
MODELS_DIR = DATA_DIR / "models"          # trained GANs and IDS models (can be large)
# Scratch space for large intermediate files that are rebuilt on demand. On Colab
# it points to the temporary local disk, to spare the Google Drive quota.
TMP_DIR = Path(os.environ.get("IDS_TMP_DIR", DATA_DIR / "tmp"))

# results/ is committed: small tables and figures used in the report.
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, AUGMENTED_DIR, MODELS_DIR,
              TMP_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset: CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity & CSE)
# ---------------------------------------------------------------------------
# Public, anonymous-read AWS S3 bucket (no account needed).
S3_BUCKET_URL = "https://cse-cic-ids2018.s3.amazonaws.com"
S3_PREFIX = "Processed Traffic Data for ML Algorithms/"

# One CSV per capture day. The attacks listed are the ones documented by the
# dataset authors for that day (benign traffic is present every day).
# Note: "Thuesday" is a typo in the official file name, kept on purpose.
DAY_FILES = {
    "2018-02-14": ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv", ["FTP-BruteForce", "SSH-Bruteforce"]),
    "2018-02-15": ("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv", ["DoS-GoldenEye", "DoS-Slowloris"]),
    "2018-02-16": ("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv", ["DoS-SlowHTTPTest", "DoS-Hulk"]),
    "2018-02-20": ("Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", ["DDoS-LOIC-HTTP", "DDoS-LOIC-UDP"]),
    "2018-02-21": ("Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv", ["DDoS-LOIC-UDP", "DDoS-HOIC"]),
    "2018-02-22": ("Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv", ["BruteForce-Web", "BruteForce-XSS", "SQL-Injection"]),
    "2018-02-23": ("Friday-23-02-2018_TrafficForML_CICFlowMeter.csv", ["BruteForce-Web", "BruteForce-XSS", "SQL-Injection"]),
    "2018-02-28": ("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", ["Infiltration"]),
    "2018-03-01": ("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv", ["Infiltration"]),
    "2018-03-02": ("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv", ["Bot"]),
}

# Official attack schedule (dataset page, "Table 2"), local wall-clock time,
# no time zone stated by the authors. Used in the EDA to check the timestamps.
# (label as written in the CSVs, day, start, end)
DOCUMENTED_ATTACK_WINDOWS = [
    ("FTP-BruteForce",           "2018-02-14", "10:32", "12:09"),
    ("SSH-Bruteforce",           "2018-02-14", "14:01", "15:31"),
    ("DoS attacks-GoldenEye",    "2018-02-15", "09:26", "10:09"),
    ("DoS attacks-Slowloris",    "2018-02-15", "10:59", "11:40"),
    ("DoS attacks-SlowHTTPTest", "2018-02-16", "10:12", "11:08"),
    ("DoS attacks-Hulk",         "2018-02-16", "13:45", "14:19"),
    ("DDoS attacks-LOIC-HTTP",   "2018-02-20", "10:12", "11:17"),
    ("DDOS attack-LOIC-UDP",     "2018-02-20", "13:13", "13:32"),
    ("DDOS attack-LOIC-UDP",     "2018-02-21", "10:09", "10:43"),
    ("DDOS attack-HOIC",         "2018-02-21", "14:05", "15:05"),
    ("Brute Force -Web",         "2018-02-22", "10:17", "11:24"),
    ("Brute Force -XSS",         "2018-02-22", "13:50", "14:29"),
    ("SQL Injection",            "2018-02-22", "16:15", "16:29"),
    ("Brute Force -Web",         "2018-02-23", "10:03", "11:03"),
    ("Brute Force -XSS",         "2018-02-23", "13:00", "14:10"),
    ("SQL Injection",            "2018-02-23", "15:05", "15:18"),
    ("Infilteration",            "2018-02-28", "10:50", "12:05"),
    ("Infilteration",            "2018-02-28", "13:42", "14:40"),
    ("Infilteration",            "2018-03-01", "09:57", "10:55"),
    ("Infilteration",            "2018-03-01", "14:00", "15:37"),
    ("Bot",                      "2018-03-02", "10:11", "11:34"),
    ("Bot",                      "2018-03-02", "14:24", "15:55"),
]

# Excel's maximum number of rows (including the header line). A CSV with
# exactly this many lines was very likely saved through Excel and truncated.
EXCEL_MAX_ROWS = 1_048_576

# Columns that identify hosts/connections rather than describe behaviour.
# They only exist in the 2018-02-20 file. We drop them because:
#   1. schema consistency with the 9 other files;
#   2. privacy (the course forbids publishing IP addresses);
#   3. shortcut learning: a model could learn "IP x = attacker" from the
#      testbed, which does not generalise to any other network.
IDENTIFIER_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

TIMESTAMP_COLUMN = "Timestamp"
LABEL_COLUMN = "Label"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

# The 78 flow features produced by CICFlowMeter (header order of the CSVs).
FEATURE_COLUMNS = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
assert len(FEATURE_COLUMNS) == 78


# ---------------------------------------------------------------------------
# Step 2 - preprocessing decisions (justified in notebooks/02_preprocessing.ipynb)
# ---------------------------------------------------------------------------
# D4: features with a single value over all 16.2 M flows (EDA step 1, section 4).
CONSTANT_FEATURES = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
]
MODEL_FEATURES = [c for c in FEATURE_COLUMNS if c not in CONSTANT_FEATURES]  # 70

# Time-valued features, in microseconds. None of them can be negative, and the
# inter-arrival times (IAT) cannot exceed the flow timeout used by CICFlowMeter.
TIME_FEATURES = [c for c in FEATURE_COLUMNS
                 if c == "Flow Duration" or "IAT" in c or c.startswith(("Active", "Idle"))]
IAT_FEATURES = [c for c in FEATURE_COLUMNS if "IAT" in c]
FLOW_TIMEOUT_US = 120_000_000   # 120 s: the maximum Flow Duration observed in the data

# D1: the capture ran 08:00-17:59, written on a 12-hour clock without AM/PM.
# Hours 01-07 are therefore afternoon hours (13-19); 06-07 never occur.
PM_HOURS = range(1, 8)

# Clean, consistent class names (the official labels have typos and mixed case).
LABEL_CANONICAL = {
    "Benign": "Benign",
    "FTP-BruteForce": "BruteForce-FTP",
    "SSH-Bruteforce": "BruteForce-SSH",
    "DoS attacks-GoldenEye": "DoS-GoldenEye",
    "DoS attacks-Slowloris": "DoS-Slowloris",
    "DoS attacks-SlowHTTPTest": "DoS-SlowHTTPTest",
    "DoS attacks-Hulk": "DoS-Hulk",
    "DDoS attacks-LOIC-HTTP": "DDoS-LOIC-HTTP",
    "DDOS attack-LOIC-UDP": "DDoS-LOIC-UDP",
    "DDOS attack-HOIC": "DDoS-HOIC",
    "Brute Force -Web": "Web-BruteForce",
    "Brute Force -XSS": "Web-XSS",
    "SQL Injection": "Web-SQLi",
    "Infilteration": "Infiltration",
    "Bot": "Bot",
}
LABELS = list(LABEL_CANONICAL.values())        # 15 fine-grained classes

# Main classification target: 7 attack families (decision "labels = families").
LABEL_FAMILY = {lab: lab.split("-")[0] for lab in LABELS}
FAMILIES = ["Benign", "DoS", "DDoS", "BruteForce", "Web", "Bot", "Infiltration"]
assert set(LABEL_FAMILY.values()) == set(FAMILIES)

# D6: chronological split inside every (day, class) group.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)            # train / validation / test
SPLITS = ("train", "val", "test")

# D7: Benign is downsampled in the TRAINING split only (val/test keep the real
# class proportions, so that precision and false-alarm rate stay realistic).
TRAIN_BENIGN_CAP = 1_000_000
