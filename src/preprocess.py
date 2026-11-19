"""
Step 2 - Preprocessing: from the raw Parquet copy to train / val / test sets.

Three passes, so that the whole dataset never has to be in memory at once:

  Pass 1  clean_day()          one day at a time
          - drop invalid timestamps (1970)            -> D1
          - drop NaN/inf rows and impossible times    -> D4
          - normalise label names, add the family label
          - fix LOIC-UDP flows mislabelled as LOIC-HTTP (UDP protocol)
          - convert the 12-hour clock to 24 h         -> D1
          - drop constant features, store as float32
          - hash every flow (for pass 2), sort the day chronologically

  Pass 2  plan_dedup_and_split()   all days, but only 3 small columns
          - remove conflicting feature vectors        -> D3
          - remove exact duplicates (keep the first in time) -> D2
          - chronological 70/15/15 split in every (day, class) -> D6
          - downsample Benign in the training split   -> D7

  Pass 3  write_splits()       one day at a time
          - write data/processed/{train,val,test}.parquet

Every removal is counted per class, so that the report can show exactly
what each decision cost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .eda import iter_batches

LAB, TS = config.LABEL_COLUMN, config.TIMESTAMP_COLUMN
FEATS = config.MODEL_FEATURES
LABEL_DTYPE = pd.CategoricalDtype(config.LABELS)
FAMILY_DTYPE = pd.CategoricalDtype(config.FAMILIES)

# Codes used by pass 2 for every cleaned flow.
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
REMOVED_DUPLICATE, REMOVED_CONFLICT, REMOVED_BENIGN_SAMPLING = -1, -2, -3


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------
def fix_clock(ts: pd.Series) -> pd.Series:
    """12-hour clock without AM/PM -> 24 h: hours 01-07 are afternoon hours."""
    pm = ts.dt.hour.isin(config.PM_HOURS).to_numpy()
    shift = np.where(pm, np.timedelta64(12, "h"), np.timedelta64(0, "h"))
    return pd.Series(ts.to_numpy() + shift, index=ts.index, name=ts.name)


def _log(log: list, df: pd.DataFrame, mask, day: str, rule: str):
    """Record how many flows of each class a rule removes/changes, then return the kept rows."""
    mask = np.asarray(mask)
    if mask.any():
        counts = df.loc[mask, LAB].value_counts()
        log.extend({"day": day, "rule": rule, "label": k, "flows": int(v)}
                   for k, v in counts.items() if v > 0)
    return df.loc[~mask]


def clean_day(day: str, src: Path, dst: Path, batch_size: int = 500_000) -> list[dict]:
    """Pass 1 for one day. Writes `dst` (sorted, cleaned) and returns the log."""
    log: list[dict] = []
    parts = []
    for df in iter_batches(src, batch_size):
        # Normalise the labels first, so that every log entry uses the clean names.
        df[LAB] = df[LAB].map(config.LABEL_CANONICAL)
        if df[LAB].isna().any():
            raise ValueError(f"{day}: unknown label(s) found")

        # D1: invalid timestamps (Unix epoch values).
        df = _log(log, df, df[TS].dt.year != 2018, day, "removed: invalid timestamp (1970)")

        # D4: NaN / inf (rates computed with a zero duration).
        X = df[config.FEATURE_COLUMNS].to_numpy()
        df = _log(log, df, ~np.isfinite(X).all(axis=1), day, "removed: NaN or inf value")

        # D4: impossible times (negative, or an inter-arrival time longer than the flow timeout).
        T = df[config.TIME_FEATURES].to_numpy()
        I = df[config.IAT_FEATURES].to_numpy()
        bad = (T < 0).any(axis=1) | (I > config.FLOW_TIMEOUT_US).any(axis=1)
        df = _log(log, df, bad, day, "removed: impossible time value")

        # Label correction: HTTP runs over TCP, so a "LOIC-HTTP" flow using UDP (protocol 17)
        # can only be the LOIC-UDP attack documented that afternoon (13:13-13:32).
        wrong = ((df[LAB] == "DDoS-LOIC-HTTP") & (df["Protocol"] == 17)).to_numpy()
        if wrong.any():
            _log(log, df, wrong, day, "relabelled: LOIC-HTTP over UDP -> DDoS-LOIC-UDP")
            df.loc[wrong, LAB] = "DDoS-LOIC-UDP"

        # Fingerprint of the 70 informative features, computed on the exact float64
        # values (before the float32 conversion below), used for deduplication in pass 2.
        fhash = pd.util.hash_pandas_object(df[FEATS], index=False).to_numpy()

        out = pd.DataFrame({
            "day": day,
            "row_in_file": df["row_in_file"].to_numpy(),
            TS: fix_clock(df[TS]).to_numpy(),
        })
        # float32 halves memory and disk; it is the precision models use anyway
        # (relative error ~1e-7, far below any meaningful difference between flows).
        feats = df[FEATS].astype(np.float32).reset_index(drop=True)
        out = pd.concat([out, feats], axis=1)
        out[LAB] = pd.Categorical(df[LAB].to_numpy(), dtype=LABEL_DTYPE)
        out["family"] = pd.Categorical(df[LAB].map(config.LABEL_FAMILY).to_numpy(), dtype=FAMILY_DTYPE)
        out["fhash"] = fhash
        parts.append(out)

    day_df = pd.concat(parts, ignore_index=True)
    # Chronological order (corrected clock); the original file order breaks ties,
    # because timestamps only have a 1-second resolution.
    day_df = day_df.sort_values([TS, "row_in_file"], kind="stable", ignore_index=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    day_df.to_parquet(dst, index=False)
    log.append({"day": day, "rule": "kept after cleaning", "label": "ALL", "flows": len(day_df)})
    return log


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------
def plan_dedup_and_split(clean_paths: dict[str, Path],
                         fractions=config.SPLIT_FRACTIONS,
                         train_benign_cap: int = config.TRAIN_BENIGN_CAP,
                         seed: int = config.SEED):
    """
    Decide the fate of every cleaned flow. Returns:
      codes : {day: int8 array}  0/1/2 = train/val/test, negative = removed
      stats : DataFrame with, per label, the number of flows removed at each stage
    Only three small columns of all 16 M flows are loaded (~200 MB).
    """
    days = list(clean_paths)
    H = pd.concat([pd.read_parquet(p, columns=["day", "fhash", LAB]) for p in clean_paths.values()],
                  ignore_index=True)
    H["day"] = H["day"].astype(pd.CategoricalDtype(days))
    # Days are concatenated in chronological order and each day is sorted, so the
    # row order of H is the global chronological order.

    # D3: feature vectors that carry more than one label cannot be learned.
    labels_per_vector = H.drop_duplicates(["fhash", LAB])["fhash"].value_counts()
    conflict = H["fhash"].isin(labels_per_vector.index[labels_per_vector > 1]).to_numpy()

    # D2: exact duplicates (same features AND same label): keep the first occurrence.
    duplicate = H.duplicated(["fhash", LAB], keep="first").to_numpy() & ~conflict

    code = np.full(len(H), REMOVED_DUPLICATE, dtype=np.int8)
    code[conflict] = REMOVED_CONFLICT
    keep = ~conflict & ~duplicate

    # D6: chronological split inside every (day, class) group.
    K = H.loc[keep, ["day", LAB]]
    grp = K.groupby(["day", LAB], observed=True)
    position = grp.cumcount().to_numpy()
    size = grp[LAB].transform("size").to_numpy()
    frac = position / size
    t1, t2 = fractions[0], fractions[0] + fractions[1]
    code[keep] = np.where(frac < t1, 0, np.where(frac < t2, 1, 2)).astype(np.int8)

    # D7: random Benign downsampling, training split only (seeded -> reproducible).
    benign_train = np.flatnonzero((code == 0) & (H[LAB] == "Benign").to_numpy())
    n_drop = max(0, len(benign_train) - train_benign_cap)
    if n_drop:
        rng = np.random.default_rng(seed)
        code[rng.choice(benign_train, size=n_drop, replace=False)] = REMOVED_BENIGN_SAMPLING

    # Per-label accounting.
    names = {REMOVED_CONFLICT: "removed: conflicting label", REMOVED_DUPLICATE: "removed: exact duplicate",
             REMOVED_BENIGN_SAMPLING: "removed: Benign train downsampling",
             0: "train", 1: "val", 2: "test"}
    stats = (pd.crosstab(H[LAB], pd.Series(code, name="stage").map(names))
               .reindex(columns=list(names.values()), fill_value=0))
    stats.insert(0, "after cleaning", stats.sum(axis=1))

    # Split the code array back per day.
    lengths = H.groupby("day", observed=False).size().reindex(days).to_numpy()
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    codes = {d: code[bounds[i]:bounds[i + 1]] for i, d in enumerate(days)}
    return codes, stats


# ---------------------------------------------------------------------------
# Pass 3
# ---------------------------------------------------------------------------
def write_splits(clean_paths: dict[str, Path], codes: dict[str, np.ndarray], out_dir: Path) -> dict:
    """Write train/val/test Parquet files, streaming one day at a time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    rows = {s: 0 for s in config.SPLITS}
    try:
        for day, path in clean_paths.items():
            df = pd.read_parquet(path)
            c = codes[day]
            assert len(c) == len(df), f"{day}: code/row mismatch"
            for split, k in SPLIT_CODE.items():
                part = df.loc[c == k]
                if part.empty:
                    continue
                table = pa.Table.from_pandas(part, preserve_index=False)
                if split not in writers:
                    writers[split] = pq.ParquetWriter(out_dir / f"{split}.parquet", table.schema,
                                                      compression="zstd")
                writers[split].write_table(table.cast(writers[split].schema))
                rows[split] += len(part)
    finally:
        for w in writers.values():
            w.close()
    return rows
