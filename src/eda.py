"""
Step 1 - Exploratory Data Analysis helpers.

The full dataset (16.2 M flows x 78 features, float64) is ~10 GB in memory,
which does not fit in Colab's RAM. We therefore use two complementary tools:

1. A *streamed audit* (`audit_dataset`): one pass over every Parquet file,
   batch by batch, that computes EXACT statistics on 100% of the data:
   NaN / inf / negative counts, min / max per feature, timestamp
   anomalies, flow counts per hour, and a 64-bit hash of every row
   (used to find duplicates without holding the rows in memory).

2. A *stratified sample* (collected during the same pass): up to N flows
   per class, used for everything that needs the actual values
   (distributions, correlations, ports). Stratified, because a plain
   random sample of 1% would contain ~1 SQL-Injection flow.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config

F = config.FEATURE_COLUMNS
LAB = config.LABEL_COLUMN
TS = config.TIMESTAMP_COLUMN


def iter_batches(path: Path, batch_size: int = 500_000, columns=None):
    """Yield a Parquet file as successive pandas DataFrames."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def audit_dataset(day_paths: dict[str, Path], label_totals: dict[str, int],
                  sample_per_class: int = 30_000, seed: int = config.SEED,
                  batch_size: int = 500_000) -> dict:
    """
    Single streamed pass over all days. Returns a dict of results
    (see keys at the end of the function).
    """
    rng = np.random.default_rng(seed)
    labels = sorted(label_totals)
    label_code = {lab: i for i, lab in enumerate(labels)}
    days = list(day_paths)
    # Sampling probability per class, so that each class contributes about
    # `sample_per_class` flows (or all of them, if it is smaller).
    p_keep = {lab: min(1.0, sample_per_class / n) for lab, n in label_totals.items()}

    n_nan = np.zeros(len(F), dtype=np.int64)
    n_posinf = np.zeros(len(F), dtype=np.int64)
    n_neginf = np.zeros(len(F), dtype=np.int64)
    n_neg = np.zeros(len(F), dtype=np.int64)
    vmin = np.full(len(F), np.inf)
    vmax = np.full(len(F), -np.inf)

    hash_parts, hour_parts, window_parts, samples = [], [], [], []
    bad_ts = []

    for d_idx, (day, path) in enumerate(day_paths.items()):
        for df in iter_batches(path, batch_size):
            X = df[F].to_numpy(dtype=np.float64)

            # --- data quality counters (exact, 100% of rows) ---------------
            n_nan += np.isnan(X).sum(axis=0)
            n_posinf += np.isposinf(X).sum(axis=0)
            n_neginf += np.isneginf(X).sum(axis=0)
            finite = np.where(np.isfinite(X), X, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
                n_neg += (finite < 0).sum(axis=0)
                vmin = np.fmin(vmin, np.nanmin(finite, axis=0))
                vmax = np.fmax(vmax, np.nanmax(finite, axis=0))

            # --- row hashes for duplicate detection ------------------------
            # Hash of the 78 feature values only (timestamp excluded): two
            # flows with identical features are indistinguishable for a model.
            hash_parts.append(pd.DataFrame({
                "day": np.int8(d_idx),
                "fhash": pd.util.hash_pandas_object(df[F], index=False).to_numpy(),
                "label": df[LAB].map(label_code).to_numpy(np.int16),
            }))

            # --- timestamps ------------------------------------------------
            ts = df[TS]
            wrong_year = ts.dt.year != 2018
            if wrong_year.any():
                bad_ts.append(df.loc[wrong_year, [TS, LAB, "row_in_file"]].assign(day=day))
            ok = ~wrong_year
            hour_parts.append(
                df.loc[ok].groupby([df.loc[ok, TS].dt.hour.rename("hour"), LAB]).size()
                  .rename("flows").reset_index().assign(day=day))
            window_parts.append(
                df.loc[ok].groupby(LAB)[TS].agg(["min", "max", "size"]).reset_index().assign(day=day))

            # --- stratified sample -----------------------------------------
            keep = rng.random(len(df)) < df[LAB].map(p_keep).to_numpy()
            samples.append(df.loc[keep])

    quality = pd.DataFrame({
        "nan": n_nan, "+inf": n_posinf, "-inf": n_neginf, "negative": n_neg,
        "min": vmin, "max": vmax}, index=F)
    quality["constant"] = quality["min"] == quality["max"]

    hours = (pd.concat(hour_parts).groupby(["day", "hour", LAB])["flows"].sum().reset_index())
    win = pd.concat(window_parts)
    windows = (win.groupby(["day", LAB])
                  .agg(first=("min", "min"), last=("max", "max"), flows=("size", "sum"))
                  .reset_index())

    hashes = pd.concat(hash_parts, ignore_index=True)
    hashes["day"] = pd.Categorical.from_codes(hashes["day"], categories=days)
    hashes["label"] = pd.Categorical.from_codes(hashes["label"], categories=labels)

    return {
        "quality": quality,                      # per-feature data quality (exact)
        "hours": hours,                          # flows per (day, hour, label)
        "windows": windows,                      # first/last timestamp per (day, label)
        "bad_timestamps": (pd.concat(bad_ts) if bad_ts else pd.DataFrame()),
        "hashes": hashes,                        # one row per flow: day, fhash, label
        "sample": pd.concat(samples, ignore_index=True),
    }


def duplicate_report(hashes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two kinds of duplicates, computed on all 16 M flows via their hashes:

    * exact duplicates: same features AND same label. Harmless for the
      label, but they inflate the dataset, bias the model towards repeated
      patterns and, worst of all, can end up in both train and test
      (the model is then tested on flows it has memorised).

    * conflicting duplicates: same features but DIFFERENT labels.
      No model can classify both correctly: this is direct evidence of
      label noise (or of features that are too coarse to tell them apart).
    """
    dup = hashes.duplicated(["fhash", "label"], keep="first")
    per_label = (pd.DataFrame({"flows": hashes.groupby("label", observed=True).size(),
                               "exact_duplicates": hashes[dup].groupby("label", observed=True).size()})
                   .fillna(0).astype(int))
    per_label["dup_percent"] = (100 * per_label["exact_duplicates"] / per_label["flows"]).round(2)

    uniq = hashes.drop_duplicates(["fhash", "label"])
    n_labels = uniq.groupby("fhash")["label"].transform("size")
    conflicts = uniq[n_labels > 1]
    # Which labels collide? Encode the set of labels of each feature vector
    # as a bitmask (one bit per class) - much faster than joining strings
    # group by group when there are hundreds of thousands of conflicts.
    cats = list(hashes["label"].cat.categories)
    bits = np.left_shift(np.int64(1), conflicts["label"].cat.codes.to_numpy().astype(np.int64))
    masks = pd.Series(bits, index=conflicts["fhash"].to_numpy()).groupby(level=0).sum()
    pairs = masks.value_counts().rename("distinct_feature_vectors").to_frame()
    pairs.index = [" | ".join(c for i, c in enumerate(cats) if (m >> i) & 1) for m in pairs.index]
    pairs.index.name = "labels sharing identical features"
    per_label["conflicting_feature_vectors"] = (
        conflicts.groupby("label", observed=True).size().reindex(per_label.index).fillna(0).astype(int))
    return per_label.sort_values("flows", ascending=False), pairs


def signed_log(x):
    """sign(x) * log10(1 + |x|): a log scale that tolerates 0 and negatives."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1 + np.abs(x))


def documented_vs_observed(windows: pd.DataFrame) -> pd.DataFrame:
    """Compare the official attack schedule with the timestamps found in the data."""
    doc = pd.DataFrame(config.DOCUMENTED_ATTACK_WINDOWS,
                       columns=["label", "day", "doc_start", "doc_end"])
    # Several documented windows per (day, label) are merged: we compare
    # with the earliest start and the latest end.
    doc = doc.groupby(["day", "label"]).agg(doc_start=("doc_start", "min"),
                                            doc_end=("doc_end", "max")).reset_index()
    obs = windows.rename(columns={LAB: "label"})
    obs = obs[obs["label"] != "Benign"].copy()
    obs["obs_first"] = obs["first"].dt.strftime("%H:%M")
    obs["obs_last"] = obs["last"].dt.strftime("%H:%M")
    out = doc.merge(obs[["day", "label", "obs_first", "obs_last", "flows"]],
                    on=["day", "label"], how="outer")
    return out.sort_values(["day", "doc_start"]).reset_index(drop=True)
