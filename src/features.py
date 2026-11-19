"""
Step 3 - Feature selection (fitted on the TRAINING split only).

Pipeline:
  1. candidate features = the 70 non-constant features, minus `Dst Port` (D5)
  2. redundancy filter : for each pair with |Spearman rho| >= 0.95, drop the
                         feature that carries less information about the target
                         (mutual information)
  3. ranking           : average of two rankings, mutual information (filter
                         method) and Random Forest importance (embedded method)
  4. choice of k       : macro-F1 on the VALIDATION split for k = 5, 10, 15, ...;
                         keep the smallest k within 0.005 of the best score
                         (parsimony: fewer features = faster, simpler, easier to explain)

Selection uses a class-stratified subsample of train (at most N flows per
family): with 1 M Benign flows, a plain sample would hide the rare families.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

from . import config

TARGET = "family"


def load_split(split: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / f"{split}.parquet", columns=columns)


def stratified_subsample(df: pd.DataFrame, target: str, n_per_class: int,
                         seed: int = config.SEED) -> pd.DataFrame:
    """At most `n_per_class` rows of every class (all rows of smaller classes)."""
    parts = [g if len(g) <= n_per_class else g.sample(n_per_class, random_state=seed)
             for _, g in df.groupby(target, observed=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle


def mutual_information(X: pd.DataFrame, y: pd.Series, n_max: int = 60_000,
                       seed: int = config.SEED) -> pd.Series:
    """MI between each feature and the target (k-NN estimator; subsampled for speed)."""
    if len(X) > n_max:
        idx = X.sample(n_max, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    mi = mutual_info_classif(X.to_numpy(), y.cat.codes.to_numpy(), random_state=seed)
    return pd.Series(mi, index=X.columns, name="mutual_information").sort_values(ascending=False)


def drop_redundant(X: pd.DataFrame, mi: pd.Series, threshold: float = 0.95):
    """
    Greedy redundancy filter. Pairs are processed from the most to the least
    correlated; in each pair still intact, the feature with the lower MI is dropped.
    Returns (kept features, table explaining every drop).
    """
    rho = X.corr(method="spearman").abs()
    upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1)).stack()
    pairs = upper[upper >= threshold].sort_values(ascending=False)
    dropped, log = set(), []
    for (a, b), r in pairs.items():
        if a in dropped or b in dropped:
            continue
        loser, winner = (a, b) if mi[a] < mi[b] else (b, a)
        dropped.add(loser)
        log.append({"dropped": loser, "kept_instead": winner, "spearman_rho": round(float(r), 4),
                    "mi_dropped": round(float(mi[loser]), 4), "mi_kept": round(float(mi[winner]), 4)})
    kept = [c for c in X.columns if c not in dropped]
    return kept, pd.DataFrame(log)


def rf_importance(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100,
                  seed: int = config.SEED) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                                class_weight="balanced_subsample", min_samples_leaf=2)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns, name="rf_importance").sort_values(ascending=False)


def combined_ranking(mi: pd.Series, imp: pd.Series) -> pd.DataFrame:
    r = pd.DataFrame({"mutual_information": mi, "rf_importance": imp}).dropna()
    r["rank_mi"] = r["mutual_information"].rank(ascending=False)
    r["rank_rf"] = r["rf_importance"].rank(ascending=False)
    r["mean_rank"] = (r["rank_mi"] + r["rank_rf"]) / 2
    return r.sort_values("mean_rank")


def macro_f1_for_k(ranked: list[str], ks, X_tr, y_tr, X_va, y_va,
                   n_estimators: int = 60, seed: int = config.SEED) -> pd.DataFrame:
    """Validation macro-F1 of a Random Forest trained on the top-k features, for each k."""
    rows = []
    for k in ks:
        cols = ranked[:k]
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                                    class_weight="balanced_subsample", min_samples_leaf=2)
        rf.fit(X_tr[cols], y_tr)
        pred = rf.predict(X_va[cols])
        rows.append({"k": k, "val_macro_f1": f1_score(y_va, pred, average="macro"),
                     "val_web_f1": f1_score(y_va, pred, labels=["Web"], average="macro")})
    return pd.DataFrame(rows)


def choose_k(curve: pd.DataFrame, tolerance: float = 0.005) -> int:
    best = curve["val_macro_f1"].max()
    return int(curve.loc[curve["val_macro_f1"] >= best - tolerance, "k"].min())


def save_selection(features: list[str], extra: dict, path: Path | None = None) -> Path:
    path = path or (config.PROCESSED_DIR / "selected_features.json")
    path.write_text(json.dumps({"features": features, **extra}, indent=2))
    return path


def load_selection(path: Path | None = None) -> dict:
    path = path or (config.PROCESSED_DIR / "selected_features.json")
    return json.loads(path.read_text())
