"""
Step 0 - Data acquisition.

Two responsibilities:
  1. download the official CSVs from the public AWS S3 bucket;
  2. convert each CSV into a *lossless* Parquet file.

"Lossless" is important: at this stage we do NOT clean anything (no NaN/Inf
removal, no deduplication, no sampling). Those are methodological decisions
that we will take in the preprocessing step, after looking at the data (EDA),
and that we must be able to justify. Here we only make the data easier and
faster to work with:
  - repeated header lines that exist inside some CSVs are removed
    (they are file-format artefacts, not network flows);
  - text is converted to proper types (float64 features, datetime timestamp);
  - identifier columns (IPs, Flow ID) are dropped - see config.IDENTIFIER_COLUMNS;
  - two columns are added: `day` and `row_in_file` (the original order of the
    flow in its CSV), needed later for the time-aware split.
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def list_remote_files() -> pd.DataFrame:
    """List the CSV files available in the bucket, with their size in bytes."""
    url = f"{config.S3_BUCKET_URL}/?list-type=2&prefix={quote(config.S3_PREFIX)}"
    with urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    rows = []
    for item in root.findall("s3:Contents", _S3_NS):
        key = item.find("s3:Key", _S3_NS).text
        size = int(item.find("s3:Size", _S3_NS).text)
        if key.endswith(".csv"):
            rows.append({"file": key[len(config.S3_PREFIX):], "size_bytes": size})
    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    df["size_MB"] = (df["size_bytes"] / 1e6).round(1)
    return df


def download_file(file_name: str, dest_dir: Path, expected_size: int | None = None) -> Path:
    """Stream one file from the bucket to disk. Skips files already downloaded."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"[skip] {file_name} already downloaded")
        return dest

    url = f"{config.S3_BUCKET_URL}/{quote(config.S3_PREFIX + file_name)}"
    tmp = dest.with_suffix(dest.suffix + ".part")  # avoids half-written files
    print(f"[download] {file_name} ...", flush=True)
    with urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=16 * 1024 * 1024)
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise IOError(f"Incomplete download for {file_name}: "
                      f"{tmp.stat().st_size} != {expected_size} bytes")
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# 2. CSV -> Parquet conversion
# ---------------------------------------------------------------------------
def _parquet_schema() -> pa.Schema:
    fields = [pa.field("day", pa.string()),
              pa.field("row_in_file", pa.int64()),
              pa.field(config.TIMESTAMP_COLUMN, pa.timestamp("ns"))]
    fields += [pa.field(c, pa.float64()) for c in config.FEATURE_COLUMNS]
    fields += [pa.field(config.LABEL_COLUMN, pa.string())]
    return pa.schema(fields)


def csv_to_parquet(csv_path: Path, out_path: Path, day: str,
                   chunksize: int = 500_000) -> dict:
    """
    Convert one daily CSV to Parquet, chunk by chunk (the largest file is
    ~4 GB, so it cannot be loaded at once in Colab's RAM).

    Returns a small summary dict used to build the data inventory.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _parquet_schema()
    expected_cols = set(config.FEATURE_COLUMNS) | {config.TIMESTAMP_COLUMN, config.LABEL_COLUMN}

    summary = {"day": day, "file": csv_path.name, "rows": 0,
               "header_rows_removed": 0, "unparsed_timestamps": 0,
               "dropped_identifier_columns": []}
    label_counts: dict[str, int] = {}
    t_min, t_max = None, None
    next_row = 0

    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    try:
        # dtype=str: read everything as text first, so that a repeated header
        # line or a value like "Infinity" never breaks type inference.
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str,
                                 skipinitialspace=True):
            chunk.columns = chunk.columns.str.strip()

            # --- schema check & identifier columns -------------------------
            ids = [c for c in config.IDENTIFIER_COLUMNS if c in chunk.columns]
            if ids:
                chunk = chunk.drop(columns=ids)
                summary["dropped_identifier_columns"] = ids
            unexpected = set(chunk.columns) ^ expected_cols
            if unexpected:
                raise ValueError(f"{csv_path.name}: unexpected columns {sorted(unexpected)}")

            # Original position of each line in the file (before any removal).
            row_in_file = pd.RangeIndex(next_row, next_row + len(chunk))
            next_row += len(chunk)

            # --- repeated header lines -------------------------------------
            is_header = chunk[config.LABEL_COLUMN] == config.LABEL_COLUMN
            summary["header_rows_removed"] += int(is_header.sum())
            keep = ~is_header.to_numpy()
            chunk = chunk.loc[keep]
            row_in_file = row_in_file[keep]

            # --- typing -----------------------------------------------------
            ts = pd.to_datetime(chunk[config.TIMESTAMP_COLUMN],
                                format=config.TIMESTAMP_FORMAT, errors="coerce")
            summary["unparsed_timestamps"] += int(ts.isna().sum())
            # "Infinity"/"NaN" strings become +inf/NaN floats: kept on purpose,
            # they will be counted and handled during preprocessing.
            feats = chunk[config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

            out = pd.DataFrame({"day": day, "row_in_file": row_in_file.to_numpy(),
                                config.TIMESTAMP_COLUMN: ts.to_numpy()})
            out = pd.concat([out, feats.reset_index(drop=True)], axis=1)
            out[config.LABEL_COLUMN] = chunk[config.LABEL_COLUMN].str.strip().to_numpy()

            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))

            # --- bookkeeping ------------------------------------------------
            summary["rows"] += len(out)
            for lab, n in out[config.LABEL_COLUMN].value_counts().items():
                label_counts[lab] = label_counts.get(lab, 0) + int(n)
            if ts.notna().any():
                t_min = ts.min() if t_min is None else min(t_min, ts.min())
                t_max = ts.max() if t_max is None else max(t_max, ts.max())
    finally:
        writer.close()

    summary["first_timestamp"] = str(t_min)
    summary["last_timestamp"] = str(t_max)
    summary["label_counts"] = label_counts
    return summary
