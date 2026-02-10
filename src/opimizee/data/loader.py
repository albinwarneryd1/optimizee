from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from opimizee.utils.config import DATA_RAW, DataSchema


@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    source_files: list[Path]


def _pick_datetime_column(df: pd.DataFrame) -> str:
    candidates = ["timestamp", "datetime", "date", "time", "Date", "Datetime", "Timestamp"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column that looks like datetime-like strings
    return df.columns[0]


def _pick_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "consumption_kwh", "kwh", "consumption", "energy", "Energy", "Load", "load",
        "power", "Power", "usage", "Usage"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last numeric column
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return numeric_cols[-1]
    raise ValueError("Could not infer target column (no known names and no numeric columns).")


def load_raw_data(schema: DataSchema = DataSchema()) -> LoadResult:
    """
    Loads all CSV files in data/raw and concatenates them.
    Expects at least one datetime-like column and one numeric target column.
    """
    raw_dir = DATA_RAW
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Put your raw dataset(s) there.")

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file"] = f.name
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # infer columns
    dt_col = schema.datetime_col if schema.datetime_col in df_all.columns else _pick_datetime_column(df_all)
    y_col = schema.target_col if schema.target_col in df_all.columns else _pick_target_column(df_all)

    # normalize column names
    if dt_col != schema.datetime_col:
        df_all = df_all.rename(columns={dt_col: schema.datetime_col})
    if y_col != schema.target_col:
        df_all = df_all.rename(columns={y_col: schema.target_col})

    # parse datetime
    df_all[schema.datetime_col] = pd.to_datetime(df_all[schema.datetime_col], errors="coerce", utc=False)
    df_all = df_all.dropna(subset=[schema.datetime_col, schema.target_col])

    # ensure numeric target
    df_all[schema.target_col] = pd.to_numeric(df_all[schema.target_col], errors="coerce")
    df_all = df_all.dropna(subset=[schema.target_col])

    return LoadResult(df=df_all, source_files=files)
