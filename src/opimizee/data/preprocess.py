from __future__ import annotations

import pandas as pd

from opimizee.utils.config import DataSchema


def add_time_features(df: pd.DataFrame, schema: DataSchema = DataSchema()) -> pd.DataFrame:
    ts = df[schema.datetime_col]
    out = df.copy()

    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Rolling signals (helpful for forecasting-ish behavior)
    out = out.sort_values(schema.datetime_col)
    out["lag_1"] = out[schema.target_col].shift(1)
    out["lag_24"] = out[schema.target_col].shift(24)
    out["rolling_mean_24"] = out[schema.target_col].rolling(24, min_periods=1).mean()
    out["rolling_std_24"] = out[schema.target_col].rolling(24, min_periods=1).std().fillna(0)

    out = out.dropna(subset=["lag_1"])  # keep it simple: drop first row
    return out


def preprocess_data(df: pd.DataFrame, schema: DataSchema = DataSchema()) -> pd.DataFrame:
    """
    Minimal, stable preprocessing:
    - time-based features
    - lags / rolling stats
    """
    df = df.copy()
    df = df.drop_duplicates(subset=[schema.datetime_col]).sort_values(schema.datetime_col)
    df = add_time_features(df, schema=schema)
    return df
