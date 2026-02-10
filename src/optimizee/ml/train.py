from __future__ import annotations

from dataclasses import dataclass
import math
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from optimizee.ml.model import EnergyModel
from optimizee.utils.config import MODELS_DIR, DataSchema


@dataclass(frozen=True)
class TrainResult:
    mae: float
    rmse: float
    model_path: str


def train_model(df: pd.DataFrame, schema: DataSchema = DataSchema()) -> TrainResult:
    feature_cols = ["hour", "dayofweek", "month", "is_weekend", "lag_1", "lag_24", "rolling_mean_24", "rolling_std_24"]

    # simple time split (no shuffle)
    df = df.sort_values(schema.datetime_col)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    model = EnergyModel(feature_cols=feature_cols)
    model.train(train_df, target_col=schema.target_col)

    preds = model.predict(test_df)
    y_true = test_df[schema.target_col].tolist()

    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(math.sqrt(mean_squared_error(y_true, preds)))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "schema": schema,
        },
        model_path,
    )

    return TrainResult(mae=mae, rmse=rmse, model_path=str(model_path))
