from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


@dataclass
class EnergyModel:
    """
    Practical baseline model:
    - numeric features only (robust for messy datasets)
    - RandomForest for non-linear patterns
    """
    feature_cols: list[str]
    pipeline: Pipeline | None = None

    def build(self) -> None:
        numeric_features = self.feature_cols

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric_features)
            ],
            remainder="drop",
        )

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )

        self.pipeline = Pipeline(steps=[("pre", pre), ("model", model)])

    def train(self, df: pd.DataFrame, target_col: str) -> None:
        if self.pipeline is None:
            self.build()
        X = df[self.feature_cols]
        y = df[target_col]
        self.pipeline.fit(X, y)

    def predict(self, df: pd.DataFrame) -> list[float]:
        if self.pipeline is None:
            raise RuntimeError("Model not trained/loaded. Train first or load from disk.")
        X = df[self.feature_cols]
        return self.pipeline.predict(X).tolist()
