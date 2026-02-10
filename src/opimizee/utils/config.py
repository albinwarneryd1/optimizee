from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DataSchema:
    """
    Lightweight schema conventions.
    If your raw data uses different column names, we map them in loader.py.
    """
    datetime_col: str = "timestamp"
    target_col: str = "consumption_kwh"
