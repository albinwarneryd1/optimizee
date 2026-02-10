from __future__ import annotations

import argparse
import pandas as pd
import joblib

from opimizee.data.loader import load_raw_data
from opimizee.data.preprocess import preprocess_data
from opimizee.ml.train import train_model
from opimizee.utils.config import DATA_PROCESSED, MODELS_DIR, DataSchema


def cmd_preprocess() -> None:
    res = load_raw_data(schema=DataSchema())
    df = preprocess_data(res.df, schema=DataSchema())

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "processed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[ok] processed data saved -> {out_path} (rows={len(df)})")


def cmd_train() -> None:
    in_path = DATA_PROCESSED / "processed.parquet"
    if not in_path.exists():
        raise FileNotFoundError("processed.parquet not found. Run: opimizee preprocess")

    df = pd.read_parquet(in_path)
    result = train_model(df, schema=DataSchema())
    print(f"[ok] trained model saved -> {result.model_path}")
    print(f"MAE={result.mae:.4f} | RMSE={result.rmse:.4f}")


def cmd_status() -> None:
    p = DATA_PROCESSED / "processed.parquet"
    m = MODELS_DIR / "model.joblib"
    print(f"processed: {'yes' if p.exists() else 'no'} ({p})")
    print(f"model:     {'yes' if m.exists() else 'no'} ({m})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="opimizee")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preprocess")
    sub.add_parser("train")
    sub.add_parser("status")
    sub.add_parser("dash")

    args = parser.parse_args()

    if args.cmd == "preprocess":
        cmd_preprocess()
    elif args.cmd == "train":
        cmd_train()
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "dash":
        from opimizee.viz.dashboard import run_dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
