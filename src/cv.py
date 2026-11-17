"""ORIEON — expanding-window time-series cross-validation (10 folds).

Fold k uses the FIRST k*10% of the data, with an 80/20 train/test split inside.
Run:  python -m src.cv
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import train
from .prepare_data import prepare, split_indices

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "results" / os.environ.get("ORIEON_RUN_TAG", "") / "cv"
N_FOLDS = 10  # minimum 10


def run_cv(n_folds: int = N_FOLDS) -> pd.DataFrame:
    df, _ = prepare()
    n = len(df)
    CV_DIR.mkdir(parents=True, exist_ok=True)
    folds = []

    for k in range(1, n_folds + 1):
        end = int(n * k / n_folds)
        sub = df.iloc[:end].reset_index(drop=True)
        n_train, _ = split_indices(sub)
        print("=" * 72)
        print(f"FOLD {k}/{n_folds}: rows 0-{end} ({k * 100 // n_folds}% of data) "
              f"| train {n_train} | test {end - n_train}")
        print("=" * 72)

        fold_dir = CV_DIR / f"fold_{k:02d}"
        table = train.run_all_models(sub, results_dir=fold_dir)
        table.insert(0, "fold", k)
        table.insert(1, "test_start", str(sub["Date"].iloc[n_train].date()))
        table.insert(2, "test_end", str(sub["Date"].iloc[-1].date()))
        folds.append(table)

    all_folds = pd.concat(folds, ignore_index=True)
    all_folds.to_csv(CV_DIR / "cv_all_folds.csv", index=False)

    summary = (all_folds.groupby("Model")[["MAE", "RMSE", "R2", "MAPE"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.sort_values("RMSE_mean")
    summary.to_csv(CV_DIR / "cv_summary.csv")
    print("\nCV SUMMARY (mean and std over folds)")
    print(summary.to_string())
    return summary


if __name__ == "__main__":
    run_cv()
