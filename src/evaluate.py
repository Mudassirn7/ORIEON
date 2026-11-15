"""ORIEON evaluation — Section 5 of the approved protocol.

Metrics are always computed from the saved prediction arrays
(results/<ModelName>_test_predictions.csv), inverse-transformed back to the
original price scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, MSE, RMSE, R^2, MAPE (division-by-zero guarded), n."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    nz = actual != 0  # guard against division by zero
    mape = float(np.mean(np.abs(err[nz] / actual[nz])) * 100.0) if nz.any() else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "n": int(len(actual))}


def save_predictions(name: str, dates, actual, predicted,
                     results_dir: Path = RESULTS_DIR) -> Path:
    """Save results/<ModelName>_test_predictions.csv (Date, actual, predicted)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}_test_predictions.csv"
    dates = pd.Series(pd.to_datetime(list(dates))).dt.strftime("%Y-%m-%d")
    pd.DataFrame({
        "Date": dates,
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    }).to_csv(out, index=False)
    return out


def load_saved_metrics(results_dir: Path = RESULTS_DIR) -> dict:
    """Recompute metrics from the saved CSVs (metrics always come from the
    saved arrays, per protocol)."""
    metrics = {}
    for csv in sorted(results_dir.glob("*_test_predictions.csv")):
        name = csv.name.replace("_test_predictions.csv", "")
        df = pd.read_csv(csv)
        metrics[name] = compute_metrics(df["actual"].values, df["predicted"].values)
    return metrics


def summarize(records: list[dict], results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Combined results table sorted by RMSE, with parameter counts and
    training seconds. `records` carry name/n_params/train_seconds; metrics are
    recomputed from the saved prediction files."""
    metrics = load_saved_metrics(results_dir)
    rows = []
    for rec in records:
        row = {"Model": rec["name"],
               "n_params": rec.get("n_params"),
               "train_seconds": round(rec.get("train_seconds", float("nan")), 2)}
        row.update(metrics[rec["name"]])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    out = results_dir / "test_results.csv"
    table.to_csv(out, index=False)
    print("[evaluate] wrote", out)
    print(table.to_string(index=False))
    return table


def resummarize(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Rebuild results/test_results.csv from the saved prediction files alone
    (re-uses n_params / train_seconds from an existing summary if present)."""
    metrics = load_saved_metrics(results_dir)
    old = {}
    old_path = results_dir / "test_results.csv"
    if old_path.exists():
        prev = pd.read_csv(old_path)
        old = prev.set_index("Model")[["n_params", "train_seconds"]].to_dict("index")
    records = [{"name": n,
                "n_params": old.get(n, {}).get("n_params"),
                "train_seconds": old.get(n, {}).get("train_seconds")}
               for n in metrics]
    return summarize(records, results_dir)


if __name__ == "__main__":
    resummarize()
