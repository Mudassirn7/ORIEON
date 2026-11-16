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


def compute_metrics(actual: np.ndarray, predicted: np.ndarray,
                    close: np.ndarray | None = None) -> dict:
    """MAE, MSE, RMSE, R^2, MAPE (division-by-zero guarded), n.

    If `close` (today's close per test day) is given, also reports
    return-space metrics — these expose the price-level "persistence
    illusion" (a naive zero-return forecast scores R^2 ~ 0.998 on prices):
      Ret_RMSE / Ret_R2: metrics on log-returns log(pred/close) vs
                         log(actual/close). Naive scores Ret_R2 = 0 exactly.
      DirAcc:            % of days with correctly predicted move direction.
    """
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
    out = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape,
           "n": int(len(actual))}

    if close is not None:
        close = np.asarray(close, dtype=float)
        ret_a = np.log(actual / close)
        ret_p = np.log(predicted / close)
        rerr = ret_p - ret_a
        ret_rmse = float(np.sqrt(np.mean(rerr ** 2)))
        r_ss_res = float(np.sum(rerr ** 2))
        r_ss_tot = float(np.sum((ret_a - ret_a.mean()) ** 2))
        ret_r2 = float(1.0 - r_ss_res / r_ss_tot) if r_ss_tot > 0 else float("nan")
        dir_acc = float(np.mean(np.sign(ret_p) == np.sign(ret_a)) * 100.0)
        out.update({"Ret_RMSE": ret_rmse, "Ret_R2": ret_r2, "DirAcc": dir_acc})
    return out


def save_predictions(name: str, dates, actual, predicted,
                     results_dir: Path = RESULTS_DIR,
                     close: np.ndarray | None = None) -> Path:
    """Save results/<ModelName>_test_predictions.csv
    (Date, actual, predicted[, Close])."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}_test_predictions.csv"
    dates = pd.Series(pd.to_datetime(list(dates))).dt.strftime("%Y-%m-%d")
    frame = pd.DataFrame({
        "Date": dates,
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    })
    if close is not None:
        frame["Close"] = np.asarray(close, dtype=float)
    frame.to_csv(out, index=False)
    return out


def load_saved_metrics(results_dir: Path = RESULTS_DIR) -> dict:
    """Recompute metrics from the saved CSVs (metrics always come from the
    saved arrays, per protocol). Return-space metrics are added when the CSV
    carries a Close column."""
    metrics = {}
    for csv in sorted(results_dir.glob("*_test_predictions.csv")):
        name = csv.name.replace("_test_predictions.csv", "")
        df = pd.read_csv(csv)
        close = df["Close"].values if "Close" in df.columns else None
        metrics[name] = compute_metrics(df["actual"].values,
                                        df["predicted"].values, close)
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
