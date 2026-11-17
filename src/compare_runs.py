"""ORIEON — feature-set ablation comparison (e.g. market vs all).

Reads results/<tag>/test_results.csv and results/<tag>/*_test_predictions.csv
for two run tags and answers: "sentiment features ne faida diya ya nahi?"

For every model it reports both runs' metrics, the RMSE delta, and a
Diebold–Mariano test on squared errors (two-sided, normal approximation):
  DM > 0  -> run B (e.g. 'all') ke errors CHHOTE  -> sentiment helped
  DM < 0  -> run A (e.g. 'market') ke errors CHHOTE -> sentiment hurt
  |DM| ke saath p < 0.05 hona chahiye significant ke liye.

Run:  python -m src.compare_runs market all
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def dm_test(actual: pd.Series, pred_a: pd.Series, pred_b: pd.Series):
    """Diebold-Mariano on squared-error loss differential d = eA^2 - eB^2."""
    d = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
    d = d.dropna()
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0 or not math.isfinite(sd):  # identical predictions
        return float("nan"), float("nan"), n
    dm = float(d.mean() / (sd / math.sqrt(n)))
    p = 2.0 * (1.0 - norm_cdf(abs(dm)))
    return dm, p, n


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m src.compare_runs <tagA> <tagB>   e.g. market all")
        sys.exit(1)
    tag_a, tag_b = sys.argv[1], sys.argv[2]

    dir_a, dir_b = ROOT / "results" / tag_a, ROOT / "results" / tag_b
    ta = pd.read_csv(dir_a / "test_results.csv")
    tb = pd.read_csv(dir_b / "test_results.csv")

    m = ta.merge(tb, on="Model", suffixes=(f"_{tag_a}", f"_{tag_b}"))
    m["dRMSE"] = m[f"RMSE_{tag_b}"] - m[f"RMSE_{tag_a}"]
    m["dRMSE_%"] = (100.0 * m["dRMSE"] / m[f"RMSE_{tag_a}"]).round(2)

    cols = ["Model", f"MAE_{tag_a}", f"MAE_{tag_b}",
            f"RMSE_{tag_a}", f"RMSE_{tag_b}", "dRMSE", "dRMSE_%",
            f"R2_{tag_a}", f"R2_{tag_b}"]
    m = m[cols].sort_values(f"RMSE_{tag_a}").round(4)

    print("=" * 78)
    print(f"ABLATION: '{tag_a}' (features kam)  vs  '{tag_b}' (features zyada)")
    print("    dRMSE < 0  =>  tag_b behtar (features ne FAIDA diya)")
    print("=" * 78)
    print(m.to_string(index=False))

    # --- Diebold-Mariano per model on saved predictions ---------------------
    print("\nDiebold–Mariano test (squared errors; p<0.05 = significant):")
    print(f"  {'Model':26s} {'DM':>8s} {'p-value':>9s}  verdict")
    for model in m["Model"]:
        fa = dir_a / f"{model}_test_predictions.csv"
        fb = dir_b / f"{model}_test_predictions.csv"
        if not (fa.exists() and fb.exists()):
            continue
        pa, pb = pd.read_csv(fa), pd.read_csv(fb)
        j = pa.merge(pb, on="Date", suffixes=("_a", "_b"))
        actual = j["actual_a"] if "actual_a" in j else j["actual"]
        dm, p, n = dm_test(actual, j["predicted_a"], j["predicted_b"])
        if math.isnan(dm):
            verdict = "identical predictions (baselines feature-independent)"
            print(f"  {model:26s} {'—':>8s} {'—':>9s}  {verdict}")
            continue
        if p >= 0.05:
            verdict = "farq significant NAHI"
        elif dm > 0:
            verdict = f"'{tag_b}' significantly behtar"
        else:
            verdict = f"'{tag_a}' significantly behtar"
        print(f"  {model:26s} {dm:8.3f} {p:9.4f}  {verdict}")

    print("\nNote: DM yahan simple normal approx hai (no Newey–West); daily")
    print("overlapping nahi hain (1-step-ahead), is liye approx theek hai.")


if __name__ == "__main__":
    main()
