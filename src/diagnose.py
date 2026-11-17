"""ORIEON — signal diagnostic: features mein next-day return signal hai ya nahi?

Four checks (test set sirf check 2/4 mein, sirf reporting ke liye — selection nahi):
  1. Tuned hyperparameters (best_params.json) — kya tune ne tiny lr pick kiya?
  2. Zero-prediction baseline (scaled MSE of predicting 0) on fit/val/test.
     Agar kisi model ka val MSE is se better nahi => model kuch nahi seekha.
  3. Feature <-> next-day log-return correlations (TRAIN rows only) +
     return autocorrelation (market-efficiency check).
  4. Saved predictions per model: implied predicted-return std (collapse
     detector — ~0 means constant predictor) and corr with actual returns.

Run:  python -m src.diagnose        (best results ke baad; bina results ke bhi chalta hai)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import train
from .prepare_data import FEATURES, TARGET, prepare


def main() -> None:
    print("=" * 72)
    print("ORIEON — SIGNAL DIAGNOSTIC")
    print("=" * 72)

    # ------------------------------------------------------------------ 1
    print("\n[1] Tuned hyperparameters (results/best_params.json)")
    f = train.RESULTS_DIR / "best_params.json"
    if f.exists():
        best = json.load(open(f))
        for name, bp in best.items():
            print(f"  {name:26s} lr={bp.get('lr'):<10g} wd={bp.get('wd'):<10g} "
                  f"epochs={bp.get('epochs'):<5} val_mse={bp.get('val_mse'):.4f}")
        lrs = {bp.get("lr") for bp in best.values()}
        if lrs == {1e-4} or (lrs and max(lrs) <= 1e-4):
            print("  ⚠ Sab models par chhoti lr select hui — validation par "
                  "'na seekhna' hi optimum nikla (signal-absence ka sign).")
    else:
        print("  best_params.json NOT FOUND — tune.py nahi chala; defaults use hue.")

    # ------------------------------------------------------------------ 2
    df, _ = prepare()
    X, y, _, _, n_train = train.scale_data(df)
    _, ytr, _, yte = train.make_windows(X, y, n_train)
    n_fit = int(len(ytr) * (1 - train.VAL_FRACTION))
    yfit, yval = ytr[:n_fit], ytr[n_fit:]

    zero_fit, zero_val, zero_te = (float(np.mean(yfit ** 2)),
                                   float(np.mean(yval ** 2)),
                                   float(np.mean(yte ** 2)))
    print("\n[2] Zero-prediction baseline — scaled MSE of predicting 0")
    print("    (is se behtar val MSE hi 'learning' hai)")
    print(f"  fit (train 85%): {zero_fit:.4f}")
    print(f"  val (train 15%): {zero_val:.4f}   <- early stopping / tuning isi par hota hai")
    print(f"  test:            {zero_te:.4f}")

    # ------------------------------------------------------------------ 3
    print("\n[3] Feature <-> next-day log-return correlation (TRAIN rows only)")
    tr = df.iloc[:n_train]
    corr = tr[FEATURES].corrwith(tr[TARGET])
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    for feat, r in corr.items():
        flag = "  <-- strongest" if feat == corr.index[0] else ""
        print(f"  {feat:22s} r = {r:+.4f}{flag}")
    rmax = float(corr.abs().max())
    print(f"  max |r| = {rmax:.4f}  "
          f"({'koi linear signal nahi' if rmax < 0.05 else 'weak signal' if rmax < 0.1 else 'kuch signal hai'})")

    # Return autocorrelation (EMH check): kal ka return aaj se predictable?
    ret = df[TARGET]  # y_logret_t1 aligned per row; shift(1) = previous day's target
    print("\n  Return autocorrelation (poora data, sirf diagnostic):")
    for lag in (1, 2, 3, 5):
        print(f"  lag {lag}: {ret.autocorr(lag):+.4f}")

    # ------------------------------------------------------------------ 4
    print("\n[4] Saved predictions — collapse aur signal check")
    print(f"  {'Model':26s} {'pred_ret_std':>12s} {'corr(pred,actual)':>18s}")
    rows = []
    for path in sorted(train.RESULTS_DIR.glob("*_predictions.csv")):
        name = path.stem.replace("_test_predictions", "")
        p = pd.read_csv(path)
        if not {"actual", "predicted"} <= set(p.columns) or len(p) < 3:
            continue
        # row i: actual = close_{i+1}; implied predicted return = pred_i / close_i - 1
        prev_close = p["actual"].shift(1)
        pred_ret = p["predicted"] / prev_close - 1.0
        act_ret = p["actual"] / prev_close - 1.0
        std = float(np.nanstd(pred_ret.values[1:]))
        c = float(pd.Series(pred_ret.values[1:]).corr(pd.Series(act_ret.values[1:])))
        rows.append((name, std, c))
    for name, std, c in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:26s} {std:12.6f} {c:+18.4f}")
    if rows:
        neural = [(n, s, c) for n, s, c in rows
                  if n not in ("Naive-Drift", "SMA-5", "ARIMA")]
        if neural and max(s for _, s, _ in neural) < 1e-3:
            print("  ⚠ Neural models ki predicted-return std ~ 0 — sab constant "
                  "(zero-return) predictors par collapse ho chuke hain.")

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 72)
    print("VERDICT GUIDE:")
    print("  - max |r| < 0.05  AUR  val-model-MSE ~ val-zero  => features mein")
    print("    daily return signal nahi (EMH-consistent finding).")
    print("  - max |r| > 0.1 lekin models learn nahi kar rahe => pipeline/training")
    print("    issue (lr, scaling, ya alignment) investigate karo.")
    print("=" * 72)


if __name__ == "__main__":
    main()
