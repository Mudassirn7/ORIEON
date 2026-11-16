"""ORIEON — ensemble integration (stage 5 of the hybrid pipeline).

Two ensembles over the non-baseline base models (trees + MLP/LSTM/GRU + hybrids):

- **Ensemble-Avg**: equal-weight mean of base-model predictions, combined in
  log-return space, converted back to price.
- **Ensemble-Stack**: Ridge meta-learner trained ONLY on out-of-sample
  validation predictions. Base models are refit on the first 85% of the
  training data (same validation split as src/tune.py: last 15% of train,
  chronological, no shuffle); their predictions on that validation block train
  the stacker. Final test predictions come from the already-saved outputs of
  the final models (trained on the full training set). The test set is never
  touched by the stacker.

Combination happens in log-return space: ret = log(pred / Close_t), then
price = Close_t * exp(combined_ret) — consistent with the rest of the pipeline.

Run:  python -m src.ensemble        (after python run_all.py)
Called automatically by run_all.py and per fold by src/cv.py.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import evaluate, train
from .prepare_data import PRICE_TARGET, TARGET, split_indices
from .train import LOOKBACK, SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

VAL_FRACTION = 0.15  # last 15% of train = validation (same as tune.py)
# ARIMA is excluded from the stack (no validation refit path) and reported
# as a standalone linear-leg row, like the baselines.
BASELINES = {"Naive", "Naive-Drift", "SMA-5", "ARIMA"}
ENSEMBLES = {"Ensemble-Avg", "Ensemble-Stack"}
RIDGE_ALPHA = 1.0


def _base_models(results_dir: Path) -> list[str]:
    names = [p.name.replace("_test_predictions.csv", "")
             for p in sorted(results_dir.glob("*_test_predictions.csv"))]
    return [n for n in names if n not in BASELINES and n not in ENSEMBLES]


def _load_test_returns(results_dir: Path, names: list[str]):
    """Saved price predictions -> log-return matrix (days x models)."""
    rets, close_te, actual_te, dates_te = [], None, None, None
    for n in names:
        d = pd.read_csv(results_dir / f"{n}_test_predictions.csv")
        assert "Close" in d.columns, f"{n}: re-run training; Close column needed"
        rets.append(np.log(d["predicted"].values / d["Close"].values))
        if close_te is None:
            close_te = d["Close"].values
            actual_te = d["actual"].values
            dates_te = d["Date"]
    return np.column_stack(rets), close_te, actual_te, dates_te


def _val_predictions(df, X, y, n_train, y_scaler, names, tmp_dir):
    """Out-of-sample validation predictions (raw log-returns) from base models
    refit on the first 85% of the training data."""
    Xtr, ytr, _, _ = train.make_windows(X, y, n_train)
    n_fit = int(len(Xtr) * (1 - VAL_FRACTION))
    Xfit, yfit, Xval = Xtr[:n_fit], ytr[:n_fit], Xtr[n_fit:]

    # window target indices within df: LOOKBACK-1 .. n_train-1
    win_idx = np.arange(LOOKBACK - 1, n_train)
    val_idx = win_idx[n_fit:]
    y_val_raw = df[TARGET].iloc[val_idx].values

    best = train.load_best_params()
    gcn_adj = train.compute_gcn_adjacency(X[:n_train])
    registry = dict((name, (build, lr))
                    for name, build, lr in train.model_registry(gcn_adj, n_features=X.shape[1]))

    cols = {}
    for name in names:
        if name in train.tree_registry():  # tree: current-day vectors
            est = train.tree_registry()[name]
            est.fit(X[win_idx[:n_fit]], y[win_idx[:n_fit]])
            pred_scaled = est.predict(X[val_idx])
        else:  # neural: sequences
            build, lr = registry[name]
            bp = best.get(name, {})
            model, _, _ = train.train_torch(
                f"stack_{name}", build(), Xfit, yfit,
                bp.get("lr", lr), Path(tmp_dir),
                epochs=bp.get("epochs", train.EPOCHS),
                wd=bp.get("wd", train.WEIGHT_DECAY))
            pred_scaled = train.predict_torch(model, Xval)
        cols[name] = y_scaler.inverse_transform(
            np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    return np.column_stack([cols[n] for n in names]), y_val_raw


def run_ensemble(df: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    names = _base_models(results_dir)
    assert len(names) >= 2, "need at least 2 base models for an ensemble"
    print(f"[ensemble] base models: {names}")

    X, y, _, y_scaler, n_train = train.scale_data(df)
    R_te, close_te, actual_te, dates_te = _load_test_returns(results_dir, names)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="orieon_stack_") as tmp:
        R_val, y_val_raw = _val_predictions(df, X, y, n_train, y_scaler, names, tmp)
    seconds = time.perf_counter() - t0

    # --- Ensemble-Avg --------------------------------------------------------
    ret_avg = R_te.mean(axis=1)
    evaluate.save_predictions("Ensemble-Avg", dates_te, actual_te,
                              close_te * np.exp(ret_avg), results_dir,
                              close=close_te)

    # --- Ensemble-Stack (Ridge on out-of-sample validation predictions) -----
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(R_val, y_val_raw)
    ret_stack = ridge.predict(R_te)
    evaluate.save_predictions("Ensemble-Stack", dates_te, actual_te,
                              close_te * np.exp(ret_stack), results_dir,
                              close=close_te)

    weights = {
        "meta_learner": f"Ridge(alpha={RIDGE_ALPHA})",
        "validation": f"last {VAL_FRACTION:.0%} of train (out-of-sample refits)",
        "intercept": float(ridge.intercept_),
        "coefficients": {n: float(c) for n, c in zip(names, ridge.coef_)},
        "stack_refit_seconds": round(seconds, 2),
        "seed": SEED,
    }
    with open(results_dir / "ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print("[ensemble] weights:", json.dumps(weights["coefficients"], indent=2))

    return evaluate.resummarize(results_dir)


if __name__ == "__main__":
    from .prepare_data import prepare
    df, _ = prepare()
    run_ensemble(df)
