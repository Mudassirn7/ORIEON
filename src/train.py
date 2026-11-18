"""ORIEON training — GRN-MLP Ensemble project.

Models per experiment:
  Benchmarks : RandomForest, LightGBM, MLP, LSTM
  Proposed   : GRN-MLP (Gated Residual Network + MLP, gated blending)

Global training settings for all neural models (protocol, do not change):
  seed = 42 everywhere (Python, NumPy, PyTorch)
  lookback window = 20 days, batch size = 64, max 100 epochs
  early stopping on validation MSE (last 15% of train windows, patience 10);
  the best-validation-epoch model is used and saved
  optimizer AdamW (weight_decay = 1e-5), MSE loss on the scaled target
  gradient-norm clipping at 1.0, torch.set_num_threads(4)

Device: uses CUDA when available, otherwise CPU.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import evaluate
from .models import (LSTMModel, MLP, GRNModel, GRNMLPEnsemble,
                     count_parameters, LOOKBACK)
from .prepare_data import PRICE_TARGET, TARGET, split_indices

SEED = 42
EPOCHS = 100        # max epochs (early stopping decides the real stop)
PATIENCE = 10       # early-stopping patience on validation MSE
VAL_FRACTION = 0.15  # last 15% of train windows = validation (time order)
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
LR = 1e-3           # shared by all neural models

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Scaling and windowing
# ---------------------------------------------------------------------------
def scale_data(df: pd.DataFrame, features: list[str]):
    """Fit a StandardScaler for inputs and a separate one for the target on
    TRAINING rows only; transform both splits."""
    n_train, _ = split_indices(df)
    x_scaler = StandardScaler().fit(df[features].iloc[:n_train])
    y_scaler = StandardScaler().fit(df[[TARGET]].iloc[:n_train])
    X = x_scaler.transform(df[features]).astype(np.float32)
    y = y_scaler.transform(df[[TARGET]]).astype(np.float32).ravel()
    return X, y, x_scaler, y_scaler, n_train


def make_windows(X: np.ndarray, y: np.ndarray, n_train: int,
                 lookback: int = LOOKBACK):
    """Window of `lookback` days ending at day i predicts y[i].
    Train samples: i in [lookback-1, n_train). Test samples: i in [n_train, N).
    Test windows may include training-day features as context; rows are never
    shuffled across the split."""
    def build(idx):
        xs = np.stack([X[i - lookback + 1: i + 1] for i in idx])
        return xs, y[idx]

    train_idx = np.arange(lookback - 1, n_train)
    test_idx = np.arange(n_train, len(X))
    Xtr, ytr = build(train_idx)
    Xte, yte = build(test_idx)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Shared neural training loop (validation-based early stopping)
# ---------------------------------------------------------------------------
def train_torch(name: str, model: torch.nn.Module, Xtr, ytr, lr: float,
                results_dir: Path = RESULTS_DIR, epochs: int = EPOCHS,
                wd: float = WEIGHT_DECAY):
    """AdamW (wd 1e-5), MSE on the scaled target, clip 1.0, up to `epochs`
    epochs with early stopping on validation MSE (last 15% of the train
    windows, patience 10). The best-validation-epoch weights are restored,
    used, and saved — never the final epoch. Test data is never touched."""
    set_seed()
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    n_fit = int(len(Xtr) * (1 - VAL_FRACTION))
    Xfit, yfit = Xtr[:n_fit], ytr[:n_fit]
    Xval, yval = Xtr[n_fit:], ytr[n_fit:]

    ds = TensorDataset(torch.from_numpy(Xfit), torch.from_numpy(yfit))
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

    history, val_history = [], []
    best_val, best_epoch, best_state, wait = np.inf, 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            total += loss.item()
            nb += 1
        history.append(total / nb)
        val = float(np.mean((predict_torch(model, Xval) - yval) ** 2))
        val_history.append(val)
        print(f"  [{name}] epoch {epoch + 1:02d}/{epochs}  train MSE {history[-1]:.6f}  val MSE {val:.6f}")
        if val < best_val - 1e-6:
            best_val, best_epoch, wait = val, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [{name}] early stop at epoch {epoch + 1} "
                      f"(best epoch {best_epoch}, val MSE {best_val:.6f})")
                break
    seconds = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best-validation weights

    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), results_dir / f"{name}.pt")
    with open(results_dir / f"{name}_history.json", "w") as f:
        json.dump({"model": name, "lr": lr, "max_epochs": epochs,
                   "best_epoch": best_epoch, "best_val_mse": best_val,
                   "weight_decay": wd,
                   "batch_size": BATCH_SIZE, "seed": SEED,
                   "device": str(DEVICE), "train_loss": history,
                   "val_loss": val_history}, f, indent=2)
    return model, history, seconds


@torch.no_grad()
def predict_torch(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[i: i + BATCH_SIZE]).to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def to_price(close_t, scaled_ret, y_scaler):
    """scaled log-return -> next-day price = Close_t * exp(return)."""
    ret = y_scaler.inverse_transform(np.asarray(scaled_ret).reshape(-1, 1)).ravel()
    return close_t * np.exp(ret)


def _finish_neural(name, model, seconds, Xte, dates_te, close_te, actual_te,
                   y_scaler, results_dir=RESULTS_DIR):
    pred = to_price(close_te, predict_torch(model, Xte), y_scaler)
    evaluate.save_predictions(name, dates_te, actual_te, pred, results_dir)
    return {"name": name, "n_params": count_parameters(model),
            "train_seconds": seconds}


# ---------------------------------------------------------------------------
# Model registry — 5 models (4 benchmarks + proposed GRN-MLP)
# ---------------------------------------------------------------------------
def tree_registry():
    """Tree benchmarks with the protocol hyperparameters."""
    from lightgbm import LGBMRegressor

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_features=0.7, min_samples_leaf=1,
            random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, num_leaves=15,
            subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED),
    }


def neural_registry(n_features: int):
    """Neural models: 3 benchmarks (MLP, LSTM, standalone GRN) + the
    proposed Ensemble-GRN-MLP."""
    return [
        ("MLP", lambda: MLP(n_features=n_features)),
        ("LSTM", lambda: LSTMModel(n_features=n_features)),
        ("GRN", lambda: GRNModel(n_features=n_features)),
        ("Ensemble-GRN-MLP", lambda: GRNMLPEnsemble(n_features=n_features)),
    ]


# ---------------------------------------------------------------------------
# One experiment = one feature set, all 5 models
# ---------------------------------------------------------------------------
def run_experiment(df: pd.DataFrame, features: list[str],
                   results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Train all models on the given feature set and write the metrics table."""
    set_seed()
    print(f"[train] torch {torch.__version__}  device={DEVICE}  "
          f"features={len(features)}")
    X, y, _, y_scaler, n_train = scale_data(df, features)

    dates_te = df["Date"].iloc[n_train:]
    close_te = df["Gold_Close"].iloc[n_train:].values
    actual = df[PRICE_TARGET].iloc[n_train:].values

    # --- Tree benchmarks (current-day feature vector, no sequences) --------
    records = []
    for name, est in tree_registry().items():
        print(f"[trees] training {name} ...")
        t0 = time.perf_counter()
        est.fit(X[:n_train], y[:n_train])
        seconds = time.perf_counter() - t0
        pred = to_price(close_te, est.predict(X[n_train:]), y_scaler)
        evaluate.save_predictions(name, dates_te, actual, pred, results_dir)
        records.append({"name": name, "n_params": None,
                        "train_seconds": seconds})

    # --- Neural models (20-day windows) ------------------------------------
    Xtr, ytr, Xte, _ = make_windows(X, y, n_train)
    print(f"[neural] train windows={len(Xtr)}  test windows={len(Xte)}")
    for name, build in neural_registry(X.shape[1]):
        print(f"[neural] training {name} (lr={LR}, max epochs={EPOCHS}) ...")
        model, _, seconds = train_torch(name, build(), Xtr, ytr, LR, results_dir)
        records.append(_finish_neural(name, model, seconds, Xte, dates_te,
                                      close_te, actual, y_scaler, results_dir))

    table = evaluate.summarize(records, results_dir)
    return table


if __name__ == "__main__":
    from .prepare_data import FEATURES, prepare
    df, _ = prepare()
    run_experiment(df, FEATURES)
