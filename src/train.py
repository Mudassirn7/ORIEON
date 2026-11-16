"""ORIEON training — shared training loop and model groups (Section 4).

Global training settings for all neural models (protocol, do not change):
  seed = 42 everywhere (Python, NumPy, PyTorch)
  lookback window = 20 days, batch size = 64, exactly 40 epochs
  no early stopping; use the final-epoch model
  optimizer AdamW (weight_decay = 1e-5), MSE loss on the scaled target
  gradient-norm clipping at 1.0, torch.set_num_threads(4)

Device: uses CUDA when available, otherwise CPU.
"""

from __future__ import annotations

import json
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
from .models import (GRUModel, LSTMModel, MLP, GraphOnlyModel,
                     GraphTemporalModel, compute_gcn_adjacency,
                     count_parameters, LOOKBACK)
from .prepare_data import FEATURES, PRICE_TARGET, TARGET, split_indices

SEED = 42
EPOCHS = 40
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
LR_DEEP = 1e-3   # Group B
LR_GRAPH = 8e-4  # Groups C and D

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
def scale_data(df: pd.DataFrame):
    """Fit a StandardScaler for inputs and a separate one for the target on
    TRAINING rows only; transform both splits."""
    n_train, _ = split_indices(df)
    x_scaler = StandardScaler().fit(df[FEATURES].iloc[:n_train])
    y_scaler = StandardScaler().fit(df[[TARGET]].iloc[:n_train])
    X = x_scaler.transform(df[FEATURES]).astype(np.float32)
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
# Shared neural training loop
# ---------------------------------------------------------------------------
def train_torch(name: str, model: torch.nn.Module, Xtr, ytr, lr: float,
                results_dir: Path = RESULTS_DIR, epochs: int = EPOCHS,
                wd: float = WEIGHT_DECAY):
    """AdamW (wd 1e-5), MSE on the scaled target, clip 1.0, exactly 40 epochs,
    no early stopping — final-epoch model is used and saved."""
    set_seed()
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

    history = []
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
        print(f"  [{name}] epoch {epoch + 1:02d}/{epochs}  train MSE {history[-1]:.6f}")
    seconds = time.perf_counter() - t0

    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), results_dir / f"{name}.pt")
    with open(results_dir / f"{name}_history.json", "w") as f:
        json.dump({"model": name, "lr": lr, "epochs": epochs, "weight_decay": wd,
                   "batch_size": BATCH_SIZE, "seed": SEED,
                   "device": str(DEVICE), "train_loss": history}, f, indent=2)
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
    evaluate.save_predictions(name, dates_te, actual_te, pred, results_dir,
                              close=close_te)
    return {"name": name, "n_params": count_parameters(model),
            "train_seconds": seconds}


def load_best_params():
    """Tuned (lr, epochs, wd) per model from results/best_params.json.
    If the file does not exist, the old fixed settings are used."""
    f = RESULTS_DIR / "best_params.json"
    return json.load(open(f)) if f.exists() else {}


# ---------------------------------------------------------------------------
# Group A — tree baselines (current-day 15-feature vector, no sequences)
# ---------------------------------------------------------------------------
def run_group_a(df, X, y, n_train, y_scaler, results_dir=RESULTS_DIR):
    from lightgbm import LGBMRegressor

    dates_te = df["Date"].iloc[n_train:]
    close_te = df["Gold_Close"].iloc[n_train:].values
    actual = df[PRICE_TARGET].iloc[n_train:].values

    trees = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_features=0.7, min_samples_leaf=1,
            random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, num_leaves=15,
            subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED),
    }
    records = []
    for name, est in trees.items():
        print(f"[Group A] training {name} ...")
        t0 = time.perf_counter()
        est.fit(X[:n_train], y[:n_train])
        seconds = time.perf_counter() - t0
        pred = to_price(close_te, est.predict(X[n_train:]), y_scaler)
        evaluate.save_predictions(name, dates_te, actual, pred, results_dir,
                                  close=close_te)
        records.append({"name": name, "n_params": None,
                        "train_seconds": seconds})
    # --- Baselines ----------------------------------------------------------
    # Naive (zero-return): tomorrow's price = today's price
    evaluate.save_predictions("Naive", dates_te, actual, close_te, results_dir,
                              close=close_te)
    records.append({"name": "Naive", "n_params": None, "train_seconds": 0.0})

    # Naive-Drift: today's price + average historical daily move.
    # Drift = mean TRAIN log-return only (no test leakage).
    drift_ret = float(df[TARGET].iloc[:n_train].mean())
    pred = close_te * np.exp(drift_ret)
    evaluate.save_predictions("Naive-Drift", dates_te, actual, pred,
                              results_dir, close=close_te)
    records.append({"name": "Naive-Drift", "n_params": None,
                    "train_seconds": 0.0})

    # SMA-5: tomorrow's price = mean of the last 5 closes (smooths noise).
    sma5 = df["Gold_Close"].rolling(5).mean().iloc[n_train:].values
    evaluate.save_predictions("SMA-5", dates_te, actual, sma5, results_dir,
                              close=close_te)
    records.append({"name": "SMA-5", "n_params": None, "train_seconds": 0.0})
    return records


# ---------------------------------------------------------------------------
# Groups B, C, D — neural models
# ---------------------------------------------------------------------------
def run_neural_groups(df, X, y, n_train, y_scaler, results_dir=RESULTS_DIR):
    Xtr, ytr, Xte, _ = make_windows(X, y, n_train)
    dates_te = df["Date"].iloc[n_train:]
    close_te = df["Gold_Close"].iloc[n_train:].values
    actual_te = df[PRICE_TARGET].iloc[n_train:].values
    print(f"[neural] device={DEVICE}  train windows={len(Xtr)}  test windows={len(Xte)}")

    # GCN adjacency from TRAINING rows only.
    gcn_adj = compute_gcn_adjacency(X[:n_train])

    registry = [
        # Group B — deep baselines (lr 1e-3)
        ("MLP", lambda: MLP(), LR_DEEP),
        ("LSTM", lambda: LSTMModel(), LR_DEEP),
        ("GRU", lambda: GRUModel(), LR_DEEP),
        # Group C (GNN-only) removed from the comparison per updated plan —
        # classes remain in models.py; comparison is trees + MLP/LSTM/GRU +
        # hybrids + baselines, followed by an ensemble model.
        # Group D — hybrids (lr 8e-4)
        ("Dynamic-GNN+GRU", lambda: GraphTemporalModel("dynamic", "gru"), LR_GRAPH),
        ("Dynamic-GNN+LSTM", lambda: GraphTemporalModel("dynamic", "lstm"), LR_GRAPH),
        ("Static-Learned-GNN+GRU", lambda: GraphTemporalModel("static", "gru"), LR_GRAPH),
        ("Static-Learned-GNN+LSTM", lambda: GraphTemporalModel("static", "lstm"), LR_GRAPH),
    ]

    best = load_best_params()
    records = []
    for name, build, lr in registry:
        bp = best.get(name, {})
        lr = bp.get("lr", lr)
        ep = bp.get("epochs", EPOCHS)
        wd = bp.get("wd", WEIGHT_DECAY)
        print(f"[neural] training {name} (lr={lr}, epochs={ep}, wd={wd}) ...")
        model, _, seconds = train_torch(name, build(), Xtr, ytr, lr, results_dir,
                                        epochs=ep, wd=wd)
        records.append(_finish_neural(name, model, seconds, Xte, dates_te,
                                      close_te, actual_te, y_scaler, results_dir))
    return records


# ---------------------------------------------------------------------------
# Full training entry point
# ---------------------------------------------------------------------------
def run_all_models(df: pd.DataFrame, results_dir: Path = RESULTS_DIR):
    set_seed()
    print(f"[train] torch {torch.__version__}  device={DEVICE}")
    X, y, _, y_scaler, n_train = scale_data(df)

    records = []
    records += run_group_a(df, X, y, n_train, y_scaler, results_dir)
    records += run_neural_groups(df, X, y, n_train, y_scaler, results_dir)

    table = evaluate.summarize(records, results_dir)
    return table


if __name__ == "__main__":
    from .prepare_data import prepare
    df, _ = prepare()
    run_all_models(df)
