"""ORIEON - hyperparameter tuning on a VALIDATION set (test set is never used).

Validation = last 15% of the TRAIN windows (time order, no shuffle).
For every neural model we try lr x weight_decay, train up to MAX_EPOCHS with
early stopping on validation MSE, and keep the best setting and best epoch.
Result is saved to results/best_params.json and used by src/train.py.

Run:  python -m src.tune
"""

import json
import numpy as np
import torch

from . import train
from .prepare_data import prepare

LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
WD_GRID = [1e-5, 1e-3]
MAX_EPOCHS = 100
PATIENCE = 10
VAL_FRACTION = 0.15


def fit_with_val(build, Xfit, yfit, Xval, yval, lr, wd):
    train.set_seed()
    model = build().to(train.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xfit), torch.from_numpy(yfit))
    gen = torch.Generator().manual_seed(train.SEED)
    loader = torch.utils.data.DataLoader(ds, batch_size=train.BATCH_SIZE,
                                         shuffle=True, generator=gen)
    best_val, best_epoch, wait = np.inf, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(train.DEVICE), yb.to(train.DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train.CLIP_NORM)
            opt.step()
        pred = train.predict_torch(model, Xval)
        val = float(np.mean((pred - yval) ** 2))
        if val < best_val - 1e-6:
            best_val, best_epoch, wait = val, epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_val, best_epoch


def main():
    df, _ = prepare()
    X, y, _, _, n_train = train.scale_data(df)
    Xtr, ytr, _, _ = train.make_windows(X, y, n_train)
    n_fit = int(len(Xtr) * (1 - VAL_FRACTION))
    Xfit, yfit, Xval, yval = Xtr[:n_fit], ytr[:n_fit], Xtr[n_fit:], ytr[n_fit:]
    print(f"[tune] fit={len(Xfit)} val={len(Xval)}  (test not used)")
    print(f"[tune] val MSE of predicting 0 (mean) = {np.mean(yval ** 2):.4f}")

    gcn_adj = train.compute_gcn_adjacency(X[:n_train])
    M = train
    models = {
        "MLP": lambda: M.MLP(),
        "LSTM": lambda: M.LSTMModel(),
        "GRU": lambda: M.GRUModel(),
        # Group C (GNN-only) removed per updated plan; hybrids remain.
        "Dynamic-GNN+GRU": lambda: M.GraphTemporalModel("dynamic", "gru"),
        "Dynamic-GNN+LSTM": lambda: M.GraphTemporalModel("dynamic", "lstm"),
        "Static-Learned-GNN+GRU": lambda: M.GraphTemporalModel("static", "gru"),
        "Static-Learned-GNN+LSTM": lambda: M.GraphTemporalModel("static", "lstm"),
    }

    best, log = {}, []
    for name, build in models.items():
        top = None
        for lr in LR_GRID:
            for wd in WD_GRID:
                val, ep = fit_with_val(build, Xfit, yfit, Xval, yval, lr, wd)
                print(f"[tune] {name:26s} lr={lr:<7g} wd={wd:<7g} val MSE={val:.4f} best epoch={ep}")
                log.append({"model": name, "lr": lr, "wd": wd, "val_mse": val, "epochs": ep})
                if top is None or val < top["val_mse"]:
                    top = {"lr": lr, "wd": wd, "epochs": ep, "val_mse": val}
        best[name] = top
        print(f"[tune] >>> {name} best: {top}")

    train.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(best, open(train.RESULTS_DIR / "best_params.json", "w"), indent=2)
    json.dump(log, open(train.RESULTS_DIR / "tuning_log.json", "w"), indent=2)
    print("[tune] saved results/best_params.json")


if __name__ == "__main__":
    main()
