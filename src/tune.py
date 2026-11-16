"""ORIEON - hyperparameter tuning on a VALIDATION set (test set is never used).

Validation = last 15% of the TRAIN windows (time order, no shuffle).
For every neural model we search (lr, weight_decay), train up to MAX_EPOCHS
with early stopping on validation MSE, and keep the best setting and epoch.
Result is saved to results/best_params.json and used by src/train.py.

Two search methods (Stage 3 of the updated pipeline):
  TUNE_METHOD = "grid"  -> fixed lr x wd grid (original)
  TUNE_METHOD = "pso"   -> Particle Swarm Optimization over log10(lr), log10(wd)

Run:  python -m src.tune          (grid)
      TUNE_METHOD=pso python -m src.tune   (or edit the constant below)
"""

import json
import os
import numpy as np
import torch

from . import train
from .prepare_data import prepare

LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
WD_GRID = [1e-5, 1e-3]
MAX_EPOCHS = 100
PATIENCE = 10
VAL_FRACTION = 0.15

# Stage 3: bio-inspired optimizer settings (PSO over log10 space).
TUNE_METHOD = os.environ.get("TUNE_METHOD", "grid")   # "grid" | "pso"
PSO_PARTICLES = 6
PSO_ITERS = 4
PSO_BOUNDS = (np.array([-4.0, -5.0]),     # min log10(lr), log10(wd)
              np.array([-2.52, -3.0]))    # max log10(lr), log10(wd)


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


def tune_grid(build, Xfit, yfit, Xval, yval, name, log):
    top = None
    for lr in LR_GRID:
        for wd in WD_GRID:
            val, ep = fit_with_val(build, Xfit, yfit, Xval, yval, lr, wd)
            print(f"[tune] {name:26s} lr={lr:<7g} wd={wd:<7g} val MSE={val:.4f} best epoch={ep}")
            log.append({"model": name, "method": "grid", "lr": lr, "wd": wd,
                        "val_mse": val, "epochs": ep})
            if top is None or val < top["val_mse"]:
                top = {"lr": lr, "wd": wd, "epochs": ep, "val_mse": val}
    return top


def tune_pso(build, Xfit, yfit, Xval, yval, name, log):
    from pyswarms.single import GlobalBestPSO

    history = []

    def objective(swarm):
        costs = []
        for pos in swarm:
            lr, wd = float(10 ** pos[0]), float(10 ** pos[1])
            val, ep = fit_with_val(build, Xfit, yfit, Xval, yval, lr, wd)
            print(f"[tune-pso] {name:26s} lr={lr:<9.3g} wd={wd:<9.3g} val MSE={val:.4f} best epoch={ep}")
            history.append({"model": name, "method": "pso", "lr": lr, "wd": wd,
                            "val_mse": val, "epochs": ep})
            costs.append(val)
        return np.array(costs)

    opt = GlobalBestPSO(n_particles=PSO_PARTICLES, dimensions=2,
                        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
                        bounds=PSO_BOUNDS)
    best_cost, best_pos = opt.optimize(objective, iters=PSO_ITERS)
    log.extend(history)
    best_eval = min(history, key=lambda r: r["val_mse"])
    return {"lr": float(10 ** best_pos[0]), "wd": float(10 ** best_pos[1]),
            "epochs": best_eval["epochs"], "val_mse": float(best_cost)}


def main():
    df, _ = prepare()
    X, y, _, _, n_train = train.scale_data(df)
    Xtr, ytr, _, _ = train.make_windows(X, y, n_train)
    n_fit = int(len(Xtr) * (1 - VAL_FRACTION))
    Xfit, yfit, Xval, yval = Xtr[:n_fit], ytr[:n_fit], Xtr[n_fit:], ytr[n_fit:]
    print(f"[tune] method={TUNE_METHOD}  fit={len(Xfit)} val={len(Xval)}  (test not used)")
    print(f"[tune] val MSE of predicting 0 (mean) = {np.mean(yval ** 2):.4f}")

    gcn_adj = train.compute_gcn_adjacency(X[:n_train])
    n_feat = X.shape[1]
    models = dict((name, build)
                  for name, build, _ in train.model_registry(gcn_adj, n_features=n_feat))

    best, log = {}, []
    for name, build in models.items():
        if TUNE_METHOD == "pso":
            top = tune_pso(build, Xfit, yfit, Xval, yval, name, log)
        else:
            top = tune_grid(build, Xfit, yfit, Xval, yval, name, log)
        best[name] = top
        print(f"[tune] >>> {name} best: {top}")

    train.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(best, open(train.RESULTS_DIR / "best_params.json", "w"), indent=2)
    json.dump(log, open(train.RESULTS_DIR / "tuning_log.json", "w"), indent=2)
    print("[tune] saved results/best_params.json")


if __name__ == "__main__":
    main()
