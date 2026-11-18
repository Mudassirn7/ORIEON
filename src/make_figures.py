"""ORIEON figures — GRN-MLP Ensemble project (4-experiment ablation).

Generates into figures/:
  - training_loss_curves.png      2x2 grid: per-experiment train-loss curves
  - comparison_rmse.png           grouped bars: model x experiment (RMSE)
  - comparison_r2.png             grouped bars: model x experiment (R^2)
  - predictions/<exp>/<model>.png actual-vs-predicted per model (separate)
  - predictions/<exp>_combined.png  all models + actual on one axes (combined)

Runnable standalone:  python -m src.make_figures
(requires results/ to be populated by run_all.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def _experiments(results_dir: Path) -> list[Path]:
    """Experiment folders that contain a test_results.csv, in order."""
    return sorted(p for p in results_dir.iterdir()
                  if p.is_dir() and (p / "test_results.csv").exists())


def _load_all_tables(results_dir: Path) -> pd.DataFrame:
    tables = []
    for exp in _experiments(results_dir):
        t = pd.read_csv(exp / "test_results.csv")
        t.insert(0, "Experiment", exp.name)
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def plot_training_curves(results_dir: Path = RESULTS_DIR,
                         figures_dir: Path = FIGURES_DIR) -> None:
    exps = _experiments(results_dir)
    if not exps:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, exp in zip(axes.ravel(), exps):
        for path in sorted(exp.glob("*_history.json")):
            with open(path) as f:
                h = json.load(f)
            ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"],
                    label=h["model"], lw=1.2)
        ax.set_title(exp.name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train MSE")
        ax.legend(fontsize=7)
    fig.suptitle("Training loss curves (neural models, scaled-target MSE)")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_loss_curves.png", dpi=150)
    plt.close(fig)


def plot_comparison(results_dir: Path = RESULTS_DIR,
                    figures_dir: Path = FIGURES_DIR) -> None:
    df = _load_all_tables(results_dir)
    for col, fname, ylabel in [("RMSE", "comparison_rmse.png", "RMSE"),
                               ("R2", "comparison_r2.png", "R$^2$")]:
        pivot = df.pivot(index="Model", columns="Experiment", values=col)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"Model comparison — {ylabel} (test set, per experiment)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.legend(fontsize=8, title="Experiment")
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150)
        plt.close(fig)


def plot_predictions(results_dir: Path = RESULTS_DIR,
                     figures_dir: Path = FIGURES_DIR) -> None:
    """Actual-vs-predicted line graphs: separate per model + combined per
    experiment."""
    pred_root = figures_dir / "predictions"
    for exp in _experiments(results_dir):
        csvs = sorted(exp.glob("*_test_predictions.csv"))
        if not csvs:
            continue
        exp_dir = pred_root / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        fig_c, ax_c = plt.subplots(figsize=(12, 5))   # combined
        actual_plotted = False
        for csv in csvs:
            name = csv.name.replace("_test_predictions.csv", "")
            df = pd.read_csv(csv, parse_dates=["Date"])

            # --- separate figure per model -------------------------------
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(df["Date"], df["actual"], label="Actual", lw=1.4)
            ax.plot(df["Date"], df["predicted"], label="Predicted",
                    lw=1.2, alpha=0.85)
            ax.set_title(f"{exp.name} — {name}: actual vs predicted (test set)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Gold close price")
            ax.legend()
            fig.tight_layout()
            fig.savefig(exp_dir / f"{name}.png", dpi=150)
            plt.close(fig)

            # --- add to combined -----------------------------------------
            if not actual_plotted:
                ax_c.plot(df["Date"], df["actual"], label="Actual",
                          lw=2.0, color="black")
                actual_plotted = True
            ax_c.plot(df["Date"], df["predicted"], label=name, lw=1.0,
                      alpha=0.85)

        ax_c.set_title(f"{exp.name} — all models vs actual (test set)")
        ax_c.set_xlabel("Date")
        ax_c.set_ylabel("Gold close price")
        ax_c.legend(fontsize=8, ncol=2)
        fig_c.tight_layout()
        fig_c.savefig(pred_root / f"{exp.name}_combined.png", dpi=150)
        plt.close(fig_c)


def make_all(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(results_dir, figures_dir)
    plot_comparison(results_dir, figures_dir)
    plot_predictions(results_dir, figures_dir)
    print(f"[make_figures] wrote figures to {figures_dir}")


if __name__ == "__main__":
    make_all()
