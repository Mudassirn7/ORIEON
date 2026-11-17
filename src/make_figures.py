"""ORIEON figures — Section 5 of the approved protocol.

Generates into figures/:
  - actual-vs-predicted test plot per model
  - model comparison bar charts (RMSE and R^2)
  - training-loss curves for the neural models

Runnable standalone:  python -m src.make_figures
(requires results/ to be populated by a training run)
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


def plot_actual_vs_predicted(results_dir: Path = RESULTS_DIR,
                             figures_dir: Path = FIGURES_DIR) -> None:
    for csv in sorted(results_dir.glob("*_test_predictions.csv")):
        name = csv.name.replace("_test_predictions.csv", "")
        df = pd.read_csv(csv, parse_dates=["Date"])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df["Date"], df["actual"], label="Actual", lw=1.4)
        ax.plot(df["Date"], df["predicted"], label="Predicted", lw=1.2, alpha=0.85)
        ax.set_title(f"{name} — actual vs predicted (test set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Gold close price")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"avp_{name}.png", dpi=150)
        plt.close(fig)


def plot_comparison(results_dir: Path = RESULTS_DIR,
                    figures_dir: Path = FIGURES_DIR) -> None:
    table = pd.read_csv(results_dir / "test_results.csv")
    for col, fname, ylabel in [("RMSE", "comparison_rmse.png", "RMSE"),
                               ("R2", "comparison_r2.png", "R$^2$")]:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(table["Model"], table[col])
        ax.set_title(f"Model comparison — {ylabel} (test set, sorted by RMSE)")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150)
        plt.close(fig)


def plot_training_curves(results_dir: Path = RESULTS_DIR,
                         figures_dir: Path = FIGURES_DIR) -> None:
    histories = sorted(results_dir.glob("*_history.json"))
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for path in histories:
        with open(path) as f:
            h = json.load(f)
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"],
                label=h["model"], lw=1.2)
    ax.set_title("Training loss curves (neural models, scaled-target MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "training_loss_curves.png", dpi=150)
    plt.close(fig)

    # Validation curves (present since validation-based early stopping).
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for path in histories:
        with open(path) as f:
            h = json.load(f)
        if "val_loss" not in h:
            continue
        plotted = True
        ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"],
                label=h["model"], lw=1.2)
        best_ep = h.get("best_epoch")
        if best_ep:
            ax.axvline(best_ep, color=ax.lines[-1].get_color(),
                       lw=0.6, alpha=0.35)
    if plotted:
        ax.set_title("Validation loss curves (vertical line = best epoch, model restored from there)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / "validation_loss_curves.png", dpi=150)
    plt.close(fig)


def make_all(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(results_dir, figures_dir)
    plot_comparison(results_dir, figures_dir)
    plot_training_curves(results_dir, figures_dir)
    print(f"[make_figures] wrote figures to {figures_dir}")


if __name__ == "__main__":
    make_all()
