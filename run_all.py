"""ORIEON — one command runs the full approved protocol:

    data preparation -> Group A (trees) -> Group B (deep baselines)
    -> Group C (GNN-only) -> Group D (hybrids) -> metrics -> figures

Usage:
    python run_all.py

Data files must be placed in data/ first:
    data/gold_clean.csv
    data/news_with_scores.csv

Individual stages can also be run separately:
    python -m src.prepare_data
    python -m src.train
    python -m src.evaluate   (re-summarizes from saved predictions)
    python -m src.make_figures
"""

from __future__ import annotations

import time

from src import evaluate, make_figures, prepare_data, train


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 72)
    print("ORIEON — Gold Price Forecasting (approved protocol run)")
    print("=" * 72)

    print("\n[1/3] Data preparation")
    df, _ = prepare_data.prepare()

    print("\n[2/3] Model training and evaluation")
    train.run_all_models(df)

    print("\n[3/3] Figures")
    make_figures.make_all()

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} minutes.")
    print("See results/test_results.csv and figures/.")

    # Honesty note (protocol Section 4, Group D): no final hybrid is selected
    # from the test table because there is no validation set.
    print("\nNOTE: per the protocol, no 'final hybrid' is selected from the test "
          "table (no validation set). All four hybrids are reported "
          "transparently; any future selection needs a separate documented "
          "selection protocol.")


if __name__ == "__main__":
    main()
