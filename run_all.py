"""ORIEON — one command runs the full pipeline:

    data preparation -> Group A (trees) -> Group B (deep baselines)
    -> Group D (hybrids) -> baselines -> metrics -> figures

Usage:
    python run_all.py

Data files must be placed in data/ first:
    data/gold_clean.csv
    data/news_with_scores.csv

Optional first step (validation-based tuning, writes results/best_params.json):
    python -m src.tune

Individual stages can also be run separately:
    python -m src.prepare_data
    python -m src.train
    python -m src.evaluate    (re-summarizes from saved predictions)
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

    # Honesty note: no 'final hybrid' is crowned from the test table (no
    # test-based selection); all models are reported transparently.
    print("\nNOTE: no model is selected as 'final' from the test table; all "
          "models are reported transparently.")


if __name__ == "__main__":
    main()
