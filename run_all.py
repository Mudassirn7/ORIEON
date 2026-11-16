"""ORIEON — one command runs the full pipeline:

    data preparation -> Group A (trees) -> Group B (deep baselines)
    -> Group D (hybrids) -> baselines -> ensembles -> metrics -> figures

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
    python -m src.ensemble    (after src.train; adds Ensemble-Avg / Ensemble-Stack)
    python -m src.evaluate    (re-summarizes from saved predictions)
    python -m src.make_figures
"""

from __future__ import annotations

import time

from src import ensemble, evaluate, make_figures, prepare_data, train


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 72)
    print("ORIEON — Gold Price Forecasting (approved protocol run)")
    print("=" * 72)

    print("\n[1/4] Data preparation")
    df, _ = prepare_data.prepare()

    print("\n[2/4] Model training and evaluation")
    train.run_all_models(df)

    print("\n[3/4] Ensembles (average + Ridge stacking on out-of-sample validation)")
    ensemble.run_ensemble(df)

    print("\n[4/4] Figures")
    make_figures.make_all()

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} minutes.")
    print("See results/test_results.csv and figures/.")

    # Honesty notes:
    # - No 'final hybrid' is crowned from the test table (no test-based
    #   selection); all models are reported transparently.
    # - Ensemble-Stack weights are learned on out-of-sample validation
    #   predictions only (see results/ensemble_weights.json).
    print("\nNOTE: no model is selected as 'final' from the test table; all "
          "models and both ensembles are reported transparently. Ensemble "
          "weights come from validation-only refits (results/ensemble_weights.json).")


if __name__ == "__main__":
    main()
