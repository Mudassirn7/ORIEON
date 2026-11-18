"""ORIEON — one command runs the full GRN-MLP Ensemble study:

    data preparation -> 4 ablation experiments x 5 models -> metrics -> figures

Experiments (Ablation Study Design):
    Exp1_Price     — Gold_Close only            (1 feature)
    Exp2_Macro     — Gold_Close + 6 macro       (7 features)
    Exp3_Sentiment — Gold_Close + 8 sentiment   (9 features)
    Exp4_All       — full feature set          (15 features)

Models: RandomForest, LightGBM, MLP, LSTM (benchmarks) + GRN-MLP (proposed).

Usage:
    python run_all.py

Data files must be placed in data/ first:
    data/gold_clean.csv
    data/news_with_scores.csv

Outputs:
    results/results_all_experiments.csv   <- single experiment-wise table
    results/<exp>/test_results.csv        <- per-experiment table
    results/<exp>/*_test_predictions.csv  <- saved predictions
    figures/                              <- all plots (see src/make_figures.py)
"""

from __future__ import annotations

import time

import pandas as pd

from src import make_figures, prepare_data, train


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 72)
    print("ORIEON — Gold Price Prediction (GRN-MLP Ensemble, 4-experiment ablation)")
    print("=" * 72)

    print("\n[1/3] Data preparation")
    df, _ = prepare_data.prepare()

    print("\n[2/3] Experiments (4 feature modes x 5 models)")
    tables = []
    for exp_name, feats in prepare_data.EXPERIMENTS.items():
        print("\n" + "-" * 72)
        print(f"EXPERIMENT {exp_name}: {len(feats)} features -> {feats}")
        print("-" * 72)
        exp_dir = train.RESULTS_DIR / exp_name
        table = train.run_experiment(df, feats, results_dir=exp_dir)
        table.insert(0, "Experiment", exp_name)
        table.insert(1, "n_features", len(feats))
        tables.append(table)

    summary = pd.concat(tables, ignore_index=True)
    out = train.RESULTS_DIR / "results_all_experiments.csv"
    summary.to_csv(out, index=False)
    print(f"\n[summary] wrote {out}")
    print(summary.round(4).to_string(index=False))

    print("\n[3/3] Figures")
    make_figures.make_all()

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} minutes.")
    print("See results/results_all_experiments.csv and figures/.")

    # Honesty note: no model is selected as 'final' from the test table (no
    # test-based selection); all models in all experiments are reported
    # transparently.
    print("\nNOTE: no model is selected as 'final' from the test table; all "
          "models and all experiments are reported transparently.")


if __name__ == "__main__":
    main()
