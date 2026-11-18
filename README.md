# Gold Price Prediction Using a Hybrid GRN-MLP Ensemble with Multi-Dimensional Sentiment Analysis

## Project Overview

Financial markets—especially Gold—exhibit both stable macroeconomic trends and
sudden volatility driven by geopolitical news and market sentiment. Traditional
static models assign fixed weights to input features, failing to adapt during
sudden market shocks.

This repository implements a **Hybrid Gated Residual Network and Multi-Layer
Perceptron (GRN-MLP) Ensemble** for gold price forecasting using **15 features**
(1 base price + 6 external macroeconomic features + 8 sentiment features),
alongside a comprehensive **ablation study** and benchmark comparisons against
standard machine learning and deep learning models.

> **Status:** no numbers in this repository are pre-computed. Every table and
> figure is produced by running `python run_all.py`.

## Dataset & Feature Description (15 Features)

### 1. Base & External Macroeconomic Features (7)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `Gold_Close` | Daily Gold closing price (USD) |
| 2 | `DXY` | US Dollar Index |
| 3 | `Silver` | Daily Silver closing price |
| 4 | `CrudeOil` | Crude Oil price |
| 5 | `UST10Y` | US 10-Year Treasury Yield |
| 6 | `VIX` | CBOE Volatility Index (market fear gauge) |
| 7 | `SP500` | S&P 500 stock market index |

### 2. Multi-Dimensional Sentiment Features (8)

| # | Feature | Description |
|---|---------|-------------|
| 8 | `Sentiment_Mean` | Daily average sentiment score of gold-related news |
| 9 | `Sentiment_Std` | Std of sentiment scores (headline disagreement) |
| 10 | `Sentiment_Min` | Most negative headline sentiment score of the day |
| 11 | `Sentiment_Max` | Most positive headline sentiment score of the day |
| 12 | `Positive_Ratio` | Proportion of positive news headlines |
| 13 | `Negative_Ratio` | Proportion of negative news headlines |
| 14 | `News_Available` | 1 if gold-related news is available, else 0 |
| 15 | `News_Count_Log` | Log-transformed daily news volume: log(1 + count) |

Sentiment scores come from FinBERT-scored headlines, filtered by a fixed
gold/macro relevance vocabulary and aggregated per trading day (see
`src/prepare_data.py`).

## Proposed Architecture: GRN-MLP Ensemble

- **Branch 1 — Gated Residual Network (GRN):** the dynamic neural component.
  Gated Linear Units (GLU) with skip connections filter input features
  conditioned on the market state — suppressing noisy sentiment signals during
  calm periods and amplifying them during high-volatility news shocks.
- **Branch 2 — Multi-Layer Perceptron (MLP):** captures deep, non-linear
  static relationships across macroeconomic indicators and historical
  Gold_Close context.
- **Ensemble Fusion (Dynamic Gated Blending):** a final per-sample gating
  layer weighs the GRN and MLP branch outputs for each time step, producing an
  adaptive final forecast.

Implementation: `src/models.py` (`GLU`, `GRNBlock`, `GRNMLPEnsemble`).

## Benchmark Models

1. Random Forest (RF)
2. LightGBM (LGBM)
3. Long Short-Term Memory (LSTM)
4. Multi-Layer Perceptron (MLP)

## Ablation Study Design (4 Experiments)

| Experiment | Features used | Count |
|---|---|---|
| **Exp1 — Price Only** | Gold_Close | 1 |
| **Exp2 — Macro** | Gold_Close + 6 macro indicators | 7 |
| **Exp3 — Sentiment** | Gold_Close + 8 sentiment features | 9 |
| **Exp4 — All** | full feature set | 15 |

The error reduction (RMSE, MAE, MAPE) between **Exp2 (Macro)** and
**Exp4 (All)** directly quantifies the contribution of multi-dimensional
sentiment dynamics.

## Evaluation Metrics

RMSE, MAE, MAPE, and R² — always computed on the **price scale** from saved
prediction arrays.

## Methodological notes (leakage control)

- Models predict the **next-day log-return**; predictions are converted back
  to price (`Close_t * exp(r)`) before metrics — this keeps the regression
  target stationary while metrics stay interpretable.
- Scalers are fit on **training rows only**; the split is chronological 80/20,
  never shuffled.
- Neural models use **validation-based early stopping** (last 15% of train
  windows, patience 10, max 100 epochs); test data is never used for any
  selection.

## Repository layout

```
ORIEON/
├── README.md
├── requirements.txt        # pinned versions, Python 3.10–3.12
├── run_all.py              # one command: data -> 4 experiments x 5 models -> table + figures
├── src/
│   ├── prepare_data.py     # relevance filter + daily sentiment aggregation + 4 feature modes
│   ├── models.py           # MLP, LSTM (benchmarks), GRN-MLP Ensemble (proposed)
│   ├── train.py            # shared training loop (early stopping, fixed protocol)
│   ├── evaluate.py         # metrics + prediction saving + results tables
│   └── make_figures.py     # loss curves, RMSE/R2 comparisons, prediction line graphs
├── data/                   # gold_clean.csv and news_with_scores.csv
├── results/                # results_all_experiments.csv + per-experiment details
└── figures/                # generated plots
```

## Exact run instructions

```bash
# Python 3.10–3.12
pip install -r requirements.txt

# data/gold_clean.csv and data/news_with_scores.csv must be present, then:
python run_all.py
```

On Colab (torch/CUDA preinstalled) only LightGBM is needed:

```python
!pip install -q lightgbm
!python run_all.py
```

## Outputs

- `results/results_all_experiments.csv` — **single table**: 4 experiments × 5
  models, all metrics
- `results/<Exp>/test_results.csv` — per-experiment table
- `results/<Exp>/*_test_predictions.csv` — saved test predictions per model
- `figures/training_loss_curves.png` — per-experiment training curves
- `figures/comparison_rmse.png`, `figures/comparison_r2.png` — model ×
  experiment comparison bars
- `figures/predictions/<Exp>/<Model>.png` — actual vs predicted, per model
  (separate)
- `figures/predictions/<Exp>_combined.png` — all models vs actual, per
  experiment (combined)
