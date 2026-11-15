# ORIEON — Gold Price Forecasting with ML, DL, and Graph Neural Networks using Financial-News Sentiment

ORIEON predicts the **next-day gold closing price** from a 15-feature daily
dataset that combines market variables (gold, DXY, silver, crude oil, UST10Y,
VIX, S&P 500) with 8 aggregated FinBERT news-sentiment features. Fourteen
models are trained under one fixed, supervisor-approved protocol: tree
baselines, deep sequence baselines, five GNN-only candidates over a daily
15-node feature graph, and four GNN + recurrent hybrids.

> **Status:** the results table below is intentionally blank — it is filled in
> after running `python run_all.py`. No numbers in this repository are
> pre-computed; everything is produced by the pipeline itself.

## Repository layout

```
ORIEON/
├── README.md
├── requirements.txt        # pinned versions, Python 3.10–3.12
├── run_all.py              # one command: data prep -> A -> B -> C -> D -> metrics -> figures
├── src/
│   ├── prepare_data.py     # relevance filter + daily sentiment aggregation + sanity checks
│   ├── models.py           # MLP/LSTM/GRU, GraphOnlyModel, GraphTemporalModel
│   ├── train.py            # shared training loop with the fixed protocol settings
│   ├── evaluate.py         # metrics + prediction saving + combined results table
│   └── make_figures.py
├── data/                   # gold_clean.csv and news_with_scores.csv
├── results/                # generated outputs (predictions, metrics, weights, histories)
└── figures/                # generated plots
```

## Exact run instructions

```bash
# Python 3.10–3.12
pip install -r requirements.txt

# data/gold_clean.csv and data/news_with_scores.csv must be present, then:
python run_all.py
```

Each stage is also runnable individually:

```bash
python -m src.prepare_data    # builds results/processed_15_feature_data.csv + data_protocol.json
python -m src.train           # trains all 14 models, saves weights/predictions/metrics
python -m src.evaluate        # rebuilds results/test_results.csv from saved predictions
python -m src.make_figures    # regenerates figures/ from saved outputs
```

**Hardware:** the code automatically uses CUDA if available (`device=cuda`),
otherwise CPU. All protocol settings (`torch.set_num_threads(4)`, batch size
64, 40 epochs) are identical on both. Report your library versions and device
alongside results (they are printed at run start and stored in each
`*_history.json`).

## Protocol summary

- **Target:** `y_close_t1` (next-day gold close), regression.
- **Features (exactly 15):** `Gold_Close, DXY, Silver, CrudeOil, UST10Y, VIX,
  SP500` + 8 sentiment features (`Sentiment_Mean/Std/Min/Max`,
  `Positive_Ratio`, `Negative_Ratio`, `News_Available`, `News_Count_Log`)
  built by filtering headlines against a fixed relevance-term list and
  aggregating FinBERT scores per calendar day (left-joined onto trading days,
  missing days filled with 0, `News_Count_Log = log1p(count)`).
- **Split:** chronological 80/20 (4,027 train / 1,007 test), no validation
  set, never shuffled. `StandardScaler`s for inputs and target are fit on
  training rows only; metrics are computed on inverse-transformed predictions.
- **Neural settings (all groups):** seed 42 (Python/NumPy/PyTorch), lookback
  20 days, batch 64, exactly 40 epochs, no early stopping (final-epoch model),
  AdamW (weight_decay 1e-5), MSE on the scaled target, gradient clipping 1.0,
  `torch.set_num_threads(4)`. Head: `Dropout(0.2) → Linear(in, 32) → ReLU →
  Linear(32, 1)`.
- **Group A — trees (current-day vector):** RandomForest (500 trees,
  max_features 0.7, seed 42); LightGBM (400 rounds, lr 0.05, depth 3,
  15 leaves, subsample 0.85, colsample 0.8, reg_lambda 1.0).
- **Group B — deep baselines (lr 1e-3, input 20×15):** MLP (300→128→64),
  LSTM (hidden 64), GRU (hidden 64).
- **Group C — GNN-only (lr 8e-4):** the 15 features are nodes of a daily
  graph; `h = tanh(value_proj(x) + node_embedding)` with WIDTH 48; readout
  `concat(mean-pool(z), z[Gold_Close]) → Linear(96,48) → Tanh`, then mean over
  the 20 days. Variants: GCN (fixed |Pearson| adjacency from training rows,
  diag 1, D^-1/2 A D^-1/2), GraphSAGE (neighbour mean excluding self), GAT
  (additive attention), Static-Learned (softmax over learnable logits + I),
  Dynamic (softmax(QKᵀ/√48 + I)).
- **Group D — hybrids (lr 8e-4):** the 20 daily 48-dim readouts feed a GRU or
  LSTM (hidden 64) instead of day-averaging; last hidden state → head. All
  four combinations of {Dynamic, Static-Learned} × {GRU, LSTM}.

## Results (filled in after running)

> ⚠️ **To be filled after running `python run_all.py`.** Paste the contents of
> the generated `results/test_results.csv` here. Do not edit the numbers.

| Model | n_params | train_seconds | MAE | MSE | RMSE | R² | MAPE | n |
|---|---|---|---|---|---|---|---|---|
| _to be filled after running_ | | | | | | | | |

Reference values from the original approved run (sanity check only — never
hard-coded or targeted; small deviations across library versions/hardware are
expected): RandomForest RMSE 1438.41 · LightGBM 1426.53 · MLP 612.35
(R² = 0.6462) · LSTM 1084.42 · GRU 960.94 · GCN-only 1337.21 ·
GraphSAGE-only 1295.53 · GAT-only 1356.79 · Static-Learned-GNN-only 1274.50 ·
Dynamic-GNN-only 1245.09. Data sanity references: 5,034 rows; 11,663 relevant
headlines of 14,173; 3,489 gold days with retained news.

## Limitations and integrity notes

- **Regime shift:** 674 of 1,007 test targets (66.9%) exceed the maximum
  training target (train max ≈ 2,069 vs test max ≈ 5,318). Weak or negative
  R² for some models is expected and legitimate; the split is deliberately
  kept chronological and is not altered to "fix" this.
- **No validation set:** the protocol uses a single chronological 80/20 split.
  The test set is never used for hyperparameter tuning, early stopping,
  checkpoint selection, or model selection.
- **No "final hybrid" is selected:** because there is no validation set, no
  winner is crowned from the test table. All four hybrids are reported
  transparently; any future final-hybrid selection requires a separate,
  documented selection protocol.
- **Reproducibility:** exact numbers vary slightly across library versions and
  hardware; always report the versions printed at run start.
