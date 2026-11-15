"""ORIEON data preparation — Section 2 of the approved protocol.

Reads data/gold_clean.csv and data/news_with_scores.csv, builds the 15-feature
daily dataset with news-sentiment aggregation, runs sanity checks, and writes
results/processed_15_feature_data.csv and results/data_protocol.json.

Runnable standalone:  python -m src.prepare_data   (or)  python src/prepare_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Protocol constants (do not change)
# --------------------------------------------------------------------------
SEED = 42

BASE_FEATURES = ["Gold_Close", "DXY", "Silver", "CrudeOil", "UST10Y", "VIX", "SP500"]

SENTIMENT_FEATURES = [
    "Sentiment_Mean", "Sentiment_Std", "Sentiment_Min", "Sentiment_Max",
    "Positive_Ratio", "Negative_Ratio", "News_Available", "News_Count_Log",
]

FEATURES = BASE_FEATURES + SENTIMENT_FEATURES  # exactly 15
TARGET = "y_close_t1"

# Exact relevance terms from the protocol — literal substring matches.
# NOTE: "fed " intentionally has a trailing space.
RELEVANCE_TERMS = [
    "gold", "xau", "bullion", "precious metal", "federal reserve", "fed ",
    "interest rate", "inflation", "consumer price", "cpi", "central bank",
    "us dollar", "u.s. dollar", "dollar index", "treasury yield", "bond yield",
    "monetary policy", "recession", "economic growth", "nonfarm", "payroll",
    "unemployment", "geopolitical", "safe haven", "safe-haven",
]

TRAIN_FRACTION = 0.8

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_raw(data_dir: Path = DATA_DIR):
    """1. Read both CSVs, parse Date, sort by Date, reset index."""
    gold = pd.read_csv(data_dir / "gold_clean.csv", parse_dates=["Date"])
    news = pd.read_csv(data_dir / "news_with_scores.csv", parse_dates=["Date"])
    gold = gold.sort_values("Date").reset_index(drop=True)
    news = news.sort_values("Date").reset_index(drop=True)
    return gold, news


def filter_relevant(news: pd.DataFrame) -> pd.DataFrame:
    """2. Keep only headlines whose lowercased Clean text contains >= 1 term."""
    clean = news["Clean"].fillna("").astype(str).str.lower()
    mask = clean.apply(lambda t: any(term in t for term in RELEVANCE_TERMS))
    return news.loc[mask].copy()


def aggregate_daily(news_rel: pd.DataFrame) -> pd.DataFrame:
    """3-4. Helper columns + per-calendar-day aggregation."""
    news_rel["positive"] = (news_rel["Score"] > 0).astype(int)
    news_rel["negative"] = (news_rel["Score"] < 0).astype(int)

    daily = (
        news_rel.groupby("Date")
        .agg(
            Sentiment_Mean=("Score", "mean"),
            Sentiment_Std=("Score", "std"),
            Sentiment_Min=("Score", "min"),
            Sentiment_Max=("Score", "max"),
            Positive_Ratio=("positive", "mean"),
            Negative_Ratio=("negative", "mean"),
            News_Count=("Score", "size"),
        )
        .reset_index()
    )
    return daily


def build_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """Full Section-2 pipeline. Returns (merged_df, protocol_info)."""
    gold, news = load_raw(data_dir)
    news_rel = filter_relevant(news)
    daily = aggregate_daily(news_rel)

    # 5. Left-join daily aggregation onto the gold trading days.
    df = gold.merge(daily, on="Date", how="left")

    # 6. News_Available flag.
    df["News_Available"] = df["News_Count"].notna().astype(int)

    # 7. Fill missing values with 0.0.
    fill_zero = [
        "Sentiment_Mean", "Sentiment_Std", "Sentiment_Min", "Sentiment_Max",
        "Positive_Ratio", "Negative_Ratio", "News_Count",
    ]
    df[fill_zero] = df[fill_zero].fillna(0.0)

    # 8. News_Count_Log = log1p(News_Count).
    df["News_Count_Log"] = np.log1p(df["News_Count"])

    # Keep only protocol columns.
    out = df[["Date"] + FEATURES + [TARGET]].copy()

    # 9. Sanity checks: all 15 features + target exist, no NaN.
    assert len(FEATURES) == 15, f"expected 15 features, got {len(FEATURES)}"
    missing_cols = [c for c in FEATURES + [TARGET] if c not in out.columns]
    assert not missing_cols, f"missing columns: {missing_cols}"
    nan_counts = out[FEATURES + [TARGET]].isna().sum()
    assert int(nan_counts.sum()) == 0, f"NaN found:\n{nan_counts[nan_counts > 0]}"

    info = {
        "seed": SEED,
        "feature_list": FEATURES,
        "target": TARGET,
        "n_rows_total": int(len(out)),
        "date_range": [str(out["Date"].min().date()), str(out["Date"].max().date())],
        "n_headlines_total": int(len(news)),
        "n_headlines_relevant": int(len(news_rel)),
        "n_gold_days_with_news": int((out["News_Available"] == 1).sum()),
        "train_fraction": TRAIN_FRACTION,
        "split": {},  # filled by add_split_info
    }
    return out, info


def split_indices(df: pd.DataFrame) -> tuple[int, int]:
    """Chronological 80/20 split, no validation set, never shuffled."""
    n = len(df)
    n_train = int(n * TRAIN_FRACTION)
    return n_train, n - n_train


def add_split_info(df: pd.DataFrame, info: dict) -> dict:
    n_train, n_test = split_indices(df)
    info["split"] = {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "train_date_range": [str(df["Date"].iloc[0].date()),
                             str(df["Date"].iloc[n_train - 1].date())],
        "test_date_range": [str(df["Date"].iloc[n_train].date()),
                            str(df["Date"].iloc[-1].date())],
        "train_target_max": float(df[TARGET].iloc[:n_train].max()),
        "test_target_max": float(df[TARGET].iloc[n_train:].max()),
        "n_test_targets_above_train_max": int(
            (df[TARGET].iloc[n_train:] > df[TARGET].iloc[:n_train].max()).sum()
        ),
    }
    return info


def prepare(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    df, info = build_dataset(data_dir)
    info = add_split_info(df, info)

    out_csv = results_dir / "processed_15_feature_data.csv"
    df.to_csv(out_csv, index=False)

    with open(results_dir / "data_protocol.json", "w") as f:
        json.dump(info, f, indent=2)

    print("[prepare_data] wrote", out_csv)
    print(json.dumps(info, indent=2))
    return df, info


if __name__ == "__main__":
    prepare()
