"""
data_loader.py
Handles all data loading & feature engineering.
Uses st.cache_data to avoid reloading on every interaction.

Key design: load_featured_data() tries the pre-built CSV first.
If it doesn't exist (common on Streamlit Cloud where processed/
files are gitignored), it builds features on-the-fly from train.csv.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    RAW_DATA_PATH, FEATURED_DATA_PATH,
    DATE_COL, TARGET_COL, FEATURE_COLS,
    LSTM_WINDOW,
)


# ── Raw data ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the raw train.csv. Raises a clean error if missing."""
    if not os.path.exists(RAW_DATA_PATH):
        st.error(
            f"❌ Raw data file not found: `{RAW_DATA_PATH}`\n\n"
            "Make sure `data/raw/train.csv` is committed to your GitHub repo."
        )
        st.stop()
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=[DATE_COL])
    df.sort_values(DATE_COL, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_featured_data() -> pd.DataFrame:
    """
    Load the feature-engineered dataset.
    - If data/processed/featured_data.csv exists → load it directly (fast).
    - Otherwise → build features from train.csv on-the-fly (slower, ~30s, cached after).
    This handles the common case where processed/ is gitignored.
    """
    if os.path.exists(FEATURED_DATA_PATH):
        df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=[DATE_COL])
        df.sort_values([DATE_COL], inplace=True)
        return df

    # Fallback: build from raw data
    st.info(
        "⚙️ `data/processed/featured_data.csv` not found — "
        "building features from `train.csv`. This runs once and is then cached.",
        icon="ℹ️",
    )
    raw = load_raw_data()
    featured = _build_features(raw)
    return featured


# ── Feature engineering (mirrors the notebook pipeline exactly) ───────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: time → lag → rolling → aggregate."""
    df = df.copy()
    df = df.sort_values(["store", "item", DATE_COL]).reset_index(drop=True)

    # 1. Time features
    df["year"]       = df[DATE_COL].dt.year
    df["month"]      = df[DATE_COL].dt.month
    df["week"]       = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day"]        = df[DATE_COL].dt.day
    df["dayofweek"]  = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # 2. Lag features
    grp = df.groupby(["store", "item"])[TARGET_COL]
    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    # 3. Rolling features (shift(1) avoids data leakage)
    shifted = grp.shift(1)
    for w in [7, 14, 28]:
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean().values
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std().values

    # 4. Store-item average (aggregate feature)
    store_item_avg = (
        df.groupby(["store", "item"])[TARGET_COL]
        .mean()
        .rename("store_item_avg")
        .reset_index()
    )
    df = df.merge(store_item_avg, on=["store", "item"], how="left")

    return df


# ── Public aliases (kept for backward compatibility) ─────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return _build_features(df)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]       = df[DATE_COL].dt.year
    df["month"]      = df[DATE_COL].dt.month
    df["week"]       = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day"]        = df[DATE_COL].dt.day
    df["dayofweek"]  = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby(["store", "item"])[TARGET_COL].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows=(7, 14, 28)) -> pd.DataFrame:
    df = df.copy()
    shifted = df.groupby(["store", "item"])[TARGET_COL].shift(1)
    for w in windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean().values
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std().values
    return df

def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    agg = (
        df.groupby(["store", "item"])[TARGET_COL]
        .mean()
        .rename("store_item_avg")
        .reset_index()
    )
    return df.merge(agg, on=["store", "item"], how="left")


# ── Train / Val splits ────────────────────────────────────────────────────────

def train_val_split(df: pd.DataFrame, val_start: str = "2017-01-01"):
    train = df[df[DATE_COL] < val_start].copy()
    val   = df[df[DATE_COL] >= val_start].copy()
    return train, val


def get_X_y(df: pd.DataFrame):
    """Return feature matrix X and target vector y, dropping NaN rows."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset = df[available + [TARGET_COL]].dropna()
    return subset[available], subset[TARGET_COL]


# ── LSTM sequence builder ─────────────────────────────────────────────────────

def create_sequences(series: np.ndarray, window: int = LSTM_WINDOW):
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window, 0])
    return np.array(X), np.array(y)


# ── Uploaded CSV validation ───────────────────────────────────────────────────

def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    required = {DATE_COL, "store", "item"}
    missing  = required - set(df.columns)
    if missing:
        return False, f"Missing required columns: {missing}"
    try:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    except Exception:
        return False, "Could not parse 'date' column as datetime."
    if df["store"].nunique() > 10 or df["item"].nunique() > 50:
        return False, "store/item values exceed training range (10 stores, 50 items)."
    return True, "OK"
