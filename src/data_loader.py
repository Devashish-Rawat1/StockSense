"""
data_loader.py — All data loading & feature engineering.

Feature names match EXACTLY what the XGBoost/LightGBM models were trained on:
  store_avg_sales  = mean sales per store (across all items/dates)
  item_avg_sales   = mean sales per item  (across all stores/dates)
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
    if not os.path.exists(RAW_DATA_PATH):
        st.error(
            f"❌ Raw data not found: `{RAW_DATA_PATH}`\n\n"
            "Ensure `data/raw/train.csv` is committed to your GitHub repo."
        )
        st.stop()
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=[DATE_COL])
    df.sort_values(DATE_COL, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_featured_data() -> pd.DataFrame:
    """
    Load featured dataset. Falls back to building from train.csv if
    the processed CSV isn't in the repo (e.g. gitignored due to size).
    """
    if os.path.exists(FEATURED_DATA_PATH):
        df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=[DATE_COL])
        # Rename legacy column names if the CSV was built with old code
        df = _rename_legacy_columns(df)
        df.sort_values(DATE_COL, inplace=True)
        return df

    st.info(
        "⚙️ `data/processed/featured_data.csv` not found — "
        "building features from `train.csv`. Cached after first run.",
        icon="ℹ️",
    )
    raw = load_raw_data()
    return _build_features(raw)


def _rename_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle old column names that may exist in a pre-built featured_data.csv.
    Maps them to the names the saved models actually expect.
    """
    rename_map = {
        "store_item_avg":  "store_avg_sales",   # old name → correct name
        "rolling_std_7":   "rolling_std_7",      # these are fine, just listed
        "rolling_std_14":  "rolling_std_14",
        "rolling_std_28":  "rolling_std_28",
    }
    # Only rename columns that actually exist
    actual_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    if actual_renames:
        df = df.rename(columns=actual_renames)
    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Output columns match EXACTLY what the trained models expect (see FEATURE_COLS).
    """
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

    # 3. Rolling mean features (shift(1) prevents data leakage)
    shifted = grp.shift(1)
    for w in [7, 14, 28]:
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean().values

    # 4. Aggregate features — named to match what the model was trained on
    store_avg = (
        df.groupby("store")[TARGET_COL]
        .mean()
        .rename("store_avg_sales")
        .reset_index()
    )
    item_avg = (
        df.groupby("item")[TARGET_COL]
        .mean()
        .rename("item_avg_sales")
        .reset_index()
    )
    df = df.merge(store_avg, on="store", how="left")
    df = df.merge(item_avg,  on="item",  how="left")

    return df


# ── Public aliases ────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return _build_features(df)


# ── Train / Val split ─────────────────────────────────────────────────────────

def train_val_split(df: pd.DataFrame, val_start: str = "2017-01-01"):
    return (
        df[df[DATE_COL] < val_start].copy(),
        df[df[DATE_COL] >= val_start].copy(),
    )


def get_X_y(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset    = df[available + [TARGET_COL]].dropna()
    return subset[available], subset[TARGET_COL]


# ── LSTM helpers ──────────────────────────────────────────────────────────────

def create_sequences(series: np.ndarray, window: int = LSTM_WINDOW):
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window, 0])
    return np.array(X), np.array(y)


# ── Upload validation ─────────────────────────────────────────────────────────

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
