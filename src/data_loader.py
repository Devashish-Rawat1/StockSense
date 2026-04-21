"""
data_loader.py
Handles all data loading & feature engineering.
Uses st.cache_data to avoid reloading on every interaction.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    RAW_DATA_PATH, FEATURED_DATA_PATH,
    DATE_COL, TARGET_COL, FEATURE_COLS,
    LSTM_WINDOW,
)


# ── Raw data ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the raw train.csv."""
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=[DATE_COL])
    df.sort_values(DATE_COL, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def load_featured_data() -> pd.DataFrame:
    """Load the pre-engineered feature CSV."""
    df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=[DATE_COL])
    df.sort_values([DATE_COL], inplace=True)
    return df


# ── Feature engineering (mirrors notebook logic) ─────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]      = df[DATE_COL].dt.year
    df["month"]     = df[DATE_COL].dt.month
    df["week"]      = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["day"]       = df[DATE_COL].dt.day
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"]= df["dayofweek"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"sales_lag_{lag}"] = (
            df.groupby(["store", "item"])[TARGET_COL].shift(lag)
        )
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 14, 28)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        shifted = df.groupby(["store", "item"])[TARGET_COL].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std()
    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    agg = (
        df.groupby(["store", "item"])[TARGET_COL]
        .mean()
        .rename("store_item_avg")
        .reset_index()
    )
    df = df.merge(agg, on=["store", "item"], how="left")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline — mirrors the notebook."""
    df = df.sort_values(["store", "item", DATE_COL])
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_aggregate_features(df)
    return df


# ── Train / Val splits ────────────────────────────────────────────────────────

def train_val_split(df: pd.DataFrame, val_start: str = "2017-01-01"):
    train = df[df[DATE_COL] < val_start].copy()
    val   = df[df[DATE_COL] >= val_start].copy()
    return train, val


def get_X_y(df: pd.DataFrame):
    """Return feature matrix X and target vector y, dropping NaN rows."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset = df[available + [TARGET_COL]].dropna()
    X = subset[available]
    y = subset[TARGET_COL]
    return X, y


# ── LSTM sequence builder ─────────────────────────────────────────────────────

def create_sequences(series: np.ndarray, window: int = LSTM_WINDOW):
    """
    Convert a 1-D (or 2-D) scaled array into (X, y) sequences.
    series : shape (N,) or (N,1)
    returns: X shape (M, window, features), y shape (M,)
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window, 0])
    return np.array(X), np.array(y)


# ── User-uploaded CSV helper ──────────────────────────────────────────────────

def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Basic validation for user-uploaded batch prediction CSV.
    Expects at least: date, store, item columns.
    Returns (is_valid, message).
    """
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
