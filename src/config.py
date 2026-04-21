"""
config.py — central place for all constants, paths, and model metadata.
Import this everywhere instead of hard-coding paths.
"""

import os

# ── Root paths ───────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

RAW_DATA_PATH      = os.path.join(DATA_DIR, "raw", "train.csv")
FEATURED_DATA_PATH = os.path.join(DATA_DIR, "processed", "featured_data.csv")

# ── Model file paths ─────────────────────────────────────────────────────────
MODEL_PATHS = {
    "XGBoost":           os.path.join(MODELS_DIR, "xgb_model.joblib"),
    "LightGBM":          os.path.join(MODELS_DIR, "lgbm_model.joblib"),
    "Univariate LSTM":   os.path.join(MODELS_DIR, "lstm_model.keras"),
    "Multivariate LSTM": os.path.join(MODELS_DIR, "multi_lstm_model.keras"),
}

SCALER_PATHS = {
    "Univariate LSTM":   os.path.join(MODELS_DIR, "lstm_scaler.save"),
    "Multivariate LSTM": os.path.join(MODELS_DIR, "multi_lstm_scaler.save"),
}

PREDS_PATHS = {
    "XGBoost":           os.path.join(MODELS_DIR, "xgb_preds.joblib"),
    "LightGBM":          os.path.join(MODELS_DIR, "lgbm_preds.joblib"),
    "Univariate LSTM":   os.path.join(MODELS_DIR, "lstm_preds.joblib"),
    "Multivariate LSTM": os.path.join(MODELS_DIR, "multi_lstm_preds.joblib"),
    "actual":            os.path.join(MODELS_DIR, "y_val_actual.joblib"),
}

# ── Model metadata (for UI display) ─────────────────────────────────────────
MODEL_METRICS = {
    "XGBoost":           {"RMSE": 7.9126,  "MAPE": 12.4361, "type": "tree"},
    "LightGBM":          {"RMSE": 7.9421,  "MAPE": 12.5623, "type": "tree"},
    "Univariate LSTM":   {"RMSE": 8.7666,  "MAPE": 13.9078, "type": "lstm"},
    "Multivariate LSTM": {"RMSE": 8.3659,  "MAPE": 13.1059, "type": "lstm"},
}

BEST_MODEL = "XGBoost"

# ── Feature config (must match feature_engineering notebook) ─────────────────
FEATURE_COLS = [
    "store", "item",
    "year", "month", "week", "day", "dayofweek", "is_weekend",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "store_item_avg",
]

TARGET_COL   = "sales"
DATE_COL     = "date"
LSTM_WINDOW  = 28          # look-back window used during training
NUM_STORES   = 10
NUM_ITEMS    = 50

# ── Inventory defaults ────────────────────────────────────────────────────────
INV_DEFAULTS = {
    "ordering_cost":          50.0,   # $ per order
    "holding_cost_per_unit":   2.0,   # $ / unit / year
    "lead_time_days":          7,     # days
    "service_level_z":         1.65,  # 95% service level
}
