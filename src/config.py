"""
config.py — central place for all constants, paths, and model metadata.

Path strategy for Streamlit Cloud:
  - Streamlit Cloud clones your repo to /mount/src/<repo-name>/
  - All paths must be resolved relative to this repo root at runtime,
    NOT relative to this file's location inside src/.
  - We use an environment variable REPO_ROOT if set, otherwise walk up
    from this file's directory.
"""

import os

# ── Resolve repo root dynamically ────────────────────────────────────────────
# This file lives at: <repo_root>/src/config.py  (one level deep)
# So: parent of parent = repo root... unless src/ is deeper.
# Walk up until we find a known anchor file (requirements.txt or train.csv).

def _find_repo_root() -> str:
    """Walk up from this file until we find the repo root."""
    candidate = os.path.dirname(os.path.abspath(__file__))  # src/
    for _ in range(5):  # max 5 levels up
        candidate = os.path.dirname(candidate)
        if os.path.exists(os.path.join(candidate, "requirements.txt")):
            return candidate
    # Last resort: current working directory (where streamlit is launched from)
    return os.getcwd()

ROOT_DIR   = _find_repo_root()
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

# ── Model metadata ────────────────────────────────────────────────────────────
MODEL_METRICS = {
    "XGBoost":           {"RMSE": 7.9126,  "MAPE": 12.4361, "type": "tree"},
    "LightGBM":          {"RMSE": 7.9421,  "MAPE": 12.5623, "type": "tree"},
    "Univariate LSTM":   {"RMSE": 8.7666,  "MAPE": 13.9078, "type": "lstm"},
    "Multivariate LSTM": {"RMSE": 8.3659,  "MAPE": 13.1059, "type": "lstm"},
}

BEST_MODEL = "XGBoost"

# ── Feature config ────────────────────────────────────────────────────────────
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
LSTM_WINDOW  = 28
NUM_STORES   = 10
NUM_ITEMS    = 50

# ── Inventory defaults ────────────────────────────────────────────────────────
INV_DEFAULTS = {
    "ordering_cost":          50.0,
    "holding_cost_per_unit":   2.0,
    "lead_time_days":          7,
    "service_level_z":         1.65,
}
