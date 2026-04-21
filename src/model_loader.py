"""
model_loader.py
Lazy-loads each of the 4 trained models with Streamlit caching.
Handles missing model files gracefully (demo mode).
"""

import os
import numpy as np
import streamlit as st

from src.config import MODEL_PATHS, SCALER_PATHS, PREDS_PATHS


# ── Tree models (XGBoost / LightGBM) ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_xgb_model():
    import joblib
    path = MODEL_PATHS["XGBoost"]
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_lgbm_model():
    import joblib
    path = MODEL_PATHS["LightGBM"]
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ── LSTM models ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_uni_lstm():
    """Returns (model, scaler) or (None, None)."""
    import joblib
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None, None

    model_path  = MODEL_PATHS["Univariate LSTM"]
    scaler_path = SCALER_PATHS["Univariate LSTM"]

    model  = load_model(model_path)  if os.path.exists(model_path)  else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


@st.cache_resource(show_spinner=False)
def load_multi_lstm():
    """Returns (model, scaler) or (None, None)."""
    import joblib
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None, None

    model_path  = MODEL_PATHS["Multivariate LSTM"]
    scaler_path = SCALER_PATHS["Multivariate LSTM"]

    model  = load_model(model_path)  if os.path.exists(model_path)  else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


# ── Saved predictions (for comparison page) ──────────────────────────────────

@st.cache_data(show_spinner=False)
def load_saved_predictions() -> dict:
    """
    Loads pre-saved .joblib prediction arrays for the val set.
    Returns a dict: { model_name -> np.ndarray }.
    """
    import joblib
    preds = {}
    for name, path in PREDS_PATHS.items():
        if os.path.exists(path):
            arr = joblib.load(path)
            preds[name] = np.array(arr).flatten()
    return preds


# ── Unified predict interface ─────────────────────────────────────────────────

def predict_tree(model_name: str, X) -> np.ndarray:
    """
    Run prediction for XGBoost or LightGBM.
    X : pd.DataFrame with FEATURE_COLS columns.
    Returns np.ndarray of predictions (clipped ≥ 0).
    """
    if model_name == "XGBoost":
        model = load_xgb_model()
    elif model_name == "LightGBM":
        model = load_lgbm_model()
    else:
        raise ValueError(f"Unknown tree model: {model_name}")

    if model is None:
        raise FileNotFoundError(f"Model file not found for {model_name}.")

    preds = model.predict(X)
    return np.clip(preds, 0, None)


def predict_uni_lstm(series: np.ndarray, n_steps: int = 1) -> np.ndarray:
    """
    Univariate LSTM rolling forecast.
    series : raw (unscaled) 1-D sales array of length ≥ LSTM_WINDOW.
    Returns n_steps predictions (inverse-scaled).
    """
    from src.config import LSTM_WINDOW

    model, scaler = load_uni_lstm()
    if model is None or scaler is None:
        raise FileNotFoundError("Univariate LSTM model/scaler not found.")

    scaled = scaler.transform(series.reshape(-1, 1))
    window = list(scaled[-LSTM_WINDOW:].flatten())
    results = []

    for _ in range(n_steps):
        X_input = np.array(window[-LSTM_WINDOW:]).reshape(1, LSTM_WINDOW, 1)
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        results.append(pred_scaled)
        window.append(pred_scaled)

    preds = scaler.inverse_transform(np.array(results).reshape(-1, 1)).flatten()
    return np.clip(preds, 0, None)
