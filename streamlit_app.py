"""
StockSense — Streamlit App
Entry point: runs the multi-page navigation shell.
"""

import streamlit as st

st.set_page_config(
    page_title="StockSense",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("📦 StockSense")
st.sidebar.caption("StockSense Dashboard")

PAGES = {
    " Home":               "pages/home.py",
    " EDA":                "pages/eda.py",
    " Forecast":           "pages/forecast.py",
    " Model Comparison":   "pages/model_comparison.py",
    " Inventory Planner":  "pages/inventory.py",
    " Batch Predict":      "pages/batch_predict.py",
    " SHAP Explainer":     "pages/shap_explainer.py",
}

page = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.info("Models: XGBoost · LightGBM · Uni-LSTM · Multi-LSTM")

# ── Dynamic page loader ─────────────────────────────────────────────────────
import importlib.util, sys, os

def load_page(path: str):
    abs_path = os.path.join(os.path.dirname(__file__), path)
    spec = importlib.util.spec_from_file_location("page_module", abs_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

load_page(PAGES[page])
