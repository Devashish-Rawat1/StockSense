"""pages/home.py — Landing / overview page."""

import streamlit as st
from src.config import MODEL_METRICS, BEST_MODEL


def render():
    st.title("StockSense — Demand Forecasting & Inventory Optimization")
    st.markdown(
        "**Predict future product demand to optimize inventory and reduce "
        "stockouts & overstocking.**"
    )
    st.markdown("---")

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",    BEST_MODEL)
    c2.metric("Best RMSE",     f"{MODEL_METRICS[BEST_MODEL]['RMSE']:.4f}")
    c3.metric("Best MAPE",     f"{MODEL_METRICS[BEST_MODEL]['MAPE']:.2f}%")
    c4.metric("Models Trained","4")

    st.markdown("---")

    # ── Project overview ─────────────────────────────────────────────────────
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("🗂️ Project Pipeline")
        st.markdown("""
        ```
        Raw Data (train.csv)
             │
             ▼
        Exploratory Data Analysis
             │
             ▼
        Feature Engineering
        (Lag · Rolling · Calendar · Aggregate)
             │
             ├──► XGBoost        RMSE: 7.91  MAPE: 12.44%
             ├──► LightGBM       RMSE: 7.94  MAPE: 12.56%
             ├──► Univariate LSTM RMSE: 8.77 MAPE: 13.91%
             └──► Multivariate LSTM RMSE: 8.37 MAPE: 13.11%
                                    │
                                    ▼
                            Inventory Planner
                        (EOQ · ROP · Safety Stock)
        ```
        """)

    with col2:
        st.subheader("📂 Dataset Facts")
        st.markdown("""
        | Attribute | Value |
        |-----------|-------|
        | Rows | 913,000 |
        | Columns | 4 |
        | Stores | 10 |
        | Items | 50 |
        | Date Range | 2013–2017 |
        | Missing Values | None |
        | Engineered Features | 19 |
        """)

    st.markdown("---")

    # ── Navigation guide ─────────────────────────────────────────────────────
    st.subheader("🚀 What you can do here")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.info("**📊 EDA**\nExplore sales trends, seasonality, and store-item distributions interactively.")
    with g2:
        st.info("**🤖 Forecast**\nSelect a model, store, and item to generate demand forecasts.")
    with g3:
        st.info("**🏭 Inventory**\nConvert forecasts into EOQ, Reorder Points, and Safety Stock per item.")

    g4, g5, _ = st.columns(3)
    with g4:
        st.info("**📈 Model Comparison**\nSide-by-side RMSE/MAPE comparisons and forecast plots.")
    with g5:
        st.info("**📤 Batch Predict**\nUpload a CSV and get bulk forecasts for any store–item combination.")

    st.markdown("---")
    st.caption("Built with Streamlit · XGBoost · LightGBM · TensorFlow/Keras")


render()
