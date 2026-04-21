"""pages/model_comparison.py — Side-by-side model performance comparison."""

import numpy as np
import streamlit as st

from src.config import MODEL_METRICS, BEST_MODEL
from src.model_loader import load_saved_predictions
from src.plots import (
    plot_all_models_comparison,
    plot_metrics_bar,
    plot_residuals,
    plot_forecast_vs_actual,
)


def render():
    st.title("📈 Model Comparison")
    st.markdown(
        "Compare all four models — XGBoost, LightGBM, Univariate LSTM, "
        "and Multivariate LSTM — on the 2017 validation set."
    )

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("📐 Performance Metrics")
    cols = st.columns(4)
    for i, (name, m) in enumerate(MODEL_METRICS.items()):
        medal = " 🏆" if name == BEST_MODEL else ""
        cols[i].metric(f"{name}{medal}", f"RMSE {m['RMSE']}", f"MAPE {m['MAPE']}%")

    st.plotly_chart(plot_metrics_bar(), use_container_width=True)

    # ── Load saved predictions ────────────────────────────────────────────────
    with st.spinner("Loading saved predictions..."):
        preds = load_saved_predictions()

    if not preds or "actual" not in preds:
        st.error(
            "Saved prediction files not found in models/. "
            "Please ensure xgb_preds.joblib, lgbm_preds.joblib, "
            "lstm_preds.joblib, multi_lstm_preds.joblib, and y_val_actual.joblib exist."
        )
        return

    # Align lengths
    min_len = min(len(v) for v in preds.values())
    preds   = {k: v[:min_len] for k, v in preds.items()}

    # ── All-model overlay plot ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Forecast Overlay")
    n_pts = st.slider("Time steps to show", 50, 500, 200, key="comp_slider")
    st.plotly_chart(plot_all_models_comparison(preds, n_pts), use_container_width=True)

    # ── Individual model deep-dive ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Individual Model Deep-Dive")
    selected = st.selectbox("Choose a model", list(MODEL_METRICS.keys()))

    if selected in preds:
        tab1, tab2 = st.tabs(["Forecast vs Actual", "Residuals"])
        with tab1:
            st.plotly_chart(
                plot_forecast_vs_actual(
                    preds["actual"], preds[selected], selected, n_points=n_pts
                ),
                use_container_width=True,
            )
        with tab2:
            st.plotly_chart(
                plot_residuals(preds["actual"], preds[selected], selected),
                use_container_width=True,
            )
    else:
        st.warning(f"Predictions for {selected} not found in saved files.")

    # ── Insight callout ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Key Takeaways")
    st.markdown(f"""
    | Rank | Model | RMSE | MAPE | Type |
    |------|-------|------|------|------|
    | 🥇 1 | **XGBoost** | 7.9126 | 12.44% | Gradient Boosted Tree |
    | 🥈 2 | LightGBM | 7.9421 | 12.56% | Gradient Boosted Tree |
    | 🥉 3 | Multivariate LSTM | 8.3659 | 13.11% | Deep Learning |
    | 4 | Univariate LSTM | 8.7666 | 13.91% | Deep Learning |

    - **Tree models outperform LSTMs** on this structured tabular data — lag and rolling features give them rich temporal context without needing sequences.
    - **Multivariate LSTM > Univariate LSTM** — using all engineered features (not just raw sales) helps the network.
    - XGBoost's **MAPE of 12.4%** makes it production-ready for inventory planning.
    """)


render()
