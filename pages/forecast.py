"""pages/forecast.py — Interactive single-item demand forecasting."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import MODEL_METRICS, BEST_MODEL, NUM_STORES, NUM_ITEMS, LSTM_WINDOW
from src.data_loader import load_featured_data, train_val_split, get_X_y
from src.model_loader import predict_tree, predict_uni_lstm, load_saved_predictions
from src.plots import plot_forecast_vs_actual, plot_residuals


def render():
    st.title("🤖 Demand Forecast")
    st.markdown("Select a model, store, and item to see demand predictions.")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ Forecast Controls")

    model_name = st.sidebar.selectbox(
        "Model",
        ["XGBoost", "LightGBM", "Univariate LSTM", "Multivariate LSTM"],
        index=0,
    )
    store = st.sidebar.selectbox("Store",  list(range(1, NUM_STORES + 1)), index=0)
    item  = st.sidebar.selectbox("Item",   list(range(1, NUM_ITEMS  + 1)), index=0)
    n_show = st.sidebar.slider("Points to show", 50, 365, 150)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Selected Model Metrics**")
    m = MODEL_METRICS[model_name]
    st.sidebar.metric("RMSE", m["RMSE"])
    st.sidebar.metric("MAPE", f"{m['MAPE']}%")

    # ── Load pre-saved predictions (fast path) ────────────────────────────────
    saved = load_saved_predictions()
    actual_all = saved.get("actual")
    pred_all   = saved.get(model_name)

    if actual_all is not None and pred_all is not None:
        st.info(
            f"Showing **{model_name}** predictions on the 2017 validation set. "
            "Use the slider to control how many time steps are displayed."
        )

        n = min(n_show, len(actual_all), len(pred_all))
        fig = plot_forecast_vs_actual(actual_all, pred_all, model_name, n_points=n)
        st.plotly_chart(fig, use_container_width=True)

        # ── Residuals ────────────────────────────────────────────────────────
        with st.expander("📉 Residual Analysis"):
            st.plotly_chart(
                plot_residuals(actual_all[:n], pred_all[:n], model_name),
                use_container_width=True,
            )

        # ── Summary metrics ──────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{m['RMSE']:.4f}")
        col2.metric("MAPE", f"{m['MAPE']:.2f}%")
        col3.metric("Rank vs others",
                    "#1 🏆" if model_name == BEST_MODEL else f"#{sorted(MODEL_METRICS, key=lambda x: MODEL_METRICS[x]['RMSE']).index(model_name)+1}")

    else:
        # ── Live predict fallback ─────────────────────────────────────────────
        st.warning(
            "Pre-saved predictions not found. Running live prediction on the "
            "featured dataset. This may take a moment."
        )
        with st.spinner("Loading featured data..."):
            df = load_featured_data()

        subset = df[(df["store"] == store) & (df["item"] == item)].copy()
        if subset.empty:
            st.error("No data found for the selected store/item combination.")
            return

        _, val = train_val_split(subset)
        X_val, y_val = get_X_y(val)

        if model_name in ["XGBoost", "LightGBM"]:
            try:
                preds = predict_tree(model_name, X_val)
                fig   = plot_forecast_vs_actual(y_val.values, preds, model_name, n_points=n_show)
                st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError as e:
                st.error(str(e))

        elif model_name == "Univariate LSTM":
            try:
                series = subset["sales"].values.astype(float)
                if len(series) < LSTM_WINDOW + 50:
                    st.error("Not enough data for LSTM (need ≥ 78 points).")
                    return
                preds = predict_uni_lstm(series[:-50], n_steps=50)
                actual = series[-50:]
                fig = plot_forecast_vs_actual(actual, preds, model_name, n_points=50)
                st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError as e:
                st.error(str(e))
        else:
            st.info("Multivariate LSTM live inference requires the full scaler. "
                    "Please ensure models/ folder is present.")

    # ── Future forecast widget ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔭 Future Demand Forecast (Univariate LSTM)")
    st.markdown(
        "Uses the Univariate LSTM to roll forward N days beyond the training data."
    )

    n_future = st.slider("Forecast horizon (days)", 7, 90, 30)

    if st.button("▶ Run Future Forecast"):
        with st.spinner("Generating future forecast..."):
            try:
                df = load_featured_data()
                series = (
                    df[(df["store"] == store) & (df["item"] == item)]
                    ["sales"]
                    .values
                    .astype(float)
                )
                preds = predict_uni_lstm(series, n_steps=n_future)
                last_date = df["date"].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), periods=n_future
                )
                fig = go.Figure()
                # last 60 days of actuals for context
                fig.add_trace(go.Scatter(
                    x=pd.date_range(end=last_date, periods=60),
                    y=series[-60:],
                    name="Historical (last 60d)",
                    line=dict(color="#2ECC71", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=preds,
                    name="Future Forecast",
                    line=dict(color="#F39C12", width=2, dash="dash"),
                ))
                fig.update_layout(
                    title=f"🔭 {n_future}-Day Future Forecast — Store {store} · Item {item}",
                    xaxis_title="Date", yaxis_title="Sales",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                out = pd.DataFrame({"date": future_dates, "forecast": preds})
                st.download_button(
                    "⬇️ Download Forecast CSV",
                    data=out.to_csv(index=False),
                    file_name=f"forecast_store{store}_item{item}.csv",
                    mime="text/csv",
                )
            except (FileNotFoundError, Exception) as e:
                st.error(f"Could not run future forecast: {e}")


render()
