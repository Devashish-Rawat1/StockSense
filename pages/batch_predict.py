"""pages/batch_predict.py — Upload a CSV, get bulk demand forecasts."""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import validate_uploaded_csv
from src.feature_engineering import engineer_features, get_model_ready
from src.model_loader import predict_tree
from src.config import FEATURE_COLS


SAMPLE_CSV = """date,store,item
2017-01-01,1,1
2017-01-02,1,1
2017-01-03,1,2
2017-01-04,2,5
2017-01-05,3,10
"""


def render():
    st.title("📤 Batch Predict")
    st.markdown(
        "Upload a CSV of **(date, store, item)** rows and download demand "
        "forecasts for all of them in one go."
    )

    # ── Template download ─────────────────────────────────────────────────────
    st.subheader("1️⃣  Download the template")
    st.download_button(
        "⬇️ Download sample_input.csv",
        data=SAMPLE_CSV,
        file_name="sample_input.csv",
        mime="text/csv",
    )
    st.caption(
        "Your CSV must contain **date** (YYYY-MM-DD), **store** (1–10), "
        "and **item** (1–50) columns. A `sales` column is optional "
        "(used to compute accuracy metrics if provided)."
    )

    st.markdown("---")

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("2️⃣  Upload your CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Waiting for file upload…")
        return

    try:
        user_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    st.success(f"Loaded **{len(user_df):,}** rows.")
    st.dataframe(user_df.head(10), use_container_width=True)

    # Validate
    is_valid, msg = validate_uploaded_csv(user_df)
    if not is_valid:
        st.error(f"Validation failed: {msg}")
        return

    st.markdown("---")

    # ── Model choice ──────────────────────────────────────────────────────────
    st.subheader("3️⃣  Choose a model")
    model_name = st.selectbox(
        "Model for prediction",
        ["XGBoost", "LightGBM"],
        help="Tree models support batch inference on new data. "
             "LSTM models require a historical time-series context "
             "and cannot be applied to arbitrary date ranges here.",
    )

    if st.button("▶ Run Batch Forecast"):
        with st.spinner("Engineering features & predicting…"):
            try:
                result_df = _run_batch(user_df.copy(), model_name)
            except FileNotFoundError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.success(f"Done! Generated **{len(result_df):,}** predictions.")

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader("📋 Results Preview")
        st.dataframe(result_df.head(50), use_container_width=True)

        # ── Accuracy (if actuals provided) ────────────────────────────────────
        if "sales" in result_df.columns and result_df["sales"].notna().any():
            st.subheader("📐 Accuracy Metrics")
            y_true = result_df["sales"].dropna().values
            y_pred = result_df.loc[result_df["sales"].notna(), "forecast"].values
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mask = y_true != 0
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
            c1, c2 = st.columns(2)
            c1.metric("RMSE", f"{rmse:.4f}")
            c2.metric("MAPE", f"{mape:.2f}%")

        # ── Chart ─────────────────────────────────────────────────────────────
        st.subheader("📈 Forecast Chart")
        plot_df = result_df.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])

        if len(plot_df["store"].unique()) == 1 and len(plot_df["item"].unique()) == 1:
            fig = px.line(
                plot_df, x="date", y="forecast",
                title="Demand Forecast",
                template="plotly_dark",
                color_discrete_sequence=["#3498DB"],
            )
            if "sales" in plot_df.columns:
                fig.add_scatter(
                    x=plot_df["date"], y=plot_df["sales"],
                    mode="lines", name="Actual",
                    line=dict(color="#2ECC71", width=2),
                )
        else:
            agg = plot_df.groupby("date")["forecast"].sum().reset_index()
            fig = px.line(
                agg, x="date", y="forecast",
                title="Total Forecast (all stores & items)",
                template="plotly_dark",
                color_discrete_sequence=["#3498DB"],
            )
        st.plotly_chart(fig, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.subheader("4️⃣  Download Results")
        csv_out = result_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download forecast_results.csv",
            data=csv_out,
            file_name="forecast_results.csv",
            mime="text/csv",
        )


def _run_batch(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Feature-engineer the uploaded dataframe and run tree model inference.
    Falls back to NaN for rows where lag/rolling features cannot be computed
    (i.e., not enough history).
    """
    df["date"] = pd.to_datetime(df["date"])
    has_sales   = "sales" in df.columns

    # We need historical context to compute lag/rolling features.
    # Strategy: load the raw training data, append user rows, engineer, slice back.
    from src.data_loader import load_raw_data
    raw = load_raw_data()[["date", "store", "item", "sales"]].copy()

    user_tag = "__user__"
    user_df  = df[["date", "store", "item"]].copy()
    if has_sales:
        user_df["sales"] = df["sales"]
    else:
        user_df["sales"] = np.nan

    combined = pd.concat([raw, user_df], ignore_index=True)
    combined = engineer_features(combined)

    # Slice back only the user rows
    user_mask  = combined["date"].isin(user_df["date"]) & \
                 combined["store"].isin(user_df["store"]) & \
                 combined["item"].isin(user_df["item"])
    user_engineered = combined[user_mask].copy()

    # Drop rows where features are still NaN (too little history)
    available_features = [c for c in FEATURE_COLS if c in user_engineered.columns]
    user_clean = user_engineered.dropna(subset=available_features)

    if user_clean.empty:
        raise ValueError(
            "All rows have NaN features after engineering. "
            "This usually means the uploaded dates are too far before "
            "the training data — not enough lag/rolling history."
        )

    X = user_clean[available_features]
    preds = predict_tree(model_name, X)
    user_clean = user_clean.copy()
    user_clean["forecast"] = np.clip(preds, 0, None).round(2)

    out_cols = ["date", "store", "item", "forecast"]
    if has_sales:
        out_cols.insert(3, "sales")

    return user_clean[out_cols].reset_index(drop=True)


render()
