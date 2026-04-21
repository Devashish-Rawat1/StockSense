"""
pages/shap_explainer.py — Demand Anomaly Explainer using SHAP.
This is the bonus "killer feature" page.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import NUM_STORES, NUM_ITEMS, FEATURE_COLS
from src.data_loader import load_featured_data, train_val_split, get_X_y
from src.model_loader import load_xgb_model


def render():
    st.title("🔍 Demand Explainer (SHAP)")
    st.markdown(
        "**Why did the model predict this demand value?**  \n"
        "SHAP (SHapley Additive exPlanations) breaks down every prediction "
        "into the contribution of each feature — so you can understand *why* "
        "demand is high or low, not just *what* it is."
    )

    # ── Check SHAP is installed ───────────────────────────────────────────────
    try:
        import shap
    except ImportError:
        st.error(
            "SHAP is not installed. Add `shap` to your requirements.txt and redeploy."
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ SHAP Controls")
    store     = st.sidebar.selectbox("Store", list(range(1, NUM_STORES + 1)))
    item      = st.sidebar.selectbox("Item",  list(range(1, NUM_ITEMS  + 1)))
    n_explain = st.sidebar.slider("Rows to explain", 1, 50, 10)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "SHAP values show how much each feature **pushed the prediction "
        "up (red) or down (blue)** from the base (average) prediction."
    )

    # ── Load data + model ─────────────────────────────────────────────────────
    with st.spinner("Loading model and data…"):
        model = load_xgb_model()
        if model is None:
            st.error("XGBoost model not found. Ensure models/xgb_model.joblib exists.")
            return

        df       = load_featured_data()
        train_df, val_df = train_val_split(df)
        X_train, _ = get_X_y(train_df)

        subset_val = val_df[(val_df["store"] == store) & (val_df["item"] == item)]
        X_val, y_val = get_X_y(subset_val)

    if X_val.empty:
        st.warning("No validation data found for this store/item combination.")
        return

    X_explain = X_val.head(n_explain)
    y_explain = y_val.head(n_explain)

    # ── Compute SHAP values ───────────────────────────────────────────────────
    with st.spinner("Computing SHAP values (this may take ~10 seconds)…"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        shap_df     = pd.DataFrame(shap_values, columns=X_explain.columns)

    # ── Global feature importance (mean |SHAP|) ───────────────────────────────
    st.markdown("---")
    st.subheader("🌐 Global Feature Importance (Mean |SHAP|)")
    st.markdown(
        "Average absolute SHAP value across all explained rows — "
        "the higher the bar, the more influential the feature."
    )

    import plotly.graph_objects as go
    mean_shap = shap_df.abs().mean().sort_values(ascending=True)
    fig_global = go.Figure(go.Bar(
        x=mean_shap.values,
        y=mean_shap.index,
        orientation="h",
        marker_color="#3498DB",
    ))
    fig_global.update_layout(
        title="Mean |SHAP| Value per Feature",
        xaxis_title="Mean |SHAP|",
        template="plotly_dark",
        height=450,
    )
    st.plotly_chart(fig_global, use_container_width=True)

    # ── Per-row waterfall explanation ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Single-Prediction Waterfall")
    st.markdown(
        "Pick any one prediction to see exactly how each feature contributed."
    )

    row_idx    = st.slider("Select row to explain", 0, len(X_explain) - 1, 0)
    row_X      = X_explain.iloc[row_idx]
    row_shap   = shap_df.iloc[row_idx]
    actual_val = float(y_explain.iloc[row_idx])
    pred_val   = float(model.predict(row_X.values.reshape(1, -1))[0])
    base_val   = float(explainer.expected_value)

    col1, col2, col3 = st.columns(3)
    col1.metric("Actual Sales",   f"{actual_val:.1f}")
    col2.metric("Predicted Sales", f"{pred_val:.1f}")
    col3.metric("Base (avg) Prediction", f"{base_val:.1f}")

    # Build waterfall chart
    sorted_idx  = row_shap.abs().sort_values(ascending=False).index[:12]
    feat_names  = list(sorted_idx)
    feat_shap   = [float(row_shap[f]) for f in feat_names]
    feat_vals   = [float(row_X[f])    for f in feat_names]
    colors      = ["#E74C3C" if s > 0 else "#3498DB" for s in feat_shap]
    labels      = [f"{f}<br><sub>val={v:.2f}</sub>" for f, v in zip(feat_names, feat_vals)]

    fig_wf = go.Figure(go.Bar(
        x=feat_shap,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:+.3f}" for s in feat_shap],
        textposition="outside",
    ))
    fig_wf.add_vline(x=0, line_color="white", line_dash="dash")
    fig_wf.update_layout(
        title=f"SHAP Waterfall — Row {row_idx} "
              f"(Store {store}, Item {item})",
        xaxis_title="SHAP Value (impact on prediction)",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── SHAP table ────────────────────────────────────────────────────────────
    with st.expander("📋 Full SHAP Values Table"):
        display = pd.DataFrame({
            "Feature":    feat_names,
            "Feature Value": [float(row_X[f]) for f in feat_names],
            "SHAP Value":    [float(row_shap[f]) for f in feat_names],
        }).sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(display, use_container_width=True)

    # ── Summary heatmap across all explained rows ─────────────────────────────
    st.markdown("---")
    st.subheader(f"🗺️ SHAP Heatmap — All {n_explain} Explained Rows")
    st.markdown("Each cell = SHAP contribution of that feature for that row.")

    import plotly.express as px
    top_feats = shap_df.abs().mean().nlargest(12).index.tolist()
    heat_df   = shap_df[top_feats].T
    heat_df.columns = [f"Row {i}" for i in range(len(heat_df.columns))]

    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="SHAP Heatmap (top 12 features × explained rows)",
        template="plotly_dark",
        aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Interpretation guide ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How to Interpret SHAP Values"):
        st.markdown("""
        | Concept | Meaning |
        |---------|---------|
        | **Base value** | Average model prediction across all training data |
        | **Positive SHAP (red)** | This feature **increased** the predicted demand |
        | **Negative SHAP (blue)** | This feature **decreased** the predicted demand |
        | **Large \|SHAP\|** | Feature had a large impact on this specific prediction |
        | **Prediction = base + sum of all SHAP values** | All contributions add up |

        **Example interpretation:**  
        If `sales_lag_7` has SHAP = +5.2, it means "because sales 7 days ago were high,  
        today's predicted demand is 5.2 units *higher* than the average."
        """)


render()
