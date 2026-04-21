"""
pages/shap_explainer.py — Demand Anomaly Explainer using SHAP.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.config import NUM_STORES, NUM_ITEMS, FEATURE_COLS
from src.data_loader import load_featured_data, train_val_split, get_X_y
from src.model_loader import load_xgb_model


def render():
    st.title("🔍 Demand Explainer (SHAP)")
    st.markdown(
        "**Why did the model predict this demand value?**  \n"
        "SHAP breaks down every prediction into the contribution of each "
        "feature — so you know *why* demand is high or low, not just *what* it is."
    )

    # ── Check SHAP installed ──────────────────────────────────────────────────
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
        st.warning("SHAP not installed — showing native XGBoost feature importance instead.")

    # ── Controls ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚙️ SHAP Controls")
    store     = st.sidebar.selectbox("Store", list(range(1, NUM_STORES + 1)))
    item      = st.sidebar.selectbox("Item",  list(range(1, NUM_ITEMS  + 1)))
    n_explain = st.sidebar.slider("Rows to explain", 1, 50, 10)

    # ── Load model + data ─────────────────────────────────────────────────────
    with st.spinner("Loading model and data…"):
        model = load_xgb_model()
        if model is None:
            st.error("XGBoost model not found. Ensure `models/xgb_model.joblib` is in your repo.")
            return

        df = load_featured_data()
        _, val_df = train_val_split(df)
        subset_val = val_df[(val_df["store"] == store) & (val_df["item"] == item)]
        X_val, y_val = get_X_y(subset_val)

    if X_val.empty:
        st.warning("No validation data for this store/item combination.")
        return

    X_explain = X_val.head(n_explain)
    y_explain = y_val.head(n_explain)

    # ── Try SHAP, fall back to native importance ──────────────────────────────
    shap_ok = False
    if shap_available:
        try:
            with st.spinner("Computing SHAP values…"):
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_explain)
                shap_df     = pd.DataFrame(shap_values, columns=X_explain.columns)
                base_val    = float(explainer.expected_value)
            shap_ok = True
        except Exception as e:
            st.warning(
                f"⚠️ SHAP TreeExplainer failed (likely an XGBoost version mismatch): `{e}`\n\n"
                "Showing **native XGBoost feature importance** instead. "
                "To fully fix this, re-save your model with `xgboost==2.1.4` and redeploy."
            )

    # ── SHAP plots ────────────────────────────────────────────────────────────
    if shap_ok:
        _render_shap_plots(model, X_explain, y_explain, shap_df, base_val, store, item, n_explain)
    else:
        _render_native_importance(model, X_explain, y_explain)


def _render_shap_plots(model, X_explain, y_explain, shap_df, base_val, store, item, n_explain):
    import plotly.graph_objects as go
    import plotly.express as px

    # Global importance
    st.markdown("---")
    st.subheader("🌐 Global Feature Importance (Mean |SHAP|)")
    mean_shap = shap_df.abs().mean().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=mean_shap.values, y=mean_shap.index,
        orientation="h", marker_color="#3498DB",
    ))
    fig.update_layout(title="Mean |SHAP| per Feature",
                      xaxis_title="Mean |SHAP|", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Waterfall
    st.markdown("---")
    st.subheader("🔬 Single-Prediction Waterfall")
    row_idx  = st.slider("Select row", 0, len(X_explain) - 1, 0)
    row_X    = X_explain.iloc[row_idx]
    row_shap = shap_df.iloc[row_idx]
    pred_val = float(model.predict(row_X.values.reshape(1, -1))[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual Sales",       f"{float(y_explain.iloc[row_idx]):.1f}")
    c2.metric("Predicted Sales",    f"{pred_val:.1f}")
    c3.metric("Base (avg) Pred",    f"{base_val:.1f}")

    top_idx   = row_shap.abs().sort_values(ascending=False).index[:12]
    feat_shap = [float(row_shap[f]) for f in top_idx]
    feat_vals = [float(row_X[f])    for f in top_idx]
    colors    = ["#E74C3C" if s > 0 else "#3498DB" for s in feat_shap]
    labels    = [f"{f}<br><sub>val={v:.2f}</sub>" for f, v in zip(top_idx, feat_vals)]

    fig_wf = go.Figure(go.Bar(
        x=feat_shap, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s:+.3f}" for s in feat_shap], textposition="outside",
    ))
    fig_wf.add_vline(x=0, line_color="white", line_dash="dash")
    fig_wf.update_layout(
        title=f"SHAP Waterfall — Row {row_idx} (Store {store}, Item {item})",
        xaxis_title="SHAP Value", template="plotly_dark", height=500,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Heatmap
    st.markdown("---")
    st.subheader(f"🗺️ SHAP Heatmap — All {n_explain} Rows")
    top_feats = shap_df.abs().mean().nlargest(12).index.tolist()
    heat_df   = shap_df[top_feats].T
    heat_df.columns = [f"Row {i}" for i in range(len(heat_df.columns))]
    fig_heat = px.imshow(
        heat_df, color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0, template="plotly_dark", aspect="auto",
        title="SHAP Heatmap (top 12 features × explained rows)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 Full SHAP Values Table"):
        row_shap_sorted = row_shap.abs().sort_values(ascending=False)
        st.dataframe(pd.DataFrame({
            "Feature":       row_shap_sorted.index,
            "Feature Value": [float(row_X[f]) for f in row_shap_sorted.index],
            "SHAP Value":    [float(row_shap[f]) for f in row_shap_sorted.index],
        }), use_container_width=True)

    with st.expander("📖 How to Read SHAP Values"):
        st.markdown("""
        | | Meaning |
        |-|---------|
        | **Red bar (positive SHAP)** | Feature **increased** predicted demand |
        | **Blue bar (negative SHAP)** | Feature **decreased** predicted demand |
        | **Bar length** | How much impact this feature had |
        | **Base value** | Average prediction across all training data |
        | **Prediction** = base + sum of all SHAP values | Everything adds up |
        """)


def _render_native_importance(model, X_explain, y_explain):
    """Fallback: show XGBoost's built-in feature importance when SHAP fails."""
    import plotly.graph_objects as go

    st.markdown("---")
    st.subheader("📊 XGBoost Native Feature Importance")
    st.caption("SHAP is unavailable due to a version mismatch — showing built-in importance scores.")

    importance = pd.Series(
        model.feature_importances_,
        index=X_explain.columns,
    ).sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        marker_color="#3498DB",
    ))
    fig.update_layout(
        title="XGBoost Feature Importance (gain)",
        xaxis_title="Importance Score",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Simple prediction table
    st.markdown("---")
    st.subheader("🔮 Predictions vs Actual")
    preds = model.predict(X_explain).clip(0)
    result = X_explain.copy()
    result["actual"]    = y_explain.values
    result["predicted"] = preds.round(2)
    result["error"]     = (result["actual"] - result["predicted"]).round(2)
    st.dataframe(result[["actual", "predicted", "error"]].reset_index(drop=True),
                 use_container_width=True)

    st.info(
        "💡 **To restore full SHAP explanations:** re-save your XGBoost model "
        "locally using `xgboost==2.1.4`, push the new `models/xgb_model.joblib` "
        "to GitHub, and redeploy."
    )


render()
