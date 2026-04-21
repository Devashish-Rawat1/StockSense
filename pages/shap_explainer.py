"""
pages/shap_explainer.py — Demand Explainer using SHAP.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.config import NUM_STORES, NUM_ITEMS
from src.data_loader import load_featured_data, train_val_split
from src.model_loader import load_xgb_model


def _get_model_features(model) -> list[str]:
    """
    Safely retrieve the feature names the model was actually trained on.
    XGBoost stores these in model.feature_names_in_ or get_booster().feature_names.
    This is the ground truth — we use this, NOT our config's FEATURE_COLS.
    """
    # Try sklearn-style attribute first
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        return list(model.feature_names_in_)
    # Try booster-level feature names
    try:
        names = model.get_booster().feature_names
        if names:
            return names
    except Exception:
        pass
    return None


def _align_X(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    """
    Keep only the columns the model was trained on, in the right order.
    Drop NaN rows after selecting those columns.
    """
    available = [f for f in model_features if f in df.columns]
    missing   = [f for f in model_features if f not in df.columns]
    if missing:
        st.warning(f"⚠️ {len(missing)} feature(s) missing from data: `{missing[:5]}...`")
    X = df[available].dropna()
    return X


def render():
    st.title("🔍 Demand Explainer (SHAP)")
    st.markdown(
        "**Why did the model predict this demand value?** "
        "SHAP breaks down every prediction into the contribution of each feature."
    )

    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False

    st.sidebar.markdown("## ⚙️ Controls")
    store     = st.sidebar.selectbox("Store", list(range(1, NUM_STORES + 1)))
    item      = st.sidebar.selectbox("Item",  list(range(1, NUM_ITEMS  + 1)))
    n_explain = st.sidebar.slider("Rows to explain", 1, 50, 10)

    # ── Load model ────────────────────────────────────────────────────────────
    with st.spinner("Loading model…"):
        model = load_xgb_model()
    if model is None:
        st.error("XGBoost model not found. Ensure `models/xgb_model.joblib` exists.")
        return

    # ── Get exact features the model expects ──────────────────────────────────
    model_features = _get_model_features(model)
    if model_features:
        st.sidebar.success(f"Model uses **{len(model_features)} features**")
    else:
        st.sidebar.warning("Could not read model feature names — will use all available columns.")

    # ── Load + filter data ────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        df = load_featured_data()
    _, val_df = train_val_split(df)
    subset = val_df[(val_df["store"] == store) & (val_df["item"] == item)]

    if subset.empty:
        st.warning("No validation data for this store/item combination.")
        return

    # Align to model's exact feature set
    if model_features:
        X_explain = _align_X(subset, model_features)
    else:
        # Fallback: drop non-numeric / non-feature columns
        drop_cols = {"date", "sales"}
        num_cols  = [c for c in subset.columns if c not in drop_cols
                     and subset[c].dtype != object]
        X_explain = subset[num_cols].dropna()

    y_explain = subset.loc[X_explain.index, "sales"] if "sales" in subset.columns else None
    X_explain = X_explain.head(n_explain)
    if y_explain is not None:
        y_explain = y_explain.loc[X_explain.index]

    if X_explain.empty:
        st.error("No rows left after dropping NaNs. Try a different store/item.")
        return

    st.caption(f"Explaining **{len(X_explain)}** rows with **{X_explain.shape[1]}** features.")

    # ── Try SHAP ──────────────────────────────────────────────────────────────
    shap_ok  = False
    shap_df  = None
    base_val = None

    if shap_available:
        try:
            with st.spinner("Computing SHAP values…"):
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_explain)
                shap_df     = pd.DataFrame(shap_values, columns=X_explain.columns)
                base_val    = float(explainer.expected_value)
            shap_ok = True
        except Exception as e:
            st.warning(f"⚠️ SHAP failed: `{str(e)[:200]}`\n\nShowing native feature importance instead.")

    # ── Render ────────────────────────────────────────────────────────────────
    if shap_ok:
        _render_shap(model, X_explain, y_explain, shap_df, base_val, store, item, n_explain)
    else:
        _render_native(model, X_explain, y_explain)


def _render_shap(model, X_explain, y_explain, shap_df, base_val, store, item, n_explain):
    # Global importance
    st.markdown("---")
    st.subheader("🌐 Global Feature Importance (Mean |SHAP|)")
    mean_shap = shap_df.abs().mean().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=mean_shap.values, y=mean_shap.index,
        orientation="h", marker_color="#3498DB",
    ))
    fig.update_layout(title="Mean |SHAP| per Feature",
                      xaxis_title="Mean |SHAP|", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Waterfall
    st.markdown("---")
    st.subheader("🔬 Single-Prediction Waterfall")
    row_idx  = st.slider("Select row", 0, len(X_explain) - 1, 0)
    row_X    = X_explain.iloc[row_idx]
    row_shap = shap_df.iloc[row_idx]
    pred_val = float(model.predict(row_X.values.reshape(1, -1))[0])

    c1, c2, c3 = st.columns(3)
    if y_explain is not None:
        c1.metric("Actual Sales", f"{float(y_explain.iloc[row_idx]):.1f}")
    c2.metric("Predicted Sales", f"{pred_val:.1f}")
    c3.metric("Base (avg) Pred", f"{base_val:.1f}")

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
        title="SHAP Heatmap (top 12 features × rows)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📖 How to Read SHAP Values"):
        st.markdown("""
        | | Meaning |
        |-|---------|
        | 🔴 **Red (positive SHAP)** | Feature **increased** predicted demand |
        | 🔵 **Blue (negative SHAP)** | Feature **decreased** predicted demand |
        | **Bar length** | How much impact this feature had |
        | **Prediction = base + Σ SHAP values** | All contributions sum up |
        """)


def _render_native(model, X_explain, y_explain):
    """Native XGBoost importance — uses the booster's own feature list, crash-proof."""
    st.markdown("---")
    st.subheader("📊 XGBoost Native Feature Importance")

    # Use the model's own feature names to build the Series safely
    model_features = _get_model_features(model)
    importances    = model.feature_importances_  # always length = n_features_in_

    if model_features and len(model_features) == len(importances):
        index = model_features
    else:
        # Last resort: generic names
        index = [f"f{i}" for i in range(len(importances))]

    importance = pd.Series(importances, index=index).sort_values(ascending=True)
    top20      = importance.tail(20)  # top 20 most important

    fig = go.Figure(go.Bar(
        x=top20.values, y=top20.index,
        orientation="h", marker_color="#3498DB",
    ))
    fig.update_layout(
        title="Top 20 Features by XGBoost Importance (gain)",
        xaxis_title="Importance Score",
        template="plotly_dark", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Predictions vs Actual table
    if not X_explain.empty:
        st.markdown("---")
        st.subheader("🔮 Predictions vs Actual")
        try:
            preds = model.predict(X_explain).clip(0)
            result = pd.DataFrame({
                "predicted": preds.round(2),
            }, index=X_explain.index)
            if y_explain is not None:
                result.insert(0, "actual", y_explain.values)
                result["error"] = (result["actual"] - result["predicted"]).round(2)
            st.dataframe(result.reset_index(drop=True), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.info(
        "💡 **To get full SHAP waterfall charts**, ensure the features passed "
        "to the explainer exactly match what the model was trained on. "
        "Check that `src/config.py` → `FEATURE_COLS` matches your training notebook."
    )


render()
