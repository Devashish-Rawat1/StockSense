"""pages/inventory.py — Inventory planning (EOQ · ROP · Safety Stock)."""

import streamlit as st
import pandas as pd
import numpy as np

from src.config import INV_DEFAULTS
from src.data_loader import load_featured_data
from src.model_loader import load_saved_predictions
from src.inventory import compute_inventory_plan
from src.plots import plot_eoq_by_item, plot_rop_safety


def render():
    st.title("🏭 Inventory Planner")
    st.markdown(
        "Convert XGBoost demand forecasts into actionable inventory decisions: "
        "**EOQ**, **Reorder Point**, and **Safety Stock** per item."
    )

    st.sidebar.markdown("## ⚙️ Inventory Parameters")
    ordering_cost = st.sidebar.number_input(
        "Ordering Cost ($ per order)", value=INV_DEFAULTS["ordering_cost"], min_value=1.0
    )
    holding_cost = st.sidebar.number_input(
        "Holding Cost ($ / unit / year)", value=INV_DEFAULTS["holding_cost_per_unit"], min_value=0.1
    )
    lead_time = st.sidebar.number_input(
        "Lead Time (days)", value=INV_DEFAULTS["lead_time_days"], min_value=1, max_value=90
    )
    service_level = st.sidebar.selectbox(
        "Service Level", ["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"], index=1
    )
    z_map = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "99% (Z=2.33)": 2.33}
    z_score = z_map[service_level]

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**EOQ** = √(2DS/h)\n\n"
        "**Safety Stock** = Z × σ × √L\n\n"
        "**ROP** = d̄ × L + Safety Stock"
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading forecast data..."):
        saved = load_saved_predictions()
        df    = load_featured_data()

    val_df = df[df["date"] >= "2017-01-01"].copy()

    if "XGBoost" in saved:
        xgb_preds = saved["XGBoost"]
        # Align lengths
        n = min(len(val_df), len(xgb_preds))
        val_df = val_df.iloc[:n].copy()
        val_df["y_pred"] = np.array(xgb_preds[:n])
        val_df["y_pred"] = val_df["y_pred"].clip(lower=0)

        st.success(f"Using **XGBoost** predictions (best model · MAPE ≈ 12.4%) "
                   f"on {n:,} validation rows.")
    else:
        # Fallback: use actual sales as a proxy
        st.warning(
            "XGBoost predictions not found — using actual sales as demand proxy."
        )
        val_df["y_pred"] = val_df["sales"]

    # ── Run inventory plan ────────────────────────────────────────────────────
    with st.spinner("Computing inventory metrics..."):
        inv_df = compute_inventory_plan(
            forecast_df=val_df,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
            lead_time_days=int(lead_time),
            z=z_score,
        )

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items Analyzed", len(inv_df))
    c2.metric("Avg EOQ (units)",          f"{inv_df['EOQ'].mean():.0f}")
    c3.metric("Avg Reorder Point (units)", f"{inv_df['ROP'].mean():.0f}")
    c4.metric("Avg Safety Stock (units)",  f"{inv_df['safety_stock'].mean():.0f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    top_n = st.slider("Items to display in charts", 5, 50, 20)

    with col1:
        st.plotly_chart(plot_eoq_by_item(inv_df, top_n), use_container_width=True)
    with col2:
        st.plotly_chart(plot_rop_safety(inv_df, top_n),  use_container_width=True)

    # ── Full table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Inventory Plan")

    display_cols = [
        "item", "annual_forecast_demand", "avg_daily_demand",
        "std_daily_demand", "cv", "EOQ", "safety_stock", "ROP",
    ]
    rename = {
        "item":                     "Item",
        "annual_forecast_demand":   "Annual Demand",
        "avg_daily_demand":         "Avg Daily Demand",
        "std_daily_demand":         "Demand Std Dev",
        "cv":                       "CV",
        "EOQ":                      "EOQ (units)",
        "safety_stock":             "Safety Stock (units)",
        "ROP":                      "Reorder Point (units)",
    }
    st.dataframe(
        inv_df[display_cols].rename(columns=rename),
        use_container_width=True,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    csv = inv_df[display_cols].rename(columns=rename).to_csv(index=False)
    st.download_button(
        "⬇️ Download Inventory Plan CSV",
        data=csv,
        file_name="inventory_plan.csv",
        mime="text/csv",
    )

    # ── Formulas reference ────────────────────────────────────────────────────
    with st.expander("📖 Formula Reference"):
        st.markdown("""
        | Formula | Description |
        |---------|-------------|
        | **EOQ = √(2DS/h)** | D=Annual Demand, S=Ordering Cost, h=Holding Cost |
        | **Safety Stock = Z × σ × √L** | Z=Service Level Z-score, σ=Demand Std Dev, L=Lead Time |
        | **ROP = d̄ × L + Safety Stock** | d̄=Avg Daily Demand, L=Lead Time |

        **CV (Coefficient of Variation)** = σ / d̄  
        Items with CV > 0.5 have high demand variability — consider increasing safety stock.
        """)


render()
