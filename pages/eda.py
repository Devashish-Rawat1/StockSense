"""pages/eda.py — Exploratory Data Analysis page."""

import streamlit as st
import pandas as pd

from src.data_loader import load_raw_data
from src.plots import (
    plot_sales_trend,
    plot_sales_distribution,
    plot_store_sales,
    plot_monthly_seasonality,
    plot_top_items,
)


def render():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Interactive exploration of the raw sales dataset (2013–2017).")

    with st.spinner("Loading data..."):
        df = load_raw_data()

    # ── Dataset overview ──────────────────────────────────────────────────────
    st.subheader("🔍 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",          f"{len(df):,}")
    c2.metric("Stores",        df["store"].nunique())
    c3.metric("Items",         df["item"].nunique())
    c4.metric("Date Range",    f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    c5.metric("Missing Values", df.isnull().sum().sum())

    with st.expander("📄 Raw Data Sample"):
        st.dataframe(df.sample(min(200, len(df))), use_container_width=True)

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    st.subheader("🎛️ Filters")
    col1, col2 = st.columns(2)
    with col1:
        stores = st.multiselect(
            "Select Stores", options=sorted(df["store"].unique()),
            default=sorted(df["store"].unique()),
        )
    with col2:
        items = st.multiselect(
            "Select Items (top 10 shown by default)",
            options=sorted(df["item"].unique()),
            default=sorted(df["item"].unique())[:10],
        )

    filtered = df[df["store"].isin(stores) & df["item"].isin(items)]
    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    st.caption(f"Showing **{len(filtered):,}** rows after filters.")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Sales Trend")
    st.markdown(
        "**Key insight:** Clear upward trend + strong yearly seasonality across 2013–2017."
    )
    st.plotly_chart(plot_sales_trend(filtered), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📊 Sales Distribution")
        st.markdown("Right-skewed — most days see moderate demand (0–100 units).")
        st.plotly_chart(plot_sales_distribution(filtered), use_container_width=True)

    with col_b:
        st.subheader("🏪 Store Performance")
        st.plotly_chart(plot_store_sales(filtered), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("📅 Monthly Seasonality")
        st.plotly_chart(plot_monthly_seasonality(filtered), use_container_width=True)

    with col_d:
        st.subheader("🏆 Top Items by Sales")
        top_n = st.slider("Number of items", 5, 20, 10, key="top_items_slider")
        st.plotly_chart(plot_top_items(filtered, top_n), use_container_width=True)

    # ── EDA conclusions ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 EDA Conclusions")
    st.markdown("""
    1. Clean time-series dataset — no missing or duplicate values.
    2. Sales show a clear **upward trend** (growing demand year over year).
    3. **Strong yearly seasonality** — mid-year peaks, lower winter demand.
    4. Most daily sales fall in the **0–100 unit** range; a long right tail indicates sporadic spikes.
    5. **All stores contain outliers** — genuine demand spikes that should be retained.
    6. Store-specific and item-specific patterns suggest disaggregated modeling improves accuracy.
    """)


render()
