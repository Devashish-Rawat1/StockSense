"""
plots.py
Reusable Plotly chart builders used across pages.
All functions return plotly.graph_objects.Figure objects.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import MODEL_METRICS


PALETTE = {
    "Actual":            "#2ECC71",
    "XGBoost":           "#3498DB",
    "LightGBM":          "#E74C3C",
    "Univariate LSTM":   "#9B59B6",
    "Multivariate LSTM": "#F39C12",
}


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("date")["sales"].mean().reset_index()
    fig = px.line(
        daily, x="date", y="sales",
        title="📈 Average Daily Sales Over Time",
        labels={"sales": "Avg Sales", "date": "Date"},
        template="plotly_dark",
    )
    fig.update_traces(line_color="#3498DB", line_width=1.5)
    return fig


def plot_sales_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="sales", nbins=50,
        title="📊 Sales Distribution",
        labels={"sales": "Sales"},
        template="plotly_dark",
        color_discrete_sequence=["#3498DB"],
    )
    mean_val = df["sales"].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean={mean_val:.1f}", annotation_position="top right")
    return fig


def plot_store_sales(df: pd.DataFrame) -> go.Figure:
    store_avg = df.groupby("store")["sales"].mean().reset_index()
    fig = px.bar(
        store_avg, x="store", y="sales",
        title="🏪 Average Sales per Store",
        labels={"store": "Store", "sales": "Avg Sales"},
        template="plotly_dark",
        color="sales",
        color_continuous_scale="Blues",
    )
    return fig


def plot_monthly_seasonality(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")["sales"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])
    fig = px.bar(
        monthly, x="month_name", y="sales",
        title="📅 Monthly Seasonality",
        labels={"month_name": "Month", "sales": "Avg Sales"},
        template="plotly_dark",
        color_discrete_sequence=["#2ECC71"],
    )
    return fig


def plot_top_items(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    item_sales = df.groupby("item")["sales"].sum().nlargest(top_n).reset_index()
    fig = px.bar(
        item_sales, x="item", y="sales",
        title=f"🏆 Top {top_n} Items by Total Sales",
        labels={"item": "Item ID", "sales": "Total Sales"},
        template="plotly_dark",
        color_discrete_sequence=["#E74C3C"],
    )
    return fig


# ── Forecast plots ────────────────────────────────────────────────────────────

def plot_forecast_vs_actual(
    actual:    np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    n_points:  int = 200,
) -> go.Figure:
    actual    = np.array(actual).flatten()[:n_points]
    predicted = np.array(predicted).flatten()[:n_points]
    x = list(range(len(actual)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=actual, name="Actual",
                             line=dict(color=PALETTE["Actual"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=predicted, name=model_name,
                             line=dict(color=PALETTE.get(model_name, "#FFF"), width=1.5,
                                       dash="dash")))
    fig.update_layout(
        title=f"🔮 {model_name} — Forecast vs Actual",
        xaxis_title="Time Step",
        yaxis_title="Sales",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_all_models_comparison(preds: dict, n_points: int = 200) -> go.Figure:
    fig = go.Figure()
    actual = preds.get("actual")
    if actual is not None:
        fig.add_trace(go.Scatter(
            x=list(range(n_points)), y=actual[:n_points],
            name="Actual", line=dict(color=PALETTE["Actual"], width=2.5)
        ))
    for name, arr in preds.items():
        if name == "actual":
            continue
        fig.add_trace(go.Scatter(
            x=list(range(n_points)), y=arr[:n_points],
            name=name, line=dict(color=PALETTE.get(name, "#AAA"), width=1.5, dash="dot")
        ))
    fig.update_layout(
        title="📊 All Models — Forecast Comparison (First 200 Steps)",
        xaxis_title="Time Step", yaxis_title="Sales",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_metrics_bar() -> go.Figure:
    models = list(MODEL_METRICS.keys())
    rmse   = [MODEL_METRICS[m]["RMSE"] for m in models]
    mape   = [MODEL_METRICS[m]["MAPE"] for m in models]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["RMSE ↓ (lower is better)",
                                        "MAPE % ↓ (lower is better)"])
    colors = [PALETTE.get(m, "#AAA") for m in models]

    fig.add_trace(go.Bar(x=models, y=rmse, name="RMSE",
                         marker_color=colors), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=mape, name="MAPE",
                         marker_color=colors, showlegend=False), row=1, col=2)

    fig.update_layout(template="plotly_dark", showlegend=False,
                      title_text="📐 Model Performance Metrics")
    return fig


def plot_residuals(actual: np.ndarray, predicted: np.ndarray,
                   model_name: str) -> go.Figure:
    residuals = np.array(actual).flatten() - np.array(predicted).flatten()
    fig = px.histogram(
        x=residuals, nbins=60, title=f"📉 Residuals — {model_name}",
        labels={"x": "Residual (Actual − Predicted)"},
        template="plotly_dark", color_discrete_sequence=["#F39C12"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    return fig


# ── Inventory plots ───────────────────────────────────────────────────────────

def plot_eoq_by_item(inv_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = inv_df.nlargest(top_n, "EOQ")
    fig  = px.bar(
        data, x="item", y="EOQ",
        title=f"📦 EOQ — Top {top_n} Items",
        labels={"item": "Item ID", "EOQ": "Economic Order Qty"},
        template="plotly_dark", color_discrete_sequence=["#3498DB"],
    )
    return fig


def plot_rop_safety(inv_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = inv_df.nlargest(top_n, "ROP")
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=data["item"].astype(str), y=data["ROP"],
                         name="Reorder Point", marker_color="#E74C3C"))
    fig.add_trace(go.Bar(x=data["item"].astype(str), y=data["safety_stock"],
                         name="Safety Stock",   marker_color="#F39C12"))
    fig.update_layout(
        barmode="group",
        title=f"🛡️ Reorder Point & Safety Stock — Top {top_n} Items",
        xaxis_title="Item ID", yaxis_title="Units",
        template="plotly_dark",
    )
    return fig
