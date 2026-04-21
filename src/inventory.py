"""
inventory.py
EOQ · Reorder Point · Safety Stock calculations.
All formulas taken directly from the inventory_management notebook.
"""

import numpy as np
import pandas as pd

from src.config import INV_DEFAULTS


def compute_eoq(
    annual_demand: float,
    ordering_cost: float = INV_DEFAULTS["ordering_cost"],
    holding_cost:  float = INV_DEFAULTS["holding_cost_per_unit"],
) -> float:
    """
    Economic Order Quantity (Wilson formula).
    EOQ = sqrt(2 * D * S / h)
    """
    if holding_cost <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / holding_cost))


def compute_safety_stock(
    std_daily_demand: float,
    lead_time_days:   int   = INV_DEFAULTS["lead_time_days"],
    z:                float = INV_DEFAULTS["service_level_z"],
) -> float:
    """
    Safety Stock = Z * σ_d * sqrt(L)
    Z = service-level z-score (1.65 → 95%)
    """
    return float(z * std_daily_demand * np.sqrt(lead_time_days))


def compute_rop(
    avg_daily_demand: float,
    lead_time_days:   int   = INV_DEFAULTS["lead_time_days"],
    safety_stock:     float = 0.0,
) -> float:
    """
    Reorder Point = (avg_daily_demand × lead_time) + safety_stock
    """
    return float(avg_daily_demand * lead_time_days + safety_stock)


def compute_inventory_plan(
    forecast_df:      pd.DataFrame,
    ordering_cost:    float = INV_DEFAULTS["ordering_cost"],
    holding_cost:     float = INV_DEFAULTS["holding_cost_per_unit"],
    lead_time_days:   int   = INV_DEFAULTS["lead_time_days"],
    z:                float = INV_DEFAULTS["service_level_z"],
) -> pd.DataFrame:
    """
    Given a DataFrame with columns [item, y_pred], compute per-item
    inventory metrics and return a summary DataFrame.

    forecast_df columns:
        date    : datetime
        item    : int (1-50)
        store   : int (1-10)
        y_pred  : float (predicted daily demand per store)
    """
    # Aggregate predicted demand across all stores → item-level daily demand
    daily_item = (
        forecast_df
        .groupby(["date", "item"])["y_pred"]
        .sum()
        .reset_index()
    )

    item_stats = (
        daily_item
        .groupby("item")
        .agg(
            annual_forecast_demand=("y_pred", "sum"),
            avg_daily_demand=("y_pred", "mean"),
            std_daily_demand=("y_pred", "std"),
            days_of_data=("y_pred", "count"),
        )
        .reset_index()
    )

    item_stats["cv"] = (
        item_stats["std_daily_demand"] / item_stats["avg_daily_demand"]
    )

    # Apply formulas
    item_stats["EOQ"] = item_stats["annual_forecast_demand"].apply(
        lambda d: compute_eoq(d, ordering_cost, holding_cost)
    )
    item_stats["safety_stock"] = item_stats["std_daily_demand"].apply(
        lambda s: compute_safety_stock(s, lead_time_days, z)
    )
    item_stats["ROP"] = item_stats.apply(
        lambda r: compute_rop(r["avg_daily_demand"], lead_time_days, r["safety_stock"]),
        axis=1,
    )

    # Round for display
    for col in ["EOQ", "safety_stock", "ROP", "avg_daily_demand", "std_daily_demand", "cv"]:
        item_stats[col] = item_stats[col].round(2)

    return item_stats.sort_values("item").reset_index(drop=True)
