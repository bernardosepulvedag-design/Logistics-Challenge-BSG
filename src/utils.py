"""
Utility functions shared across modules.
"""

import numpy as np
import pandas as pd


def coverage_ratio(stock: float, demand: float, lead_time: int) -> float:
    """
    Days-of-supply ratio: how many replenishment cycles the current stock covers.
    Lower values indicate higher stockout risk.
    """
    denominator = demand * lead_time * 0.15
    if denominator == 0:
        return float("inf")
    return round(stock / denominator, 4)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (excludes zeros in y_true)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def summarize_alerts(alerts: list) -> pd.DataFrame:
    """Converts the agent's alert list into a readable summary DataFrame."""
    if not alerts:
        return pd.DataFrame(columns=["point_id", "severity", "message", "timestamp"])
    return pd.DataFrame(alerts)[["point_id", "severity", "message", "timestamp"]]


def print_section(title: str, width: int = 60):
    """Prints a formatted section header to stdout."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
