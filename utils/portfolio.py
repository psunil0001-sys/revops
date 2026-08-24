"""Portfolio math and allocation helpers."""

from __future__ import annotations

import pandas as pd

from utils.config import (
    ASSET_DATA,
    INVESTMENT,
    SECTOR_DATA,
    UNITS_HELD,
)


def calculate_portfolio(current_nav: float) -> dict[str, float]:
    current_value = round(UNITS_HELD * current_nav, 4)
    profit_loss = round(current_value - INVESTMENT, 4)
    roi = (current_value / INVESTMENT - 1) * 100
    return {
        "units": UNITS_HELD,
        "current_nav": round(float(current_nav), 4),
        "current_value": current_value,
        "profit_loss": profit_loss,
        "roi": roi,
    }


def inr(value: float, decimals: int = 4) -> str:
    """Format currency with configurable decimals (default 4 for NAV accuracy)."""
    return f"₹{value:,.{decimals}f}"


def get_sector_dataframe() -> pd.DataFrame:
    return pd.DataFrame(SECTOR_DATA)


def get_asset_dataframe() -> pd.DataFrame:
    return pd.DataFrame(ASSET_DATA)


def build_portfolio_history(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Attach portfolio value series to a normalized NAV DataFrame."""
    history = nav_df.copy()
    history["Portfolio Value"] = (history["NAV"] * UNITS_HELD).round(4)
    return history
