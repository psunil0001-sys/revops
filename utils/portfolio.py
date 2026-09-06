"""Portfolio math and allocation helpers."""

from __future__ import annotations

import pandas as pd

from utils.funds import FundConfig


def calculate_portfolio(
    current_nav: float,
    *,
    investment: float,
    units: float,
) -> dict[str, float]:
    current_value = round(units * current_nav, 4)
    profit_loss = round(current_value - investment, 4)
    roi = (current_value / investment - 1) * 100
    return {
        "units": units,
        "current_nav": round(float(current_nav), 4),
        "current_value": current_value,
        "profit_loss": profit_loss,
        "roi": roi,
    }


def inr(value: float, decimals: int = 4) -> str:
    """Format currency with configurable decimals (default 4 for NAV accuracy)."""
    return f"₹{value:,.{decimals}f}"


def get_sector_dataframe(fund: FundConfig) -> pd.DataFrame:
    return pd.DataFrame(fund.sector_dataframe_dict())


def get_asset_dataframe(fund: FundConfig) -> pd.DataFrame:
    return pd.DataFrame(fund.asset_dataframe_dict())


def build_portfolio_history(nav_df: pd.DataFrame, *, units: float) -> pd.DataFrame:
    """Attach portfolio value series to a normalized NAV DataFrame."""
    history = nav_df.copy()
    history["Portfolio Value"] = (history["NAV"] * units).round(4)
    return history
