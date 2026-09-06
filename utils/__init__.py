"""Shared utilities for the multi-fund portfolio dashboard."""

from utils.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
from utils.backtest import walk_forward_forecast_backtest
from utils.features import (
    assert_forecast_ready,
    build_nav_features,
    forecast_paths_to_dataframe,
    statistical_baseline_forecast,
)
from utils.funds import FundConfig, get_fund, load_funds
from utils.llm import forecast_nav
from utils.market import (
    attach_benchmark_features,
    fetch_benchmark_context,
    fetch_nifty_context,
)
from utils.nav import (
    fetch_nav_history,
    fetch_nav_history_for_fund,
    filter_nav_by_dates,
    get_date_range,
    normalize_nav_dataframe,
    parse_nav_response,
)
from utils.portfolio import (
    build_portfolio_history,
    calculate_portfolio,
    get_asset_dataframe,
    get_sector_dataframe,
    inr,
)

__all__ = [
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "FundConfig",
    "assert_forecast_ready",
    "attach_benchmark_features",
    "build_nav_features",
    "build_portfolio_history",
    "calculate_portfolio",
    "fetch_benchmark_context",
    "fetch_nav_history",
    "fetch_nav_history_for_fund",
    "fetch_nifty_context",
    "filter_nav_by_dates",
    "forecast_nav",
    "forecast_paths_to_dataframe",
    "get_asset_dataframe",
    "get_date_range",
    "get_fund",
    "get_sector_dataframe",
    "inr",
    "load_funds",
    "normalize_nav_dataframe",
    "parse_nav_response",
    "statistical_baseline_forecast",
    "walk_forward_forecast_backtest",
]
