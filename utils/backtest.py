"""Walk-forward backtest for bootstrap NAV scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.config import MIN_FORECAST_OBSERVATIONS
from utils.features import (
    assert_forecast_ready,
    build_nav_features,
    statistical_baseline_forecast,
)


def _terminal_nav(scenario: dict[str, Any]) -> float | None:
    path = (scenario or {}).get("nav_path") or []
    if not path:
        return None
    return float(path[-1]["nav"])


def walk_forward_forecast_backtest(
    nav_df: pd.DataFrame,
    *,
    horizon_days: int = 60,
    step: int = 21,
    min_observations: int = MIN_FORECAST_OBSERVATIONS,
) -> dict[str, Any]:
    """
    Walk-forward bootstrap-only backtest.

    At each origin ``t`` with enough history, forecast ``horizon_days`` trading
    steps and compare to the actual NAV ``horizon_days`` observations ahead.
    """
    if nav_df is None or nav_df.empty:
        return {"ready": False, "reason": "empty_nav", "n_origins": 0}

    df = nav_df.copy().sort_values("Date").reset_index(drop=True)
    n = len(df)
    # Need min history at origin + horizon ahead for actuals.
    if n < min_observations + horizon_days:
        return {
            "ready": False,
            "reason": (
                f"need>={min_observations + horizon_days} rows "
                f"(have {n})"
            ),
            "n_origins": 0,
            "horizon_days": horizon_days,
            "step": step,
        }

    coverage_hits = 0
    abs_pct_errors: list[float] = []
    origins = 0
    # Origin index is inclusive end of history slice (iloc 0..origin inclusive).
    first_origin = min_observations - 1
    last_origin = n - horizon_days - 1
    for origin in range(first_origin, last_origin + 1, step):
        hist = df.iloc[: origin + 1]
        try:
            assert_forecast_ready(hist, min_observations=min_observations)
            features = build_nav_features(hist)
            forecast = statistical_baseline_forecast(
                features, horizon_days=horizon_days
            )
        except Exception:
            continue

        scenarios = forecast.get("scenarios") or {}
        bear = _terminal_nav(scenarios.get("bear") or {})
        base = _terminal_nav(scenarios.get("base") or {})
        bull = _terminal_nav(scenarios.get("bull") or {})
        actual = float(df.iloc[origin + horizon_days]["NAV"])
        if bear is None or base is None or bull is None:
            continue

        lo, hi = min(bear, bull), max(bear, bull)
        if lo <= actual <= hi:
            coverage_hits += 1
        if base > 0:
            abs_pct_errors.append(abs(actual / base - 1.0) * 100.0)
        origins += 1

    if origins == 0:
        return {
            "ready": False,
            "reason": "no_valid_origins",
            "n_origins": 0,
            "horizon_days": horizon_days,
            "step": step,
        }

    return {
        "ready": True,
        "horizon_days": horizon_days,
        "horizon_unit": "trading_days",
        "step": step,
        "n_origins": origins,
        "coverage_pct": round(100.0 * coverage_hits / origins, 2),
        "median_abs_pct_error_base": round(float(np.median(abs_pct_errors)), 4),
        "mean_abs_pct_error_base": round(float(np.mean(abs_pct_errors)), 4),
        "method": "bootstrap_baseline",
        "label": (
            f"illustrative / backtested coverage "
            f"{round(100.0 * coverage_hits / origins, 1)}%"
        ),
    }
