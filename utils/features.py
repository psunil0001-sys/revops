"""NAV feature engineering and statistical baseline forecasts."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


def _period_return(series: pd.Series, trading_days: int) -> float | None:
    if len(series) <= trading_days:
        return None
    start = float(series.iloc[-(trading_days + 1)])
    end = float(series.iloc[-1])
    if start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def build_nav_features(
    nav_df: pd.DataFrame,
    recent_points: int = 120,
) -> dict[str, Any]:
    """
    Build a compact feature summary + recent series for LLM / charts.

    Expects columns: Date, NAV (sorted ascending).
    """
    if nav_df.empty:
        raise ValueError("Cannot build features from an empty NAV DataFrame.")

    df = nav_df.copy().sort_values("Date").reset_index(drop=True)
    nav = df["NAV"].astype(float)
    dates = pd.to_datetime(df["Date"])

    daily_ret = nav.pct_change().dropna()
    cummax = nav.cummax()
    drawdown = (nav / cummax - 1.0) * 100.0

    first_nav = float(nav.iloc[0])
    last_nav = float(nav.iloc[-1])
    first_date = dates.iloc[0].date()
    last_date = dates.iloc[-1].date()
    elapsed_days = max((last_date - first_date).days, 1)
    cagr = ((last_nav / first_nav) ** (365.25 / elapsed_days) - 1.0) * 100.0

    sma_50 = float(nav.tail(50).mean()) if len(nav) >= 50 else float(nav.mean())
    sma_200 = float(nav.tail(200).mean()) if len(nav) >= 200 else float(nav.mean())

    vol_21 = float(daily_ret.tail(21).std() * np.sqrt(252) * 100) if len(daily_ret) >= 5 else None
    vol_63 = float(daily_ret.tail(63).std() * np.sqrt(252) * 100) if len(daily_ret) >= 20 else None

    mean_daily = float(daily_ret.tail(63).mean()) if len(daily_ret) >= 5 else 0.0
    std_daily = float(daily_ret.tail(63).std()) if len(daily_ret) >= 5 else 0.0

    recent = df.tail(recent_points)[["Date", "NAV"]].copy()
    recent_series = [
        {
            "date": pd.Timestamp(row.Date).strftime("%Y-%m-%d"),
            "nav": round(float(row.NAV), 4),
        }
        for row in recent.itertuples(index=False)
    ]

    return {
        "first_date": first_date.isoformat(),
        "last_date": last_date.isoformat(),
        "first_nav": round(first_nav, 4),
        "last_nav": round(last_nav, 4),
        "observations": int(len(df)),
        "return_1m_pct": _round_or_none(_period_return(nav, 21)),
        "return_3m_pct": _round_or_none(_period_return(nav, 63)),
        "return_1y_pct": _round_or_none(_period_return(nav, 252)),
        "cagr_pct": round(cagr, 4),
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "vol_21d_ann_pct": _round_or_none(vol_21),
        "vol_63d_ann_pct": _round_or_none(vol_63),
        "sma_50": round(sma_50, 4),
        "sma_200": round(sma_200, 4),
        "price_vs_sma50_pct": round((last_nav / sma_50 - 1.0) * 100.0, 4),
        "price_vs_sma200_pct": round((last_nav / sma_200 - 1.0) * 100.0, 4),
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "recent_series": recent_series,
    }


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def statistical_baseline_forecast(
    features: dict[str, Any],
    horizon_days: int = 30,
) -> dict[str, Any]:
    """
    Project bull/base/bear NAV paths from recent mean return ± volatility.

    Paths use calendar days from last_date for charting simplicity.
    """
    horizon_days = max(1, int(horizon_days))
    last_nav = float(features["last_nav"])
    last_date = date.fromisoformat(features["last_date"])
    mu = float(features.get("mean_daily_return") or 0.0)
    sigma = float(features.get("std_daily_return") or 0.0)

    # Cap extreme drifts so short-horizon paths stay plausible.
    mu = float(np.clip(mu, -0.01, 0.01))
    sigma = float(max(sigma, 1e-6))

    scenarios: dict[str, dict[str, Any]] = {}
    drifts = {
        "bear": mu - sigma,
        "base": mu,
        "bull": mu + sigma,
    }
    rationales = {
        "bear": (
            f"Statistical bear case: mean daily return minus 1σ "
            f"(μ={mu:.5f}, σ={sigma:.5f})."
        ),
        "base": (
            f"Statistical base case: recent mean daily return "
            f"(μ={mu:.5f}) compounded over {horizon_days} days."
        ),
        "bull": (
            f"Statistical bull case: mean daily return plus 1σ "
            f"(μ={mu:.5f}, σ={sigma:.5f})."
        ),
    }

    for name, drift in drifts.items():
        path = []
        nav = last_nav
        for step in range(1, horizon_days + 1):
            nav = max(nav * (1.0 + drift), 0.01)
            path.append(
                {
                    "date": (last_date + timedelta(days=step)).isoformat(),
                    "nav": round(nav, 4),
                }
            )
        scenarios[name] = {
            "nav_path": path,
            "rationale": rationales[name],
        }

    return {
        "horizon_days": horizon_days,
        "source": "statistical_baseline",
        "scenarios": scenarios,
        "disclaimer": (
            "Statistical projection only — not investment advice. "
            "Past returns do not guarantee future performance."
        ),
    }


def forecast_paths_to_dataframe(forecast: dict[str, Any]) -> pd.DataFrame:
    """Flatten scenario nav_path lists into a long DataFrame for Altair."""
    rows: list[dict[str, Any]] = []
    scenarios = forecast.get("scenarios") or {}
    for name, payload in scenarios.items():
        for point in payload.get("nav_path") or []:
            rows.append(
                {
                    "Date": pd.to_datetime(point["date"]),
                    "NAV": round(float(point["nav"]), 4),
                    "Scenario": name.capitalize(),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Date", "NAV", "Scenario"])
    return pd.DataFrame(rows).sort_values(["Scenario", "Date"]).reset_index(drop=True)
