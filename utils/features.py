"""NAV feature engineering and bootstrap trading-day baseline forecasts."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from utils.config import (
    FORECAST_BOOTSTRAP_PATHS,
    MIN_FORECAST_OBSERVATIONS,
)


def _period_return(series: pd.Series, trading_days: int) -> float | None:
    if len(series) <= trading_days:
        return None
    start = float(series.iloc[-(trading_days + 1)])
    end = float(series.iloc[-1])
    if start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def _window_mean(daily_ret: pd.Series, window: int) -> float | None:
    if len(daily_ret) < max(5, window // 3):
        return None
    slice_ = daily_ret.tail(min(window, len(daily_ret)))
    if slice_.empty:
        return None
    return float(slice_.mean())


def _shrunk_mean_daily(daily_ret: pd.Series) -> tuple[float, dict[str, float | None]]:
    """Blend 21/63/252 window means with fixed weights; renormalize if missing."""
    parts = {
        "mu_21": _window_mean(daily_ret, 21),
        "mu_63": _window_mean(daily_ret, 63),
        "mu_252": _window_mean(daily_ret, 252),
    }
    weights = {"mu_21": 0.5, "mu_63": 0.3, "mu_252": 0.2}
    num = 0.0
    den = 0.0
    for key, w in weights.items():
        val = parts[key]
        if val is None:
            continue
        num += w * val
        den += w
    if den <= 0:
        fallback = float(daily_ret.mean()) if len(daily_ret) else 0.0
        return fallback, parts
    return num / den, parts


def assert_forecast_ready(
    nav_or_features: pd.DataFrame | dict[str, Any],
    *,
    min_observations: int = MIN_FORECAST_OBSERVATIONS,
) -> int:
    """Raise if NAV history is too short for a reliable forecast."""
    if isinstance(nav_or_features, pd.DataFrame):
        observations = int(len(nav_or_features))
    else:
        observations = int(nav_or_features.get("observations") or 0)
    if observations < min_observations:
        raise ValueError(
            f"Need at least {min_observations} NAV observations to forecast "
            f"(have {observations}). Append more history (e.g. ULIP file NAV) "
            "before running scenarios."
        )
    return observations


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

    vol_21 = (
        float(daily_ret.tail(21).std() * np.sqrt(252) * 100)
        if len(daily_ret) >= 5
        else None
    )
    vol_63 = (
        float(daily_ret.tail(63).std() * np.sqrt(252) * 100)
        if len(daily_ret) >= 20
        else None
    )

    mean_daily = float(daily_ret.tail(63).mean()) if len(daily_ret) >= 5 else 0.0
    std_daily = float(daily_ret.tail(63).std()) if len(daily_ret) >= 5 else 0.0
    shrunk_mu, mu_parts = _shrunk_mean_daily(daily_ret)

    # Historical rolling horizon move envelope (for path clamps).
    envelope_up = None
    envelope_down = None
    for horizon in (21, 63, 60):
        if len(nav) <= horizon:
            continue
        rolls = nav / nav.shift(horizon) - 1.0
        rolls = rolls.dropna()
        if rolls.empty:
            continue
        envelope_up = float(rolls.max())
        envelope_down = float(rolls.min())
        break

    recent = df.tail(recent_points)[["Date", "NAV"]].copy()
    recent_series = [
        {
            "date": pd.Timestamp(row.Date).strftime("%Y-%m-%d"),
            "nav": round(float(row.NAV), 4),
        }
        for row in recent.itertuples(index=False)
    ]

    # Store daily returns for bootstrap (last 252 preferred by caller).
    daily_returns_list = [round(float(x), 8) for x in daily_ret.tolist()]

    return {
        "first_date": first_date.isoformat(),
        "last_date": last_date.isoformat(),
        "first_nav": round(first_nav, 4),
        "last_nav": round(last_nav, 4),
        "observations": int(len(df)),
        "trading_day_count": int(len(df)),
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
        "mean_daily_return_shrunk": float(np.clip(shrunk_mu, -0.01, 0.01)),
        "mu_21": _round_or_none(mu_parts["mu_21"], 8),
        "mu_63": _round_or_none(mu_parts["mu_63"], 8),
        "mu_252": _round_or_none(mu_parts["mu_252"], 8),
        "envelope_up_pct": _round_or_none(
            envelope_up * 100.0 if envelope_up is not None else None
        ),
        "envelope_down_pct": _round_or_none(
            envelope_down * 100.0 if envelope_down is not None else None
        ),
        "daily_returns": daily_returns_list,
        "recent_series": recent_series,
    }


def _horizon_dates(last_date: date, horizon_days: int) -> list[date]:
    """Next ``horizon_days`` business days after last_date."""
    start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    idx = pd.bdate_range(start=start, periods=horizon_days)
    return [ts.date() for ts in idx]


def _apply_terminal_envelope(
    path_navs: np.ndarray,
    last_nav: float,
    features: dict[str, Any],
) -> np.ndarray:
    """Clamp terminal NAV using historical rolling horizon moves."""
    up = features.get("envelope_up_pct")
    down = features.get("envelope_down_pct")
    if up is None and down is None:
        return path_navs
    lo = last_nav * (1.0 + (float(down) / 100.0)) if down is not None else 0.01
    hi = last_nav * (1.0 + (float(up) / 100.0)) if up is not None else last_nav * 10.0
    lo = max(lo, 0.01)
    hi = max(hi, lo)
    terminals = path_navs[:, -1]
    scale = np.ones(len(terminals))
    for i, term in enumerate(terminals):
        clamped = float(np.clip(term, lo, hi))
        if term > 0:
            scale[i] = clamped / term
    return path_navs * scale[:, None]


def statistical_baseline_forecast(
    features: dict[str, Any],
    horizon_days: int = 30,
    *,
    n_paths: int = FORECAST_BOOTSTRAP_PATHS,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Bootstrap bull/base/bear NAV paths on a trading-day calendar.

    bear/base/bull = day-by-day p10 / p50 / p90 across bootstrap paths.
    """
    assert_forecast_ready(features)
    horizon_days = max(1, int(horizon_days))
    last_nav = float(features["last_nav"])
    last_date = date.fromisoformat(features["last_date"])
    returns = np.asarray(features.get("daily_returns") or [], dtype=float)
    if returns.size < 5:
        raise ValueError("Insufficient daily returns for bootstrap forecast.")

    # Prefer last 252 trading-day returns.
    pool = returns[-252:] if returns.size > 252 else returns
    # Center around shrunk mean while preserving residual noise.
    pool_mean = float(pool.mean())
    target_mu = float(features.get("mean_daily_return_shrunk") or pool_mean)
    adjusted = pool - pool_mean + target_mu
    adjusted = np.clip(adjusted, -0.05, 0.05)

    generator = rng or np.random.default_rng(42)
    samples = generator.choice(adjusted, size=(n_paths, horizon_days), replace=True)
    # Cumulative compound paths
    growth = np.cumprod(1.0 + samples, axis=1)
    path_navs = last_nav * growth
    path_navs = np.maximum(path_navs, 0.01)
    path_navs = _apply_terminal_envelope(path_navs, last_nav, features)

    dates = _horizon_dates(last_date, horizon_days)
    percentiles = {"bear": 10, "base": 50, "bull": 90}
    rationales = {
        "bear": (
            f"Bootstrap p10 path over {horizon_days} trading days "
            f"({n_paths} paths; shrunk μ={target_mu:.5f})."
        ),
        "base": (
            f"Bootstrap p50 path over {horizon_days} trading days "
            f"({n_paths} paths; shrunk μ={target_mu:.5f})."
        ),
        "bull": (
            f"Bootstrap p90 path over {horizon_days} trading days "
            f"({n_paths} paths; shrunk μ={target_mu:.5f})."
        ),
    }

    scenarios: dict[str, dict[str, Any]] = {}
    for name, pct in percentiles.items():
        curve = np.percentile(path_navs, pct, axis=0)
        path = [
            {"date": d.isoformat(), "nav": round(float(nav), 4)}
            for d, nav in zip(dates, curve)
        ]
        scenarios[name] = {"nav_path": path, "rationale": rationales[name]}

    # Enforce bear <= base <= bull at each step after percentile noise.
    bear = np.array([p["nav"] for p in scenarios["bear"]["nav_path"]], dtype=float)
    base = np.array([p["nav"] for p in scenarios["base"]["nav_path"]], dtype=float)
    bull = np.array([p["nav"] for p in scenarios["bull"]["nav_path"]], dtype=float)
    base = np.maximum(base, bear)
    bull = np.maximum(bull, base)
    for i, d in enumerate(dates):
        scenarios["bear"]["nav_path"][i] = {
            "date": d.isoformat(),
            "nav": round(float(bear[i]), 4),
        }
        scenarios["base"]["nav_path"][i] = {
            "date": d.isoformat(),
            "nav": round(float(base[i]), 4),
        }
        scenarios["bull"]["nav_path"][i] = {
            "date": d.isoformat(),
            "nav": round(float(bull[i]), 4),
        }

    return {
        "horizon_days": horizon_days,
        "horizon_unit": "trading_days",
        "source": "bootstrap_baseline",
        "bootstrap_paths": n_paths,
        "scenarios": scenarios,
        "disclaimer": (
            "Illustrative bootstrap scenarios only — not investment advice. "
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
