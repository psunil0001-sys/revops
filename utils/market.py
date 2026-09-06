"""Optional market / benchmark context for forecast prompts and features."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


def fetch_nifty_context(
    start_date: date,
    end_date: date,
) -> dict[str, Any] | None:
    """Fetch Nifty 50 (^NSEI) summary via yfinance for the same window."""
    return fetch_benchmark_context("^NSEI", start_date, end_date)


def fetch_benchmark_context(
    symbol: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any] | None:
    """
    Fetch a yfinance symbol summary for the window.

    Returns None soft-fail if yfinance is missing or the request fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
    except Exception:
        return None

    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna()
    if close.empty:
        return None

    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    ret_pct = (last / first - 1.0) * 100.0 if first else None
    recent = close.tail(21)
    recent_ret = (
        (float(recent.iloc[-1]) / float(recent.iloc[0]) - 1.0) * 100.0
        if len(recent) >= 2
        else None
    )

    closes = [
        {
            "date": pd.Timestamp(idx).date().isoformat(),
            "close": round(float(val), 4),
        }
        for idx, val in close.items()
    ]

    return {
        "symbol": symbol,
        "name": "Nifty 50" if symbol == "^NSEI" else symbol,
        "start_date": pd.Timestamp(close.index[0]).date().isoformat(),
        "end_date": pd.Timestamp(close.index[-1]).date().isoformat(),
        "start_close": round(first, 2),
        "end_close": round(last, 2),
        "return_window_pct": round(ret_pct, 4) if ret_pct is not None else None,
        "return_21d_pct": round(recent_ret, 4) if recent_ret is not None else None,
        "observations": int(len(close)),
        "closes": closes,
    }


def attach_benchmark_features(
    nav_df: pd.DataFrame,
    benchmark_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    OLS beta / residual vol / correlation of fund vs benchmark daily returns.

    Returns a dict to merge into NAV features (empty if insufficient overlap).
    """
    if not benchmark_context:
        return {}
    closes = benchmark_context.get("closes") or []
    if len(closes) < 10 or nav_df is None or nav_df.empty:
        return {
            "benchmark_symbol": benchmark_context.get("symbol"),
            "benchmark_return_window_pct": benchmark_context.get("return_window_pct"),
            "benchmark_return_21d_pct": benchmark_context.get("return_21d_pct"),
        }

    fund = nav_df.copy().sort_values("Date")
    fund["Date"] = pd.to_datetime(fund["Date"]).dt.normalize()
    fund["fund_ret"] = fund["NAV"].astype(float).pct_change()

    bench = pd.DataFrame(closes)
    bench["Date"] = pd.to_datetime(bench["date"]).dt.normalize()
    bench["bench_ret"] = bench["close"].astype(float).pct_change()

    merged = pd.merge(
        fund[["Date", "fund_ret"]],
        bench[["Date", "bench_ret"]],
        on="Date",
        how="inner",
    ).dropna()
    if len(merged) < 20:
        return {
            "benchmark_symbol": benchmark_context.get("symbol"),
            "benchmark_return_window_pct": benchmark_context.get("return_window_pct"),
            "benchmark_return_21d_pct": benchmark_context.get("return_21d_pct"),
            "benchmark_overlap_days": int(len(merged)),
        }

    y = merged["fund_ret"].to_numpy(dtype=float)
    x = merged["bench_ret"].to_numpy(dtype=float)
    x_des = np.column_stack([np.ones(len(x)), x])
    try:
        coef, _, _, _ = np.linalg.lstsq(x_des, y, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
        resid = y - (alpha + beta * x)
        resid_vol = float(np.std(resid) * np.sqrt(252) * 100)
        corr = float(np.corrcoef(y, x)[0, 1]) if len(y) > 1 else None
    except Exception:
        return {
            "benchmark_symbol": benchmark_context.get("symbol"),
            "benchmark_return_window_pct": benchmark_context.get("return_window_pct"),
        }

    return {
        "benchmark_symbol": benchmark_context.get("symbol"),
        "benchmark_return_window_pct": benchmark_context.get("return_window_pct"),
        "benchmark_return_21d_pct": benchmark_context.get("return_21d_pct"),
        "benchmark_overlap_days": int(len(merged)),
        "benchmark_beta": round(beta, 4),
        "benchmark_alpha_daily": round(alpha, 8),
        "benchmark_residual_vol_ann_pct": round(resid_vol, 4),
        "benchmark_corr": round(corr, 4) if corr is not None and not np.isnan(corr) else None,
    }
