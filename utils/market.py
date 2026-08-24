"""Optional market context (Nifty 50) for forecast prompts."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd


def fetch_nifty_context(
    start_date: date,
    end_date: date,
) -> dict[str, Any] | None:
    """
    Fetch Nifty 50 (^NSEI) summary via yfinance for the same window.

    Returns None soft-fail if yfinance is missing or the request fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        # yfinance end is exclusive; pad by one day.
        ticker = yf.Ticker("^NSEI")
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

    return {
        "symbol": "^NSEI",
        "name": "Nifty 50",
        "start_date": pd.Timestamp(close.index[0]).date().isoformat(),
        "end_date": pd.Timestamp(close.index[-1]).date().isoformat(),
        "start_close": round(first, 2),
        "end_close": round(last, 2),
        "return_window_pct": round(ret_pct, 4) if ret_pct is not None else None,
        "return_21d_pct": round(recent_ret, 4) if recent_ret is not None else None,
        "observations": int(len(close)),
    }
