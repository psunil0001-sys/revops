"""Shared yfinance market / fundamentals helpers."""

from __future__ import annotations

from typing import Any


def fetch_history(symbol: str, period: str = "6mo") -> dict[str, Any]:
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist is None or hist.empty:
        return {"symbol": symbol, "rows": [], "error": "empty_history"}
    records = []
    for idx, row in hist.tail(120).iterrows():
        if hasattr(idx, "date"):
            date_str = str(idx.date())
        else:
            date_str = str(idx)
        records.append(
            {
                "date": date_str,
                "open": float(row.get("Open") or 0),
                "high": float(row.get("High") or 0),
                "low": float(row.get("Low") or 0),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume") or 0),
            }
        )
    return {"symbol": symbol, "rows": records, "error": None}


def fetch_info(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    info = yf.Ticker(symbol).info or {}
    # Keep a compact, JSON-friendly subset
    keys = [
        "symbol",
        "shortName",
        "longName",
        "sector",
        "industry",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "marketCap",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "currentPrice",
        "previousClose",
        "currency",
        "exchange",
    ]
    slim = {k: info.get(k) for k in keys if k in info}
    slim["symbol"] = symbol
    return slim


def compute_technicals(closes: list[float]) -> dict[str, Any]:
    if len(closes) < 5:
        return {}
    last = closes[-1]
    prev = closes[-2]
    ret_1d = (last / prev - 1.0) * 100.0
    ret_1m = (
        (last / closes[-22] - 1.0) * 100.0 if len(closes) >= 22 else None
    )
    rets = [
        (closes[i] / closes[i - 1] - 1.0)
        for i in range(1, len(closes))
    ]
    window = rets[-21:] if len(rets) >= 21 else rets
    mean = sum(window) / len(window)
    var = sum((x - mean) ** 2 for x in window) / max(len(window) - 1, 1)
    vol = (var ** 0.5) * (252 ** 0.5) * 100.0

    sma20 = sum(closes[-20:]) / min(20, len(closes))
    sma50 = sum(closes[-50:]) / min(50, len(closes)) if len(closes) >= 20 else None

    # Simple RSI(14)
    rsi = None
    if len(closes) >= 15:
        gains = []
        losses = []
        for i in range(-14, 0):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
        avg_gain = sum(gains) / 14.0
        avg_loss = sum(losses) / 14.0
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

    return {
        "last": round(last, 4),
        "ret_1d_pct": round(ret_1d, 4),
        "ret_1m_pct": round(ret_1m, 4) if ret_1m is not None else None,
        "vol_21d_ann_pct": round(vol, 4),
        "sma20": round(sma20, 4),
        "sma50": round(sma50, 4) if sma50 is not None else None,
        "rsi14": round(rsi, 2) if rsi is not None else None,
        "above_sma20": bool(last >= sma20),
    }
