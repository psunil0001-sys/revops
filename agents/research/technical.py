"""Technical Analyst — OHLCV indicators via yfinance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent
from agents.contracts import Claim, ResearchBrief, TaskContract
from agents.fetchers import market as market_fetcher
from agents.memory.raw_store import RawStore
from utils.config import TECHNICAL_SYMBOLS


class TechnicalAnalyst(BaseAgent):
    agent_id = "technical_analyst"

    def run(self, contract: TaskContract, **kwargs: Any) -> ResearchBrief:
        raw_store: RawStore | None = kwargs.get("raw_store")
        run_id = kwargs.get("run_id") or (raw_store.run_id if raw_store else "unknown")
        _ = run_id

        metrics: dict[str, Any] = {}
        claims: list[Claim] = []
        sources: list[str] = []
        raw_paths: list[str] = []
        docs_for_chroma: list[dict[str, Any]] = []
        errors: list[str] = []

        for label, symbol in TECHNICAL_SYMBOLS.items():
            try:
                hist = market_fetcher.fetch_history(symbol, period="6mo")
                if raw_store is not None:
                    path = raw_store.write_raw(
                        "technical", f"yfinance_{label}", hist
                    )
                    raw_paths.append(str(path))
                closes = [float(r["close"]) for r in hist.get("rows") or []]
                tech = market_fetcher.compute_technicals(closes)
                if not tech:
                    continue
                metrics[label] = {"symbol": symbol, **tech}
                sources.append(f"yfinance:{symbol}")
                direction = "up" if float(tech["ret_1d_pct"]) >= 0 else "down"
                rsi = tech.get("rsi14")
                claims.append(
                    Claim(
                        text=(
                            f"{label} ({symbol}) {direction} "
                            f"{abs(float(tech['ret_1d_pct'])):.2f}% d/d; "
                            f"RSI14={rsi if rsi is not None else 'n/a'}; "
                            f"above SMA20={tech.get('above_sma20')}."
                        ),
                        evidence_refs=[f"yfinance:{symbol}"],
                    )
                )
                docs_for_chroma.append(
                    {
                        "title": f"Technical {label}",
                        "text": (
                            f"{label} {symbol} last={tech.get('last')} "
                            f"ret_1d={tech.get('ret_1d_pct')} "
                            f"vol={tech.get('vol_21d_ann_pct')} "
                            f"rsi14={rsi} sma20={tech.get('sma20')}"
                        ),
                        "source": f"yfinance:{symbol}",
                        "url": f"https://finance.yahoo.com/quote/{symbol}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}:{exc}")
                metrics[f"{label}_error"] = str(exc)

        nifty = metrics.get("nifty50") or {}
        if nifty:
            tilt = "risk-on" if float(nifty.get("ret_1d_pct") or 0) >= 0 else "risk-off"
            claims.append(
                Claim(
                    text=(
                        f"Technical tape into {contract.session} session appears "
                        f"{tilt} on Nifty proxies."
                    ),
                    evidence_refs=["yfinance:^NSEI"],
                )
            )

        confidence = 0.75 if metrics else 0.2
        return ResearchBrief(
            agent_id="technical_analyst",
            as_of=datetime.now(timezone.utc),
            session=contract.session,
            sources=sources[: contract.max_claims * 2],
            confidence=confidence,
            claims=claims[: contract.max_claims],
            metrics={
                **metrics,
                "errors": errors[:8],
                "chroma_documents": docs_for_chroma,
            },
            notes=(
                f"TechnicalAnalyst session={contract.session}. "
                "OHLCV indicators via yfinance; not investment advice."
            ),
            raw_paths=raw_paths,
        )
