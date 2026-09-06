"""Fundamentals Analyst — fund/sector/peer fundamentals via yfinance + FundConfig."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent
from agents.contracts import Claim, ResearchBrief, TaskContract
from agents.fetchers import market as market_fetcher
from agents.memory.raw_store import RawStore
from utils.config import FUNDAMENTAL_PROXY_SYMBOLS
from utils.funds import FundConfig


class FundamentalsAnalyst(BaseAgent):
    agent_id = "fundamentals_analyst"

    def run(self, contract: TaskContract, **kwargs: Any) -> ResearchBrief:
        fund: FundConfig = kwargs["fund"]
        raw_store: RawStore | None = kwargs.get("raw_store")

        sectors = fund.sectors
        metrics: dict[str, Any] = {
            "fund": fund.name,
            "fund_id": fund.id,
            "fund_type": fund.type,
            "equity_allocation_pct": fund.equity_pct,
            "other_allocation_pct": fund.other_pct,
            "sectors": dict(sectors),
            "proxies": {},
        }
        claims: list[Claim] = []
        sources: list[str] = [f"config:{fund.id}"]
        raw_paths: list[str] = []
        docs_for_chroma: list[dict[str, Any]] = []
        errors: list[str] = []

        claims.append(
            Claim(
                text=(
                    f"{fund.name} equity allocation "
                    f"{fund.equity_pct:.2f}% / other {fund.other_pct:.2f}% "
                    "(config snapshot — not live AUM)."
                ),
                evidence_refs=["config:allocations"],
            )
        )
        if sectors:
            top_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:3]
            claims.append(
                Claim(
                    text=(
                        "Top sector weights: "
                        + ", ".join(f"{n} {w:.1f}%" for n, w in top_sectors)
                        + "."
                    ),
                    evidence_refs=["config:sectors"],
                )
            )

        for label, symbol in FUNDAMENTAL_PROXY_SYMBOLS.items():
            try:
                info = market_fetcher.fetch_info(symbol)
                metrics["proxies"][label] = info
                if raw_store is not None:
                    path = raw_store.write_raw(
                        "fundamentals", f"info_{label}_{symbol}", info
                    )
                    raw_paths.append(str(path))
                sources.append(f"yfinance:{symbol}")
                pe = info.get("trailingPE")
                pb = info.get("priceToBook")
                name = info.get("shortName") or info.get("longName") or symbol
                claims.append(
                    Claim(
                        text=(
                            f"Proxy {name} ({symbol}): "
                            f"PE={pe if pe is not None else 'n/a'}, "
                            f"PB={pb if pb is not None else 'n/a'}."
                        ),
                        evidence_refs=[f"yfinance:{symbol}"],
                    )
                )
                docs_for_chroma.append(
                    {
                        "title": f"Fundamentals {name}",
                        "text": (
                            f"{name} sector={info.get('sector')} "
                            f"PE={pe} PB={pb} marketCap={info.get('marketCap')}"
                        ),
                        "source": f"yfinance:{symbol}",
                        "url": f"https://finance.yahoo.com/quote/{symbol}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}:{exc}")
                metrics["proxies"][f"{label}_error"] = str(exc)

        if raw_store is not None:
            path = raw_store.write_raw(
                "fundamentals",
                "summary",
                {"metrics": metrics, "errors": errors},
            )
            raw_paths.append(str(path))

        confidence = 0.7 if metrics["proxies"] else 0.4
        return ResearchBrief(
            agent_id="fundamentals_analyst",
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
                f"FundamentalsAnalyst fund={fund.id} session={contract.session}."
            ),
            raw_paths=raw_paths,
        )
