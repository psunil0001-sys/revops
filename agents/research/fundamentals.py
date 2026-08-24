"""Fundamentals Analyst — fund/sector/peer fundamentals via yfinance + config."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent
from agents.contracts import Claim, ResearchBrief, TaskContract
from agents.fetchers import market as market_fetcher
from agents.memory.raw_store import RawStore
from utils.config import (
    EQUITY_ALLOCATION,
    FUND_NAME,
    FUNDAMENTAL_PROXY_SYMBOLS,
    OTHER_ALLOCATION,
    SECTOR_DATA,
)


class FundamentalsAnalyst(BaseAgent):
    agent_id = "fundamentals_analyst"

    def run(self, contract: TaskContract, **kwargs: Any) -> ResearchBrief:
        raw_store: RawStore | None = kwargs.get("raw_store")
        run_id = kwargs.get("run_id") or (raw_store.run_id if raw_store else "unknown")
        _ = run_id

        metrics: dict[str, Any] = {
            "fund": FUND_NAME,
            "equity_allocation_pct": EQUITY_ALLOCATION,
            "other_allocation_pct": OTHER_ALLOCATION,
            "sectors": dict(zip(SECTOR_DATA["Sector"], SECTOR_DATA["Allocation"])),
            "proxies": {},
        }
        claims: list[Claim] = []
        sources: list[str] = [f"config:{FUND_NAME}"]
        raw_paths: list[str] = []
        docs_for_chroma: list[dict[str, Any]] = []
        errors: list[str] = []

        claims.append(
            Claim(
                text=(
                    f"{FUND_NAME} reported equity allocation "
                    f"{EQUITY_ALLOCATION:.2f}% / other {OTHER_ALLOCATION:.2f}% "
                    "(static factsheet snapshot in config — not live AUM)."
                ),
                evidence_refs=["config:allocations"],
            )
        )
        top_sectors = sorted(
            zip(SECTOR_DATA["Sector"], SECTOR_DATA["Allocation"]),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
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
                f"FundamentalsAnalyst session={contract.session}. "
                "Uses config allocations + yfinance peer proxies; not live holdings."
            ),
            raw_paths=raw_paths,
        )
