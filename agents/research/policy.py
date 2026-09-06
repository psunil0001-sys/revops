"""Policy / regulation analyst — SEBI, RBI, IRDAI, tax/budget rules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent
from agents.contracts import Claim, ResearchBrief, TaskContract, URL_ALLOWLIST_SUFFIXES
from agents.fetchers.news import gather_news
from agents.memory.raw_store import RawStore
from utils.embeddings import sanitize_embed_text
from utils.funds import FundConfig


class PolicyAnalyst(BaseAgent):
    agent_id = "policy_analyst"

    def run(self, contract: TaskContract, **kwargs: Any) -> ResearchBrief:
        fund: FundConfig | None = kwargs.get("fund")
        raw_store: RawStore | None = kwargs.get("raw_store")
        allowlist = contract.allowlist_domains or list(URL_ALLOWLIST_SUFFIXES)

        queries = list((fund.research_queries if fund else None) or [])
        if fund and fund.type == "ulip":
            queries.extend(
                [
                    "IRDAI ULIP regulations India",
                    "ULIP tax rules India Budget",
                    "life insurance fund investment guidelines IRDAI",
                ]
            )
        else:
            queries.extend(
                [
                    "SEBI mutual fund regulations India",
                    "SEBI multicap fund rules",
                    "RBI monetary policy India markets",
                    "mutual fund capital gains tax India",
                ]
            )
        # Deduplicate preserving order
        seen_q: set[str] = set()
        uniq_queries: list[str] = []
        for q in queries:
            key = q.lower()
            if key in seen_q:
                continue
            seen_q.add(key)
            uniq_queries.append(q)

        unique, errors, raw_by_query = gather_news(
            uniq_queries[:8],
            allowlist=allowlist,
            api_key=kwargs.get("news_api_key"),
        )

        raw_paths: list[str] = []
        if raw_store is not None:
            for query, payload in raw_by_query.items():
                safe = "".join(c if c.isalnum() else "_" for c in query)[:60]
                path = raw_store.write_raw("policy", f"rss_{safe}", payload)
                raw_paths.append(str(path))

        claims: list[Claim] = []
        sources: list[str] = []
        docs_for_chroma: list[dict[str, Any]] = []
        for item in unique[: contract.max_claims]:
            src = item.get("url") or item["title"]
            sources.append(src)
            claims.append(Claim(text=item["title"], evidence_refs=[src]))
            docs_for_chroma.append(
                {
                    "title": sanitize_embed_text(item["title"], max_chars=200),
                    "text": sanitize_embed_text(
                        item.get("summary") or item["title"], max_chars=1000
                    ),
                    "source": item.get("source") or "policy",
                    "url": item.get("url") or "",
                }
            )

        fund_label = fund.name if fund else "fund"
        return ResearchBrief(
            agent_id="policy_analyst",
            as_of=datetime.now(timezone.utc),
            session=contract.session,
            sources=sources or ["policy:no_hits"],
            confidence=0.7 if claims else 0.25,
            claims=claims,
            metrics={
                "articles": len(unique),
                "fund_type": fund.type if fund else None,
                "errors": errors[:8],
                "chroma_documents": docs_for_chroma,
            },
            notes=(
                f"PolicyAnalyst for {fund_label} "
                f"type={fund.type if fund else 'n/a'}; "
                f"{len(claims)} regulatory/policy headlines."
            ),
            raw_paths=raw_paths,
        )
