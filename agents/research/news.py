"""News Analyst — Google News RSS (+ optional NewsAPI)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from agents.contracts import Claim, ResearchBrief, TaskContract, URL_ALLOWLIST_SUFFIXES
from agents.fetchers.news import gather_news
from agents.memory.raw_store import RawStore
from agents.rag.store import DocumentStore
from utils.config import FUND_NAME
from utils.embeddings import sanitize_embed_text


class NewsAnalyst(BaseAgent):
    agent_id = "news_analyst"

    QUERIES = [
        FUND_NAME,
        "Kotak Multicap Fund",
        "India multicap mutual fund SEBI",
        "RBI interest rate India markets",
    ]

    def __init__(self, store: DocumentStore | None = None) -> None:
        self.store = store or DocumentStore()

    def run(self, contract: TaskContract, **kwargs: Any) -> ResearchBrief:
        raw_store: RawStore | None = kwargs.get("raw_store")
        run_id = kwargs.get("run_id") or (raw_store.run_id if raw_store else "unknown")
        _ = run_id
        allowlist = contract.allowlist_domains or list(URL_ALLOWLIST_SUFFIXES)
        api_key = kwargs.get("news_api_key")

        unique, errors, raw_by_query = gather_news(
            self.QUERIES,
            allowlist=allowlist,
            api_key=api_key,
        )

        raw_paths: list[str] = []
        if raw_store is not None:
            for query, payload in raw_by_query.items():
                safe = "".join(c if c.isalnum() else "_" for c in query)[:60]
                path = raw_store.write_raw("news", f"rss_{safe}", payload)
                raw_paths.append(str(path))

        claims: list[Claim] = []
        sources: list[str] = []
        docs_for_chroma: list[dict[str, Any]] = []
        for item in unique[: contract.max_claims]:
            self.store.add_document(
                title=item["title"],
                text=f"{item['title']}. {item.get('summary') or ''}",
                source=item.get("source") or "news",
                url=item.get("url") or None,
            )
            src = item.get("url") or item["title"]
            sources.append(src)
            claims.append(Claim(text=item["title"], evidence_refs=[src]))
            docs_for_chroma.append(
                {
                    "title": sanitize_embed_text(item["title"], max_chars=200),
                    "text": sanitize_embed_text(
                        item.get("summary") or item["title"], max_chars=1000
                    ),
                    "source": item.get("source") or "news",
                    "url": item.get("url") or "",
                }
            )

        local_added = self.store.ingest_local_folder(Path("data/docs"))

        return ResearchBrief(
            agent_id="news_analyst",
            as_of=datetime.now(timezone.utc),
            session=contract.session,
            sources=sources,
            confidence=0.7 if claims else 0.25,
            claims=claims,
            metrics={
                "articles": len(unique),
                "local_chunks_added": local_added,
                "errors": errors[:8],
                "chroma_documents": docs_for_chroma,
            },
            notes=(
                f"NewsAnalyst session={contract.session}; "
                f"ingested {len(claims)} headlines into RAG (Chroma upsert by Manager)."
            ),
            raw_paths=raw_paths,
        )
