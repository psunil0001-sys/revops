"""Sentiment Analyst — social (Reddit) + RAG tone (Chroma upserted by Manager)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent
from agents.contracts import Claim, RagHit, ResearchBrief, SentimentReport, TaskContract
from agents.fetchers.social import gather_social
from agents.memory.raw_store import RawStore
from agents.rag.hybrid import HybridRetriever
from agents.rag.store import DocumentStore
from utils.config import FUND_NAME, SECTOR_DATA


def _score_texts(texts: list[str]) -> tuple[float, str]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    score = 0.0
    if texts:
        compounds = [analyzer.polarity_scores(t)["compound"] for t in texts]
        score = float(sum(compounds) / len(compounds))
    if score >= 0.15:
        label = "bullish"
    elif score <= -0.15:
        label = "bearish"
    else:
        label = "neutral"
    return score, label


class SentimentAnalyst(BaseAgent):
    agent_id = "sentiment_analyst"

    def __init__(
        self,
        store: DocumentStore | None = None,
        retriever: HybridRetriever | None = None,
        *,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.store = store or DocumentStore()
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.retriever = retriever or HybridRetriever(
            self.store,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
        )

    def run(self, contract: TaskContract, **kwargs: Any) -> ResearchBrief:
        raw_store: RawStore | None = kwargs.get("raw_store")
        queries = [
            "Kotak Multicap",
            "multicap mutual fund",
            "Nifty India markets",
        ]
        posts, social_errors, social_raw = gather_social(queries)

        raw_paths: list[str] = []
        if raw_store is not None:
            path = raw_store.write_raw("sentiment", "social_reddit", social_raw)
            raw_paths.append(str(path))

        docs_for_chroma: list[dict[str, Any]] = []
        for post in posts[:40]:
            title = post.get("title") or "reddit"
            text = f"{title}. {post.get('selftext') or ''}"
            self.store.add_document(
                title=title,
                text=text,
                source=f"reddit:{post.get('subreddit') or 'search'}",
                url=post.get("permalink") or post.get("url"),
            )
            docs_for_chroma.append(
                {
                    "title": title,
                    "text": text[:1500],
                    "source": f"reddit:{post.get('subreddit') or 'search'}",
                    "url": post.get("permalink") or "",
                }
            )

        # Current corpus hybrid retrieval (BM25 + embed server)
        self.retriever.refresh()
        all_hits: list[RagHit] = []
        seen: set[str] = set()
        for query in [
            FUND_NAME,
            "multicap mutual fund India outlook",
            " ".join(SECTOR_DATA["Sector"][:3]),
        ]:
            for hit in self.retriever.search(query, top_k=4):
                if hit.doc_id in seen:
                    continue
                seen.add(hit.doc_id)
                all_hits.append(hit)

        if raw_store is not None:
            path = raw_store.write_raw(
                "sentiment",
                "rag_hits",
                [h.model_dump(mode="json") for h in all_hits[:30]],
            )
            raw_paths.append(str(path))

        social_texts = [
            f"{p.get('title')}. {p.get('selftext') or ''}" for p in posts[:30]
        ]
        hit_texts = [f"{h.title}. {h.snippet}" for h in all_hits]
        score, label = _score_texts(social_texts + hit_texts)

        report = SentimentReport(
            agent_id="sentiment_analyst",
            as_of=datetime.now(timezone.utc),
            session=contract.session,
            overall_score=round(score, 4),
            label=label,  # type: ignore[arg-type]
            hits=all_hits[:10],
            sources=[
                (h.url or h.source or h.doc_id) for h in all_hits[:10]
            ]
            or [p.get("permalink") or p.get("title") for p in posts[:5]],
            confidence=0.65 if (posts or all_hits) else 0.2,
            notes=(
                f"SentimentAnalyst posts={len(posts)} hits={len(all_hits)} "
                f"errors={len(social_errors)}"
            ),
        )
        if not report.sources:
            report.sources = ["sentiment:empty"]

        claims = [
            Claim(
                text=(
                    f"Overall sentiment tilt is {report.label} "
                    f"(score={report.overall_score}) from social+news corpus."
                ),
                evidence_refs=[h.doc_id for h in all_hits[:3]]
                or [str(p.get("id") or "social") for p in posts[:2]],
            )
        ]
        for post in posts[:4]:
            claims.append(
                Claim(
                    text=f"Social: {post.get('title')}",
                    evidence_refs=[post.get("permalink") or str(post.get("id"))],
                )
            )
        for hit in all_hits[:4]:
            claims.append(
                Claim(text=f"RAG: {hit.title}", evidence_refs=[hit.doc_id])
            )

        return ResearchBrief(
            agent_id="sentiment_analyst",
            as_of=report.as_of,
            session=contract.session,
            sources=list(report.sources) or ["sentiment:no_hits"],
            confidence=report.confidence,
            claims=claims[: contract.max_claims],
            metrics={
                "sentiment": report.model_dump(mode="json"),
                "post_count": len(posts),
                "hit_count": len(all_hits),
                "errors": social_errors[:8],
                "chroma_documents": docs_for_chroma,
            },
            notes=report.notes,
            raw_paths=raw_paths,
        )
