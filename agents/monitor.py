"""Monitor agent — NAV refresh, enriched 60d forecast, summary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents.base import BaseAgent
from agents.contracts import (
    DEFAULT_HORIZON_DAYS,
    DISCLAIMER,
    MemoryHit,
    MonitorBundle,
    ResearchBrief,
    SentimentReport,
    SessionKind,
    TaskContract,
)
from utils.config import FUND_INCEPTION_DATE, FUND_NAME, SCHEME_CODE
from utils.features import build_nav_features
from utils.llm import forecast_nav
from utils.nav import fetch_nav_history

_MAX_MEMORY_CHARS = 1600


class MonitorAgent(BaseAgent):
    agent_id = "monitor"

    def run(self, contract: TaskContract, **kwargs: Any) -> MonitorBundle:
        return self.execute(
            session=contract.session,
            briefs=list(kwargs.get("briefs") or []),
            horizon_days=contract.horizon_days,
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model"),
            use_llm=bool(kwargs.get("use_llm", True)),
            run_id=kwargs.get("run_id"),
            memory_hits=list(kwargs.get("memory_hits") or []),
        )

    def execute(
        self,
        *,
        session: SessionKind,
        briefs: list[ResearchBrief],
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        use_llm: bool = True,
        run_id: str | None = None,
        memory_hits: list[MemoryHit] | None = None,
    ) -> MonitorBundle:
        hits = list(memory_hits or [])
        nav_df = fetch_nav_history(
            scheme_code=SCHEME_CODE,
            start_date=FUND_INCEPTION_DATE,
            end_date=datetime.now(timezone.utc).date(),
        )
        features = build_nav_features(nav_df)
        sentiment = self._extract_sentiment(briefs)
        agent_context = self._build_agent_context(
            briefs, sentiment, session, run_id=run_id, memory_hits=hits
        )

        forecast = forecast_nav(
            features,
            horizon_days=horizon_days,
            base_url=base_url,
            api_key=api_key,
            model=model,
            market_context=None,
            agent_context=agent_context,
            use_llm=use_llm,
        )

        summary = self._summarize(
            session=session,
            features=features,
            briefs=briefs,
            sentiment=sentiment,
            forecast=forecast,
            horizon_days=horizon_days,
            memory_hits=hits,
        )

        latest_nav = float(nav_df.iloc[-1]["NAV"]) if not nav_df.empty else None
        return MonitorBundle(
            as_of=datetime.now(timezone.utc),
            session=session,
            horizon_days=horizon_days,
            nav_records=int(len(nav_df)),
            latest_nav=latest_nav,
            research_briefs=briefs,
            sentiment=sentiment,
            forecast=forecast,
            summary_markdown=summary,
            disclaimer=DISCLAIMER,
            agent_context=agent_context,
            run_id=run_id,
            memory_hits=hits,
            memory_hit_count=len(hits),
        )

    @staticmethod
    def _extract_sentiment(briefs: list[ResearchBrief]) -> SentimentReport | None:
        for brief in briefs:
            if brief.agent_id != "sentiment_analyst":
                continue
            payload = (brief.metrics or {}).get("sentiment")
            if isinstance(payload, dict):
                return SentimentReport.model_validate(payload)
        return None

    @staticmethod
    def _trim_memory_hits(hits: list[MemoryHit], limit: int = 8) -> list[dict[str, Any]]:
        trimmed: list[dict[str, Any]] = []
        for hit in hits[:limit]:
            trimmed.append(
                {
                    "doc_id": hit.doc_id,
                    "collection": hit.collection,
                    "title": hit.title,
                    "text": hit.text[:_MAX_MEMORY_CHARS],
                    "score": hit.score,
                }
            )
        return trimmed

    @classmethod
    def _build_agent_context(
        cls,
        briefs: list[ResearchBrief],
        sentiment: SentimentReport | None,
        session: SessionKind,
        *,
        run_id: str | None = None,
        memory_hits: list[MemoryHit] | None = None,
    ) -> dict[str, Any]:
        return {
            "fund": FUND_NAME,
            "session": session,
            "run_id": run_id,
            "briefs": [b.model_dump(mode="json") for b in briefs],
            "sentiment": sentiment.model_dump(mode="json") if sentiment else None,
            "memory_hits": cls._trim_memory_hits(list(memory_hits or [])),
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _summarize(
        *,
        session: SessionKind,
        features: dict[str, Any],
        briefs: list[ResearchBrief],
        sentiment: SentimentReport | None,
        forecast: dict[str, Any],
        horizon_days: int,
        memory_hits: list[MemoryHit] | None = None,
    ) -> str:
        lines = [
            f"### Monitor summary ({session} session)",
            f"- Fund: **{FUND_NAME}**",
            f"- Latest NAV: **{features.get('last_nav')}** as of {features.get('last_date')}",
            f"- Horizon: **{horizon_days} days**",
            f"- Forecast source: **{forecast.get('source', 'unknown')}**",
        ]
        if sentiment:
            lines.append(
                f"- Sentiment: **{sentiment.label}** ({sentiment.overall_score})"
            )
        if memory_hits:
            lines.append(f"- Related memory hits: **{len(memory_hits)}**")
        for brief in briefs:
            top = brief.claims[0].text if brief.claims else "no claims"
            lines.append(
                f"- {brief.agent_id}: confidence={brief.confidence:.2f} — {top}"
            )
        scenarios = forecast.get("scenarios") or {}
        for name in ("bear", "base", "bull"):
            path = (scenarios.get(name) or {}).get("nav_path") or []
            if path:
                lines.append(f"- {name} horizon NAV: **{path[-1].get('nav')}**")
        lines.append(f"\n_{DISCLAIMER}_")
        return "\n".join(lines)
