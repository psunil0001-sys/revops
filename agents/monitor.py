"""Monitor agent — per-fund NAV refresh, enriched 60d forecast, summary."""

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
from utils.backtest import walk_forward_forecast_backtest
from utils.config import MIN_FORECAST_OBSERVATIONS
from utils.features import assert_forecast_ready, build_nav_features
from utils.funds import FundConfig
from utils.llm import forecast_nav
from utils.market import attach_benchmark_features, fetch_benchmark_context
from utils.nav import fetch_nav_history_for_fund

_MAX_MEMORY_CHARS = 1600


class MonitorAgent(BaseAgent):
    agent_id = "monitor"

    def run(self, contract: TaskContract, **kwargs: Any) -> MonitorBundle:
        return self.execute(
            fund=kwargs["fund"],
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
        fund: FundConfig,
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
        nav_df = fetch_nav_history_for_fund(
            fund,
            start_date=fund.inception_date,
            end_date=datetime.now(timezone.utc).date(),
        )
        assert_forecast_ready(nav_df, min_observations=MIN_FORECAST_OBSERVATIONS)
        features = build_nav_features(nav_df)

        start = pd_timestamp_date(nav_df["Date"].min())
        end = pd_timestamp_date(nav_df["Date"].max())
        market_context = fetch_benchmark_context(
            fund.benchmark_symbol or "^NSEI", start, end
        )
        features.update(attach_benchmark_features(nav_df, market_context))

        sentiment = self._extract_sentiment(briefs)
        agent_context = self._build_agent_context(
            fund, briefs, sentiment, session, run_id=run_id, memory_hits=hits
        )

        forecast = forecast_nav(
            features,
            horizon_days=horizon_days,
            base_url=base_url,
            api_key=api_key,
            model=model,
            market_context=market_context,
            agent_context=agent_context,
            use_llm=use_llm,
        )

        backtest = walk_forward_forecast_backtest(
            nav_df, horizon_days=horizon_days, step=21
        )
        forecast = dict(forecast)
        forecast["backtest"] = backtest
        forecast["observations"] = int(features.get("observations") or len(nav_df))

        summary = self._summarize(
            fund=fund,
            session=session,
            features=features,
            briefs=briefs,
            sentiment=sentiment,
            forecast=forecast,
            horizon_days=horizon_days,
            memory_hits=hits,
            backtest=backtest,
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
        fund: FundConfig,
        briefs: list[ResearchBrief],
        sentiment: SentimentReport | None,
        session: SessionKind,
        *,
        run_id: str | None = None,
        memory_hits: list[MemoryHit] | None = None,
    ) -> dict[str, Any]:
        return {
            "fund": fund.name,
            "fund_id": fund.id,
            "fund_type": fund.type,
            "benchmark_symbol": fund.benchmark_symbol,
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
        fund: FundConfig,
        session: SessionKind,
        features: dict[str, Any],
        briefs: list[ResearchBrief],
        sentiment: SentimentReport | None,
        forecast: dict[str, Any],
        horizon_days: int,
        memory_hits: list[MemoryHit] | None = None,
        backtest: dict[str, Any] | None = None,
    ) -> str:
        lines = [
            f"### Monitor summary ({session} session)",
            f"- Fund: **{fund.name}** (`{fund.id}`)",
            f"- Latest NAV: **{features.get('last_nav')}** as of {features.get('last_date')}",
            f"- Observations: **{features.get('observations')}**",
            f"- Horizon: **{horizon_days} trading days**",
            f"- Forecast source: **{forecast.get('source', 'unknown')}**",
        ]
        if forecast.get("llm_fallback"):
            lines.append("- LLM refine: **fell back to bootstrap baseline**")
        if backtest and backtest.get("ready"):
            lines.append(
                f"- Backtest: **{backtest.get('label')}** "
                f"(n={backtest.get('n_origins')}, "
                f"MedAE base={backtest.get('median_abs_pct_error_base')}%)"
            )
        elif backtest:
            lines.append(
                f"- Backtest: not ready ({backtest.get('reason', 'insufficient history')})"
            )
        if features.get("benchmark_beta") is not None:
            lines.append(
                f"- Benchmark `{features.get('benchmark_symbol')}`: "
                f"beta={features.get('benchmark_beta')}, "
                f"corr={features.get('benchmark_corr')}"
            )
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


def pd_timestamp_date(value: Any):
    import pandas as pd

    return pd.Timestamp(value).date()
