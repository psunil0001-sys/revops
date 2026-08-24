"""Pydantic contracts for multi-agent research and forecasting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SessionKind = Literal["open", "close", "manual"]
AgentStatusKind = Literal["pending", "running", "ok", "error", "skipped"]
AgentId = Literal[
    "fundamentals_analyst",
    "sentiment_analyst",
    "news_analyst",
    "technical_analyst",
    "monitor",
    "manager",
]

DEFAULT_HORIZON_DAYS = 60
DISCLAIMER = (
    "Research and scenario forecasts only — not investment advice. "
    "Past performance does not guarantee future results."
)

URL_ALLOWLIST_SUFFIXES = (
    "google.com",
    "news.google.com",
    "reuters.com",
    "bloomberg.com",
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "sebi.gov.in",
    "rbi.org.in",
    "kotakmf.com",
    "amfiindia.com",
    "yahoo.com",
    "newsapi.org",
    "reddit.com",
)


class Claim(BaseModel):
    text: str
    evidence_refs: list[str] = Field(default_factory=list)


class TaskContract(BaseModel):
    agent_id: AgentId
    session: SessionKind
    horizon_days: int = DEFAULT_HORIZON_DAYS
    assigned_at: datetime
    instructions: str
    allowlist_domains: list[str] = Field(
        default_factory=lambda: list(URL_ALLOWLIST_SUFFIXES)
    )
    max_claims: int = 12

    @field_validator("horizon_days")
    @classmethod
    def _horizon_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("horizon_days must be >= 1")
        return value


class RagHit(BaseModel):
    doc_id: str
    title: str
    snippet: str
    source: str
    score: float
    url: str | None = None


class MemoryHit(BaseModel):
    doc_id: str
    collection: str
    title: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchBrief(BaseModel):
    agent_id: AgentId
    as_of: datetime
    session: SessionKind
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    claims: list[Claim] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    raw_paths: list[str] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def _non_empty_sources_when_claims(cls, sources: list[str], info):
        return sources


class SentimentReport(BaseModel):
    agent_id: AgentId = "sentiment_analyst"
    as_of: datetime
    session: SessionKind
    overall_score: float = Field(ge=-1.0, le=1.0)
    label: Literal["bearish", "neutral", "bullish"]
    hits: list[RagHit] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class AgentRunStatus(BaseModel):
    agent_id: AgentId
    status: AgentStatusKind = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    notes: str = ""


class MonitorBundle(BaseModel):
    as_of: datetime
    session: SessionKind
    horizon_days: int = DEFAULT_HORIZON_DAYS
    nav_records: int = 0
    latest_nav: float | None = None
    research_briefs: list[ResearchBrief] = Field(default_factory=list)
    sentiment: SentimentReport | None = None
    forecast: dict[str, Any] = Field(default_factory=dict)
    summary_markdown: str = ""
    disclaimer: str = DISCLAIMER
    agent_context: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    memory_hits: list[MemoryHit] = Field(default_factory=list)
    memory_hit_count: int = 0


class ManagerVerdict(BaseModel):
    accepted: bool
    session: SessionKind
    as_of: datetime
    messages: list[str] = Field(default_factory=list)
    guardrail_hits: list[str] = Field(default_factory=list)
    monitor_bundle: MonitorBundle | None = None
    disclaimer: str = DISCLAIMER
    run_id: str | None = None
    agent_statuses: list[AgentRunStatus] = Field(default_factory=list)
    memory_hit_count: int = 0
    raw_root: str | None = None


class SessionRunState(BaseModel):
    trade_date: str  # YYYY-MM-DD IST
    session: SessionKind
    completed_at: datetime
    accepted: bool
    notes: str = ""
