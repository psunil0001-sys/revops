"""Manager agent — assign 4 analysts, guardrails, memory, open/close orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from agents.contracts import (
    DEFAULT_HORIZON_DAYS,
    DISCLAIMER,
    AgentRunStatus,
    ManagerVerdict,
    ResearchBrief,
    SessionKind,
    SessionRunState,
    TaskContract,
    URL_ALLOWLIST_SUFFIXES,
)
from agents.memory.chroma_store import ChromaMemoryStore
from agents.memory.raw_store import RawStore
from agents.memory.summarizer import (
    append_summary_file,
    build_run_summary,
    persist_verdict_json,
)
from agents.monitor import MonitorAgent
from agents.research.fundamentals import FundamentalsAnalyst
from agents.research.news import NewsAnalyst
from agents.research.sentiment import SentimentAnalyst
from agents.research.technical import TechnicalAnalyst
from agents.scheduler import SessionRunStore, today_ist
from agents.server_manager import ServerManager
from utils.config import DEFAULT_EMBEDDING_BASE_URL, FUND_NAME

_GUARANTEE_WORDS = (
    "guaranteed return",
    "guaranteed returns",
    "risk-free profit",
    "sure profit",
    "cannot lose",
)

_ANALYST_ORDER = {
    "fundamentals_analyst": 0,
    "sentiment_analyst": 1,
    "news_analyst": 2,
    "technical_analyst": 3,
}


class ManagerAgent:
    agent_id = "manager"

    def __init__(
        self,
        *,
        chat_base_url: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        self.servers = ServerManager(
            chat_base_url=chat_base_url,
            embedding_base_url=embedding_base_url or DEFAULT_EMBEDDING_BASE_URL,
        )
        self.fundamentals = FundamentalsAnalyst()
        self.news = NewsAnalyst()
        self.sentiment = SentimentAnalyst(
            embedding_base_url=self.servers.embedding_base_url,
        )
        self.technical = TechnicalAnalyst()
        self.monitor = MonitorAgent()
        self.runs = SessionRunStore()
        self._chroma: ChromaMemoryStore | None = None

    @property
    def chroma(self) -> ChromaMemoryStore:
        if self._chroma is None:
            self._chroma = ChromaMemoryStore(
                embedding_base_url=self.servers.embedding_base_url,
            )
        return self._chroma

    @chroma.setter
    def chroma(self, value: ChromaMemoryStore) -> None:
        self._chroma = value

    def _contracts(self, session: SessionKind) -> list[TaskContract]:
        now = datetime.now(timezone.utc)
        common = dict(
            session=session,
            horizon_days=DEFAULT_HORIZON_DAYS,
            assigned_at=now,
            allowlist_domains=list(URL_ALLOWLIST_SUFFIXES),
        )
        return [
            TaskContract(
                agent_id="fundamentals_analyst",
                instructions=(
                    f"Analyze fund/sector/peer fundamentals for session={session}. "
                    "Use config allocations + yfinance proxies. Do not invent holdings."
                ),
                **common,
            ),
            TaskContract(
                agent_id="sentiment_analyst",
                instructions=(
                    f"Gather social + RAG/Chroma sentiment for session={session}. "
                    "Cite sources; do not fabricate posts."
                ),
                **common,
            ),
            TaskContract(
                agent_id="news_analyst",
                instructions=(
                    f"Gather fund/AMC/sector/regulatory news for session={session}. "
                    "Ingest into RAG+Chroma. Prefer allowlisted domains."
                ),
                **common,
            ),
            TaskContract(
                agent_id="technical_analyst",
                instructions=(
                    f"Compute technical indicators for India equity proxies "
                    f"at session={session}. Use yfinance only. Do not invent prices."
                ),
                **common,
            ),
        ]

    def _validate_brief(self, brief: ResearchBrief, contract: TaskContract) -> list[str]:
        hits: list[str] = []
        if brief.agent_id != contract.agent_id:
            hits.append(f"{brief.agent_id}: agent_id mismatch")
        if brief.session != contract.session:
            hits.append(f"{brief.agent_id}: session mismatch")
        if not brief.sources and brief.claims:
            hits.append(f"{brief.agent_id}: claims without sources")
        if len(brief.claims) > contract.max_claims:
            hits.append(f"{brief.agent_id}: too many claims")
        for claim in brief.claims:
            lowered = claim.text.lower()
            if any(word in lowered for word in _GUARANTEE_WORDS):
                hits.append(f"{brief.agent_id}: forbidden guarantee language")
                break
        for src in brief.sources:
            if not src.startswith("http"):
                continue
            host = urlparse(src).netloc.lower()
            if host and not any(
                host.endswith(d) or d in host for d in contract.allowlist_domains
            ):
                hits.append(f"{brief.agent_id}: source host not in allowlist ({host})")
        return hits

    def _run_research(
        self,
        contracts: list[TaskContract],
        *,
        raw_store: RawStore,
        statuses: dict[str, AgentRunStatus],
    ) -> tuple[list[ResearchBrief], list[str]]:
        agents = {
            "fundamentals_analyst": self.fundamentals,
            "sentiment_analyst": self.sentiment,
            "news_analyst": self.news,
            "technical_analyst": self.technical,
        }
        briefs: list[ResearchBrief] = []
        errors: list[str] = []

        def _one(contract: TaskContract) -> ResearchBrief:
            agent = agents[contract.agent_id]
            statuses[contract.agent_id].status = "running"
            statuses[contract.agent_id].started_at = datetime.now(timezone.utc)
            return agent.run(
                contract,
                raw_store=raw_store,
                run_id=raw_store.run_id,
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_one, c): c for c in contracts}
            for future in as_completed(futures):
                contract = futures[future]
                try:
                    brief = future.result()
                    hits = self._validate_brief(brief, contract)
                    hard = [h for h in hits if "allowlist" not in h]
                    if hard:
                        brief = agents[contract.agent_id].run(
                            contract,
                            raw_store=raw_store,
                            run_id=raw_store.run_id,
                        )
                        hits = self._validate_brief(brief, contract)
                        hard = [h for h in hits if "allowlist" not in h]
                        if hard:
                            errors.extend(hard)
                            brief.confidence = min(brief.confidence, 0.2)
                            brief.notes += " | guardrail warnings: " + "; ".join(hard)
                    briefs.append(brief)
                    statuses[contract.agent_id].status = "ok"
                    statuses[contract.agent_id].finished_at = datetime.now(timezone.utc)
                    statuses[contract.agent_id].notes = brief.notes[:200]
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{contract.agent_id} failed: {exc}")
                    statuses[contract.agent_id].status = "error"
                    statuses[contract.agent_id].finished_at = datetime.now(timezone.utc)
                    statuses[contract.agent_id].error = str(exc)

        briefs.sort(key=lambda b: _ANALYST_ORDER.get(b.agent_id, 9))
        return briefs, errors

    def _stop_servers_after_run(self, verdict: ManagerVerdict | None) -> None:
        """Halt chat+embed after pipeline; next run will ensure_ready again."""
        try:
            result = self.servers.stop_servers()
            msg = "Servers stopped after pipeline (restart on next LLM run)."
            if verdict is not None:
                verdict.messages.append(msg)
                if not result.get("stopped"):
                    verdict.guardrail_hits.append(
                        f"server_stop:{result.get('message') or 'stop failed'}"
                    )
        except Exception as exc:  # noqa: BLE001
            if verdict is not None:
                verdict.guardrail_hits.append(f"server_stop:{exc}")

    def run_pipeline(
        self,
        session: SessionKind = "manual",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_base_url: str | None = None,
        use_llm: bool = True,
        mark_schedule_complete: bool = False,
        ensure_servers: bool = True,
        stop_servers_after: bool = True,
    ) -> ManagerVerdict:
        if embedding_base_url:
            self.servers.embedding_base_url = embedding_base_url.rstrip("/")
            self.sentiment = SentimentAnalyst(
                embedding_base_url=self.servers.embedding_base_url,
            )
            self.chroma = ChromaMemoryStore(
                embedding_base_url=self.servers.embedding_base_url,
            )
        if base_url:
            self.servers.chat_base_url = base_url.rstrip("/")

        statuses: dict[str, AgentRunStatus] = {
            "manager": AgentRunStatus(
                agent_id="manager",
                status="running",
                started_at=datetime.now(timezone.utc),
            ),
            "fundamentals_analyst": AgentRunStatus(agent_id="fundamentals_analyst"),
            "sentiment_analyst": AgentRunStatus(agent_id="sentiment_analyst"),
            "news_analyst": AgentRunStatus(agent_id="news_analyst"),
            "technical_analyst": AgentRunStatus(agent_id="technical_analyst"),
            "monitor": AgentRunStatus(agent_id="monitor"),
        }

        server_messages: list[str] = []
        managed_servers = False

        def _finish(verdict: ManagerVerdict) -> ManagerVerdict:
            if managed_servers and stop_servers_after:
                self._stop_servers_after_run(verdict)
            return verdict

        if ensure_servers and use_llm:
            managed_servers = True
            try:
                status = self.servers.ensure_ready()
                server_messages.append(
                    f"Servers ready chat={status['chat']['ok']} embed={status['embed']['ok']}"
                )
            except Exception as exc:  # noqa: BLE001
                statuses["manager"].status = "error"
                statuses["manager"].error = str(exc)
                statuses["manager"].finished_at = datetime.now(timezone.utc)
                return _finish(
                    ManagerVerdict(
                        accepted=False,
                        session=session,
                        as_of=datetime.now(timezone.utc),
                        messages=["Server ensure_ready failed; pipeline aborted."],
                        guardrail_hits=[str(exc)],
                        monitor_bundle=None,
                        disclaimer=DISCLAIMER,
                        agent_statuses=list(statuses.values()),
                    )
                )

        raw_store = RawStore.start_run(session)
        run_id = raw_store.run_id

        contracts = self._contracts(session)
        briefs, errors = self._run_research(
            contracts, raw_store=raw_store, statuses=statuses
        )

        raw_store.write_meta(
            {
                "finished_research_at": datetime.now(timezone.utc).isoformat(),
                "statuses": {
                    k: v.model_dump(mode="json") for k, v in statuses.items()
                },
            }
        )

        if not briefs:
            statuses["manager"].status = "error"
            statuses["manager"].finished_at = datetime.now(timezone.utc)
            return _finish(
                ManagerVerdict(
                    accepted=False,
                    session=session,
                    as_of=datetime.now(timezone.utc),
                    messages=server_messages + ["No research briefs produced."],
                    guardrail_hits=errors,
                    monitor_bundle=None,
                    disclaimer=DISCLAIMER,
                    run_id=run_id,
                    agent_statuses=list(statuses.values()),
                    raw_root=str(raw_store.root),
                )
            )

        # Upsert research docs into Chroma (serial — not thread-safe concurrently)
        for brief in briefs:
            docs = (brief.metrics or {}).pop("chroma_documents", None) or []
            if not docs:
                continue
            try:
                self.chroma.upsert_documents(
                    list(docs),
                    run_id=run_id,
                    agent_id=brief.agent_id,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"chroma_upsert:{brief.agent_id}:{exc}")

        # Related past (+ just-ingested) memory for forecasting
        memory_hits = []
        try:
            claim_bits = " ".join(
                c.text for b in briefs for c in b.claims[:2]
            )[:800]
            memory_hits = self.chroma.query_related(
                f"{FUND_NAME} {session} {claim_bits}",
                n=8,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"memory_query:{exc}")

        statuses["monitor"].status = "running"
        statuses["monitor"].started_at = datetime.now(timezone.utc)
        try:
            bundle = self.monitor.execute(
                session=session,
                briefs=briefs,
                horizon_days=DEFAULT_HORIZON_DAYS,
                base_url=base_url,
                api_key=api_key,
                model=model,
                use_llm=use_llm,
                run_id=run_id,
                memory_hits=memory_hits,
            )
            statuses["monitor"].status = "ok"
            statuses["monitor"].finished_at = datetime.now(timezone.utc)
        except Exception as exc:  # noqa: BLE001
            statuses["monitor"].status = "error"
            statuses["monitor"].error = str(exc)
            statuses["monitor"].finished_at = datetime.now(timezone.utc)
            statuses["manager"].status = "error"
            statuses["manager"].finished_at = datetime.now(timezone.utc)
            return _finish(
                ManagerVerdict(
                    accepted=False,
                    session=session,
                    as_of=datetime.now(timezone.utc),
                    messages=server_messages + ["Monitor failed."],
                    guardrail_hits=errors + [str(exc)],
                    monitor_bundle=None,
                    disclaimer=DISCLAIMER,
                    run_id=run_id,
                    agent_statuses=list(statuses.values()),
                    raw_root=str(raw_store.root),
                )
            )

        guardrail_hits = list(errors)
        if bundle.horizon_days != DEFAULT_HORIZON_DAYS:
            guardrail_hits.append(
                f"Monitor horizon {bundle.horizon_days} != required {DEFAULT_HORIZON_DAYS}"
            )
        if not bundle.summary_markdown:
            guardrail_hits.append("Monitor summary missing")
        if DISCLAIMER.split("—")[0].strip().lower() not in bundle.disclaimer.lower():
            bundle.disclaimer = DISCLAIMER

        accepted = bundle.forecast is not None and bool(bundle.summary_markdown)

        # Persist memory summary
        try:
            summary = build_run_summary(
                run_id=run_id,
                session=session,
                briefs=briefs,
                monitor_summary=bundle.summary_markdown,
                memory_hits=memory_hits,
            )
            append_summary_file(summary, run_id=run_id, session=session)
            self.chroma.upsert_memory(
                run_id=run_id,
                session=session,
                summary=summary,
                metadata={"accepted": accepted, "fund": FUND_NAME},
            )
        except Exception as exc:  # noqa: BLE001
            guardrail_hits.append(f"memory_write:{exc}")

        statuses["manager"].status = "ok" if accepted else "error"
        statuses["manager"].finished_at = datetime.now(timezone.utc)

        verdict = ManagerVerdict(
            accepted=accepted,
            session=session,
            as_of=datetime.now(timezone.utc),
            messages=server_messages
            + [
                f"Research analysts completed: {len(briefs)}",
                f"Monitor NAV records: {bundle.nav_records}",
                f"Forecast source: {bundle.forecast.get('source')}",
                f"Horizon days: {bundle.horizon_days}",
                f"Memory hits: {len(memory_hits)}",
                f"Run id: {run_id}",
            ],
            guardrail_hits=guardrail_hits,
            monitor_bundle=bundle,
            disclaimer=DISCLAIMER,
            run_id=run_id,
            agent_statuses=list(statuses.values()),
            memory_hit_count=len(memory_hits),
            raw_root=str(raw_store.root),
        )

        try:
            persist_verdict_json(verdict.model_dump(mode="json"), run_id)
        except Exception as exc:  # noqa: BLE001
            verdict.guardrail_hits.append(f"verdict_persist:{exc}")

        raw_store.write_meta(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "accepted": accepted,
                "statuses": {
                    k: v.model_dump(mode="json") for k, v in statuses.items()
                },
            }
        )

        if mark_schedule_complete and session in ("open", "close") and accepted:
            self.runs.mark_completed(
                SessionRunState(
                    trade_date=today_ist().isoformat(),
                    session=session,
                    completed_at=datetime.now(timezone.utc),
                    accepted=True,
                    notes="; ".join(verdict.messages),
                )
            )
        return _finish(verdict)
