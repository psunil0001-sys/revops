"""Build and persist run memory summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.contracts import MemoryHit, ResearchBrief
from utils.config import FUND_NAME, MEMORY_DATA_DIR


def build_run_summary(
    *,
    run_id: str,
    session: str,
    briefs: list[ResearchBrief],
    monitor_summary: str,
    memory_hits: list[MemoryHit] | None = None,
) -> str:
    lines = [
        f"Run {run_id} session={session} fund={FUND_NAME}",
        f"As of {datetime.now(timezone.utc).isoformat()}",
        "",
        "Analyst briefs:",
    ]
    for brief in briefs:
        top = brief.claims[0].text if brief.claims else "no claims"
        lines.append(
            f"- {brief.agent_id}: conf={brief.confidence:.2f} — {top}"
        )
    if memory_hits:
        lines.append("")
        lines.append(f"Related memory hits used: {len(memory_hits)}")
        for hit in memory_hits[:5]:
            lines.append(f"- [{hit.collection}] {hit.title}")
    lines.append("")
    lines.append("Monitor summary:")
    lines.append(monitor_summary[:3000])
    return "\n".join(lines)


def append_summary_file(summary: str, *, run_id: str, session: str) -> Path:
    MEMORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DATA_DIR / "summaries.jsonl"
    record = {
        "run_id": run_id,
        "session": session,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


def persist_verdict_json(verdict: dict[str, Any], run_id: str) -> Path:
    from utils.config import AGENT_RUNS_DIR

    AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = AGENT_RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return path


def load_latest_verdict() -> dict[str, Any] | None:
    from utils.config import AGENT_RUNS_DIR

    if not AGENT_RUNS_DIR.is_dir():
        return None
    files = sorted(AGENT_RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
