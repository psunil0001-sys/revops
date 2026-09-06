"""Build and persist run memory summaries (per fund)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.contracts import MemoryHit, ResearchBrief
from utils.config import AGENT_RUNS_DIR, MEMORY_DATA_DIR


def build_run_summary(
    *,
    run_id: str,
    session: str,
    fund_name: str,
    fund_id: str,
    briefs: list[ResearchBrief],
    monitor_summary: str,
    memory_hits: list[MemoryHit] | None = None,
) -> str:
    lines = [
        f"Run {run_id} session={session} fund_id={fund_id} fund={fund_name}",
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


def append_summary_file(
    summary: str,
    *,
    run_id: str,
    session: str,
    fund_id: str,
) -> Path:
    MEMORY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DATA_DIR / "summaries.jsonl"
    record = {
        "run_id": run_id,
        "fund_id": fund_id,
        "session": session,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


def persist_verdict_json(
    verdict: dict[str, Any],
    run_id: str,
    *,
    fund_id: str,
) -> Path:
    out_dir = AGENT_RUNS_DIR / fund_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return path


def load_latest_verdict(fund_id: str | None = None) -> dict[str, Any] | None:
    if fund_id:
        folder = AGENT_RUNS_DIR / fund_id
        if not folder.is_dir():
            return None
        files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    else:
        if not AGENT_RUNS_DIR.is_dir():
            return None
        files = sorted(AGENT_RUNS_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
