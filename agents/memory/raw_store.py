"""Persist raw research payloads under data/raw/{run_id}/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from utils.config import RAW_DATA_DIR


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class RawStore:
    """One folder per pipeline run; agents write JSON payloads by name."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.root = Path(root or RAW_DATA_DIR) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths: list[str] = []

    @classmethod
    def start_run(
        cls,
        session: str,
        *,
        root: Path | None = None,
    ) -> "RawStore":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{session}_{uuid4().hex[:8]}"
        store = cls(run_id, root=root)
        store.write_meta(
            {
                "run_id": run_id,
                "session": session,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "statuses": {},
            }
        )
        return store

    def agent_dir(self, agent: str) -> Path:
        path = self.root / agent
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_raw(self, agent: str, name: str, payload: Any) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        if not safe.endswith(".json"):
            safe = f"{safe}.json"
        path = self.agent_dir(agent) / safe
        path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
        self.paths.append(str(path))
        return path

    def write_meta(self, payload: dict[str, Any]) -> Path:
        path = self.root / "meta.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(payload)
        path.write_text(
            json.dumps(existing, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    def relative_paths(self) -> list[str]:
        return list(self.paths)
