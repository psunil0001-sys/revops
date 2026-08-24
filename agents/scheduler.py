"""NSE open/close session scheduler (Asia/Kolkata)."""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from agents.contracts import SessionKind, SessionRunState

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()


def is_weekday(day: date | None = None) -> bool:
    day = day or today_ist()
    return day.weekday() < 5


def due_session(now: datetime | None = None) -> SessionKind | None:
    """
    Return which session should run now if its clock has passed and it's a weekday.

    open due after 09:15; close due after 15:30.
    """
    now = now or now_ist()
    if now.weekday() >= 5:
        return None
    local_t = now.timetz().replace(tzinfo=None)
    if local_t >= MARKET_CLOSE:
        return "close"
    if local_t >= MARKET_OPEN:
        return "open"
    return None


def next_scheduled_label(now: datetime | None = None) -> str:
    now = now or now_ist()
    if now.weekday() >= 5:
        # next Monday open
        return "next weekday 09:15 IST (open)"
    local_t = now.timetz().replace(tzinfo=None)
    if local_t < MARKET_OPEN:
        return f"today {MARKET_OPEN.strftime('%H:%M')} IST (open)"
    if local_t < MARKET_CLOSE:
        return f"today {MARKET_CLOSE.strftime('%H:%M')} IST (close)"
    return "next weekday 09:15 IST (open)"


class SessionRunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("data/agent_runs")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "sessions.jsonl"

    def _key(self, trade_date: str, session: SessionKind) -> str:
        return f"{trade_date}:{session}"

    def load_completed(self) -> set[str]:
        if not self.path.exists():
            return set()
        done: set[str] = set()
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("accepted"):
                    done.add(self._key(payload["trade_date"], payload["session"]))
        return done

    def is_completed(self, trade_date: str, session: SessionKind) -> bool:
        return self._key(trade_date, session) in self.load_completed()

    def mark_completed(self, state: SessionRunState) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(state.model_dump_json() + "\n")

    def last_completed(self) -> SessionRunState | None:
        if not self.path.exists():
            return None
        last: SessionRunState | None = None
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                last = SessionRunState.model_validate_json(line)
        return last


def pending_scheduled_session(store: SessionRunStore | None = None) -> SessionKind | None:
    store = store or SessionRunStore()
    session = due_session()
    if session is None:
        return None
    trade_date = today_ist().isoformat()
    if store.is_completed(trade_date, session):
        return None
    # If close is due and open not done, still allow close (open may have been missed).
    return session
