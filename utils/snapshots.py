"""Daily portfolio snapshot persistence under data/snapshots/."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from utils.config import DATA_DIR
from utils.funds import FundConfig

SNAPSHOTS_DIR = DATA_DIR / "snapshots"


def _ensure_dir() -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOTS_DIR


def _date_str(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def snapshot_path(snap_date: date | datetime | str) -> Path:
    return _ensure_dir() / f"{_date_str(snap_date)}.json"


def save_snapshot(
    snap_date: date | datetime | str,
    holdings: Sequence[Mapping[str, Any]],
    total_value: float,
) -> Path:
    """Persist a portfolio snapshot for ``snap_date`` (overwrites same day)."""
    path = snapshot_path(snap_date)
    payload = {
        "date": _date_str(snap_date),
        "total_value": round(float(total_value), 4),
        "holdings": [
            {
                "id": h["id"],
                "name": h["name"],
                "units": round(float(h["units"]), 6),
                "nav": round(float(h["nav"]), 4),
                "value": round(float(h["value"]), 4),
            }
            for h in holdings
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_snapshot(snap_date: date | datetime | str) -> dict[str, Any] | None:
    path = snapshot_path(snap_date)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_snapshots() -> list[str]:
    """Return snapshot dates (YYYY-MM-DD) newest first."""
    _ensure_dir()
    dates = sorted(
        (p.stem for p in SNAPSHOTS_DIR.glob("*.json") if p.is_file()),
        reverse=True,
    )
    return dates


def latest_snapshot() -> dict[str, Any] | None:
    dates = list_snapshots()
    if not dates:
        return None
    return load_snapshot(dates[0])


def day_over_day(
    prev: Mapping[str, Any] | None,
    curr: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare two snapshots; returns delta_value / delta_pct / per-holding deltas."""
    if not curr:
        return {
            "delta_value": None,
            "delta_pct": None,
            "prev_total": None,
            "curr_total": None,
            "holdings": [],
        }
    curr_total = float(curr.get("total_value") or 0.0)
    if not prev:
        return {
            "delta_value": None,
            "delta_pct": None,
            "prev_total": None,
            "curr_total": curr_total,
            "holdings": [],
        }
    prev_total = float(prev.get("total_value") or 0.0)
    delta_value = round(curr_total - prev_total, 4)
    delta_pct = (
        round((curr_total / prev_total - 1.0) * 100.0, 4) if prev_total else None
    )
    prev_by_id = {h["id"]: h for h in prev.get("holdings") or []}
    holding_deltas = []
    for h in curr.get("holdings") or []:
        prior = prev_by_id.get(h["id"])
        prior_val = float(prior["value"]) if prior else None
        curr_val = float(h["value"])
        holding_deltas.append(
            {
                "id": h["id"],
                "name": h["name"],
                "prev_value": prior_val,
                "curr_value": curr_val,
                "delta_value": (
                    round(curr_val - prior_val, 4) if prior_val is not None else None
                ),
            }
        )
    return {
        "delta_value": delta_value,
        "delta_pct": delta_pct,
        "prev_total": prev_total,
        "curr_total": curr_total,
        "holdings": holding_deltas,
    }


def build_snapshot_from_funds(
    funds: Iterable[FundConfig],
    nav_by_id: Mapping[str, float],
) -> tuple[list[dict[str, Any]], float]:
    """Build holdings rows + total from fund configs and current NAVs.

    Missing NAV falls back to ``purchase_nav``.
    """
    holdings: list[dict[str, Any]] = []
    total = 0.0
    for fund in funds:
        nav = float(nav_by_id.get(fund.id, fund.purchase_nav))
        units = float(fund.units)
        value = round(units * nav, 4)
        total += value
        holdings.append(
            {
                "id": fund.id,
                "name": fund.name,
                "units": units,
                "nav": nav,
                "value": value,
            }
        )
    return holdings, round(total, 4)
