"""Multi-fund configuration loader (config/funds.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from utils.config import _REPO_ROOT

FUNDS_YAML_PATH = _REPO_ROOT / "config" / "funds.yaml"

FundType = Literal["mutual_fund", "ulip", "nps"]
NavSource = Literal["mfapi", "file"]

# Human labels for dashboard_group ids (fallback: title-cased id).
GROUP_DISPLAY_NAMES: dict[str, str] = {
    "icici_nps": "ICICI Prudential NPS (E+C+G)",
}


class FundConfig(BaseModel):
    id: str
    name: str
    type: FundType = "mutual_fund"
    scheme_code: str | None = None
    sfin: str | None = None
    nav_source: NavSource = "mfapi"
    nav_file: str | None = None
    inception_date: date
    investment: float
    purchase_nav: float
    equity_pct: float = 0.0
    other_pct: float = 0.0
    sectors: dict[str, float] = Field(default_factory=dict)
    research_queries: list[str] = Field(default_factory=list)
    benchmark_symbol: str = "^NSEI"
    dashboard_group: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dashboard_group", "group_id", "ui_group"),
    )

    @field_validator("inception_date", mode="before")
    @classmethod
    def _parse_date(cls, value: Any) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        return date.fromisoformat(str(value))

    @model_validator(mode="after")
    def _validate_source(self) -> "FundConfig":
        if self.nav_source == "mfapi" and not self.scheme_code:
            raise ValueError(f"Fund {self.id}: mfapi requires scheme_code")
        if self.nav_source == "file" and not self.nav_file:
            raise ValueError(f"Fund {self.id}: file nav_source requires nav_file")
        if self.purchase_nav <= 0:
            raise ValueError(f"Fund {self.id}: purchase_nav must be > 0")
        if self.investment <= 0:
            raise ValueError(f"Fund {self.id}: investment must be > 0")
        return self

    @property
    def units(self) -> float:
        return self.investment / self.purchase_nav

    @property
    def display_code(self) -> str:
        if self.scheme_code:
            return self.scheme_code
        if self.sfin:
            return self.sfin
        return self.id

    def sector_dataframe_dict(self) -> dict[str, list]:
        if not self.sectors:
            return {"Sector": ["Equity"], "Allocation": [self.equity_pct or 100.0]}
        return {
            "Sector": list(self.sectors.keys()),
            "Allocation": list(self.sectors.values()),
        }

    def asset_dataframe_dict(self) -> dict[str, list]:
        equity = self.equity_pct
        other = self.other_pct if self.other_pct else max(0.0, 100.0 - equity)
        return {
            "Asset Class": ["Equity", "Others"],
            "Allocation": [equity, other],
        }

    def nav_file_path(self) -> Path | None:
        if not self.nav_file:
            return None
        path = Path(self.nav_file)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return path


@dataclass(frozen=True)
class DashboardGroup:
    """Bundle of funds that share one top-level dashboard tab."""

    group_id: str
    funds: tuple[FundConfig, ...]

    @property
    def display_name(self) -> str:
        return GROUP_DISPLAY_NAMES.get(
            self.group_id,
            self.group_id.replace("_", " ").title(),
        )

    @property
    def fund_ids(self) -> tuple[str, ...]:
        return tuple(f.id for f in self.funds)


DashboardEntry = Union[FundConfig, DashboardGroup]


@lru_cache(maxsize=1)
def load_funds(path: str | None = None) -> tuple[FundConfig, ...]:
    import yaml

    yaml_path = Path(path) if path else FUNDS_YAML_PATH
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Funds config missing: {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    items = payload.get("funds") or []
    if not items:
        raise ValueError(f"No funds listed in {yaml_path}")
    funds = tuple(FundConfig.model_validate(item) for item in items)
    ids = [f.id for f in funds]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate fund ids in funds.yaml")
    return funds


def get_fund(fund_id: str) -> FundConfig:
    for fund in load_funds():
        if fund.id == fund_id:
            return fund
    raise KeyError(f"Unknown fund id: {fund_id}")


def get_funds_in_group(group_id: str) -> tuple[FundConfig, ...]:
    """Return funds belonging to ``group_id`` in YAML order."""
    return tuple(f for f in load_funds() if f.dashboard_group == group_id)


def group_display_name(group_id: str) -> str:
    return GROUP_DISPLAY_NAMES.get(
        group_id,
        group_id.replace("_", " ").title(),
    )


def iter_dashboard_entries(
    funds: tuple[FundConfig, ...] | None = None,
) -> Iterator[DashboardEntry]:
    """Yield top-level dashboard entries: ungrouped funds or group bundles.

    Funds that share a ``dashboard_group`` collapse into one ``DashboardGroup``
    (emitted once, at the position of the first member). Ungrouped funds yield
    as themselves in YAML order.
    """
    items = funds if funds is not None else load_funds()
    seen_groups: set[str] = set()
    for fund in items:
        gid = fund.dashboard_group
        if not gid:
            yield fund
            continue
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        members = tuple(f for f in items if f.dashboard_group == gid)
        yield DashboardGroup(group_id=gid, funds=members)




# Synthetic sidebar target for the combined ICICI NPS sleeve (not a fund id).
ACTIVE_TARGET_ICICI_NPS = "icici_nps"


def is_group_target(target_id: str) -> bool:
    """True when ``target_id`` is a dashboard_group id (e.g. icici_nps)."""
    if not target_id:
        return False
    if target_id in GROUP_DISPLAY_NAMES:
        return True
    return any(f.dashboard_group == target_id for f in load_funds())


def resolve_target_funds(target_id: str) -> tuple[FundConfig, ...]:
    """Resolve sidebar ``active_target`` to one or more FundConfig rows."""
    if is_group_target(target_id):
        members = get_funds_in_group(target_id)
        if not members:
            raise KeyError(f"Unknown group target: {target_id}")
        return members
    return (get_fund(target_id),)


def default_focus_fund_id(target_id: str) -> str:
    """Pick the focus fund id for a target (prefer Scheme E inside ICICI)."""
    funds = resolve_target_funds(target_id)
    for f in funds:
        if f.id.endswith("_e") or "Scheme E" in f.name:
            return f.id
    return funds[0].id


def iter_active_targets(
    funds: tuple[FundConfig, ...] | None = None,
) -> list[tuple[str, str]]:
    """Sidebar Active-target options: ungrouped funds + one row per group.

    Returns list of ``(target_id, label)`` in YAML / dashboard order.
    """
    items = funds if funds is not None else load_funds()
    out: list[tuple[str, str]] = []
    seen_groups: set[str] = set()
    for fund in items:
        gid = fund.dashboard_group
        if not gid:
            out.append((fund.id, fund.name))
            continue
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        out.append((gid, group_display_name(gid)))
    return out


def target_label(target_id: str) -> str:
    for tid, label in iter_active_targets():
        if tid == target_id:
            return label
    if is_group_target(target_id):
        return group_display_name(target_id)
    return get_fund(target_id).name


def clear_funds_cache() -> None:
    load_funds.cache_clear()
