"""Multi-fund configuration loader (config/funds.yaml)."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from utils.config import _REPO_ROOT

FUNDS_YAML_PATH = _REPO_ROOT / "config" / "funds.yaml"

FundType = Literal["mutual_fund", "ulip"]
NavSource = Literal["mfapi", "file"]


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


def clear_funds_cache() -> None:
    load_funds.cache_clear()
