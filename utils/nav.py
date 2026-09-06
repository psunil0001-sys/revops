"""NAV history fetching and normalization (mfapi.in or local CSV)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from utils.config import MFAPI_NAV_URL
from utils.funds import FundConfig


def get_date_range(
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    default_start: date | None = None,
) -> tuple[date, date]:
    """Return inclusive start/end dates."""
    start = start_date or default_start or date(2000, 1, 1)
    end = end_date or date.today()
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}.")
    return start, end


def filter_nav_by_dates(
    nav_df: pd.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    default_start: date | None = None,
) -> pd.DataFrame:
    """Filter a normalized NAV DataFrame to an inclusive date window."""
    start, end = get_date_range(
        start_date, end_date, default_start=default_start
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = nav_df[
        (nav_df["Date"] >= start_ts) & (nav_df["Date"] <= end_ts)
    ].reset_index(drop=True)
    return filtered


def fetch_nav_history_for_fund(
    fund: FundConfig,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Load NAV for a fund via mfapi or local CSV."""
    if fund.nav_source == "file":
        return fetch_nav_from_file(
            fund.nav_file_path(),
            start_date=start_date or fund.inception_date,
            end_date=end_date,
            default_start=fund.inception_date,
        )
    if not fund.scheme_code:
        raise ValueError(f"Fund {fund.id} missing scheme_code for mfapi")
    return fetch_nav_history(
        scheme_code=fund.scheme_code,
        start_date=start_date or fund.inception_date,
        end_date=end_date,
        default_start=fund.inception_date,
        timeout=timeout,
    )


def fetch_nav_from_file(
    path: Path | None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    default_start: date | None = None,
) -> pd.DataFrame:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"NAV file missing: {path}")
    df = pd.read_csv(path)
    # File NAVs use ISO dates (YYYY-MM-DD); mfapi uses day-first.
    nav_df = normalize_nav_dataframe(df, dayfirst=False)
    nav_df = filter_nav_by_dates(
        nav_df, start_date, end_date, default_start=default_start
    )
    if nav_df.empty:
        start, end = get_date_range(
            start_date, end_date, default_start=default_start
        )
        raise ValueError(f"No NAV records in {path} between {start} and {end}.")
    return nav_df


def fetch_nav_history(
    scheme_code: str,
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    default_start: date | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch NAV history from mfapi.in and return a normalized DataFrame.

    Fetches full history then filters to [start_date, end_date].
    """
    url = MFAPI_NAV_URL.format(scheme_code=scheme_code)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "revops-portfolio-tracker/1.0",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(
            f"NAV API returned HTTP {exc.code} for scheme {scheme_code}."
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach NAV API: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "NAV API did not return valid JSON.\n"
            f"Response preview:\n{raw[:2000]}"
        ) from exc

    if isinstance(payload, dict) and payload.get("status") not in (
        None,
        "SUCCESS",
        "success",
        "OK",
        "ok",
    ):
        if not payload.get("data"):
            raise RuntimeError(
                f"NAV API status={payload.get('status')!r} with no data."
            )

    nav_df = parse_nav_response(payload)
    nav_df = normalize_nav_dataframe(nav_df)
    nav_df = filter_nav_by_dates(
        nav_df, start_date, end_date, default_start=default_start
    )

    if nav_df.empty:
        start, end = get_date_range(
            start_date, end_date, default_start=default_start
        )
        raise ValueError(f"No NAV records found between {start} and {end}.")

    return nav_df


def parse_nav_response(data: Any) -> pd.DataFrame:
    """Extract a list of NAV records from mfapi / Kotak-shaped JSON."""
    records = None

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in (
            "data",
            "Data",
            "result",
            "Result",
            "response",
            "Response",
            "nav",
            "NAV",
            "schemeNav",
            "SchemeNav",
        ):
            value = data.get(key)
            if isinstance(value, list):
                records = value
                break

        if records is None:
            records = _find_first_list(data)

    if not records:
        raise ValueError("No NAV records found in API response.")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("NAV DataFrame is empty.")
    return df


def _find_first_list(obj: Any) -> list | None:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_first_list(value)
            if found is not None:
                return found
    return None


def normalize_nav_dataframe(
    df: pd.DataFrame,
    *,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Normalize arbitrary NAV JSON/CSV columns to Date + NAV."""
    df = df.copy()
    columns = list(df.columns)

    normalized = {
        str(col).strip().lower().replace("_", "").replace(" ", "").replace("-", ""): col
        for col in columns
    }

    nav_column = None
    for candidate in ("nav", "netassetvalue", "navvalue", "currentnav"):
        if candidate in normalized:
            nav_column = normalized[candidate]
            break
    if nav_column is None:
        for col in columns:
            if "nav" in str(col).lower():
                nav_column = col
                break

    date_column = None
    for candidate in ("date", "navdate", "effectivedate", "valuedate"):
        if candidate in normalized:
            date_column = normalized[candidate]
            break
    if date_column is None:
        for col in columns:
            if "date" in str(col).lower():
                date_column = col
                break

    if nav_column is None:
        raise ValueError(f"NAV column not found. Columns: {columns}")
    if date_column is None:
        raise ValueError(f"Date column not found. Columns: {columns}")

    result = pd.DataFrame(
        {
            "Date": df[date_column],
            "NAV": df[nav_column],
        }
    )
    result["NAV"] = pd.to_numeric(result["NAV"], errors="coerce")
    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
        dayfirst=dayfirst,
    )
    result = result.dropna(subset=["Date", "NAV"])
    result = result[result["NAV"] > 0]
    result["NAV"] = result["NAV"].round(4)

    return (
        result.drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
