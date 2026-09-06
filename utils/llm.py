"""OpenAI-compatible local LLM client for NAV forecasting."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from utils.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLM_CHAT_TIMEOUT_S,
    LLM_TERMINAL_BAND_FRAC,
)
from utils.features import assert_forecast_ready, statistical_baseline_forecast


def resolve_llm_settings(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Resolve LLM settings from args, env, then defaults."""
    return {
        "base_url": (
            (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL)
            .rstrip("/")
        ),
        "api_key": api_key or os.getenv("LLM_API_KEY") or "ollama",
        "model": model or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL,
    }


def chat_completions(
    messages: list[dict[str, str]],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout: float | None = None,
) -> str:
    """Call OpenAI-compatible POST {base}/chat/completions with a finite timeout."""
    settings = resolve_llm_settings(base_url, api_key, model)
    url = f"{settings['base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings['api_key']}",
    }
    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    # Finite timeout (default 120s) so Forecast / Manager cannot hang indefinitely.
    read_timeout = LLM_CHAT_TIMEOUT_S if timeout is None else float(timeout)
    response = requests.post(
        url, headers=headers, json=payload, timeout=read_timeout
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response shape: {data!r}") from exc


def check_llm_health(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    heartbeat_timeout: float = 10.0,
) -> dict[str, object]:
    """
    Lightweight heartbeat against OpenAI-compatible /models.

    Uses a short timeout so the UI heartbeat cannot hang the app.
    """
    from datetime import datetime, timezone

    settings = resolve_llm_settings(base_url, api_key, model)
    url = f"{settings['base_url']}/models"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings['api_key']}",
    }
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=heartbeat_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        models = []
        if isinstance(payload, dict):
            data = payload.get("data") or []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("id"):
                        models.append(str(item["id"]))
        model_ok = (
            not settings["model"]
            or settings["model"] in models
            or any(settings["model"] in m for m in models)
            or not models  # some servers return empty list but are still healthy
        )
        return {
            "ok": True,
            "reachable": True,
            "model_listed": model_ok,
            "models": models,
            "requested_model": settings["model"],
            "base_url": settings["base_url"],
            "checked_at": checked_at,
            "message": (
                "Model endpoint healthy"
                if model_ok
                else f"Endpoint healthy, but model {settings['model']!r} not listed"
            ),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reachable": False,
            "model_listed": False,
            "models": [],
            "requested_model": settings["model"],
            "base_url": settings["base_url"],
            "checked_at": checked_at,
            "message": "Model endpoint unhealthy",
            "error": str(exc),
        }



def check_embedding_health(
    *,
    embedding_base_url: str | None = None,
    heartbeat_timeout: float = 10.0,
) -> dict[str, object]:
    """Heartbeat against the dedicated embedding server /models endpoint."""
    from datetime import datetime, timezone

    base = (
        embedding_base_url
        or os.getenv("EMBEDDING_BASE_URL")
        or DEFAULT_EMBEDDING_BASE_URL
    ).rstrip("/")
    chat = (os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
    checked_at = datetime.now(timezone.utc).isoformat()
    if base == chat:
        return {
            "ok": False,
            "reachable": False,
            "base_url": base,
            "checked_at": checked_at,
            "message": "Embedding URL must not equal chat URL",
            "error": "refused_chat_url",
            "models": [],
        }
    url = f"{base}/models"
    try:
        response = requests.get(url, timeout=heartbeat_timeout)
        response.raise_for_status()
        payload = response.json()
        models = []
        if isinstance(payload, dict):
            for item in payload.get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
        return {
            "ok": True,
            "reachable": True,
            "base_url": base,
            "checked_at": checked_at,
            "message": "Embedding endpoint healthy",
            "error": None,
            "models": models,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reachable": False,
            "base_url": base,
            "checked_at": checked_at,
            "message": "Embedding endpoint unhealthy",
            "error": str(exc),
            "models": [],
        }


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output (handles fences)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _validate_forecast(forecast: dict[str, Any], horizon_days: int) -> dict[str, Any]:
    scenarios = forecast.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("Forecast missing scenarios object.")

    normalized: dict[str, Any] = {
        "horizon_days": int(forecast.get("horizon_days") or horizon_days),
        "horizon_unit": "trading_days",
        "source": forecast.get("source") or "llm",
        "scenarios": {},
        "disclaimer": forecast.get("disclaimer")
        or (
            "Illustrative LLM-refined scenarios only — not investment advice. "
            "Past performance does not guarantee future results."
        ),
    }

    for name in ("bear", "base", "bull"):
        payload = scenarios.get(name)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing scenario: {name}")
        path = payload.get("nav_path") or []
        if not isinstance(path, list) or len(path) < max(1, horizon_days // 3):
            raise ValueError(f"Scenario {name} has insufficient nav_path points.")
        clean_path = []
        for point in path:
            clean_path.append(
                {
                    "date": str(point["date"])[:10],
                    "nav": round(float(point["nav"]), 4),
                }
            )
        normalized["scenarios"][name] = {
            "nav_path": clean_path,
            "rationale": str(payload.get("rationale") or "").strip()
            or f"{name.capitalize()} scenario from local LLM.",
        }
    return normalized


def _terminal(scenario: dict[str, Any]) -> float:
    path = scenario.get("nav_path") or []
    if not path:
        raise ValueError("Empty nav_path")
    return float(path[-1]["nav"])


def _rescale_path_to_terminal(
    path: list[dict[str, Any]],
    new_terminal: float,
) -> list[dict[str, Any]]:
    if not path:
        return path
    old_terminal = float(path[-1]["nav"])
    if old_terminal <= 0:
        scale = 1.0
    else:
        scale = new_terminal / old_terminal
    first = float(path[0]["nav"])
    # Keep path shape: scale deviations from start toward new end.
    out = []
    n = len(path)
    for i, point in enumerate(path):
        t = (i + 1) / n
        nav = float(point["nav"])
        # Blend multiplicative scale by progress along path.
        scaled = nav * (1.0 + (scale - 1.0) * t)
        out.append(
            {
                "date": point["date"],
                "nav": round(max(scaled, 0.01), 4),
            }
        )
    out[-1]["nav"] = round(max(new_terminal, 0.01), 4)
    # Avoid unused warning for first
    _ = first
    return out


def _clamp_to_baseline(
    llm_forecast: dict[str, Any],
    baseline: dict[str, Any],
    *,
    band_frac: float = LLM_TERMINAL_BAND_FRAC,
) -> dict[str, Any]:
    """
    Clamp LLM terminals near baseline; enforce bear <= base <= bull.

    Returns clamped forecast, or raises ValueError if irreparable.
    """
    base_sc = baseline.get("scenarios") or {}
    llm_sc = llm_forecast.get("scenarios") or {}
    b_bear = _terminal(base_sc["bear"])
    b_base = _terminal(base_sc["base"])
    b_bull = _terminal(base_sc["bull"])
    half_width = 0.5 * abs(b_bull - b_bear)
    band = max(half_width * band_frac, abs(b_base) * 0.002, 1e-4)

    targets = {}
    for name, b_term in (("bear", b_bear), ("base", b_base), ("bull", b_bull)):
        llm_term = _terminal(llm_sc[name])
        targets[name] = float(min(max(llm_term, b_term - band), b_term + band))

    # Enforce order after individual clamps.
    targets["base"] = max(targets["base"], targets["bear"])
    targets["bull"] = max(targets["bull"], targets["base"])
    if not (targets["bear"] <= targets["base"] <= targets["bull"]):
        raise ValueError("Unable to enforce bear <= base <= bull after clamp")

    clamped = dict(llm_forecast)
    clamped_scenarios = {}
    for name in ("bear", "base", "bull"):
        path = list((llm_sc[name] or {}).get("nav_path") or [])
        # Prefer baseline dates for trading-day alignment.
        base_path = list((base_sc[name] or {}).get("nav_path") or [])
        if base_path and len(base_path) == len(path):
            aligned = [
                {"date": bp["date"], "nav": pp["nav"]}
                for bp, pp in zip(base_path, path)
            ]
        elif base_path:
            aligned = [
                {"date": bp["date"], "nav": bp["nav"]} for bp in base_path
            ]
            # Seed with LLM relative move then rescale
            aligned = _rescale_path_to_terminal(aligned, targets[name])
        else:
            aligned = path
        clamped_scenarios[name] = {
            "nav_path": _rescale_path_to_terminal(aligned, targets[name]),
            "rationale": (
                str((llm_sc[name] or {}).get("rationale") or "").strip()
                + f" | Clamped terminal to baseline band (±{band_frac:.0%} half-width)."
            ).strip(" |"),
        }
    clamped["scenarios"] = clamped_scenarios
    clamped["source"] = "llm_clamped"
    clamped["horizon_unit"] = "trading_days"
    clamped["llm_clamped"] = True
    return clamped


def build_forecast_messages(
    features: dict[str, Any],
    horizon_days: int,
    baseline: dict[str, Any],
    market_context: dict[str, Any] | None = None,
    agent_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    compact_features = {
        k: v
        for k, v in features.items()
        if k not in ("recent_series", "daily_returns")
    }
    # Drop bulky close series from market_context in the prompt.
    market_compact = None
    if market_context:
        market_compact = {
            k: v for k, v in market_context.items() if k != "closes"
        }
    recent_tail = features.get("recent_series", [])[-45:]
    b_bear = _terminal((baseline.get("scenarios") or {})["bear"])
    b_bull = _terminal((baseline.get("scenarios") or {})["bull"])
    half = 0.5 * abs(b_bull - b_bear)
    band = max(half * LLM_TERMINAL_BAND_FRAC, 1e-4)
    user_payload = {
        "task": "Refine bootstrap NAV scenarios (trading days)",
        "horizon_trading_days": horizon_days,
        "features": compact_features,
        "recent_nav_tail": recent_tail,
        "bootstrap_baseline": baseline,
        "market_context": market_compact,
        "agent_context": agent_context,
        "output_schema": {
            "horizon_days": horizon_days,
            "scenarios": {
                "bear": {
                    "nav_path": [{"date": "YYYY-MM-DD", "nav": 0.0}],
                    "rationale": "string",
                },
                "base": {
                    "nav_path": [{"date": "YYYY-MM-DD", "nav": 0.0}],
                    "rationale": "string",
                },
                "bull": {
                    "nav_path": [{"date": "YYYY-MM-DD", "nav": 0.0}],
                    "rationale": "string",
                },
            },
            "disclaimer": "string",
        },
        "rules": [
            "Return ONLY valid JSON matching output_schema.",
            "Use exactly bear, base, and bull scenarios with bear <= base <= bull terminals.",
            f"Each nav_path must include about {horizon_days} trading-day points after last_date.",
            "NAV values must use 4 decimal places and stay positive.",
            "Ground rationales in features, bootstrap_baseline, and agent_context; do not invent holdings.",
            "Use research briefs/sentiment as qualitative context only; do not claim guaranteed returns.",
            f"Refine bootstrap terminals only within ±{band:.4f} NAV of each baseline terminal "
            f"(band = {LLM_TERMINAL_BAND_FRAC:.0%} of half bull−bear width).",
            "Keep path shapes close to the bootstrap baseline.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a careful quantitative assistant for mutual-fund NAV scenario "
                "analysis. You refine bootstrap baselines within a tight band. "
                "You never give personalized investment advice. "
                "Respond with JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=True),
        },
    ]


def forecast_nav(
    features: dict[str, Any],
    horizon_days: int = 30,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    market_context: dict[str, Any] | None = None,
    agent_context: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Produce bull/base/bear NAV forecast.

    Baseline is bootstrap (trading days). When use_llm=True, LLM may refine
    within a clamp band; irreparable LLM output falls back to baseline and
    keeps llm_raw_response.
    """
    assert_forecast_ready(features)
    baseline = statistical_baseline_forecast(features, horizon_days=horizon_days)
    if not use_llm:
        baseline = dict(baseline)
        baseline["llm_raw_response"] = None
        return baseline

    messages = build_forecast_messages(
        features,
        horizon_days,
        baseline,
        market_context=market_context,
        agent_context=agent_context,
    )
    try:
        raw = chat_completions(
            messages,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001 — transport / server errors
        fallback = dict(baseline)
        fallback["llm_raw_response"] = None
        fallback["llm_fallback"] = True
        fallback["source"] = "bootstrap_baseline"
        fallback["disclaimer"] = (
            "LLM unreachable or request failed — showing bootstrap baseline. "
            f"({type(exc).__name__}: {exc}) "
            "Illustrative only — not investment advice."
        )
        return fallback
    try:
        parsed = extract_json_object(raw)
        validated = _validate_forecast(parsed, horizon_days)
        clamped = _clamp_to_baseline(validated, baseline)
        clamped["llm_raw_response"] = raw
        clamped["llm_fallback"] = False
        return clamped
    except Exception:
        fallback = dict(baseline)
        fallback["llm_raw_response"] = raw
        fallback["llm_fallback"] = True
        fallback["source"] = "bootstrap_baseline"
        fallback["disclaimer"] = (
            "LLM refine failed or violated clamps — showing bootstrap baseline. "
            "Illustrative only — not investment advice."
        )
        return fallback
