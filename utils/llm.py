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
)
from utils.features import statistical_baseline_forecast


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
) -> str:
    """Call OpenAI-compatible POST {base}/chat/completions (no read timeout)."""
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
    # No overall timeout — local models may take several minutes.
    response = requests.post(url, headers=headers, json=payload, timeout=None)
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
        "source": forecast.get("source") or "llm",
        "scenarios": {},
        "disclaimer": forecast.get("disclaimer")
        or (
            "LLM-assisted projection only — not investment advice. "
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
        if k != "recent_series"
    }
    recent_tail = features.get("recent_series", [])[-45:]
    user_payload = {
        "task": "Forecast mutual fund NAV scenarios",
        "horizon_calendar_days": horizon_days,
        "features": compact_features,
        "recent_nav_tail": recent_tail,
        "statistical_baseline": baseline,
        "market_context": market_context,
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
            "Use exactly bear, base, and bull scenarios.",
            f"Each nav_path must include about {horizon_days} daily points after last_date.",
            "NAV values must use 4 decimal places and stay positive.",
            "Ground rationales in the provided features, baseline, and agent_context; do not invent holdings.",
            "Use research briefs/sentiment as qualitative context only; do not claim guaranteed returns.",
            "You may refine the statistical baseline but keep paths plausible.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a careful quantitative assistant for mutual-fund NAV scenario "
                "analysis. You never give personalized investment advice. "
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
    Produce bull/base/bear NAV forecast via the local chat LLM.

    When use_llm=False, returns the statistical baseline used as prompt context only.
    LLM failures raise — no silent statistical substitute.
    """
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
    raw = chat_completions(
        messages,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.2,
    )
    parsed = extract_json_object(raw)
    validated = _validate_forecast(parsed, horizon_days)
    validated["source"] = "llm"
    validated["llm_raw_response"] = raw
    return validated
