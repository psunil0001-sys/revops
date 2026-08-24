"""Shared OpenAI-compatible embedding helpers for :8001."""

from __future__ import annotations

import os
import re
from typing import Sequence

import requests

from utils.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_BASE_URL,
)

# llama-server embed physical batch is often 512 tokens; keep inputs short.
_MAX_EMBED_CHARS = 1200
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalize_base(url: str) -> str:
    return url.rstrip("/")


def sanitize_embed_text(text: str, *, max_chars: int = _MAX_EMBED_CHARS) -> str:
    cleaned = _HTML_RE.sub(" ", text or "")
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return " "
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def resolve_embedding_model(
    *,
    embedding_base_url: str | None = None,
    model: str | None = None,
    timeout: float = 5.0,
) -> str:
    """Prefer explicit model; else first id from /models; else config default."""
    if model:
        return model
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        return env_model
    base = normalize_base(
        embedding_base_url
        or os.getenv("EMBEDDING_BASE_URL")
        or DEFAULT_EMBEDDING_BASE_URL
    )
    response = requests.get(f"{base}/models", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    for item in payload.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            return str(item["id"])
    return DEFAULT_EMBEDDING_MODEL


def embed_via_server(
    texts: Sequence[str],
    *,
    embedding_base_url: str | None = None,
    model: str | None = None,
    batch_size: int = 1,
    timeout: float = 120.0,
) -> list[list[float]]:
    """
    POST /v1/embeddings against the dedicated embed server.

    Sends short, HTML-stripped texts (one per request by default).
    """
    if not texts:
        return []

    base = normalize_base(
        embedding_base_url
        or os.getenv("EMBEDDING_BASE_URL")
        or DEFAULT_EMBEDDING_BASE_URL
    )
    chat = normalize_base(os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL)
    if base == chat:
        raise RuntimeError(
            "Refusing embeddings against chat URL; use embed server."
        )

    cleaned = [sanitize_embed_text(t) for t in texts]
    model_name = resolve_embedding_model(
        embedding_base_url=base, model=model
    )
    url = f"{base}/embeddings"
    batch_size = max(1, min(int(batch_size), 4))
    vectors: list[list[float]] = []

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        response = requests.post(
            url,
            json={"input": batch if len(batch) > 1 else batch[0], "model": model_name},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected embeddings response: {data!r}")
        ordered = sorted(items, key=lambda x: int(x.get("index", 0)))
        for item in ordered:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(f"Missing embedding in item: {item!r}")
            vectors.append([float(v) for v in emb])

    if len(vectors) != len(cleaned):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} for {len(cleaned)} texts"
        )
    return vectors
