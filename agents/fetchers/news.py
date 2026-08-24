"""News fetchers — Google News RSS + optional NewsAPI."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree


def domain_allowed(url: str, allowlist: list[str]) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in allowlist)


def fetch_google_news_rss(query: str, limit: int = 8) -> list[dict[str, str]]:
    feed_url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    request = Request(
        feed_url,
        headers={"User-Agent": "revops-news-analyst/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        raw = response.read()
    root = ElementTree.fromstring(raw)
    items: list[dict[str, str]] = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if title:
            items.append(
                {
                    "title": title,
                    "url": link,
                    "summary": desc,
                    "published": pub,
                    "source": "google_news_rss",
                    "query": query,
                }
            )
    return items


def fetch_newsapi(query: str, api_key: str, limit: int = 8) -> list[dict[str, str]]:
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={quote_plus(query)}&language=en&pageSize={limit}&sortBy=publishedAt"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "revops-news-analyst/1.0",
            "X-Api-Key": api_key,
        },
    )
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    articles = payload.get("articles") or []
    items: list[dict[str, str]] = []
    for article in articles[:limit]:
        title = (article.get("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "url": article.get("url") or "",
                "summary": article.get("description") or "",
                "published": article.get("publishedAt") or "",
                "source": "newsapi",
                "query": query,
            }
        )
    return items


def resolve_news_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return os.getenv("NEWS_API_KEY") or None


def gather_news(
    queries: list[str],
    *,
    allowlist: list[str],
    api_key: str | None = None,
) -> tuple[list[dict[str, str]], list[str], dict[str, Any]]:
    collected: list[dict[str, str]] = []
    errors: list[str] = []
    raw_by_query: dict[str, Any] = {}

    key = resolve_news_api_key(api_key)
    for query in queries:
        bucket: dict[str, Any] = {"rss": [], "newsapi": [], "errors": []}
        try:
            rss_items = fetch_google_news_rss(query, limit=5)
            bucket["rss"] = rss_items
            collected.extend(rss_items)
        except Exception as exc:  # noqa: BLE001
            msg = f"rss:{query}:{exc}"
            errors.append(msg)
            bucket["errors"].append(msg)
        if key:
            try:
                api_items = fetch_newsapi(query, key, limit=4)
                bucket["newsapi"] = api_items
                collected.extend(api_items)
            except Exception as exc:  # noqa: BLE001
                msg = f"newsapi:{query}:{exc}"
                errors.append(msg)
                bucket["errors"].append(msg)
        raw_by_query[query] = bucket

    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in collected:
        key_title = item["title"].lower()
        if key_title in seen:
            continue
        url = item.get("url") or ""
        if url and not domain_allowed(url, allowlist):
            if "google.com" not in url.lower():
                continue
        seen.add(key_title)
        unique.append(item)
    return unique, errors, raw_by_query
