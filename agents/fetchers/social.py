"""Social media fetchers — Reddit public JSON (no OAuth)."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

DEFAULT_SUBREDDITS = (
    "IndiaInvestments",
    "mutualfunds",
    "IndianStreetBets",
)


def _user_agent() -> str:
    return os.getenv("REDDIT_USER_AGENT") or "revops-sentiment-analyst/1.0 (research)"


def search_reddit(
    query: str,
    *,
    subreddit: str | None = None,
    limit: int = 15,
) -> dict[str, Any]:
    if subreddit:
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json?"
            f"q={quote_plus(query)}&restrict_sr=1&sort=new&limit={limit}"
        )
    else:
        url = (
            "https://www.reddit.com/search.json?"
            f"q={quote_plus(query)}&sort=new&limit={limit}"
        )
    request = Request(url, headers={"User-Agent": _user_agent()})
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    children = ((payload.get("data") or {}).get("children")) or []
    posts: list[dict[str, Any]] = []
    for child in children:
        data = child.get("data") or {}
        title = (data.get("title") or "").strip()
        if not title:
            continue
        posts.append(
            {
                "id": data.get("id"),
                "title": title,
                "selftext": (data.get("selftext") or "")[:2000],
                "subreddit": data.get("subreddit"),
                "score": data.get("score"),
                "num_comments": data.get("num_comments"),
                "created_utc": data.get("created_utc"),
                "permalink": (
                    f"https://www.reddit.com{data.get('permalink')}"
                    if data.get("permalink")
                    else None
                ),
                "url": data.get("url"),
            }
        )
    return {
        "query": query,
        "subreddit": subreddit,
        "posts": posts,
        "raw_count": len(children),
    }


def gather_social(
    queries: list[str],
    *,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    posts: list[dict[str, Any]] = []
    errors: list[str] = []
    raw: dict[str, Any] = {}
    for sub in subreddits:
        for query in queries:
            key = f"{sub}:{query}"
            try:
                result = search_reddit(query, subreddit=sub, limit=8)
                raw[key] = result
                posts.extend(result.get("posts") or [])
            except Exception as exc:  # noqa: BLE001
                msg = f"reddit:{key}:{exc}"
                errors.append(msg)
                raw[key] = {"error": str(exc)}
    # Deduplicate by id/title
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for post in posts:
        pid = str(post.get("id") or post.get("title") or "").lower()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        unique.append(post)
    return unique, errors, raw
