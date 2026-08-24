"""Manage llama chat (:8000) and embedding (:8001) servers via repo scripts."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from utils.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_LLM_BASE_URL,
    START_SERVERS_SCRIPT,
    STOP_SERVERS_SCRIPT,
)


def _normalize_base(url: str) -> str:
    return url.rstrip("/")


def _port_from_base(url: str, default: int) -> int:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    if parsed.port:
        return int(parsed.port)
    return default


class ServerManager:
    """Wrap start/stop scripts and health checks for chat + embed servers."""

    def __init__(
        self,
        *,
        chat_base_url: str | None = None,
        embedding_base_url: str | None = None,
        start_script: Path | None = None,
        stop_script: Path | None = None,
    ) -> None:
        self.chat_base_url = _normalize_base(
            chat_base_url
            or os.getenv("LLM_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        )
        self.embedding_base_url = _normalize_base(
            embedding_base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or DEFAULT_EMBEDDING_BASE_URL
        )
        self.start_script = Path(start_script or START_SERVERS_SCRIPT)
        self.stop_script = Path(stop_script or STOP_SERVERS_SCRIPT)

        if self.chat_base_url == self.embedding_base_url:
            raise ValueError(
                "Chat and embedding base URLs must differ; "
                "chat server has no embeddings API."
            )

    def _probe(self, base_url: str, timeout: float = 3.0) -> dict[str, Any]:
        url = f"{base_url}/models"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            models: list[str] = []
            if isinstance(payload, dict):
                for item in payload.get("data") or []:
                    if isinstance(item, dict) and item.get("id"):
                        models.append(str(item["id"]))
            return {"ok": True, "url": url, "models": models, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "url": url, "models": [], "error": str(exc)}

    def status(self) -> dict[str, Any]:
        chat = self._probe(self.chat_base_url)
        embed = self._probe(self.embedding_base_url)
        return {
            "chat": {
                "ok": chat["ok"],
                "base_url": self.chat_base_url,
                "port": _port_from_base(self.chat_base_url, 8000),
                "models": chat["models"],
                "error": chat["error"],
            },
            "embed": {
                "ok": embed["ok"],
                "base_url": self.embedding_base_url,
                "port": _port_from_base(self.embedding_base_url, 8001),
                "models": embed["models"],
                "error": embed["error"],
            },
            "both_ok": bool(chat["ok"] and embed["ok"]),
        }

    def start_servers(self) -> dict[str, Any]:
        if not self.start_script.is_file():
            raise FileNotFoundError(f"Start script missing: {self.start_script}")
        current = self.status()
        if current["both_ok"]:
            return {"started": False, "message": "Servers already healthy", "status": current}

        # If PID files exist but servers down, stop first so start script can run.
        if not current["both_ok"]:
            try:
                self.stop_servers()
            except Exception:
                pass

        env = os.environ.copy()
        env.setdefault("CHAT_PORT", str(_port_from_base(self.chat_base_url, 8000)))
        env.setdefault("EMBED_PORT", str(_port_from_base(self.embedding_base_url, 8001)))
        completed = subprocess.run(
            ["bash", str(self.start_script)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=240,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "start_llama_servers.sh failed:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        return {
            "started": True,
            "message": "Start script completed",
            "stdout": completed.stdout[-2000:],
            "status": self.status(),
        }

    def stop_servers(self) -> dict[str, Any]:
        if not self.stop_script.is_file():
            raise FileNotFoundError(f"Stop script missing: {self.stop_script}")
        completed = subprocess.run(
            ["bash", str(self.stop_script)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "stopped": completed.returncode == 0,
            "message": "Stop script completed" if completed.returncode == 0 else "Stop script failed",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-1000:],
            "status": self.status(),
        }

    def ensure_ready(self, timeout_s: float = 180.0) -> dict[str, Any]:
        """Ensure chat + embed are healthy; start them if needed."""
        status = self.status()
        if status["both_ok"]:
            return status

        self.start_servers()
        deadline = time.time() + timeout_s
        last = self.status()
        while time.time() < deadline:
            last = self.status()
            if last["both_ok"]:
                return last
            time.sleep(2.0)

        raise RuntimeError(
            "Chat and/or embedding servers not ready after ensure_ready. "
            f"chat_ok={last['chat']['ok']} embed_ok={last['embed']['ok']} "
            f"chat_err={last['chat']['error']} embed_err={last['embed']['error']}"
        )
