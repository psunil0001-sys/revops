"""Shared paths, LLM defaults, and market proxy symbols (fund data lives in funds.yaml)."""

import os
from pathlib import Path

# Public AMFI-backed API — no CAPTCHA / browser session required.
MFAPI_NAV_URL = "https://api.mfapi.in/mf/{scheme_code}"

# Local llama-server / OpenAI-compatible endpoints (env overrides defaults).
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL") or "gemma4 4B"
DEFAULT_LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8000/v1"
).rstrip("/")


def _default_embedding_base_url(llm_base: str) -> str:
    """Prefer explicit EMBEDDING_BASE_URL; else share Ollama host; else :8001."""
    env = os.getenv("EMBEDDING_BASE_URL")
    if env:
        return env.rstrip("/")
    base = (llm_base or "").rstrip("/")
    # Ollama exposes chat + embeddings on the same OpenAI-compatible base.
    if ":11434" in base or base.endswith("11434/v1"):
        return base
    return "http://127.0.0.1:8001/v1"


DEFAULT_EMBEDDING_BASE_URL = _default_embedding_base_url(DEFAULT_LLM_BASE_URL)
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "nomic-embed"

# Finite timeout for chat/completions so Forecast cannot hang forever.
LLM_CHAT_TIMEOUT_S = float(os.getenv("LLM_CHAT_TIMEOUT_S", "120"))

# Max concurrent fund pipelines and research analysts (1 = fully queued).
MAX_PARALLEL_CALLS = max(1, int(os.getenv("REVOPS_MAX_PARALLEL_CALLS", "1")))

# Forecast reliability knobs.
MIN_FORECAST_OBSERVATIONS = 60
FORECAST_BOOTSTRAP_PATHS = 500
LLM_TERMINAL_BAND_FRAC = 0.25  # of half (bull − bear) around each baseline terminal

_REPO_ROOT = Path(__file__).resolve().parent.parent
START_SERVERS_SCRIPT = _REPO_ROOT / "scripts" / "start_llama_servers.sh"
STOP_SERVERS_SCRIPT = _REPO_ROOT / "scripts" / "stop_llama_servers.sh"

DATA_DIR = _REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MEMORY_DATA_DIR = DATA_DIR / "memory"
CHROMA_DATA_DIR = DATA_DIR / "chroma_store"
AGENT_RUNS_DIR = DATA_DIR / "agent_runs"
RAG_DATA_DIR = DATA_DIR / "rag"
NAV_DATA_DIR = DATA_DIR / "nav"

# Proxies used by Fundamentals / Technical analysts (yfinance).
FUNDAMENTAL_PROXY_SYMBOLS = {
    "nifty50": "^NSEI",
    "reliance": "RELIANCE.NS",
    "hdfcbank": "HDFCBANK.NS",
    "infy": "INFY.NS",
}
TECHNICAL_SYMBOLS = {
    "nifty50": "^NSEI",
    "india_vix": "^INDIAVIX",
    "bank_nifty": "^NSEBANK",
    "nifty_it": "^CNXIT",
}
