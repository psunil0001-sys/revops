"""Shared paths, LLM defaults, and market proxy symbols (fund data lives in funds.yaml)."""

import os
from pathlib import Path

# Public AMFI-backed API — no CAPTCHA / browser session required.
MFAPI_NAV_URL = "https://api.mfapi.in/mf/{scheme_code}"

# Local llama-server endpoints (chat has no embeddings API).
DEFAULT_LLM_MODEL = "gemma4 4B"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed"

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
