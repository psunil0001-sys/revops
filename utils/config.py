"""Fund configuration and static allocation data."""

from datetime import date
from pathlib import Path

SCHEME_CODE = "149182"
FUND_NAME = "Kotak Multicap Fund - Regular Plan - Growth"

# Official allotment / inception (Kotak / AMFI). First mfapi NAV is 2021-10-06.
FUND_INCEPTION_DATE = date(2021, 9, 29)

INVESTMENT = 99995.00
PURCHASE_NAV = 20.30
UNITS_HELD = INVESTMENT / PURCHASE_NAV

EQUITY_ALLOCATION = 99.25
OTHER_ALLOCATION = 0.75

SECTOR_DATA = {
    "Sector": [
        "Financial",
        "Consumer Discretionary",
        "Industrials",
        "Communication Services",
        "Consumer Staples",
        "Other Equity",
    ],
    "Allocation": [30.75, 15.01, 14.72, 8.55, 7.76, 22.46],
}

ASSET_DATA = {
    "Asset Class": ["Equity", "Others"],
    "Allocation": [EQUITY_ALLOCATION, OTHER_ALLOCATION],
}

# Public AMFI-backed API — no CAPTCHA / browser session required.
MFAPI_NAV_URL = "https://api.mfapi.in/mf/{scheme_code}"

# Local llama-server endpoints (chat has no embeddings API).
DEFAULT_LLM_MODEL = "gemma4 4B"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed"

_REPO_ROOT = Path(__file__).resolve().parent.parent
START_SERVERS_SCRIPT = _REPO_ROOT / "scripts" / "start_llama_servers.sh"
STOP_SERVERS_SCRIPT = _REPO_ROOT / "scripts" / "stop_llama_servers.sh"

DATA_DIR = _REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MEMORY_DATA_DIR = DATA_DIR / "memory"
CHROMA_DATA_DIR = DATA_DIR / "chroma_store"
AGENT_RUNS_DIR = DATA_DIR / "agent_runs"
RAG_DATA_DIR = DATA_DIR / "rag"

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
