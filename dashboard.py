"""Kotak Multicap Fund portfolio tracker — interactive Streamlit UI."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from utils.charts import (
    allocation_bar_chart,
    forecast_overlay_chart,
    nav_history_chart,
    portfolio_value_chart,
)
from utils.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    EQUITY_ALLOCATION,
    FUND_INCEPTION_DATE,
    FUND_NAME,
    INVESTMENT,
    PURCHASE_NAV,
    SCHEME_CODE,
    UNITS_HELD,
)
from utils.features import (
    build_nav_features,
    forecast_paths_to_dataframe,
)
from agents.contracts import DEFAULT_HORIZON_DAYS, DISCLAIMER
from agents.manager import ManagerAgent
from agents.memory.summarizer import load_latest_verdict
from agents.scheduler import (
    SessionRunStore,
    next_scheduled_label,
    pending_scheduled_session,
)
from agents.server_manager import ServerManager
from utils.llm import check_embedding_health, check_llm_health, forecast_nav
from utils.market import fetch_nifty_context
from utils.nav import fetch_nav_history, filter_nav_by_dates
from utils.portfolio import (
    build_portfolio_history,
    calculate_portfolio,
    get_asset_dataframe,
    get_sector_dataframe,
    inr,
)

st.set_page_config(
    page_title="Kotak Multicap Portfolio",
    page_icon=":material/account_balance:",
    layout="wide",
)


def _init_session_state() -> None:
    defaults = {
        "nav_df_full": None,
        "nav_error": None,
        "auto_loaded": False,
        "forecast": None,
        "forecast_error": None,
        "focus_forecast": False,
        "llm_base_url": DEFAULT_LLM_BASE_URL,
        "embedding_base_url": DEFAULT_EMBEDDING_BASE_URL,
        "llm_model": DEFAULT_LLM_MODEL,
        "llm_api_key": "",
        "llm_health": None,
        "embed_health": None,
        "server_status": None,
        "manager_verdict": None,
        "agent_pipeline_error": None,
        "agent_run_store": None,
        "focus_agents": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=3600, show_spinner=False)
def load_full_nav_history(scheme_code: str) -> pd.DataFrame:
    return fetch_nav_history(
        scheme_code=scheme_code,
        start_date=FUND_INCEPTION_DATE,
        end_date=date.today(),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_nifty_context(start: date, end: date):
    return fetch_nifty_context(start, end)


def refresh_nav(*, bust_cache: bool = False) -> None:
    if bust_cache:
        load_full_nav_history.clear()
    try:
        st.session_state.nav_df_full = load_full_nav_history(SCHEME_CODE)
        st.session_state.nav_error = None
    except Exception as error:  # noqa: BLE001
        st.session_state.nav_df_full = None
        st.session_state.nav_error = str(error)


@st.fragment(run_every=60)
def server_heartbeat(heartbeat_slot) -> None:
    """Ping chat (:8000) and embed (:8001) /models every 60s."""
    chat = check_llm_health(
        base_url=st.session_state.llm_base_url,
        api_key=st.session_state.llm_api_key or None,
        model=st.session_state.llm_model,
    )
    embed = check_embedding_health(
        embedding_base_url=st.session_state.embedding_base_url,
    )
    st.session_state.llm_health = chat
    st.session_state.embed_health = embed
    try:
        st.session_state.server_status = ServerManager(
            chat_base_url=st.session_state.llm_base_url,
            embedding_base_url=st.session_state.embedding_base_url,
        ).status()
    except Exception as exc:  # noqa: BLE001
        st.session_state.server_status = {"both_ok": False, "error": str(exc)}

    with heartbeat_slot.container():
        c1, c2 = st.columns(2)
        with c1:
            if chat["ok"]:
                st.success(f"Chat :8000 OK · {chat['message']}")
            else:
                st.error(
                    f"Chat down · {chat.get('error') or chat['message']}"
                )
        with c2:
            if embed["ok"]:
                st.success(f"Embed :8001 OK · {embed['message']}")
            else:
                st.error(
                    f"Embed down · {embed.get('error') or embed['message']}"
                )
        st.caption(
            f"Checked at {chat['checked_at']} · refreshes every 60s"
        )


_init_session_state()

if st.session_state.agent_run_store is None:
    st.session_state.agent_run_store = SessionRunStore()

if st.session_state.manager_verdict is None:
    latest = load_latest_verdict()
    if latest:
        st.session_state.manager_verdict = latest
        bundle = latest.get("monitor_bundle") or {}
        if bundle.get("forecast"):
            st.session_state.forecast = bundle["forecast"]


def _apply_manager_verdict(verdict) -> None:
    st.session_state.manager_verdict = verdict.model_dump(mode="json")
    bundle = verdict.monitor_bundle
    if bundle is None:
        st.session_state.focus_agents = True
        st.session_state.focus_forecast = False
        st.session_state.pop("main_dashboard_tabs", None)
        st.session_state.pop("main_dashboard_tabs_agents", None)
        st.session_state.pop("main_dashboard_tabs_forecast", None)
        return
    st.session_state.forecast = bundle.forecast
    st.session_state.forecast_error = None
    st.session_state.focus_agents = True
    st.session_state.focus_forecast = False
    st.session_state.pop("main_dashboard_tabs", None)
    st.session_state.pop("main_dashboard_tabs_agents", None)
    st.session_state.pop("main_dashboard_tabs_forecast", None)
    # Refresh NAV after monitor cycle
    refresh_nav(bust_cache=True)


def _render_agents_workspace() -> None:
    """Full Agents tab: statuses, briefs, raw paths, memory hits."""
    st.caption(DISCLAIMER)
    if st.session_state.agent_pipeline_error:
        st.error(st.session_state.agent_pipeline_error)

    verdict = st.session_state.manager_verdict
    if verdict is None:
        st.info(
            "No agent run yet. Use **Ask Manager: run now** in the sidebar "
            "or wait for the NSE open/close schedule."
        )
        return

    accepted = verdict.get("accepted")
    st.badge(
        "Accepted" if accepted else "Rejected / warnings",
        color="green" if accepted else "orange",
    )
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Session", str(verdict.get("session") or "—"))
    with meta_cols[1]:
        st.metric("Run id", str(verdict.get("run_id") or "—")[:18])
    with meta_cols[2]:
        st.metric("Memory hits", str(verdict.get("memory_hit_count") or 0))
    with meta_cols[3]:
        raw_root = verdict.get("raw_root") or "—"
        st.caption(f"Raw: `{raw_root}`")

    for msg in verdict.get("messages") or []:
        st.write(f"- {msg}")
    hits = verdict.get("guardrail_hits") or []
    if hits:
        st.warning("Guardrail hits:\n\n- " + "\n- ".join(hits))

    statuses = verdict.get("agent_statuses") or []
    if statuses:
        st.subheader("Agent status")
        status_rows = []
        for item in statuses:
            status_rows.append(
                {
                    "Agent": item.get("agent_id"),
                    "Status": item.get("status"),
                    "Error": item.get("error") or "",
                    "Notes": (item.get("notes") or "")[:120],
                }
            )
        st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch")

    bundle = verdict.get("monitor_bundle") or {}
    if bundle.get("summary_markdown"):
        st.subheader("Monitor summary")
        st.markdown(bundle.get("summary_markdown"))

    briefs = bundle.get("research_briefs") or []
    st.subheader("Analyst responses")
    if not briefs:
        st.info("No research briefs in this run.")
    for brief in briefs:
        with st.expander(
            f"{brief.get('agent_id')} · confidence={brief.get('confidence')}",
            expanded=True,
        ):
            st.caption(
                f"session={brief.get('session')} · as_of={brief.get('as_of')}"
            )
            st.write(brief.get("notes") or "")
            claims = brief.get("claims") or []
            for claim in claims:
                st.write(f"• {claim.get('text')}")
            sources = brief.get("sources") or []
            if sources:
                st.markdown("**Sources**")
                for src in sources[:12]:
                    st.write(f"- {src}")
            raw_paths = brief.get("raw_paths") or []
            if raw_paths:
                st.markdown("**Raw files**")
                for path in raw_paths:
                    st.code(path, language=None)
            metrics = brief.get("metrics") or {}
            if metrics:
                with st.expander("Metrics JSON"):
                    st.json(metrics)

    sentiment = bundle.get("sentiment")
    if sentiment:
        st.badge(
            f"Sentiment: {sentiment.get('label')}",
            color=(
                "green"
                if sentiment.get("label") == "bullish"
                else "red"
                if sentiment.get("label") == "bearish"
                else "gray"
            ),
        )

    memory_hits = bundle.get("memory_hits") or []
    st.subheader("Related memory used for forecast")
    if not memory_hits:
        st.caption("No related memory hits for this run.")
    else:
        for hit in memory_hits:
            with st.container(border=True):
                st.markdown(
                    f"**{hit.get('title')}** · `{hit.get('collection')}`"
                )
                st.caption(f"score={hit.get('score')} · id={hit.get('doc_id')}")
                st.write(hit.get("text") or "")


@st.fragment(run_every=60)
def manager_scheduler(status_slot) -> None:
    """Poll NSE open/close schedule every 60s and run Manager when due."""
    store = st.session_state.agent_run_store
    due = pending_scheduled_session(store)
    last = store.last_completed()
    with status_slot.container():
        st.caption(f"Next slot: {next_scheduled_label()}")
        if last:
            st.caption(
                f"Last completed: {last.trade_date} {last.session} "
                f"({'ok' if last.accepted else 'rejected'})"
            )
        else:
            st.caption("Last completed: none yet")
        if due is None:
            st.info("No scheduled research due right now.")
            return
        st.warning(f"Scheduled session due: **{due}** — running agents…")
        manager = ManagerAgent(
            chat_base_url=st.session_state.llm_base_url,
            embedding_base_url=st.session_state.embedding_base_url,
        )
        manager.runs = store
        with st.spinner(f"Manager running {due} pipeline…"):
            verdict = manager.run_pipeline(
                session=due,
                base_url=st.session_state.llm_base_url,
                embedding_base_url=st.session_state.embedding_base_url,
                api_key=st.session_state.llm_api_key or None,
                model=st.session_state.llm_model,
                use_llm=True,
                mark_schedule_complete=True,
                ensure_servers=True,
            )
        _apply_manager_verdict(verdict)
        if verdict.accepted:
            st.success(f"Scheduled {due} pipeline accepted.")
        else:
            st.error("Scheduled pipeline rejected by guardrails.")


st.title("Kotak Multicap Fund portfolio tracker")
st.caption(
    f"{FUND_NAME} · Scheme `{SCHEME_CODE}` · "
    f"Inception {FUND_INCEPTION_DATE.strftime('%d-%b-%Y')}"
)

today = date.today()
full_df = st.session_state.nav_df_full
min_available = FUND_INCEPTION_DATE
max_available = today
if full_df is not None and not full_df.empty:
    min_available = max(
        FUND_INCEPTION_DATE,
        pd.Timestamp(full_df["Date"].min()).date(),
    )
    max_available = min(today, pd.Timestamp(full_df["Date"].max()).date())

with st.sidebar:
    st.header("Controls")
    st.subheader("Date range")
    start_date = st.date_input(
        "Start date",
        value=min_available,
        min_value=FUND_INCEPTION_DATE,
        max_value=today,
    )
    end_date = st.date_input(
        "End date",
        value=today,
        min_value=FUND_INCEPTION_DATE,
        max_value=today,
    )
    if start_date > end_date:
        st.error("Start date must be on or before end date.")

    st.caption(
        "NAV from mfapi.in (AMFI-backed). "
        f"Official inception {FUND_INCEPTION_DATE.isoformat()}; "
        "first published NAV may be a few days later."
    )

    if st.button(
        "Load / refresh NAV",
        type="primary",
        width="stretch",
        icon=":material/refresh:",
    ):
        with st.spinner("Fetching NAV history…"):
            refresh_nav(bust_cache=True)
        if st.session_state.nav_error is None:
            st.success("NAV history updated.")
            st.session_state.forecast = None
        else:
            st.error("NAV fetch failed.")

    st.divider()
    st.subheader("Local LLM servers")
    st.session_state.llm_base_url = st.text_input(
        "Chat endpoint (completions)",
        value=st.session_state.llm_base_url,
        help="OpenAI-compatible chat base, e.g. http://127.0.0.1:8000/v1",
    )
    st.session_state.embedding_base_url = st.text_input(
        "Embedding endpoint",
        value=st.session_state.embedding_base_url,
        help="Dedicated embed server, e.g. http://127.0.0.1:8001/v1",
    )
    st.session_state.llm_model = st.text_input(
        "Chat model name",
        value=st.session_state.llm_model,
        help="Recommended: Gemma / Mistral-Nemo Instruct",
    )
    st.session_state.llm_api_key = st.text_input(
        "API key (optional)",
        value=st.session_state.llm_api_key,
        type="password",
    )

    start_col, stop_col = st.columns(2)
    with start_col:
        start_servers = st.button(
            "Start servers",
            width="stretch",
            icon=":material/play_arrow:",
            help="Run scripts/start_llama_servers.sh (chat :8000, embed :8001)",
        )
    with stop_col:
        stop_servers = st.button(
            "Stop servers",
            width="stretch",
            icon=":material/stop:",
            help="Run scripts/stop_llama_servers.sh",
        )

    if start_servers:
        with st.spinner("Starting chat + embedding servers…"):
            try:
                sm = ServerManager(
                    chat_base_url=st.session_state.llm_base_url,
                    embedding_base_url=st.session_state.embedding_base_url,
                )
                result = sm.ensure_ready(timeout_s=180)
                st.session_state.server_status = result
                if result.get("both_ok"):
                    st.success("Chat and embedding servers ready.")
                else:
                    st.warning("Servers started but health check incomplete.")
            except Exception as error:  # noqa: BLE001
                st.error(f"Start servers failed: {error}")

    if stop_servers:
        with st.spinner("Stopping llama servers…"):
            try:
                sm = ServerManager(
                    chat_base_url=st.session_state.llm_base_url,
                    embedding_base_url=st.session_state.embedding_base_url,
                )
                result = sm.stop_servers()
                st.session_state.server_status = result.get("status")
                if result.get("stopped"):
                    st.info("Servers stopped.")
                else:
                    st.warning(result.get("message") or "Stop may have failed.")
            except Exception as error:  # noqa: BLE001
                st.error(f"Stop servers failed: {error}")

    heartbeat_slot = st.empty()
    server_heartbeat(heartbeat_slot)

    st.divider()
    st.subheader("Ask Manager — run now")
    st.caption(
        "Bypass the NSE clock: Manager starts chat+embed if needed, assigns "
        "Fundamentals / Sentiment / News / Technical analysts, then Monitor "
        "refreshes NAV and runs a **60-day** forecast. Servers are **stopped** "
        "when the run finishes and restarted on the next LLM run. "
        "Does **not** mark open/close as completed."
    )
    use_llm = st.toggle(
        "Use local LLM (Manager starts servers, then stops them after the run)",
        value=True,
    )
    run_agents = st.button(
        "Ask Manager: run now",
        width="stretch",
        icon=":material/play_circle:",
        type="primary",
        help=(
            "Manual Manager pipeline anytime — ignores scheduled 09:15 / 15:30 IST. "
            "Starts servers if down when LLM is on."
        ),
    )

    st.divider()
    st.subheader("Scheduled research (NSE)")
    st.caption(
        "Automatic twice daily: open **09:15 IST** and close **15:30 IST** "
        "(weekdays). Same Manager pipeline as Run now."
    )
    schedule_slot = st.empty()
    manager_scheduler(schedule_slot)

    st.divider()
    st.subheader("Forecast only")
    horizon_days = st.segmented_control(
        "Forecast horizon",
        options=[30, 60, 90],
        default=DEFAULT_HORIZON_DAYS,
        required=True,
        format_func=lambda d: f"{d} days",
    )
    run_forecast = st.button(
        "Run forecast only",
        width="stretch",
        icon=":material/psychology:",
        type="secondary",
        help="Skip research agents; statistical/LLM forecast only",
    )

if st.session_state.nav_df_full is None and not st.session_state.auto_loaded:
    st.session_state.auto_loaded = True
    with st.spinner("Loading NAV history…"):
        refresh_nav()

full_df = st.session_state.nav_df_full

if st.session_state.nav_error:
    st.error(st.session_state.nav_error)

if full_df is None or full_df.empty:
    st.info("Click **Load / refresh NAV** to populate the dashboard.")
    st.stop()

if start_date > end_date:
    st.stop()

nav_df = filter_nav_by_dates(full_df, start_date, end_date)
if nav_df.empty:
    st.warning("No NAV rows in the selected date range.")
    st.stop()

latest = nav_df.iloc[-1]
current_nav = float(latest["NAV"])
nav_date = latest["Date"]
portfolio = calculate_portfolio(current_nav)
portfolio_history = build_portfolio_history(nav_df)
features = build_nav_features(nav_df)
nav_spark = nav_df["NAV"].tail(30).tolist()
value_spark = portfolio_history["Portfolio Value"].tail(30).tolist()

if run_agents:
    with st.spinner(
        "Manager (manual): ensure servers → research → monitor → 60-day forecast…"
    ):
        try:
            manager = ManagerAgent(
                chat_base_url=st.session_state.llm_base_url,
                embedding_base_url=st.session_state.embedding_base_url,
            )
            manager.runs = st.session_state.agent_run_store
            verdict = manager.run_pipeline(
                session="manual",
                base_url=st.session_state.llm_base_url,
                embedding_base_url=st.session_state.embedding_base_url,
                api_key=st.session_state.llm_api_key or None,
                model=st.session_state.llm_model,
                use_llm=use_llm,
                mark_schedule_complete=False,
                ensure_servers=bool(use_llm),
            )
            _apply_manager_verdict(verdict)
            st.session_state.agent_pipeline_error = None
            if verdict.accepted:
                st.success(
                    "Manager manual run accepted "
                    "(schedule slots unchanged)."
                )
            else:
                st.warning(
                    "Manager manual run finished with guardrail warnings/rejection."
                )
                if verdict.guardrail_hits:
                    st.caption("; ".join(verdict.guardrail_hits[:3]))
        except Exception as error:  # noqa: BLE001
            st.session_state.agent_pipeline_error = str(error)
            st.error(f"Manager manual run failed: {error}")

if run_forecast:
    with st.spinner("Building forecast…"):
        market_ctx = load_nifty_context(
            pd.Timestamp(nav_df["Date"].min()).date(),
            pd.Timestamp(nav_df["Date"].max()).date(),
        )
        try:
            st.session_state.forecast = forecast_nav(
                features,
                horizon_days=int(horizon_days or 30),
                base_url=st.session_state.llm_base_url,
                api_key=st.session_state.llm_api_key or None,
                model=st.session_state.llm_model,
                market_context=market_ctx,
                use_llm=use_llm,
            )
            st.session_state.forecast_error = None
            st.session_state.focus_forecast = True
            # Remount tabs so Forecast becomes the active tab.
            st.session_state.pop("main_dashboard_tabs", None)
        except Exception as error:  # noqa: BLE001
            st.session_state.forecast = None
            st.session_state.forecast_error = str(error)
            st.session_state.focus_forecast = True
            st.session_state.pop("main_dashboard_tabs", None)

# --- Overview metrics ---
# Color accents (user-requested visible coloring on dark theme)
st.html(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(34,211,238,0.12), rgba(30,41,59,0.55));
        border: 1px solid rgba(34,211,238,0.35) !important;
        border-radius: 12px;
        padding: 0.35rem 0.6rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #22D3EE !important;
        border-bottom-color: #22D3EE !important;
    }
    .revops-banner {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.4rem 0 0.9rem 0;
        font-weight: 600;
    }
    .revops-loss {
        background: rgba(239,68,68,0.18);
        border: 1px solid #EF4444;
        color: #FCA5A5;
    }
    .revops-gain {
        background: rgba(34,197,94,0.18);
        border: 1px solid #22C55E;
        color: #86EFAC;
    }
    .revops-forecast-ready {
        background: rgba(34,211,238,0.16);
        border: 1px solid #22D3EE;
        color: #A5F3FC;
    }
    </style>
    """
)

pl_positive = portfolio["profit_loss"] >= 0
if pl_positive:
    st.html('<div class="revops-banner revops-gain">Status: Profit</div>')
else:
    st.html('<div class="revops-banner revops-loss">Status: Loss</div>')

if st.session_state.forecast is not None:
    st.html(
        '<div class="revops-banner revops-forecast-ready">'
        'Forecast ready — open the <b>Forecast</b> tab for scenarios and LLM response.'
        '</div>'
    )

st.subheader("Fund snapshot")
with st.container(horizontal=True):
    st.metric("Scheme code", SCHEME_CODE, border=True, icon=":material/tag:")
    st.metric(
        "Current NAV",
        inr(current_nav, 4),
        border=True,
        chart_data=nav_spark,
        chart_type="line",
        icon=":material/payments:",
    )
    st.metric(
        "NAV date",
        pd.Timestamp(nav_date).strftime("%d-%b-%Y"),
        border=True,
        icon=":material/calendar_month:",
    )
    st.metric(
        "Units held",
        f"{UNITS_HELD:,.4f}",
        border=True,
        icon=":material/toll:",
    )

st.subheader("Portfolio performance")
with st.container(horizontal=True):
    st.metric("Investment", inr(INVESTMENT, 4), border=True)
    st.metric(
        "Current value",
        inr(portfolio["current_value"], 4),
        border=True,
        chart_data=value_spark,
        chart_type="line",
    )
    st.metric(
        "Profit / loss",
        inr(portfolio["profit_loss"], 4),
        delta=f"{portfolio['roi']:.2f}%",
        border=True,
    )
    st.metric(
        "ROI",
        f"{portfolio['roi']:.2f}%",
        delta=f"{portfolio['roi']:.2f}%",
        border=True,
    )

TAB_OVERVIEW = ":material/dashboard: Overview"
TAB_CHARTS = ":material/show_chart: Charts"
TAB_ALLOC = ":material/pie_chart: Allocations"
TAB_AGENTS = ":material/smart_toy: Agents"
TAB_FORECAST = ":material/psychology: Forecast"

if st.session_state.focus_agents:
    default_tab = TAB_AGENTS
elif st.session_state.focus_forecast:
    default_tab = TAB_FORECAST
else:
    default_tab = TAB_OVERVIEW

if st.session_state.focus_agents:
    tabs_key = "main_dashboard_tabs_agents"
elif st.session_state.focus_forecast:
    tabs_key = "main_dashboard_tabs_forecast"
else:
    tabs_key = "main_dashboard_tabs"

tab_overview, tab_charts, tab_alloc, tab_agents, tab_forecast = st.tabs(
    [TAB_OVERVIEW, TAB_CHARTS, TAB_ALLOC, TAB_AGENTS, TAB_FORECAST],
    default=default_tab,
    key=tabs_key,
)

with tab_overview:
    st.badge("Overview", color="blue", icon=":material/dashboard:")
    st.subheader("Investment details")
    investment_df = pd.DataFrame(
        {
            "Metric": [
                "Investment",
                "Purchase NAV",
                "Units purchased",
                "Current NAV",
                "Current value",
                "Profit / loss",
                "ROI",
            ],
            "Value": [
                inr(INVESTMENT, 4),
                inr(PURCHASE_NAV, 4),
                f"{UNITS_HELD:,.4f}",
                inr(current_nav, 4),
                inr(portfolio["current_value"], 4),
                inr(portfolio["profit_loss"], 4),
                f"{portfolio['roi']:.2f}%",
            ],
        }
    )
    st.dataframe(investment_df, hide_index=True, width="stretch")

    st.subheader("Key statistics")
    with st.container(horizontal=True):
        st.metric("CAGR", f"{features['cagr_pct']:.2f}%", border=True)
        st.metric(
            "Max drawdown",
            f"{features['max_drawdown_pct']:.2f}%",
            border=True,
        )
        st.metric(
            "1Y return",
            (
                f"{features['return_1y_pct']:.2f}%"
                if features["return_1y_pct"] is not None
                else "—"
            ),
            border=True,
        )
        st.metric(
            "Vol (63d ann.)",
            (
                f"{features['vol_63d_ann_pct']:.2f}%"
                if features["vol_63d_ann_pct"] is not None
                else "—"
            ),
            border=True,
        )

    with st.expander("NAV history data"):
        st.dataframe(
            nav_df,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="DD-MMM-YYYY"),
                "NAV": st.column_config.NumberColumn("NAV", format="₹%.4f"),
            },
            hide_index=True,
            width="stretch",
        )

with tab_charts:
    st.badge("Charts", color="blue", icon=":material/show_chart:")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("NAV history")
            st.altair_chart(
                nav_history_chart(nav_df),
                width="stretch",
                theme=None,
            )
    with c2:
        with st.container(border=True):
            st.subheader("Portfolio value history")
            st.altair_chart(
                portfolio_value_chart(portfolio_history),
                width="stretch",
                theme=None,
            )

with tab_alloc:
    st.badge("Allocations", color="orange", icon=":material/pie_chart:")
    asset_df = get_asset_dataframe()
    sector_df = get_sector_dataframe()
    sector_display = sector_df.copy()
    sector_display["Approx. Fund Value"] = (
        portfolio["current_value"]
        * EQUITY_ALLOCATION
        / 100
        * sector_display["Allocation"]
        / 100
    ).round(4)

    a1, a2 = st.columns(2)
    with a1:
        with st.container(border=True):
            st.subheader("Asset allocation")
            st.dataframe(
                asset_df,
                column_config={
                    "Asset Class": st.column_config.TextColumn("Asset class"),
                    "Allocation": st.column_config.NumberColumn(
                        "Allocation",
                        format="%.2f%%",
                    ),
                },
                hide_index=True,
                width="stretch",
            )
            st.altair_chart(
                allocation_bar_chart(asset_df, "Asset Class"),
                width="stretch",
                theme=None,
            )
    with a2:
        with st.container(border=True):
            st.subheader("Equity sector allocation")
            st.dataframe(
                sector_display,
                column_config={
                    "Sector": st.column_config.TextColumn("Sector"),
                    "Allocation": st.column_config.NumberColumn(
                        "Allocation",
                        format="%.2f%%",
                    ),
                    "Approx. Fund Value": st.column_config.NumberColumn(
                        "Approx. fund value",
                        format="₹%.4f",
                    ),
                },
                hide_index=True,
                width="stretch",
            )
            st.altair_chart(
                allocation_bar_chart(sector_df, "Sector"),
                width="stretch",
                theme=None,
            )
    st.caption(
        "Sector allocation is a percentage of the fund's equity portfolio. "
        f"Equity itself is {EQUITY_ALLOCATION:.2f}% of the fund."
    )

with tab_agents:
    st.badge("Agents", color="blue", icon=":material/smart_toy:")
    st.subheader("Research analysts & Monitor")
    _render_agents_workspace()

with tab_forecast:
    st.badge("Forecast", color="violet", icon=":material/psychology:")
    st.subheader("NAV scenario forecast")
    st.caption(
        "Uses your local OpenAI-compatible endpoint. "
        "Recommended model: Mistral-Nemo-Instruct-2407 (~12B)."
    )

    if st.session_state.forecast_error:
        st.error(st.session_state.forecast_error)

    # Nested tabs always visible so Forecast is clearly a dedicated workspace
    sub_scenarios, sub_agents, sub_llm, sub_data = st.tabs(
        [
            ":material/candlestick_chart: Scenarios",
            ":material/smart_toy: Agents",
            ":material/chat: LLM response",
            ":material/table: Data",
        ],
        key="forecast_subtabs",
    )

    forecast = st.session_state.forecast

    with sub_scenarios:
        if forecast is None:
            st.info(
                "No forecast yet. Set the endpoint in the sidebar and click "
                "**Ask Manager: run now** or **Run forecast only**."
            )
        else:
            source = forecast.get("source", "unknown")
            source_color = "green" if source == "llm" else "orange"
            st.badge(
                f"Source: {source}",
                icon=":material/analytics:",
                color=source_color,
            )
            if forecast.get("llm_error"):
                st.error(f"LLM error: {forecast['llm_error']}")

            forecast_df = forecast_paths_to_dataframe(forecast)
            scenarios = forecast.get("scenarios") or {}
            scenario_colors = {
                "bear": "red",
                "base": "yellow",
                "bull": "green",
            }

            st.altair_chart(
                forecast_overlay_chart(nav_df, forecast_df),
                width="stretch",
                theme=None,
            )

            cols = st.columns(3)
            for col, name in zip(cols, ("bear", "base", "bull")):
                payload = scenarios.get(name) or {}
                path = payload.get("nav_path") or []
                end_nav = float(path[-1]["nav"]) if path else None
                with col:
                    with st.container(border=True):
                        st.badge(
                            name.capitalize(),
                            color=scenario_colors[name],
                        )
                        if end_nav is not None:
                            delta_pct = (end_nav / current_nav - 1.0) * 100.0
                            st.metric(
                                "Horizon NAV",
                                inr(end_nav, 4),
                                delta=f"{delta_pct:.2f}%",
                            )
                            st.metric(
                                "Horizon value",
                                inr(end_nav * UNITS_HELD, 4),
                            )
                        st.write(payload.get("rationale") or "—")

            st.caption(forecast.get("disclaimer") or "")

    with sub_agents:
        st.info(
            "Full analyst status, raw file paths, and memory hits are on the "
            "top-level **Agents** tab."
        )
        verdict = st.session_state.manager_verdict
        if verdict is None:
            st.caption("No agent run yet.")
        else:
            st.write(
                f"Last run: `{verdict.get('run_id')}` · "
                f"session={verdict.get('session')} · "
                f"accepted={verdict.get('accepted')}"
            )
            bundle = verdict.get("monitor_bundle") or {}
            briefs = bundle.get("research_briefs") or []
            for brief in briefs:
                st.write(
                    f"- **{brief.get('agent_id')}** "
                    f"(conf={brief.get('confidence')}): "
                    f"{(brief.get('claims') or [{}])[0].get('text', '—')}"
                )

    with sub_llm:
        if forecast is None:
            st.info("Run a forecast to see the raw LLM response here.")
        else:
            raw = forecast.get("llm_raw_response")
            if raw:
                st.badge("LLM response received", color="green")
                st.code(raw, language="json")
                st.text_area(
                    "LLM response (scrollable)",
                    value=raw,
                    height=360,
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.info(
                    "No raw LLM response for this run. "
                    "Enable **Use local LLM**, ensure the endpoint is reachable, "
                    "then click **Run forecast** again."
                )
                if forecast.get("llm_error"):
                    st.error(forecast["llm_error"])

    with sub_data:
        if forecast is None:
            st.info("Run a forecast to see path data and model features.")
        else:
            forecast_df = forecast_paths_to_dataframe(forecast)
            st.subheader("Forecast path data")
            st.dataframe(
                forecast_df,
                column_config={
                    "Date": st.column_config.DatetimeColumn(format="DD-MMM-YYYY"),
                    "NAV": st.column_config.NumberColumn(format="₹%.4f"),
                    "Scenario": st.column_config.TextColumn("Scenario"),
                },
                hide_index=True,
                width="stretch",
            )
            st.subheader("Features sent to the model")
            display_features = {
                k: v for k, v in features.items() if k != "recent_series"
            }
            st.json(display_features)

st.caption("Kotak Multicap Fund — interactive portfolio tracker")
