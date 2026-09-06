"""Multi-fund portfolio tracker — interactive Streamlit UI."""

from __future__ import annotations

from datetime import date, datetime, timezone

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
    MAX_PARALLEL_CALLS,
    MIN_FORECAST_OBSERVATIONS,
)
from utils.concurrency import run_queued
from utils.backtest import walk_forward_forecast_backtest
from utils.features import (
    assert_forecast_ready,
    build_nav_features,
    forecast_paths_to_dataframe,
)
from agents.contracts import DEFAULT_HORIZON_DAYS, DISCLAIMER, SessionRunState
from agents.manager import ManagerAgent
from agents.memory.summarizer import load_latest_verdict
from agents.scheduler import (
    SessionRunStore,
    next_scheduled_label,
    pending_scheduled_session,
    today_ist,
)
from agents.server_manager import ServerManager
from utils.funds import FundConfig, get_fund, load_funds
from utils.llm import check_embedding_health, check_llm_health, forecast_nav
from utils.market import (
    attach_benchmark_features,
    fetch_benchmark_context,
    fetch_nifty_context,
)
from utils.nav import fetch_nav_history_for_fund, filter_nav_by_dates
from utils.portfolio import (
    build_portfolio_history,
    calculate_portfolio,
    get_asset_dataframe,
    get_sector_dataframe,
    inr,
)

FUNDS = load_funds()

st.set_page_config(
    page_title="Multi-fund portfolio tracker",
    page_icon=":material/account_balance:",
    layout="wide",
)

TAB_OVERVIEW = ":material/dashboard: Overview"
TAB_CHARTS = ":material/show_chart: Charts"
TAB_ALLOC = ":material/pie_chart: Allocations"
TAB_AGENTS = ":material/smart_toy: Agents"
TAB_FORECAST = ":material/psychology: Forecast"


def _default_fund_state() -> dict:
    return {
        "nav_df_full": None,
        "nav_error": None,
        "forecast": None,
        "forecast_error": None,
        "manager_verdict": None,
        "agent_pipeline_error": None,
        "auto_loaded": False,
        "focus_agents": False,
        "focus_forecast": False,
    }


def _fs(fund_id: str) -> dict:
    """Return per-fund session state, creating defaults if missing."""
    states = st.session_state.fund_states
    if fund_id not in states:
        states[fund_id] = _default_fund_state()
    return states[fund_id]


def _init_session_state() -> None:
    defaults = {
        "fund_states": {},
        "active_fund_id": FUNDS[0].id,
        "llm_base_url": DEFAULT_LLM_BASE_URL,
        "embedding_base_url": DEFAULT_EMBEDDING_BASE_URL,
        "llm_model": DEFAULT_LLM_MODEL,
        "llm_api_key": "",
        "llm_health": None,
        "embed_health": None,
        "server_status": None,
        "agent_run_store": None,
        "max_parallel_calls": MAX_PARALLEL_CALLS,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(ttl=3600, show_spinner=False)
def load_full_nav_history(fund_id: str) -> pd.DataFrame:
    fund = get_fund(fund_id)
    return fetch_nav_history_for_fund(
        fund,
        start_date=fund.inception_date,
        end_date=date.today(),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_nifty_context(start: date, end: date):
    return fetch_nifty_context(start, end)


def refresh_nav(fund: FundConfig, *, bust_cache: bool = False) -> None:
    fs = _fs(fund.id)
    if bust_cache:
        load_full_nav_history.clear()
    try:
        fs["nav_df_full"] = load_full_nav_history(fund.id)
        fs["nav_error"] = None
    except Exception as error:  # noqa: BLE001
        fs["nav_df_full"] = None
        fs["nav_error"] = str(error)


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


def _apply_manager_verdict(fund: FundConfig, verdict) -> None:
    fs = _fs(fund.id)
    fs["manager_verdict"] = verdict.model_dump(mode="json")
    bundle = verdict.monitor_bundle
    fs["focus_agents"] = True
    fs["focus_forecast"] = False
    st.session_state.pop(f"main_tabs_{fund.id}", None)
    st.session_state.pop(f"main_tabs_{fund.id}_agents", None)
    st.session_state.pop(f"main_tabs_{fund.id}_forecast", None)
    if bundle is None:
        return
    fs["forecast"] = bundle.forecast
    fs["forecast_error"] = None
    refresh_nav(fund, bust_cache=True)


def _render_agents_workspace(fund: FundConfig) -> None:
    """Full Agents tab: statuses, briefs, raw paths, memory hits."""
    fs = _fs(fund.id)
    st.caption(DISCLAIMER)
    if fs["agent_pipeline_error"]:
        st.error(fs["agent_pipeline_error"])

    verdict = fs["manager_verdict"]
    if verdict is None:
        st.info(
            "No agent run yet. Use **Ask Manager: run now** in the sidebar "
            "(targets the active fund) or wait for the NSE open/close schedule."
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
    """Poll NSE open/close schedule every 60s; queue Manager for all funds when due."""
    store = st.session_state.agent_run_store
    due = pending_scheduled_session(store)
    last = store.last_completed()
    max_parallel = int(st.session_state.max_parallel_calls)
    with status_slot.container():
        st.caption(f"Next slot: {next_scheduled_label()}")
        st.caption(f"Queue concurrency: {max_parallel} parallel call(s)")
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
        st.warning(f"Scheduled session due: **{due}** — queuing all funds…")

        servers = ServerManager(
            chat_base_url=st.session_state.llm_base_url,
            embedding_base_url=st.session_state.embedding_base_url,
        )
        any_accepted = False
        with st.spinner(
            f"Manager {due}: start servers → queue {len(FUNDS)} funds "
            f"(max {max_parallel} parallel)…"
        ):
            try:
                servers.ensure_ready()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Scheduled pipeline aborted — servers not ready: {exc}")
                return

            def _run_one(fund: FundConfig):
                manager = ManagerAgent(
                    chat_base_url=st.session_state.llm_base_url,
                    embedding_base_url=st.session_state.embedding_base_url,
                )
                manager.runs = store
                return manager.run_pipeline(
                    session=due,
                    fund=fund,
                    base_url=st.session_state.llm_base_url,
                    embedding_base_url=st.session_state.embedding_base_url,
                    api_key=st.session_state.llm_api_key or None,
                    model=st.session_state.llm_model,
                    use_llm=True,
                    mark_schedule_complete=False,
                    ensure_servers=False,
                    stop_servers_after=False,
                    max_parallel_calls=max_parallel,
                )

            outcomes = run_queued(FUNDS, _run_one, max_parallel=max_parallel)
            for fund, outcome in outcomes:
                if isinstance(outcome, BaseException):
                    _fs(fund.id)["agent_pipeline_error"] = str(outcome)
                    continue
                _apply_manager_verdict(fund, outcome)
                if outcome.accepted:
                    any_accepted = True

            try:
                servers.stop_servers()
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Servers stop after batch: {exc}")

        if any_accepted:
            store.mark_completed(
                SessionRunState(
                    trade_date=today_ist().isoformat(),
                    session=due,
                    completed_at=datetime.now(timezone.utc),
                    accepted=True,
                    notes=f"multi-fund schedule {due} max_parallel={max_parallel}",
                )
            )
            st.success(f"Scheduled {due} pipeline accepted for at least one fund.")
        else:
            st.error("Scheduled pipeline rejected for all funds.")


def _render_fund_workspace(
    fund: FundConfig,
    start_date: date,
    end_date: date,
) -> None:
    """Independent workspace for one fund (own NAV / agents / forecast state)."""
    fs = _fs(fund.id)

    st.caption(
        f"`{fund.display_code}` · {fund.type} · "
        f"Inception {fund.inception_date.strftime('%d-%b-%Y')}"
    )

    if fs["nav_df_full"] is None and not fs["auto_loaded"]:
        fs["auto_loaded"] = True
        with st.spinner(f"Loading NAV for {fund.name}…"):
            refresh_nav(fund)

    if fs["nav_error"]:
        st.error(fs["nav_error"])

    full_df = fs["nav_df_full"]
    if full_df is None or full_df.empty:
        st.info(
            "Click **Load / refresh NAV** in the sidebar (with this fund active) "
            "to populate this workspace."
        )
        return

    if start_date > end_date:
        return

    nav_df = filter_nav_by_dates(
        full_df,
        start_date,
        end_date,
        default_start=fund.inception_date,
    )
    if nav_df.empty:
        st.warning("No NAV rows in the selected date range.")
        return

    latest = nav_df.iloc[-1]
    current_nav = float(latest["NAV"])
    nav_date = latest["Date"]
    portfolio = calculate_portfolio(
        current_nav,
        investment=fund.investment,
        units=fund.units,
    )
    portfolio_history = build_portfolio_history(nav_df, units=fund.units)
    features = build_nav_features(nav_df)
    nav_spark = nav_df["NAV"].tail(30).tolist()
    value_spark = portfolio_history["Portfolio Value"].tail(30).tolist()

    pl_positive = portfolio["profit_loss"] >= 0
    if pl_positive:
        st.html('<div class="revops-banner revops-gain">Status: Profit</div>')
    else:
        st.html('<div class="revops-banner revops-loss">Status: Loss</div>')

    if fs["forecast"] is not None:
        st.html(
            '<div class="revops-banner revops-forecast-ready">'
            "Forecast ready — open the <b>Forecast</b> tab for scenarios "
            "and LLM response."
            "</div>"
        )

    st.subheader("Fund snapshot")
    with st.container(horizontal=True):
        st.metric(
            "Code",
            fund.display_code,
            border=True,
            icon=":material/tag:",
        )
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
            f"{fund.units:,.4f}",
            border=True,
            icon=":material/toll:",
        )

    st.subheader("Portfolio performance")
    with st.container(horizontal=True):
        st.metric("Investment", inr(fund.investment, 4), border=True)
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

    if fs["focus_agents"]:
        default_tab = TAB_AGENTS
        tabs_key = f"main_tabs_{fund.id}_agents"
    elif fs["focus_forecast"]:
        default_tab = TAB_FORECAST
        tabs_key = f"main_tabs_{fund.id}_forecast"
    else:
        default_tab = TAB_OVERVIEW
        tabs_key = f"main_tabs_{fund.id}"

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
                    inr(fund.investment, 4),
                    inr(fund.purchase_nav, 4),
                    f"{fund.units:,.4f}",
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
                    "Date": st.column_config.DatetimeColumn(
                        "Date", format="DD-MMM-YYYY"
                    ),
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
        asset_df = get_asset_dataframe(fund)
        sector_df = get_sector_dataframe(fund)
        sector_display = sector_df.copy()
        equity_pct = fund.equity_pct or 100.0
        sector_display["Approx. Fund Value"] = (
            portfolio["current_value"]
            * equity_pct
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
            f"Equity itself is {equity_pct:.2f}% of the fund."
        )

    with tab_agents:
        st.badge("Agents", color="blue", icon=":material/smart_toy:")
        st.subheader("Research analysts & Monitor")
        _render_agents_workspace(fund)

    with tab_forecast:
        st.badge("Forecast", color="violet", icon=":material/psychology:")
        st.subheader("NAV scenario forecast")
        st.caption(
            "Bootstrap trading-day scenarios (p10/p50/p90). LLM may refine "
            "within a clamp band. Illustrative only — not investment advice. "
            f"Requires ≥{MIN_FORECAST_OBSERVATIONS} NAV observations."
        )

        if fs["forecast_error"]:
            st.error(fs["forecast_error"])

        sub_scenarios, sub_agents, sub_llm, sub_data = st.tabs(
            [
                ":material/candlestick_chart: Scenarios",
                ":material/smart_toy: Agents",
                ":material/chat: LLM response",
                ":material/table: Data",
            ],
            key=f"forecast_subtabs_{fund.id}",
        )

        forecast = fs["forecast"]

        with sub_scenarios:
            if forecast is None:
                st.info(
                    "No forecast yet. Select this fund as active in the sidebar "
                    "and click **Ask Manager: run now** or **Run forecast only**."
                )
            else:
                source = forecast.get("source", "unknown")
                source_color = (
                    "green"
                    if source in ("llm_clamped", "llm")
                    else "orange"
                )
                st.badge(
                    f"Source: {source}",
                    icon=":material/analytics:",
                    color=source_color,
                )
                obs = forecast.get("observations") or features.get("observations")
                st.badge(
                    f"Observations: {obs}",
                    icon=":material/database:",
                    color="gray",
                )
                st.badge(
                    f"Horizon: {forecast.get('horizon_days')} trading days",
                    icon=":material/calendar_month:",
                    color="blue",
                )
                backtest = forecast.get("backtest") or {}
                if backtest.get("ready"):
                    st.badge(
                        backtest.get("label")
                        or f"illustrative / backtested coverage "
                        f"{backtest.get('coverage_pct')}%",
                        icon=":material/verified:",
                        color="green",
                    )
                elif backtest:
                    st.badge(
                        f"Backtest not ready: {backtest.get('reason', 'n/a')}",
                        icon=":material/info:",
                        color="gray",
                    )
                if forecast.get("llm_fallback"):
                    st.warning(
                        "LLM refine failed or violated clamps — showing "
                        "bootstrap baseline."
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
                                    inr(end_nav * fund.units, 4),
                                )
                            st.write(payload.get("rationale") or "—")

                st.caption(forecast.get("disclaimer") or "")

                if backtest.get("ready"):
                    with st.expander("Walk-forward backtest metrics", expanded=False):
                        st.write(
                            f"- Origins: **{backtest.get('n_origins')}** "
                            f"(step={backtest.get('step')})"
                        )
                        st.write(
                            f"- Coverage (actual in [bear, bull]): "
                            f"**{backtest.get('coverage_pct')}%**"
                        )
                        st.write(
                            f"- Median abs % error (base): "
                            f"**{backtest.get('median_abs_pct_error_base')}%**"
                        )
                        st.write(
                            f"- Mean abs % error (base): "
                            f"**{backtest.get('mean_abs_pct_error_base')}%**"
                        )
                        st.caption(
                            "Bootstrap-only walk-forward on this fund's NAV history."
                        )

        with sub_agents:
            st.info(
                "Full analyst status, raw file paths, and memory hits are on the "
                "top-level **Agents** tab."
            )
            verdict = fs["manager_verdict"]
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
                        key=f"llm_raw_ta_{fund.id}",
                    )
                else:
                    st.info(
                        "No raw LLM response for this run. "
                        "Enable **Use local LLM**, ensure the endpoint is reachable, "
                        "then click **Run forecast only** again."
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
                        "Date": st.column_config.DatetimeColumn(
                            format="DD-MMM-YYYY"
                        ),
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


# ---------------------------------------------------------------------------
# App body
# ---------------------------------------------------------------------------

_init_session_state()

if st.session_state.agent_run_store is None:
    st.session_state.agent_run_store = SessionRunStore()

for _fund in FUNDS:
    _fund_fs = _fs(_fund.id)
    if _fund_fs["manager_verdict"] is None:
        _latest = load_latest_verdict(_fund.id)
        if _latest:
            _fund_fs["manager_verdict"] = _latest
            _bundle = _latest.get("monitor_bundle") or {}
            if _bundle.get("forecast"):
                _fund_fs["forecast"] = _bundle["forecast"]

st.title("Multi-fund portfolio tracker")
st.caption(f"{len(FUNDS)} funds · NAV + agents + forecast per fund")

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

today = date.today()
active_fund = get_fund(st.session_state.active_fund_id)
active_fs = _fs(active_fund.id)
active_full = active_fs["nav_df_full"]
min_available = active_fund.inception_date
max_available = today
if active_full is not None and not active_full.empty:
    min_available = max(
        active_fund.inception_date,
        pd.Timestamp(active_full["Date"].min()).date(),
    )
    max_available = min(today, pd.Timestamp(active_full["Date"].max()).date())

with st.sidebar:
    st.header("Controls")
    st.selectbox(
        "Active fund",
        options=[f.id for f in FUNDS],
        format_func=lambda fid: next(f.name for f in FUNDS if f.id == fid),
        key="active_fund_id",
        help="Sidebar actions (Load NAV, Ask Manager, Forecast) target this fund.",
    )
    # Re-resolve after selectbox may have updated session state
    active_fund = get_fund(st.session_state.active_fund_id)
    active_fs = _fs(active_fund.id)

    st.subheader("Date range")
    start_date = st.date_input(
        "Start date",
        value=min_available,
        min_value=active_fund.inception_date,
        max_value=today,
    )
    end_date = st.date_input(
        "End date",
        value=today,
        min_value=active_fund.inception_date,
        max_value=today,
    )
    if start_date > end_date:
        st.error("Start date must be on or before end date.")

    st.caption(
        "NAV from mfapi.in (AMFI-backed) or local CSV. "
        f"Active fund inception {active_fund.inception_date.isoformat()}; "
        "first published NAV may be a few days later."
    )

    if st.button(
        "Load / refresh NAV",
        type="primary",
        width="stretch",
        icon=":material/refresh:",
    ):
        with st.spinner(f"Fetching NAV for {active_fund.name}…"):
            refresh_nav(active_fund, bust_cache=True)
        if active_fs["nav_error"] is None:
            st.success(f"NAV updated for {active_fund.name}.")
            active_fs["forecast"] = None
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
        f"Runs for **{active_fund.name}** only. Manager starts chat+embed if "
        "needed, assigns analysts, then Monitor refreshes NAV and runs a "
        "**60-day** forecast. Servers are **stopped** when the run finishes. "
        "Does **not** mark open/close as completed."
    )
    st.number_input(
        "Max parallel calls",
        min_value=1,
        max_value=5,
        step=1,
        key="max_parallel_calls",
        help=(
            "Limits concurrent fund pipelines (scheduled) and research "
            "analysts (per fund). Default 1 queues everything for local llama. "
            "Override with env REVOPS_MAX_PARALLEL_CALLS."
        ),
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
            "Manual Manager pipeline for the active fund — ignores scheduled "
            "09:15 / 15:30 IST. Starts servers if down when LLM is on."
        ),
    )

    st.divider()
    st.subheader("Scheduled research (NSE)")
    st.caption(
        "Automatic twice daily: open **09:15 IST** and close **15:30 IST** "
        "(weekdays). Queues the Manager pipeline for **all funds** with "
        f"**{st.session_state.max_parallel_calls}** parallel call(s)."
    )
    schedule_slot = st.empty()
    manager_scheduler(schedule_slot)

    st.divider()
    st.subheader("Forecast only")
    st.caption(f"Targets active fund: **{active_fund.name}**")
    horizon_days = st.segmented_control(
        "Forecast horizon (trading days)",
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
        help=(
            "Skip research agents; bootstrap/LLM forecast on full NAV history "
            f"(needs ≥{MIN_FORECAST_OBSERVATIONS} observations)"
        ),
    )

# Sidebar actions for active fund
if run_agents:
    with st.spinner(
        f"Manager (manual) for {active_fund.name}: "
        "ensure servers → research → monitor → 60-day forecast…"
    ):
        try:
            manager = ManagerAgent(
                chat_base_url=st.session_state.llm_base_url,
                embedding_base_url=st.session_state.embedding_base_url,
            )
            manager.runs = st.session_state.agent_run_store
            verdict = manager.run_pipeline(
                session="manual",
                fund=active_fund,
                base_url=st.session_state.llm_base_url,
                embedding_base_url=st.session_state.embedding_base_url,
                api_key=st.session_state.llm_api_key or None,
                model=st.session_state.llm_model,
                use_llm=use_llm,
                mark_schedule_complete=False,
                ensure_servers=bool(use_llm),
                max_parallel_calls=int(st.session_state.max_parallel_calls),
            )
            _apply_manager_verdict(active_fund, verdict)
            active_fs["agent_pipeline_error"] = None
            if verdict.accepted:
                st.success(
                    f"Manager manual run accepted for {active_fund.name} "
                    "(schedule slots unchanged)."
                )
            else:
                st.warning(
                    "Manager manual run finished with guardrail warnings/rejection."
                )
                if verdict.guardrail_hits:
                    st.caption("; ".join(verdict.guardrail_hits[:3]))
        except Exception as error:  # noqa: BLE001
            active_fs["agent_pipeline_error"] = str(error)
            st.error(f"Manager manual run failed: {error}")

if run_forecast:
    afull = active_fs["nav_df_full"]
    if afull is None or afull.empty:
        st.error("Load NAV for the active fund before running a forecast.")
    else:
        with st.spinner(f"Building forecast for {active_fund.name}…"):
            try:
                assert_forecast_ready(
                    afull, min_observations=MIN_FORECAST_OBSERVATIONS
                )
                afeatures = build_nav_features(afull)
                market_ctx = fetch_benchmark_context(
                    active_fund.benchmark_symbol or "^NSEI",
                    pd.Timestamp(afull["Date"].min()).date(),
                    pd.Timestamp(afull["Date"].max()).date(),
                )
                afeatures.update(
                    attach_benchmark_features(afull, market_ctx)
                )
                hz = int(horizon_days or DEFAULT_HORIZON_DAYS)
                result = forecast_nav(
                    afeatures,
                    horizon_days=hz,
                    base_url=st.session_state.llm_base_url,
                    api_key=st.session_state.llm_api_key or None,
                    model=st.session_state.llm_model,
                    market_context=market_ctx,
                    use_llm=use_llm,
                )
                result = dict(result)
                result["backtest"] = walk_forward_forecast_backtest(
                    afull, horizon_days=hz, step=21
                )
                result["observations"] = int(
                    afeatures.get("observations") or len(afull)
                )
                active_fs["forecast"] = result
                active_fs["forecast_error"] = None
                active_fs["focus_forecast"] = True
                active_fs["focus_agents"] = False
                st.session_state.pop(f"main_tabs_{active_fund.id}", None)
                st.session_state.pop(
                    f"main_tabs_{active_fund.id}_agents", None
                )
                st.session_state.pop(
                    f"main_tabs_{active_fund.id}_forecast", None
                )
            except Exception as error:  # noqa: BLE001
                active_fs["forecast"] = None
                active_fs["forecast_error"] = str(error)
                active_fs["focus_forecast"] = True
                st.session_state.pop(f"main_tabs_{active_fund.id}", None)
                st.session_state.pop(
                    f"main_tabs_{active_fund.id}_forecast", None
                )

fund_tabs = st.tabs([f.name for f in FUNDS])
for tab, fund in zip(fund_tabs, FUNDS):
    with tab:
        _render_fund_workspace(fund, start_date, end_date)

st.caption("Multi-fund portfolio tracker")
