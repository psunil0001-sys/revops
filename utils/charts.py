"""Altair chart helpers for the portfolio dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd

HISTORY_COLOR = "#22D3EE"
BULL_COLOR = "#4ADE80"
BASE_COLOR = "#FACC15"
BEAR_COLOR = "#FB7185"
VALUE_COLOR = "#60A5FA"
ALLOC_COLORS = [
    "#22D3EE",
    "#4ADE80",
    "#FACC15",
    "#FB7185",
    "#FB923C",
    "#818CF8",
]


def nav_history_chart(nav_df: pd.DataFrame, height: int = 360) -> alt.Chart:
    data = nav_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    return (
        alt.Chart(data)
        .mark_line(color=HISTORY_COLOR, strokeWidth=3)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("NAV:Q", title="NAV (₹)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d-%b-%Y"),
                alt.Tooltip("NAV:Q", title="NAV", format=".4f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )


def portfolio_value_chart(history_df: pd.DataFrame, height: int = 360) -> alt.Chart:
    data = history_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    return (
        alt.Chart(data)
        .mark_area(
            color=VALUE_COLOR,
            opacity=0.55,
            line={"color": VALUE_COLOR, "strokeWidth": 3},
        )
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(
                "Portfolio Value:Q",
                title="Portfolio value (₹)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d-%b-%Y"),
                alt.Tooltip("Portfolio Value:Q", title="Value", format=".4f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )


def allocation_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str = "Allocation",
    height: int = 280,
) -> alt.Chart:
    data = df.copy()
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=18)
        .encode(
            y=alt.Y(f"{category_col}:N", sort="-x", title=None),
            x=alt.X(f"{value_col}:Q", title="Allocation (%)"),
            color=alt.Color(
                f"{category_col}:N",
                scale=alt.Scale(range=ALLOC_COLORS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N"),
                alt.Tooltip(f"{value_col}:Q", format=".2f"),
            ],
        )
        .properties(height=height)
    )


def forecast_overlay_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    height: int = 420,
) -> alt.Chart:
    """Overlay recent history with bull/base/bear forecast paths."""
    hist = history_df.tail(120).copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist["Scenario"] = "History"

    hist_chart = (
        alt.Chart(hist)
        .mark_line(strokeWidth=3.5)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("NAV:Q", title="NAV (₹)", scale=alt.Scale(zero=False)),
            color=alt.value(HISTORY_COLOR),
            tooltip=[
                alt.Tooltip("Date:T", format="%d-%b-%Y"),
                alt.Tooltip("NAV:Q", format=".4f"),
                alt.Tooltip("Scenario:N"),
            ],
        )
    )

    if forecast_df is None or forecast_df.empty:
        return hist_chart.properties(height=height).interactive()

    fc = forecast_df.copy()
    fc["Date"] = pd.to_datetime(fc["Date"])
    color_scale = alt.Scale(
        domain=["Bear", "Base", "Bull"],
        range=[BEAR_COLOR, BASE_COLOR, BULL_COLOR],
    )
    fc_chart = (
        alt.Chart(fc)
        .mark_line(strokeDash=[7, 4], strokeWidth=3)
        .encode(
            x="Date:T",
            y=alt.Y("NAV:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Scenario:N", scale=color_scale, title="Scenario"),
            tooltip=[
                alt.Tooltip("Date:T", format="%d-%b-%Y"),
                alt.Tooltip("NAV:Q", format=".4f"),
                alt.Tooltip("Scenario:N"),
            ],
        )
    )
    return (hist_chart + fc_chart).properties(height=height).interactive()
