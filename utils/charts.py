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
    "#A78BFA",
    "#34D399",
]


def _dark_props(height: int) -> dict:
    return {
        "height": height,
        "background": "transparent",
        "padding": {"left": 8, "right": 8, "top": 8, "bottom": 8},
    }


def nav_history_chart(nav_df: pd.DataFrame, height: int = 360) -> alt.Chart:
    data = nav_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["Date"], empty=False
    )
    line = (
        alt.Chart(data)
        .mark_line(color=HISTORY_COLOR, strokeWidth=3)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("NAV:Q", title="NAV (₹)", scale=alt.Scale(zero=False)),
        )
    )
    points = (
        alt.Chart(data)
        .mark_circle(size=64, color=HISTORY_COLOR)
        .encode(
            x="Date:T",
            y=alt.Y("NAV:Q", scale=alt.Scale(zero=False)),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d-%b-%Y"),
                alt.Tooltip("NAV:Q", title="NAV", format=".4f"),
            ],
        )
        .add_params(nearest)
    )
    rules = (
        alt.Chart(data)
        .mark_rule(color="rgba(34,211,238,0.35)")
        .encode(x="Date:T")
        .transform_filter(nearest)
    )
    return (line + points + rules).properties(**_dark_props(height)).interactive()


def portfolio_value_chart(history_df: pd.DataFrame, height: int = 360) -> alt.Chart:
    data = history_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["Date"], empty=False
    )
    area = (
        alt.Chart(data)
        .mark_area(
            color=VALUE_COLOR,
            opacity=0.45,
            line={"color": VALUE_COLOR, "strokeWidth": 3},
        )
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y(
                "Portfolio Value:Q",
                title="Portfolio value (₹)",
                scale=alt.Scale(zero=False),
            ),
        )
    )
    points = (
        alt.Chart(data)
        .mark_circle(size=64, color=VALUE_COLOR)
        .encode(
            x="Date:T",
            y=alt.Y("Portfolio Value:Q", scale=alt.Scale(zero=False)),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%d-%b-%Y"),
                alt.Tooltip("Portfolio Value:Q", title="Value", format=".4f"),
            ],
        )
        .add_params(nearest)
    )
    return (area + points).properties(**_dark_props(height)).interactive()


def allocation_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str = "Allocation",
    height: int = 280,
) -> alt.Chart:
    data = df.copy()
    highlight = alt.selection_point(fields=[category_col], on="pointerover", empty=True)
    click = alt.selection_point(fields=[category_col], toggle=True)
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=6, size=20)
        .encode(
            y=alt.Y(f"{category_col}:N", sort="-x", title=None),
            x=alt.X(f"{value_col}:Q", title="Allocation (%)"),
            color=alt.Color(
                f"{category_col}:N",
                scale=alt.Scale(range=ALLOC_COLORS),
                legend=None,
            ),
            opacity=alt.condition(
                highlight | click, alt.value(1.0), alt.value(0.55)
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N"),
                alt.Tooltip(f"{value_col}:Q", title="%", format=".2f"),
            ],
        )
        .add_params(highlight, click)
        .properties(**_dark_props(height))
    )


def allocation_pie_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str = "Allocation",
    height: int = 300,
) -> alt.Chart:
    """Interactive donut / pie with legend toggle and hover highlight."""
    data = df.copy()
    if value_col not in data.columns and "Value" in data.columns:
        value_col = "Value"
    highlight = alt.selection_point(fields=[category_col], on="pointerover", empty=True)
    legend_sel = alt.selection_point(fields=[category_col], bind="legend")
    base = (
        alt.Chart(data)
        .mark_arc(innerRadius=55, outerRadius=110, stroke="#0f172a", strokeWidth=2)
        .encode(
            theta=alt.Theta(f"{value_col}:Q", stack=True),
            color=alt.Color(
                f"{category_col}:N",
                scale=alt.Scale(range=ALLOC_COLORS),
                legend=alt.Legend(title=None, orient="bottom", columns=2),
            ),
            opacity=alt.condition(
                highlight & legend_sel, alt.value(1.0), alt.value(0.45)
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title="Slice"),
                alt.Tooltip(f"{value_col}:Q", title="Value", format=",.2f"),
            ],
            order=alt.Order(f"{value_col}:Q", sort="descending"),
        )
        .add_params(highlight, legend_sel)
        .properties(**_dark_props(height))
    )
    return base


def stacked_sleeve_value_chart(
    long_df: pd.DataFrame,
    *,
    date_col: str = "Date",
    series_col: str = "Scheme",
    value_col: str = "Value",
    height: int = 300,
) -> alt.Chart:
    """Stacked area of sleeve values over time (or single-point levels)."""
    data = long_df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    legend_sel = alt.selection_point(fields=[series_col], bind="legend")
    highlight = alt.selection_point(
        fields=[series_col], on="pointerover", empty=True
    )
    return (
        alt.Chart(data)
        .mark_area(opacity=0.75, line={"strokeWidth": 1.5})
        .encode(
            x=alt.X(f"{date_col}:T", title="Date"),
            y=alt.Y(f"{value_col}:Q", title="Value (₹)", stack="zero"),
            color=alt.Color(
                f"{series_col}:N",
                scale=alt.Scale(range=ALLOC_COLORS),
                legend=alt.Legend(title="Scheme", orient="bottom"),
            ),
            opacity=alt.condition(
                legend_sel & highlight, alt.value(0.9), alt.value(0.35)
            ),
            tooltip=[
                alt.Tooltip(f"{date_col}:T", format="%d-%b-%Y"),
                alt.Tooltip(f"{series_col}:N"),
                alt.Tooltip(f"{value_col}:Q", format=",.2f"),
            ],
        )
        .add_params(legend_sel, highlight)
        .properties(**_dark_props(height))
        .interactive()
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

    legend_sel = alt.selection_point(fields=["Scenario"], bind="legend")

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
        return hist_chart.properties(**_dark_props(height)).interactive()

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
            opacity=alt.condition(legend_sel, alt.value(1.0), alt.value(0.25)),
            tooltip=[
                alt.Tooltip("Date:T", format="%d-%b-%Y"),
                alt.Tooltip("NAV:Q", format=".4f"),
                alt.Tooltip("Scenario:N"),
            ],
        )
        .add_params(legend_sel)
    )
    return (hist_chart + fc_chart).properties(**_dark_props(height)).interactive()
