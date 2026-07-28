"""
analytics.py
------------
Builds Plotly charts for the Analytics section of the dashboard.
All charts share a minimal, professional visual style and support both
light and dark themes via the `dark` parameter.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CATEGORY_COLORS = {
    "Urgent": "#DC2626",
    "Job/Internship": "#7C3AED",
    "Follow-Up": "#F59E0B",
    "News & Promotions": "#EC4899",
    "Spam": "#9CA3AF",
}

CHART_FONT_FAMILY = "Inter, Helvetica, Arial, sans-serif"


def _base_layout(fig, title, dark: bool = False):
    bg = "#1A1D23" if dark else "white"
    title_color = "#F3F4F6" if dark else "#111827"
    font_color = "#D1D5DB" if dark else "#1F2937"
    grid_color = "#2A2E37" if dark else "#F3F4F6"
    axis_line_color = "#2A2E37" if dark else "#E5E7EB"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family=CHART_FONT_FAMILY, color=title_color)),
        font=dict(family=CHART_FONT_FAMILY, size=13, color=font_color),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=font_color),
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=axis_line_color)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False)
    return fig


def category_distribution_chart(df: pd.DataFrame, dark: bool = False):
    """Donut chart showing the share of emails per category."""
    if df.empty:
        return go.Figure()

    counts = df["category"].value_counts().reindex(CATEGORY_COLORS.keys(), fill_value=0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.6,
                marker=dict(colors=[CATEGORY_COLORS[c] for c in counts.index]),
                textinfo="label+percent",
            )
        ]
    )
    return _base_layout(fig, "Category Distribution", dark=dark)


def daily_volume_chart(df: pd.DataFrame, dark: bool = False):
    """Bar chart of email volume per day."""
    if df.empty:
        return go.Figure()

    daily = df.copy()
    daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
    volume = daily.groupby("date").size().reset_index(name="count")

    fig = px.bar(volume, x="date", y="count")
    fig.update_traces(marker_color="#7C3AED")
    return _base_layout(fig, "Daily Email Volume", dark=dark)


def sender_frequency_chart(df: pd.DataFrame, top_n: int = 10, dark: bool = False):
    """Horizontal bar chart of the most frequent senders."""
    if df.empty:
        return go.Figure()

    freq = df["sender"].value_counts().head(top_n).sort_values()
    fig = go.Figure(
        go.Bar(
            x=freq.values,
            y=freq.index,
            orientation="h",
            marker_color="#EC4899",
        )
    )
    return _base_layout(fig, f"Top {top_n} Senders by Frequency", dark=dark)


def category_trend_chart(df: pd.DataFrame, dark: bool = False):
    """Line chart showing how each category trends over time."""
    if df.empty:
        return go.Figure()

    trend = df.copy()
    trend["date"] = pd.to_datetime(trend["timestamp"]).dt.date
    grouped = trend.groupby(["date", "category"]).size().reset_index(name="count")

    fig = go.Figure()
    for category, color in CATEGORY_COLORS.items():
        subset = grouped[grouped["category"] == category]
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["count"],
                mode="lines+markers",
                name=category,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )
    return _base_layout(fig, "Email Trends Over Time", dark=dark)


