"""
utils.py
--------
Small shared helper functions used across the app: formatting, CSV export,
and summary statistics.
"""

import io
import pandas as pd
from classifier import CATEGORY_ORDER


def build_dataframe(classified_emails):
    """Converts a list of classified email dicts into a display-ready DataFrame."""
    if not classified_emails:
        return pd.DataFrame(
            columns=[
                "message_id", "sender", "subject", "snippet", "body",
                "timestamp", "is_unread", "category", "confidence",
                "suggested_action", "explanation",
            ]
        )
    df = pd.DataFrame(classified_emails)
    # Gmail dates include timezone offsets (e.g. +0530), and different emails
    # can carry different offsets. Normalize everything to UTC first, then
    # drop the timezone so the rest of the app can treat timestamps as plain
    # (tz-naive) datetimes without pandas raising a conversion error.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def get_summary_stats(df: pd.DataFrame):
    """Returns counts used by the summary cards on the Dashboard page."""
    stats = {"total": 0 if df.empty else int(len(df))}
    counts = df["category"].value_counts() if not df.empty else {}
    for cat in CATEGORY_ORDER:
        stats[cat] = int(counts.get(cat, 0)) if not df.empty else 0
    return stats


def count_unread_urgent(df: pd.DataFrame) -> int:
    """Counts emails that are both unread and classified as Urgent."""
    if df.empty or "is_unread" not in df.columns or "category" not in df.columns:
        return 0
    return int(((df["category"] == "Urgent") & (df["is_unread"] == True)).sum())  # noqa: E712


def truncate_text(text: str, max_chars: int = 90) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def format_sender(sender: str) -> str:
    """Strips the display name down to 'Name <email>' -> 'Name' for compact display."""
    if "<" in sender:
        return sender.split("<")[0].strip().strip('"')
    return sender


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializes the DataFrame to CSV bytes for st.download_button."""
    buffer = io.StringIO()
    export_df = df.drop(columns=["body"], errors="ignore")
    export_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def apply_filters(df, categories=None, sender_query=None, date_range=None, min_confidence=0):
    """Applies the Inbox page filters to the classified emails DataFrame."""
    filtered = df.copy()

    if categories:
        filtered = filtered[filtered["category"].isin(categories)]

    if sender_query:
        filtered = filtered[
            filtered["sender"].str.contains(sender_query, case=False, na=False)
        ]

    if date_range and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)
        ]

    if min_confidence:
        filtered = filtered[filtered["confidence"] >= min_confidence]

    return filtered


def apply_search(df, query):
    """Searches sender, subject, and body/snippet for a free-text query."""
    if not query:
        return df
    query = query.lower()
    mask = (
        df["sender"].str.lower().str.contains(query, na=False)
        | df["subject"].str.lower().str.contains(query, na=False)
        | df["snippet"].str.lower().str.contains(query, na=False)
    )
    return df[mask]


