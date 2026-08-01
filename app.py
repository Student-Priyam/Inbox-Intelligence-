"""
app.py
------
Main Streamlit entry point for the AI Email Management Dashboard.
Ties together auth, gmail_fetch, classifier, analytics, and utils into a
single-page-app style dashboard with sidebar navigation.
"""
import auth

# --- top of app.py, before any UI is drawn ---
try:
    auth.handle_redirect()
except auth.AuthError as e:
    st.error(str(e))
    st.stop()

if not auth.is_authenticated():
    # ...render your existing landing page markup here...
    try:
        login_url = auth.get_login_url()
        st.link_button("Sign in with Google", login_url)
    except auth.AuthError as e:
        st.error(str(e))
    st.stop()

# From here on, the user is authenticated — existing Refresh Inbox /
# Sign out logic can stay exactly as it was:
if st.sidebar.button("Sign out"):
    auth.sign_out()
    st.rerun()
import streamlit as st
import pandas as pd

from auth import get_gmail_service, is_authenticated, sign_out, AuthError
from gmail_fetch import fetch_emails, GmailFetchError
from classifier import classify_emails, CATEGORY_ORDER
from analytics import (
    category_distribution_chart,
    daily_volume_chart,
    sender_frequency_chart,
    category_trend_chart,
)
from utils import (
    build_dataframe,
    get_summary_stats,
    truncate_text,
    format_sender,
    to_csv_bytes,
    apply_filters,
    apply_search,
    count_unread_urgent,
)

st.set_page_config(
    page_title="Inbox Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Global styling — minimal, enterprise-grade look (Stripe/Linear/Vercel-ish)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Theme (light / dark mode)
# --------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

LIGHT_THEME = {
    "app_bg": "#FFFFFF",
    "sidebar_bg": "#FAFAFA",
    "sidebar_border": "#E5E7EB",
    "text_primary": "#111827",
    "text_secondary": "#6B7280",
    "card_bg": "#FFFFFF",
    "card_border": "#E5E7EB",
    "divider": "#E5E7EB",
    "hero_start": "#F5F3FF",
    "hero_end": "#FDF2F8",
    "hero_border": "#EDE9FE",
    "input_bg": "#FFFFFF",
    "input_border": "#D1D5DB",
    "alert_bg": "#F9FAFB",
}

DARK_THEME = {
    "app_bg": "#0F1115",
    "sidebar_bg": "#15171C",
    "sidebar_border": "#262A33",
    "text_primary": "#F3F4F6",
    "text_secondary": "#9CA3AF",
    "card_bg": "#1A1D23",
    "card_border": "#2A2E37",
    "divider": "#2A2E37",
    "hero_start": "#241B36",
    "hero_end": "#33202C",
    "hero_border": "#3A2A4D",
    "input_bg": "#1A1D23",
    "input_border": "#374151",
    "alert_bg": "#1A1D23",
}


def build_custom_css(dark: bool) -> str:
    t = DARK_THEME if dark else LIGHT_THEME
    return f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}

    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }}

    .stApp {{
        background-color: {t['app_bg']};
        color: {t['text_primary']};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']};
        border-right: 1px solid {t['sidebar_border']};
    }}

    section[data-testid="stSidebar"] .stRadio > label {{
        font-weight: 500;
        color: {t['text_primary']};
    }}

    p, span, label, div[data-testid="stMarkdownContainer"] {{
        color: {t['text_primary']};
    }}

    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {t['text_secondary']} !important;
    }}

    hr {{ border-color: {t['divider']}; }}

    /* Brand mark — the "Inbox Intelligence" wordmark, gradient text.
       Stays the same violet-to-pink gradient in both themes since it's
       the fixed brand identity, not a themed surface. */
    .brand-mark {{
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
        background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
        margin-bottom: 0.15rem;
    }}
    .brand-mark-sm {{ font-size: 1.55rem; }}
    .brand-mark-lg {{ font-size: 3.4rem; }}

    /* Dashboard gets a slightly larger, brand-gradient heading since it's
       the app's home view */
    .dashboard-title {{
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
        margin-bottom: 0.15rem;
    }}

    /* Page headings (Dashboard / Inbox / Analytics / Settings) */
    .app-title {{
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {t['text_primary']};
        margin-bottom: 0.15rem;
    }}

    .app-subtitle {{
        font-size: 0.95rem;
        color: {t['text_secondary']};
        margin-bottom: 1.6rem;
    }}

    .metric-card {{
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-top: 3px solid #7C3AED;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}

    .metric-label {{
        font-size: 0.78rem;
        color: {t['text_secondary']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}

    .metric-value {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {t['text_primary']};
        margin-top: 4px;
    }}

    .section-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {t['text_primary']};
        margin: 1.8rem 0 0.7rem 0;
        letter-spacing: -0.01em;
    }}

    .pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }}

    /* Pill badges keep their pastel colors in both themes for consistent,
       reliable legibility as small accent chips. */
    .pill-urgent {{ background: #FEE2E2; color: #B91C1C; }}
    .pill-followup {{ background: #FEF3C7; color: #B45309; }}
    .pill-news {{ background: #FCE7F3; color: #BE185D; }}
    .pill-job {{ background: #EDE9FE; color: #6D28D9; }}
    .pill-spam {{ background: #F3F4F6; color: #4B5563; }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {t['card_border']};
        border-radius: 8px;
    }}

    div[data-testid="stAlert"] {{
        background-color: {t['alert_bg']};
        border: 1px solid {t['card_border']};
    }}

    div[data-testid="stExpander"] {{
        background-color: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 10px;
    }}

    input, textarea {{
        background-color: {t['input_bg']} !important;
        color: {t['text_primary']} !important;
        border-color: {t['input_border']} !important;
    }}

    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid {t['card_border']};
        background-color: {t['card_bg']};
        color: {t['text_primary']};
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%);
        border: none;
        color: #FFFFFF;
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.30);
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.92;
        color: #FFFFFF;
    }}
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] div,
    .stButton > button[kind="primary"] span {{
        color: #FFFFFF !important;
    }}

    /* ---------------- Landing page ---------------- */
    .landing-hero {{
        background: linear-gradient(135deg, {t['hero_start']} 0%, {t['hero_end']} 100%);
        border: 1px solid {t['hero_border']};
        border-radius: 28px;
        padding: 72px 40px 56px 40px;
        text-align: center;
        margin-bottom: 2.2rem;
    }}

    .landing-tagline {{
        font-size: 1.15rem;
        color: {t['text_secondary']};
        max-width: 620px;
        margin: 14px auto 0 auto;
        line-height: 1.55;
    }}

    .landing-badge {{
        display: inline-block;
        background: {t['card_bg']};
        border: 1px solid {t['hero_border']};
        color: #7C3AED;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 999px;
        margin-bottom: 18px;
    }}

    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 0 0 2rem 0;
    }}

    .feature-card {{
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}

    .feature-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-bottom: 10px;
    }}
    .feature-dot-urgent {{ background: #DC2626; }}
    .feature-dot-job {{ background: #7C3AED; }}
    .feature-dot-news {{ background: #EC4899; }}
    .feature-dot-spam {{ background: #9CA3AF; }}

    .feature-title {{
        font-weight: 700;
        font-size: 0.95rem;
        color: {t['text_primary']};
        margin-bottom: 4px;
    }}

    .feature-desc {{
        font-size: 0.83rem;
        color: {t['text_secondary']};
        line-height: 1.45;
    }}

    .trust-line {{
        text-align: center;
        font-size: 0.82rem;
        color: {t['text_secondary']};
        margin-top: 0.6rem;
    }}
</style>
"""


st.markdown(build_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)

PILL_CLASS = {
    "Urgent": "pill-urgent",
    "Job/Internship": "pill-job",
    "Follow-Up": "pill-followup",
    "News & Promotions": "pill-news",
    "Spam": "pill-spam",
}

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "emails_df" not in st.session_state:
    st.session_state.emails_df = pd.DataFrame()
if "selected_message_id" not in st.session_state:
    st.session_state.selected_message_id = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None


# --------------------------------------------------------------------------
# Core actions
# --------------------------------------------------------------------------
def refresh_inbox(max_results=50):
    """Fetches emails from Gmail and runs them through the classifier."""
    st.session_state.last_error = None
    try:
        with st.spinner("Connecting to Gmail..."):
            service = get_gmail_service()

        with st.spinner(f"Fetching latest {max_results} emails..."):
            raw_emails = fetch_emails(service, max_results=max_results)

        if not raw_emails:
            st.session_state.emails_df = pd.DataFrame()
            st.session_state.last_error = "empty_inbox"
            return

        progress_bar = st.progress(0, text="Classifying emails...")

        def _on_progress(done, total):
            progress_bar.progress(done / total, text=f"Classifying emails... ({done}/{total})")

        classified = classify_emails(raw_emails, progress_callback=_on_progress)
        progress_bar.empty()

        st.session_state.emails_df = build_dataframe(classified)

    except AuthError as exc:
        st.session_state.last_error = str(exc)
    except GmailFetchError as exc:
        st.session_state.last_error = str(exc)
    except Exception as exc:
        st.session_state.last_error = f"Unexpected error: {exc}"


# --------------------------------------------------------------------------
# Landing page (shown before sign-in)
# --------------------------------------------------------------------------
def render_landing_page():
    st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-badge">AI-Powered Inbox Triage</div>
            <div class="brand-mark brand-mark-lg">Inbox Intelligence</div>
            <div class="landing-tagline">
                Your Gmail inbox, automatically sorted into what actually needs
                you — deadlines and mandatory sessions, job and internship
                alerts, newsletters, and the noise you can safely ignore.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <span class="feature-dot feature-dot-urgent"></span>
                <div class="feature-title">Urgent</div>
                <div class="feature-desc">
                    Real deadlines, mandatory sessions, and interview calls
                    surfaced first — never buried in your feed.
                </div>
            </div>
            <div class="feature-card">
                <span class="feature-dot feature-dot-job"></span>
                <div class="feature-title">Job &amp; Internship</div>
                <div class="feature-desc">
                    LinkedIn, Naukri, Internshala and other alerts sorted
                    into their own lane automatically.
                </div>
            </div>
            <div class="feature-card">
                <span class="feature-dot feature-dot-news"></span>
                <div class="feature-title">News &amp; Promotions</div>
                <div class="feature-desc">
                    Webinars, newsletters, and announcements — read them
                    when you actually have time.
                </div>
            </div>
            <div class="feature-card">
                <span class="feature-dot feature-dot-spam"></span>
                <div class="feature-title">Spam</div>
                <div class="feature-desc">
                    Promotional noise and phishing attempts filtered out
                    before they waste your time.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1, 1])
    with center:
        if st.button("Sign in with Google", type="primary", use_container_width=True):
            with st.spinner("Waiting for Google sign-in in your browser..."):
                try:
                    get_gmail_service()
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))

    st.markdown(
        """<div class="trust-line">
            Read-only Gmail access. Your credentials never leave your machine.
            First time here? See README.md for the Gmail API and OAuth setup guide.
        </div>""",
        unsafe_allow_html=True,
    )


if not is_authenticated():
    render_landing_page()
    st.stop()


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='brand-mark brand-mark-sm'>Inbox Intelligence</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>AI email management</div>", unsafe_allow_html=True
    )
    page = st.radio(
        "Navigation",
        ["Dashboard", "Inbox", "Analytics", "Settings"],
        label_visibility="collapsed",
    )
    st.divider()
    max_results = st.slider("Emails to fetch", min_value=10, max_value=100, value=50, step=10)
    if st.button("Refresh Inbox", type="primary", use_container_width=True):
        refresh_inbox(max_results=max_results)
    st.divider()
    if st.button("Sign out", use_container_width=True):
        sign_out()
        st.session_state.emails_df = pd.DataFrame()
        st.rerun()

df = st.session_state.emails_df

# --------------------------------------------------------------------------
# Error / empty states
# --------------------------------------------------------------------------
if st.session_state.last_error == "empty_inbox":
    st.info("Your inbox is empty, or no messages matched the fetch request.")
elif st.session_state.last_error:
    st.error(st.session_state.last_error)

if df.empty and st.session_state.last_error is None:
    st.markdown("<div class='app-title'>Welcome to Inbox Intelligence</div>", unsafe_allow_html=True)
    st.write(
        "Click **Refresh Inbox** in the sidebar to fetch and classify your latest emails."
    )
    st.stop()


# --------------------------------------------------------------------------
# Urgent-unread notification
# --------------------------------------------------------------------------
if not df.empty:
    unread_urgent = count_unread_urgent(df)
    if unread_urgent > 0:
        plural = "s" if unread_urgent != 1 else ""
        st.warning(
            f"You have {unread_urgent} unread urgent email{plural} that may need your attention."
        )
        if not st.session_state.get("urgent_toast_shown", False):
            st.toast(f"{unread_urgent} unread urgent email{plural} waiting", icon="🔔")
            st.session_state.urgent_toast_shown = True


# --------------------------------------------------------------------------
# Dashboard page
# --------------------------------------------------------------------------
def render_dashboard(df):
    st.markdown("<div class='dashboard-title'>Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Overview of your inbox activity</div>",
        unsafe_allow_html=True,
    )

    stats = get_summary_stats(df)
    labels = ["Total Emails"] + CATEGORY_ORDER
    keys = ["total"] + CATEGORY_ORDER

    cols = st.columns(len(labels))
    for col, label, key in zip(cols, labels, keys):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{stats[key]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    # Each chart already renders its own title (e.g. "Category Distribution",
    # "Daily Email Volume"), so no extra markdown header is added here —
    # that previously produced two overlapping "Category Distribution" labels.
    left, right = st.columns([1, 1.4])
    with left:
        st.plotly_chart(category_distribution_chart(df, dark=st.session_state.dark_mode), use_container_width=True, key="dashboard_category_chart")
    with right:
        st.plotly_chart(daily_volume_chart(df, dark=st.session_state.dark_mode), use_container_width=True, key="dashboard_volume_chart")

    st.markdown("<div class='section-title'>Needs Attention</div>", unsafe_allow_html=True)
    urgent_df = df[df["category"] == "Urgent"].head(5)
    if urgent_df.empty:
        st.caption("No urgent emails right now.")
    else:
        for _, row in urgent_df.iterrows():
            st.markdown(
                f"""<div class="metric-card" style="margin-bottom:8px;">
                        <span class="pill pill-urgent">Urgent</span>
                        <span style="margin-left:8px; font-weight:600;">{row['subject']}</span>
                        <div style="color:#6B7280; font-size:0.85rem; margin-top:4px;">
                            {format_sender(row['sender'])} · {row['timestamp'].strftime('%b %d, %Y %H:%M')}
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Inbox page
# --------------------------------------------------------------------------
def render_inbox(df):
    st.markdown("<div class='app-title'>Inbox</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Search, filter, and review classified emails</div>",
        unsafe_allow_html=True,
    )

    search_col, export_col = st.columns([3, 1])
    with search_col:
        query = st.text_input("Search by sender, subject, or content", placeholder="Search emails...")
    with export_col:
        st.write("")
        st.download_button(
            "Export CSV",
            data=to_csv_bytes(df),
            file_name="classified_emails.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            categories = st.multiselect(
                "Category", CATEGORY_ORDER, default=[]
            )
        with f2:
            sender_query = st.text_input("Filter by sender contains")
        with f3:
            min_confidence = st.slider("Minimum confidence (%)", 0, 100, 0)

        min_date = df["timestamp"].dt.date.min()
        max_date = df["timestamp"].dt.date.max()
        date_range = st.date_input("Date range", value=(min_date, max_date))

    filtered = apply_filters(
        df,
        categories=categories or None,
        sender_query=sender_query or None,
        date_range=date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None,
        min_confidence=min_confidence,
    )
    filtered = apply_search(filtered, query)

    st.caption(f"Showing {len(filtered)} of {len(df)} emails")

    display_df = filtered.copy()
    display_df["Sender"] = display_df["sender"].apply(format_sender)
    display_df["Subject"] = display_df["subject"]
    display_df["Date"] = display_df["timestamp"].dt.strftime("%b %d, %Y %H:%M")
    display_df["Category"] = display_df["category"]
    display_df["Confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1f}%")
    display_df["Suggested Action"] = display_df["suggested_action"]

    table = display_df[
        ["Sender", "Subject", "Date", "Category", "Confidence", "Suggested Action"]
    ]

    event = st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        selected_idx = filtered.index[selected_rows[0]]
        render_email_detail(df.loc[selected_idx])


def render_email_detail(row):
    st.markdown("<div class='section-title'>Email Detail</div>", unsafe_allow_html=True)
    pill_class = PILL_CLASS.get(row["category"], "pill-news")

    st.markdown(
        f"""<div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:700; font-size:1.05rem;">{row['subject']}</div>
                        <div style="color:#6B7280; font-size:0.85rem; margin-top:2px;">
                            {format_sender(row['sender'])} &middot; {row['timestamp'].strftime('%b %d, %Y %H:%M')}
                        </div>
                    </div>
                    <span class="pill {pill_class}">{row['category']}</span>
                </div>
                <hr style="border-color:#E5E7EB;">
                <p style="color:#374151; line-height:1.5;">{row['snippet']}</p>
                <hr style="border-color:#E5E7EB;">
                <div style="font-size:0.85rem; color:#6B7280;">
                    <b>Confidence:</b> {row['confidence']:.1f}%&nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Suggested Action:</b> {row['suggested_action']}
                </div>
                <div style="font-size:0.85rem; color:#6B7280; margin-top:6px;">
                    <b>AI Explanation:</b> {row['explanation']}
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Analytics page
# --------------------------------------------------------------------------
def render_analytics(df):
    st.markdown("<div class='app-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Trends and patterns across your inbox</div>",
        unsafe_allow_html=True,
    )

    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.plotly_chart(category_distribution_chart(df, dark=st.session_state.dark_mode), use_container_width=True, key="analytics_category_chart")
    with row1_right:
        st.plotly_chart(daily_volume_chart(df, dark=st.session_state.dark_mode), use_container_width=True, key="analytics_volume_chart")

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.plotly_chart(sender_frequency_chart(df, dark=st.session_state.dark_mode), use_container_width=True, key="analytics_sender_chart")
    with row2_right:
        st.plotly_chart(category_trend_chart(df, dark=st.session_state.dark_mode), use_container_width=True, key="analytics_trend_chart")


# --------------------------------------------------------------------------
# Settings page
# --------------------------------------------------------------------------
def render_settings():
    st.markdown("<div class='app-title'>Settings</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Account and application preferences</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Appearance</div>", unsafe_allow_html=True)
    st.toggle("Dark Mode", key="dark_mode")
    st.caption("Switches the dashboard between light and dark color schemes.")

    st.markdown("<div class='section-title'>Account</div>", unsafe_allow_html=True)
    st.write("Signed in with Google. Access scope: Gmail read-only.")
    if st.button("Sign out"):
        sign_out()
        st.session_state.emails_df = pd.DataFrame()
        st.rerun()

    st.markdown("<div class='section-title'>Classification Model</div>", unsafe_allow_html=True)
    st.write(
        "Emails are classified into five categories tailored for a student "
        "inbox — Urgent, Job/Internship, Follow-Up, News & Promotions, and "
        "Spam. A few reliable rule-based checks (e.g. LinkedIn/Naukri/"
        "Internshala senders, webinar/livestream language) run first, and "
        "everything else falls back to the pre-trained zero-shot model "
        "`facebook/bart-large-mnli` from Hugging Face. No training or "
        "fine-tuning is performed."
    )

    st.markdown("<div class='section-title'>Data & Privacy</div>", unsafe_allow_html=True)
    st.write(
        "Emails are fetched on demand and processed locally in this session. "
        "Nothing is sent to third parties other than Google (to fetch mail) "
        "and the local Hugging Face model (to classify it)."
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
if page == "Dashboard":
    render_dashboard(df)
elif page == "Inbox":
    render_inbox(df)
elif page == "Analytics":
    render_analytics(df)
elif page == "Settings":
    render_settings()



