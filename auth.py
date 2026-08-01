"""
auth.py
-------
Handles Google OAuth 2.0 authentication for Gmail API access using the
**Web application** flow, which works both locally and on Streamlit
Community Cloud.

Why this replaces the old InstalledAppFlow implementation:
- InstalledAppFlow.run_local_server() spawns a local browser and listens
  on a local port — there is no "local browser" on a remote Streamlit
  Cloud server, so it can never complete there.
- credentials.json was excluded from git (correctly, for security) but
  that also means it's never available on Cloud at all.
- token.json as a file assumes one user on one machine. On Streamlit
  Cloud, many users can hit the same running app, so a shared file would
  leak one user's Gmail access to another. Credentials must be kept
  per-browser-session instead.

This version:
1. Reads the OAuth client secret from `st.secrets` (never committed to git).
2. Sends the user's browser to Google's consent screen.
3. Google redirects back to the app's own URL with `?code=...`.
4. The code is exchanged for credentials, which are stored in
   `st.session_state` — scoped to that one user's session only.
"""

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class AuthError(Exception):
    """Raised when authentication cannot be completed."""
    pass


def _redirect_uri() -> str:
    """
    Must exactly match a URI registered on the OAuth client in Google
    Cloud Console (no trailing slash mismatch, correct http/https).
    Read from secrets so it's correct per-environment (local vs. Cloud)
    without touching code.
    """
    try:
        return st.secrets["google_oauth"]["redirect_uri"]
    except (KeyError, FileNotFoundError) as exc:
        raise AuthError(
            "Missing `redirect_uri` under [google_oauth] in Streamlit "
            "secrets. See README.md > Deploying to Streamlit Cloud."
        ) from exc


def _build_flow() -> Flow:
    try:
        client_config = {
            "web": {
                "client_id": st.secrets["google_oauth"]["client_id"],
                "client_secret": st.secrets["google_oauth"]["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [_redirect_uri()],
            }
        }
    except (KeyError, FileNotFoundError) as exc:
        raise AuthError(
            "Missing Google OAuth credentials in Streamlit secrets. Add a "
            "[google_oauth] section with client_id, client_secret, and "
            "redirect_uri — see README.md > Deploying to Streamlit Cloud."
        ) from exc

    flow = Flow.from_client_config(client_config, scopes=SCOPES)
    flow.redirect_uri = _redirect_uri()
    return flow


def get_login_url() -> str:
    """Returns the Google consent screen URL to send the user to."""
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    return auth_url


def handle_redirect() -> bool:
    """
    Call once near the top of app.py on every run, before rendering the
    page. If Google just redirected back with `?code=...`, exchanges it
    for credentials and stores them in session state.

    Returns True if the user is signed in (just now, or from earlier
    this session); False otherwise.
    """
    if "credentials" in st.session_state:
        return True

    code = st.query_params.get("code")
    if not code:
        return False

    flow = _build_flow()
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        raise AuthError(f"Google sign-in failed: {exc}") from exc

    st.session_state["credentials"] = flow.credentials
    st.query_params.clear()  # scrub ?code=...&state=... from the URL bar
    return True


def get_credentials() -> Credentials:
    """Returns valid credentials for the current session, refreshing if needed."""
    creds = st.session_state.get("credentials")
    if not creds:
        raise AuthError("Not signed in yet.")

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        st.session_state["credentials"] = creds

    return creds


def get_gmail_service():
    """Returns an authenticated Gmail API service client for this session."""
    creds = get_credentials()
    try:
        return build("gmail", "v1", credentials=creds)
    except Exception as exc:
        raise AuthError(f"Could not connect to Gmail API: {exc}") from exc


def is_authenticated() -> bool:
    """Quick check used by the UI to decide whether to show the sign-in screen."""
    return "credentials" in st.session_state


def sign_out():
    """Clears this session's credentials, forcing re-authentication next time."""
    st.session_state.pop("credentials", None)
