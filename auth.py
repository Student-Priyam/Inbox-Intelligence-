"""
auth.py
-------
Handles Google OAuth 2.0 authentication for Gmail API access.

Flow:
1. User provides a `credentials.json` (OAuth client secret) downloaded from
   Google Cloud Console and places it in the `credentials/` folder.
2. On first run, a local browser window opens asking the user to sign in
   and grant read-only Gmail access.
3. A `token.json` is cached in `credentials/` so the user does not need to
   re-authenticate on every run (refreshed automatically when expired).
"""

import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only scope is sufficient for fetching and classifying emails.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CREDENTIALS_DIR = "credentials"
CLIENT_SECRET_FILE = os.path.join(CREDENTIALS_DIR, "credentials.json")
TOKEN_FILE = os.path.join(CREDENTIALS_DIR, "token.json")


class AuthError(Exception):
    """Raised when authentication cannot be completed."""
    pass


def _load_cached_credentials():
    """Load cached OAuth credentials from token.json, if present."""
    if os.path.exists(TOKEN_FILE):
        try:
            return Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def _save_credentials(creds):
    os.makedirs(CREDENTIALS_DIR, exist_ok=True)
    with open(TOKEN_FILE, "w") as token_file:
        token_file.write(creds.to_json())


def get_credentials():
    """
    Returns valid Google OAuth credentials, refreshing or re-authenticating
    as needed. Raises AuthError with a user-friendly message on failure.
    """
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise AuthError(
            "Missing credentials/credentials.json. "
            "Download your OAuth client secret from Google Cloud Console "
            "and place it at credentials/credentials.json. "
            "See README.md > Google OAuth Setup for step-by-step instructions."
        )

    creds = _load_cached_credentials()

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds)
            return creds
        except Exception as exc:
            raise AuthError(f"Could not refresh Google session: {exc}")

    # No valid cached credentials -> run the interactive OAuth flow.
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        _save_credentials(creds)
        return creds
    except Exception as exc:
        raise AuthError(f"Google sign-in failed: {exc}")


def get_gmail_service():
    """Returns an authenticated Gmail API service client."""
    creds = get_credentials()
    try:
        return build("gmail", "v1", credentials=creds)
    except Exception as exc:
        raise AuthError(f"Could not connect to Gmail API: {exc}")


def is_authenticated():
    """Quick check used by the UI to decide whether to show the sign-in screen."""
    creds = _load_cached_credentials()
    return bool(creds and (creds.valid or (creds.expired and creds.refresh_token)))


def sign_out():
    """Deletes the cached token, forcing re-authentication next time."""
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
