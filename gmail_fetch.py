"""
gmail_fetch.py
--------------
Fetches recent emails from Gmail and normalizes them into a list of dicts
that the rest of the app (classifier, analytics, UI) can consume.
"""

import base64
from datetime import datetime
from email.utils import parsedate_to_datetime

from googleapiclient.errors import HttpError


class GmailFetchError(Exception):
    """Raised when emails cannot be retrieved from Gmail."""
    pass


def _get_header(headers, name):
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_body_part(data):
    try:
        return base64.urlsafe_b64decode(data.encode("UTF-8")).decode("UTF-8", errors="ignore")
    except Exception:
        return ""


def _extract_plain_text(payload):
    """Walks the MIME payload tree and returns the best plain-text body found."""
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})

    if mime_type == "text/plain" and body.get("data"):
        return _decode_body_part(body["data"])

    for part in payload.get("parts", []) or []:
        text = _extract_plain_text(part)
        if text:
            return text

    # Fallback: html or nothing
    if mime_type == "text/html" and body.get("data"):
        return _decode_body_part(body["data"])

    return ""


def _parse_message(msg):
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    sender = _get_header(headers, "From")
    subject = _get_header(headers, "Subject") or "(no subject)"
    date_str = _get_header(headers, "Date")

    try:
        timestamp = parsedate_to_datetime(date_str) if date_str else None
    except Exception:
        timestamp = None

    if timestamp is None:
        # Fallback to Gmail's internalDate (epoch millis)
        internal_date = msg.get("internalDate")
        if internal_date:
            timestamp = datetime.fromtimestamp(int(internal_date) / 1000)
        else:
            timestamp = datetime.now()

    body_text = _extract_plain_text(payload)
    snippet = msg.get("snippet", "")
    label_ids = msg.get("labelIds", []) or []

    return {
        "message_id": msg.get("id"),
        "sender": sender,
        "subject": subject,
        "snippet": snippet,
        "body": body_text[:2000] if body_text else snippet,
        "timestamp": timestamp,
        "is_unread": "UNREAD" in label_ids,
    }


def fetch_emails(service, max_results=50):
    """
    Fetches the most recent `max_results` emails from the user's inbox.

    Returns a list of dicts with keys:
        message_id, sender, subject, snippet, body, timestamp
    """
    try:
        response = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX"], maxResults=max_results)
            .execute()
        )
    except HttpError as exc:
        raise GmailFetchError(f"Gmail API request failed: {exc}")
    except Exception as exc:
        raise GmailFetchError(f"Network error while contacting Gmail: {exc}")

    message_refs = response.get("messages", [])
    if not message_refs:
        return []

    emails = []
    for ref in message_refs:
        try:
            full_msg = (
                service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            emails.append(_parse_message(full_msg))
        except HttpError:
            # Skip individual messages that fail to load rather than
            # failing the whole batch.
            continue

    return emails

