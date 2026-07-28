"""
classifier.py
-------------
Classifies emails into categories tailored for a student/undergrad inbox:
Urgent, Job/Internship, Follow-Up, News & Promotions, Spam.

Two layers are used together:
1. Lightweight rule-based checks for patterns a general-purpose model
   struggles with (job platform senders, webinar/livestream broadcasts).
   These are cheap, explainable, and highly reliable for their narrow cases.
2. Zero-shot classification (facebook/bart-large-mnli) as the fallback for
   everything else, using natural-language hypotheses per category.

No training or fine-tuning is performed anywhere.
"""

import re
import streamlit as st
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

# Single source of truth for category order used across the app
# (dashboard cards, filters, chart colors, etc.)
CATEGORY_ORDER = ["Urgent", "Job/Internship", "Follow-Up", "News & Promotions", "Spam"]

CATEGORY_HYPOTHESES = {
    "Urgent": (
        "This email requires the specific reader to personally take action "
        "very soon, such as confirming attendance, submitting a document, "
        "attending an interview, or responding to a real deadline or "
        "schedule change that directly affects them. It is not a mass "
        "announcement, newsletter, webinar invite, or advertisement."
    ),
    "Job/Internship": (
        "This email is about a job or internship opportunity, a job "
        "application status update, a recruiter message, or a job alert "
        "from a company or a platform such as LinkedIn, Naukri, "
        "Internshala, Indeed, or Glassdoor."
    ),
    "Follow-Up": (
        "This email asks the reader to reply, confirm, review, or approve "
        "something within the next day or two, but is not extremely "
        "time-critical and is not a mass announcement."
    ),
    "News & Promotions": (
        "This email is a general announcement, newsletter, webinar or "
        "livestream invite, event promotion, product update, or marketing "
        "message sent to a broad audience. It does not require the "
        "specific reader to personally act before a real deadline."
    ),
    "Spam": (
        "This email is unsolicited advertising, a suspicious or phishing "
        "message, or clearly irrelevant promotional spam."
    ),
}

SUGGESTED_ACTIONS = {
    "Urgent": "Respond Immediately",
    "Job/Internship": "Review Opportunity",
    "Follow-Up": "Reply Within 24 Hours",
    "News & Promotions": "Read When Convenient",
    "Spam": "Ignore or Delete",
}

EXPLANATION_TEMPLATES = {
    "Urgent": (
        "The email asks you personally to act soon, such as a deadline, "
        "interview, or schedule change, rather than being a broad "
        "announcement."
    ),
    "Job/Internship": (
        "The email is a job or internship alert, application update, or "
        "recruiter message."
    ),
    "Follow-Up": (
        "The email asks for a reply, confirmation, or review, so it "
        "should be addressed soon but is not time-critical."
    ),
    "News & Promotions": (
        "The email is a broadcast-style announcement, newsletter, or "
        "event/webinar promotion sent to a wide audience."
    ),
    "Spam": (
        "The email shows characteristics of promotional or unsolicited "
        "content and is unlikely to require your attention."
    ),
}

# --------------------------------------------------------------------------
# Rule-based pre-checks (run before the ML model, and skip it when they hit)
# --------------------------------------------------------------------------

# Real obligation/deadline language always wins first — this is checked
# BEFORE the broadcast/newsletter check below, so a mandatory college
# session or a hard deadline is never demoted just because the email is
# formatted like an announcement (subject lines with "Session", "Launch",
# a scheduled date, a meeting link, etc. all look like broadcasts on the
# surface, but "mandatory" / "without fail" / "closely monitored" are the
# actual signal that the reader must personally act).
URGENCY_KEYWORDS = [
    "mandatory", "compulsory", "without fail", "must attend",
    "required to attend", "attendance will be", "closely monitored",
    "before the deadline", "before the prescribed deadline", "last date to",
    "final reminder", "final notice", "failure to", "strictly mandatory",
    "non-negotiable", "will be marked absent", "action required",
    "immediate action", "respond by", "submit by", "due by",
]

JOB_PLATFORM_KEYWORDS = [
    "linkedin", "naukri", "internshala", "indeed", "glassdoor",
    "wellfound", "angel.co", "hirist", "cutshort", "unstop",
    "shine.com", "monster.com",
]

BROADCAST_KEYWORDS = [
    "watch live", "livestream", "live stream", "tune in", "premiere",
    "webinar", "registration open", "rsvp", "save the date",
]

URGENCY_PATTERN = re.compile(
    "|".join(re.escape(k) for k in URGENCY_KEYWORDS), re.IGNORECASE
)
JOB_HYPOTHESIS_SENDER_PATTERN = re.compile(
    "|".join(re.escape(k) for k in JOB_PLATFORM_KEYWORDS), re.IGNORECASE
)
BROADCAST_SUBJECT_PATTERN = re.compile(
    "|".join(re.escape(k) for k in BROADCAST_KEYWORDS), re.IGNORECASE
)


def _rule_based_category(email):
    """Returns (category, confidence, explanation) or None if no rule fires."""
    sender = email.get("sender", "") or ""
    subject = email.get("subject", "") or ""
    body = email.get("body") or email.get("snippet", "") or ""

    # 1. Real obligation/deadline language always wins first.
    if URGENCY_PATTERN.search(subject) or URGENCY_PATTERN.search(body):
        return (
            "Urgent",
            90.0,
            "Detected mandatory-attendance or hard-deadline language "
            "(e.g. \"mandatory\", \"without fail\", \"before the deadline\") "
            "in the email.",
        )

    # 2. Job/internship platform senders.
    if JOB_HYPOTHESIS_SENDER_PATTERN.search(sender):
        return (
            "Job/Internship",
            92.0,
            "Detected as a job/internship alert based on the sending platform.",
        )

    # 3. Public/spectator broadcast language (only when no urgency signal above).
    if BROADCAST_SUBJECT_PATTERN.search(subject):
        return (
            "News & Promotions",
            85.0,
            "Detected as a broadcast-style announcement based on "
            "webinar/livestream/event language in the subject.",
        )

    return None


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Loads and caches the zero-shot classification pipeline (CPU by default)."""
    return pipeline("zero-shot-classification", model=MODEL_NAME)


def _build_text(email):
    """Combines subject and body/snippet into one string for classification."""
    subject = email.get("subject", "")
    body = email.get("body") or email.get("snippet", "")
    return f"Subject: {subject}\n\n{body}"[:1500]


def classify_email(email, classifier=None):
    """
    Classifies a single email dict (with subject/body/snippet keys).

    Returns a dict with: category, confidence (0-100 float), suggested_action,
    explanation.
    """
    rule_hit = _rule_based_category(email)
    if rule_hit:
        category, confidence, explanation = rule_hit
        return {
            "category": category,
            "confidence": confidence,
            "suggested_action": SUGGESTED_ACTIONS[category],
            "explanation": explanation,
        }

    classifier = classifier or load_classifier()
    text = _build_text(email)

    hypotheses = list(CATEGORY_HYPOTHESES.values())
    result = classifier(text, candidate_labels=hypotheses, multi_label=False)

    top_hypothesis = result["labels"][0]
    top_score = result["scores"][0]

    category = next(
        cat for cat, hyp in CATEGORY_HYPOTHESES.items() if hyp == top_hypothesis
    )

    return {
        "category": category,
        "confidence": round(top_score * 100, 1),
        "suggested_action": SUGGESTED_ACTIONS[category],
        "explanation": EXPLANATION_TEMPLATES[category],
    }


def classify_emails(emails, progress_callback=None):
    """
    Classifies a list of email dicts, merging classification results into
    each email dict. Optionally reports progress via progress_callback(i, n).
    """
    classifier = load_classifier()
    total = len(emails)
    classified = []

    for i, email in enumerate(emails):
        result = classify_email(email, classifier=classifier)
        merged = {**email, **result}
        classified.append(merged)
        if progress_callback:
            progress_callback(i + 1, total)

    return classified

