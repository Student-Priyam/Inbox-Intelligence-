# Inbox Intelligence — AI Email Management Dashboard

A Streamlit dashboard that connects to Gmail, fetches your recent inbox,
and uses a pre-trained Hugging Face transformer model to classify each
email as **Urgent**, **Follow-Up**, **FYI**, or **Spam**, with a confidence
score and a suggested action.

---

## 1. Project Structure

```
email-management-dashboard/
├── app.py              # Streamlit UI — pages, layout, routing
├── auth.py              # Google OAuth 2.0 authentication
├── gmail_fetch.py        # Gmail API email fetching + parsing
├── classifier.py         # Zero-shot classification (Hugging Face)
├── analytics.py          # Plotly chart builders
├── utils.py              # Formatting, filtering, CSV export
├── requirements.txt
├── README.md
├── assets/
├── .streamlit/config.toml   # Theme
└── credentials/              # credentials.json + token.json go here
```

---

## 2. How Classification Works

Rather than training a model from scratch, the app uses
**zero-shot classification** with the pre-trained
[`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)
model. Each email's subject + body is compared against four natural-language
category descriptions, and the model scores how well each one fits — no
labeled training data or fine-tuning required.

The first time you run the app, this model (~1.6 GB) downloads automatically
from Hugging Face and is cached locally by `transformers`.

---

## 3. Prerequisites

- Python 3.10+
- A Google account with Gmail
- ~2 GB free disk space (for the model download)

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Gmail API Setup Guide

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. In the left menu, go to **APIs & Services > Library**.
4. Search for **Gmail API** and click **Enable**.

---

## 5. Google OAuth 2.0 Setup Guide

1. In **APIs & Services > OAuth consent screen**:
   - Choose **External** (unless you have a Google Workspace org).
   - Fill in an app name, support email, and developer contact email.
   - Under **Scopes**, add `https://www.googleapis.com/auth/gmail.readonly`.
   - Under **Test users**, add the Gmail address you'll sign in with
     (required while the app is in "Testing" mode).
2. In **APIs & Services > Credentials**:
   - Click **Create Credentials > OAuth client ID**.
   - Application type: **Desktop app**.
   - Name it (e.g. "Inbox Intelligence Local") and click **Create**.
   - Click **Download JSON**.
3. Rename the downloaded file to `credentials.json` and place it in the
   project's `credentials/` folder:

   ```
   email-management-dashboard/credentials/credentials.json
   ```

> **Note:** While the OAuth consent screen is in "Testing" status, only the
> test users you added can sign in, and tokens expire after 7 days —
> you'll simply be prompted to sign in again. Publish the app (or move to
> production) if you need long-lived, non-test access.

---

## 6. Running the App Locally

```bash
streamlit run app.py
```

On first load, click **Sign in with Google**. A browser window opens for
you to authorize read-only Gmail access. After granting access, a
`token.json` is cached in `credentials/` so you won't need to sign in again
until it expires.

Click **Refresh Inbox** in the sidebar to fetch and classify your latest
emails (default: 50).

---

## 7. Deployment Guide (Streamlit Community Cloud)

Because OAuth's "Desktop app" flow opens a local browser window, it's best
suited to running **locally** or on a machine you control. To deploy:

1. Push this project to a GitHub repository (do **not** commit the
   `credentials/` folder — see `.gitignore` notes below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your
   GitHub account.
3. Select the repository and set the main file path to `app.py`.
4. Under **Advanced settings > Secrets**, you have two options:
   - **Simplest:** keep using it as a personal/local tool only.
   - **For shared deployment:** switch the OAuth client type to **Web
     application**, add your deployed app's URL as an authorized redirect
     URI in Google Cloud Console, and adapt `auth.py` to use
     `Flow.from_client_config()` with that redirect URI instead of
     `run_local_server()`.
5. Add a `packages.txt` file with `build-essential` if you hit a build error
   installing `torch` on Streamlit Cloud's default image.

**Recommended `.gitignore`:**

```
credentials/credentials.json
credentials/token.json
venv/
__pycache__/
```

---

## 8. Troubleshooting

| Issue | Fix |
|---|---|
| `Missing credentials/credentials.json` | Complete the OAuth setup steps above and confirm the file path/name. |
| `redirect_uri_mismatch` | Make sure the OAuth client type is "Desktop app" for local use. |
| Sign-in works but fetch fails | Confirm the Gmail API is enabled in your Google Cloud project. |
| Model download is slow / fails | Check your internet connection; the model is ~1.6 GB and downloads once. |
| "Access blocked: app not verified" | Add your Google account as a test user on the OAuth consent screen. |

---

## 9. Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Auth:** Google OAuth 2.0 (`google-auth-oauthlib`)
- **Email Access:** Gmail API (`google-api-python-client`)
- **AI/NLP:** Hugging Face Transformers (`facebook/bart-large-mnli`, zero-shot)
- **Data:** Pandas
- **Visualization:** Plotly
