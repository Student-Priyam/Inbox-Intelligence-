# Inbox Intelligence — AI Email Management Dashboard

An AI-powered Streamlit dashboard that connects to Gmail, fetches your recent inbox, and automatically classifies each email into **Urgent**, **Job/Internship**, **Follow-Up**, **News & Promotions**, or **Spam** using a fine-tuned DistilBERT model — complete with a confidence score and a suggested action for every email.

---

## ✨ Features

- 🔐 **Secure Gmail sign-in** via Google OAuth 2.0 — read-only access, credentials never leave your machine
- 🧠 **AI-powered classification** — a fine-tuned DistilBERT model backed by high-precision rule-based pre-checks (including automatic detection of job/internship alerts from platforms like LinkedIn, Naukri, and Internshala)
- 📊 **Interactive dashboard** — category breakdown, daily email volume, top senders, and category trends over time
- 🔍 **Searchable, filterable inbox** — filter by category, sender, date range, or confidence score
- 📤 **CSV export** — download your classified inbox for further analysis
- ⚙️ **Fully local inference** — once trained, the model runs entirely on your machine; no email content is sent to a third party

---

## 🏗️ Architecture

```
Gmail API ──▶ gmail_fetch.py ──▶ classifier.py ──┬─▶ rule-based pre-checks (deadline language / job-platform senders / webinar-broadcast language)
                                                   └─▶ src/inference.py ──▶ models/best_model/ (fine-tuned DistilBERT)
                                                                                     ▲
                                                                     training/train_model.py
                                                                                     ▲
                                              src/dataset.py + src/preprocessing.py
                                                                                     ▲
                                                          dataset/{train,validation,test}.csv
                                                                                     ▲
                                                                    data_generation.py
```

Every email is first checked against a small set of high-precision rules — hard-deadline / mandatory-attendance language → **Urgent**; sender is a known job platform (LinkedIn, Naukri, Internshala, Indeed, etc.) → **Job/Internship**; webinar/livestream/RSVP language in the subject → **News & Promotions**. Everything else is classified by the fine-tuned model.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| Auth | Google OAuth 2.0 (`google-auth-oauthlib`) |
| Email access | Gmail API (`google-api-python-client`) |
| AI / NLP | HuggingFace Transformers — fine-tuned `distilbert-base-uncased` |
| Training | PyTorch, HuggingFace `Trainer`, scikit-learn |
| Data | Pandas |
| Visualization | Plotly (dashboard), Matplotlib (training/eval plots) |

---

## 📁 Project Structure

```
inbox-intelligence/
├── app.py                     # Streamlit UI — dashboard, inbox, analytics, settings
├── auth.py                    # Google OAuth 2.0
├── gmail_fetch.py              # Gmail API fetching + parsing
├── classifier.py                # Rule-based pre-checks + trained-model inference
├── analytics.py                  # Plotly chart builders
├── utils.py                       # Dashboard formatting/filtering helpers
├── config.py                       # Paths, category labels, hyperparameter defaults
├── data_generation.py               # Builds the labeled dataset
├── requirements.txt
├── dataset/
│   ├── train.csv                     # Training split
│   ├── validation.csv                 # Validation split
│   └── test.csv                        # Held-out test split
├── models/
│   └── best_model/                      # Fine-tuned weights + tokenizer (created after training)
├── src/
│   ├── preprocessing.py                  # HTML/URL/signature cleaning
│   ├── dataset.py                         # CSV loading + tokenized dataset construction
│   ├── model.py                            # Model construction / loading
│   ├── train.py                             # Training loop (AdamW, LR schedule, early stopping)
│   ├── evaluate.py                           # Metrics, confusion matrix, ROC-AUC, plots
│   ├── inference.py                           # Lightweight predict() wrapper
│   └── utils.py                                # Seeding, logging helpers
├── training/
│   ├── train_model.py                            # CLI: fine-tune and save the best checkpoint
│   └── evaluate_model.py                          # CLI: full evaluation report
├── outputs/
│   ├── plots/                                       # Confusion matrix, ROC curves, training history
│   └── logs/                                         # Training logs, evaluation reports
├── assets/                                             # Screenshots used in this README
└── credentials/                                         # credentials.json + token.json (gitignored)
```

---

## 📊 Dataset

The classifier is trained on a labeled dataset of realistic business emails — workplace, HR, banking, phishing, promotional, meeting requests, and follow-up conversations — generated by `data_generation.py` from templates with interchangeable slot values (names, companies, dates, amounts) so examples aren't exact duplicates.

| Category | Examples |
|---|---|
| Urgent | 30 |
| Job/Internship | 30 |
| Follow-Up | 30 |
| News & Promotions | 30 |
| Spam | 30 |
| **Total** | **150** |

Split 70/15/15 (train/validation/test) using **stratified sampling**, so every split keeps the same class balance.

> **Note:** this is a small starter dataset intended to exercise the full pipeline end-to-end. For stronger real-world accuracy, grow it with more examples per category — either by generating more via `data_generation.py --per-class N` or by appending your own labeled emails to `dataset/full_dataset.csv`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google account with Gmail
- ~500 MB free disk space for the model weights

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/inbox-intelligence.git
cd inbox-intelligence
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up Gmail API & OAuth

1. Go to the [Google Cloud Console](https://console.cloud.google.com/), create or select a project.
2. Enable the **Gmail API** under **APIs & Services > Library**.
3. Configure the **OAuth consent screen** (External), add scope `https://www.googleapis.com/auth/gmail.readonly`, and add your Gmail address as a test user.
4. Under **APIs & Services > Credentials**, create an **OAuth Client ID** (Application type: **Desktop app**) and download the JSON.
5. Save it as `credentials/credentials.json`.

### 3. Generate the dataset

```bash
python data_generation.py --per-class 30 --seed 42
```

### 4. Train the model

```bash
python training/train_model.py
```

### 5. Evaluate

```bash
python training/evaluate_model.py
```

### 6. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`, click **Sign in with Google**, authorize read-only access, then **Refresh Inbox**.

---

## 🧪 Model Details

- **Base model:** `distilbert-base-uncased`, fully fine-tuned (not zero-shot, not feature extraction)
- **Optimizer:** AdamW with weight decay (0.01)
- **LR schedule:** linear decay with warmup
- **Regularization:** label smoothing, gradient clipping, early stopping on validation macro-F1
- **Evaluation:** accuracy, precision/recall/F1 (per-class + macro), confusion matrix, ROC-AUC, training/validation loss curves

Run a single ad-hoc prediction:

```bash
python training/evaluate_model.py --subject "Mandatory dept meeting tomorrow" --body "All students must attend."
```

---

## 🗺️ Roadmap

- [ ] Grow the dataset with real, de-identified labeled emails
- [ ] Evaluate `bert-base-uncased` for a possible accuracy gain
- [ ] Active-learning loop — correct low-confidence predictions directly in the UI
- [ ] Multi-label support for emails that are genuinely both Urgent and Follow-Up
- [ ] Deploy guide for Streamlit Community Cloud with Web-application OAuth flow

---
# Inbox Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inbox-intelligen-tzgngcybz4n6trwve5mnwz.streamlit.app)

## Live Demo

https://inbox-intelligen-tzgngcybz4n6trwve5mnwz.streamlit.app
---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss what you'd like to change before submitting a pull request.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Inbox Intelligence — sorting the noise, so what matters reaches you first.</i>
</p>
