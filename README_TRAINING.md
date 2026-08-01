# Training the classifier (new)

## What changed
Previously `classifier.py` called `facebook/bart-large-mnli` **zero-shot**
at inference time — no training step, no labeled data, just prompting a
general-purpose model with category descriptions on every email.

Now there's an actual training pipeline:

```
data/build_seed_dataset.py   → writes data/training_data.csv (labeled examples)
train_model.py                → fine-tunes distilbert-base-uncased on that CSV,
                                 saves the trained model to model/email_classifier/
test_model.py                  → evaluates the trained model (report or single prediction)
classifier.py                  → loads model/email_classifier/ instead of calling
                                  the internet-hosted zero-shot model
```

**Nothing else changed.** Gmail fetching (`gmail_fetch.py`, `auth.py`), the
UI (`app.py`), analytics (`analytics.py`), and helpers (`utils.py`) are
untouched — `classifier.py` still exposes the same `classify_emails()`,
`classify_email()`, and `CATEGORY_ORDER` that the rest of the app imports,
so it's a drop-in replacement.

## One-time setup

```bash
pip install -r requirements.txt

# 1. Generate the seed labeled dataset (50 examples, 10 per category)
python data/build_seed_dataset.py

# 2. Train the model — takes a few minutes on CPU
python train_model.py

# 3. Check how well it did
python test_model.py
```

`train_model.py` prints accuracy/precision/recall/F1 on a held-out 20%
test split as it trains. `test_model.py` re-runs that same evaluation with
a full classification report and confusion matrix, or you can pass a
single email to sanity-check a prediction:

```bash
python test_model.py --subject "Mandatory dept meeting tomorrow" --body "All students must attend."
```

Once `model/email_classifier/` exists, run the Streamlit app as usual
(`streamlit run app.py`) — `classify_emails()` will load the trained model
automatically. If the model hasn't been trained yet, the app will show a
clear error telling you to run `train_model.py` first, instead of
silently falling back to the old zero-shot behavior.

## Improving accuracy
The seed dataset (50 rows) is enough to get training working end-to-end,
but it's small. To get real improvements over the old zero-shot approach:

1. Export some of your own classified emails (there's already a
   **Download CSV** button in the app's sidebar) and manually correct any
   wrong labels.
2. Append those rows to `data/training_data.csv` (same three columns:
   `subject,body,category`, category must be one of the five in
   `CATEGORY_ORDER`).
3. Re-run `python train_model.py` — it retrains from scratch on the full
   updated file.

The rule-based layer in `classifier.py` (mandatory/deadline language, job
platform senders, webinar/livestream keywords) still runs *before* the
trained model and is unchanged — it only reaches the model for emails none
of those rules catch.
