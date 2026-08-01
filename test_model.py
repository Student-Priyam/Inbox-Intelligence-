"""
test_model.py
--------------
Evaluates the trained model saved in model/email_classifier/, either:

1. On the held-out split of data/training_data.csv (full classification
   report, confusion matrix), or
2. On a single ad-hoc email you type in, for a quick sanity check.

Usage:
    python test_model.py                       # full report on training_data.csv
    python test_model.py --subject "..." --body "..."   # single prediction
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).parent / "model" / "email_classifier"
DATA_PATH = Path(__file__).parent / "data" / "training_data.csv"


def load_model():
    if not MODEL_DIR.exists():
        raise SystemExit(
            f"No trained model found at {MODEL_DIR}.\n"
            "Run `python train_model.py` first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    with open(MODEL_DIR / "label_map.json") as f:
        category_order = json.load(f)["category_order"]
    return tokenizer, model, category_order


def predict(tokenizer, model, category_order, subject, body):
    text = f"Subject: {subject}\n\n{body}"[:1500]
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_idx = int(torch.argmax(probs))
    return category_order[top_idx], float(probs[top_idx]) * 100, probs


def full_report(tokenizer, model, category_order):
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["category"].map({c: i for i, c in enumerate(category_order)})
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        category, confidence, _ = predict(
            tokenizer, model, category_order, row["subject"], row["body"]
        )
        y_true.append(row["category"])
        y_pred.append(category)

    print(f"Evaluated on {len(test_df)} held-out examples (same split used in training)\n")
    print(classification_report(y_true, y_pred, labels=category_order, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=category_order)
    header = "".join(f"{c[:10]:>12}" for c in category_order)
    print(" " * 18 + header)
    for cat, row_counts in zip(category_order, cm):
        print(f"{cat[:16]:>16}  " + "".join(f"{n:>12}" for n in row_counts))


def main():
    parser = argparse.ArgumentParser(description="Test the trained email classifier.")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--body", type=str, default=None)
    args = parser.parse_args()

    tokenizer, model, category_order = load_model()

    if args.subject or args.body:
        category, confidence, probs = predict(
            tokenizer, model, category_order, args.subject or "", args.body or ""
        )
        print(f"Predicted category: {category} ({confidence:.1f}% confidence)")
        print("\nAll scores:")
        for cat, p in zip(category_order, probs.tolist()):
            print(f"  {cat:<20} {p * 100:5.1f}%")
    else:
        full_report(tokenizer, model, category_order)


if __name__ == "__main__":
    main()
