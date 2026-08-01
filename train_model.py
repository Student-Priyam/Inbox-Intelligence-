"""
train_model.py
---------------
Fine-tunes a small transformer (distilbert-base-uncased) on the labeled
email data in data/training_data.csv, and saves the resulting TRAINED
model to model/email_classifier/.

This replaces the old approach in classifier.py, which called the
general-purpose facebook/bart-large-mnli model zero-shot at inference
time with no training step at all. classifier.py now loads the model
this script produces instead.

Usage:
    python train_model.py
    python train_model.py --data data/training_data.csv --epochs 6

After training, run test_model.py to see accuracy / precision / recall
on the held-out test split, or try ad-hoc predictions on your own text.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Keep this in sync with classifier.CATEGORY_ORDER so label ids line up
# consistently across training runs.
CATEGORY_ORDER = ["Urgent", "Job/Internship", "Follow-Up", "News & Promotions", "Spam"]

BASE_MODEL = "distilbert-base-uncased"
DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "training_data.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "model" / "email_classifier"


def load_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing_cols = {"subject", "body", "category"} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"training_data.csv is missing required columns: {missing_cols}. "
            "Expected columns: subject, body, category."
        )
    unknown_categories = set(df["category"]) - set(CATEGORY_ORDER)
    if unknown_categories:
        raise ValueError(
            f"Found categories in the data that aren't in CATEGORY_ORDER: "
            f"{unknown_categories}. Fix the CSV or update CATEGORY_ORDER."
        )
    df["text"] = "Subject: " + df["subject"].fillna("") + "\n\n" + df["body"].fillna("")
    df["label"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the email classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    df = load_dataframe(args.data)
    print(f"Loaded {len(df)} labeled examples across {df['category'].nunique()} categories.")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    print(f"Train: {len(train_df)} examples | Test: {len(test_df)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(CATEGORY_ORDER),
        id2label={i: c for i, c in enumerate(CATEGORY_ORDER)},
        label2id={c: i for i, c in enumerate(CATEGORY_ORDER)},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output).parent / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Final evaluation on held-out test split:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    with open(args.output / "label_map.json", "w") as f:
        json.dump({"category_order": CATEGORY_ORDER}, f, indent=2)

    with open(args.output / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTrained model saved to {args.output}")
    print("classifier.py will automatically load it from that path.")
    print("Run test_model.py for a more detailed evaluation report.")


if __name__ == "__main__":
    main()