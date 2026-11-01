import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

from datasets import load_dataset

SEED = 42
DATASET = "VladislavMarinovich/polaris-support-tickets-v2"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"

SIZES = {"train": 4000, "valid": 500, "test": 500}

PROMPT = """Classify the customer support ticket into three attributes.
Return only a valid JSON object with exactly these keys:
category, sentiment, urgency.

Allowed category labels:
connectors, dashboards, alerts, reports, attribution,
billing, users_workspace, northstar

Allowed sentiment labels:
neutral, frustrated, confused, angry, overwhelmed, anxious

Allowed urgency labels:
low, medium, high, critical

"""


def make_example(record):
    subject = (record["subject"] or "").strip()
    body = (record["body"] or "").strip()

    ticket = f"Subject: {subject}\n" if subject else ""
    ticket += f"Body: {body}"

    answer = {
        "category": record["topic"],
        "sentiment": record["sentiment"],
        "urgency": record["priority"],
    }

    return {
        "messages": [
            {"role": "user", "content": PROMPT + ticket},
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False),
            },
        ]
    }



def main():
    rows = load_dataset(DATASET, split="train")

    cleaned = []
    seen = set()

    for row in rows:
        # Exclude event-linked tickets in the initial experiment.
        if row["event_id"] is not None and str(row["event_id"]).strip():
            continue

        subject = (row["subject"] or "").strip()
        body = (row["body"] or "").strip()

        if not body:
            continue

        if not all(row[k] for k in ["topic", "sentiment", "priority"]):
            continue

        key = " ".join((subject + " " + body).lower().split())

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(row)

    print("Eligible records:", len(cleaned))

    # Select 5,000 samples, stratified by topic.
    selected, _ = train_test_split(
        cleaned,
        train_size=5000,
        stratify=[r["topic"] for r in cleaned],
        random_state=SEED,
    )

    # Split into 4,000 train and 1,000 held-out samples.
    train, remaining = train_test_split(
        selected,
        test_size=1000,
        stratify=[r["topic"] for r in selected],
        random_state=SEED,
    )

    # Split held-out samples into validation and test sets.
    valid, test = train_test_split(
        remaining,
        test_size=500,
        stratify=[r["topic"] for r in remaining],
        random_state=SEED,
    )

    partitions = {
        "train": train,
        "valid": valid,
        "test": test,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, records in partitions.items():
        path = OUT_DIR / f"{name}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for record in records:
                example = make_example(record)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"\n{name}: {len(records)} samples")

        for field in ["topic", "sentiment", "priority"]:
            counts = Counter(r[field] for r in records)
            print(f"{field}: {dict(sorted(counts.items()))}")

    print("\nData preparation completed.")


if __name__ == "__main__":
    main()