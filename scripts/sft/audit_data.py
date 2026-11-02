import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/sft/original"
DATASET = "VladislavMarinovich/polaris-support-tickets-v2"


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def make_key(subject, body):
    return (normalize(subject), normalize(body))


def extract_ticket(example):
    prompt = example["messages"][0]["content"]

    # Match the start of the ticket, not line breaks
    # that may appear inside the body.
    ticket_start = prompt.find("\n\nSubject: ")
    if ticket_start == -1:
        ticket_start = prompt.find("\n\nBody: ")

    if ticket_start == -1:
        raise ValueError(
            "Cannot locate ticket block:\n"
            + repr(prompt[-300:])
        )

    ticket = prompt[ticket_start + 2:]

    if ticket.startswith("Subject: "):
        subject_part, separator, body = ticket.partition("\nBody: ")
        if not separator:
            raise ValueError("Body marker not found")

        subject = subject_part[len("Subject: "):]
    else:
        subject = ""
        body = ticket[len("Body: "):]

    return make_key(subject, body)


def load_split(name):
    path = DATA_DIR / f"{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    splits = {
        name: load_split(name)
        for name in ("train", "valid", "test")
    }

    keys = {
        name: [extract_ticket(row) for row in rows]
        for name, rows in splits.items()
    }

    print("=== SPLIT AUDIT ===")

    for name, items in keys.items():
        print(
            f"{name}: {len(items)} records, "
            f"{len(set(items))} unique"
        )

    for a, b in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        overlap = set(keys[a]) & set(keys[b])
        print(f"{a} vs {b}: {len(overlap)} overlaps")

    used = set().union(*(set(v) for v in keys.values()))

    print("\n=== SOURCE MATCHING ===")

    source = load_dataset(DATASET, split="train")

    ordinary = [
        row for row in source
        if not str(row.get("event_id") or "").strip()
    ]

    source_keys = [
        make_key(row["subject"], row["body"])
        for row in ordinary
    ]

    source_key_set = set(source_keys)
    missing = used - source_key_set

    print("Ordinary source records:", len(ordinary))
    print("Unique used tickets:", len(used))
    print("Used tickets found in source:", len(used) - len(missing))
    print("Used tickets NOT found:", len(missing))

    if missing:
        print("\nExample unmatched keys:")
        for item in list(sorted(missing))[:3]:
            print(repr(item)[:250])

        raise RuntimeError(
            "Source matching incomplete. "
            "Do not create a diagnostic set yet."
        )

    remaining = [
        row for row, key in zip(ordinary, source_keys)
        if key not in used
    ]

    counts = Counter(row["priority"] for row in remaining)

    print("\n=== UNUSED ORDINARY RECORDS ===")
    print("Records:", len(remaining))
    print("Priority distribution:", dict(counts))
    print("Critical candidates:", counts["critical"])

    print("\nAudit passed.")


if __name__ == "__main__":
    main()