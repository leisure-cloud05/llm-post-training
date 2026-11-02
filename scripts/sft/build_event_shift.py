import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/sft/original"
SOURCE = "VladislavMarinovich/polaris-support-tickets-v2"

SEED = 42
PER_EVENT = 40


def norm(text):
    return re.sub(
        r"\s+", " ", str(text or "").strip().casefold()
    )


def ticket_key(subject, body):
    return norm(subject), norm(body)


def parse_existing_prompt(prompt):
    match = re.search(
        r"\n\n(?=Subject: |Body: )", prompt
    )
    if match is None:
        raise ValueError("Ticket block not found")

    prefix = prompt[:match.end()]
    ticket = prompt[match.end():]

    if ticket.startswith("Subject: "):
        part, separator, body = ticket.partition("\nBody: ")
        if not separator:
            raise ValueError("Body marker not found")
        subject = part[len("Subject: "):]
    else:
        subject = ""
        body = ticket[len("Body: "):]

    return prefix, ticket_key(subject, body)


def load_existing():
    keys = set()
    prefix = None

    # Include the previously constructed diagnostic sets.
    for name in (
        "train", "valid", "test",
        "diagnostic_dev", "diagnostic_test"
    ):
        path = DATA / f"{name}.jsonl"

        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                current_prefix, key = parse_existing_prompt(
                    record["messages"][0]["content"]
                )

                if prefix is None:
                    prefix = current_prefix
                elif current_prefix != prefix:
                    raise ValueError("Inconsistent prompt template")

                keys.add(key)

    return keys, prefix


def make_example(row, prefix):
    subject = str(row["subject"] or "").strip()
    body = str(row["body"] or "").strip()

    ticket = (
        f"Subject: {subject}\nBody: {body}"
        if subject else f"Body: {body}"
    )

    answer = {
        "category": row["topic"],
        "sentiment": row["sentiment"],
        "urgency": row["priority"],
    }

    return {
        "messages": [
            {
                "role": "user",
                "content": prefix + ticket,
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    answer, ensure_ascii=False
                ),
            },
        ],
    }


def main():
    rng = random.Random(SEED)
    used_keys, prefix = load_existing()

    source = load_dataset(SOURCE, split="train")
    groups = defaultdict(list)
    seen = set(used_keys)

    excluded_overlap = 0
    excluded_invalid = 0

    for index, row in enumerate(source):
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue

        if not all(
            str(row.get(field) or "").strip()
            for field in (
                "body", "topic", "sentiment", "priority"
            )
        ):
            excluded_invalid += 1
            continue

        key = ticket_key(row["subject"], row["body"])

        if key in seen:
            excluded_overlap += 1
            continue

        seen.add(key)
        groups[event_id].append((index, row))

    if not groups:
        raise RuntimeError("No eligible event-linked tickets")

    selected = []
    event_counts = {}

    for event_id in sorted(groups):
        candidates = groups[event_id]
        n = min(PER_EVENT, len(candidates))

        chosen = rng.sample(candidates, n)
        selected.extend(
            (event_id, index, row)
            for index, row in chosen
        )

        event_counts[event_id] = {
            "eligible": len(candidates),
            "selected": n,
        }

    rng.shuffle(selected)

    records = [
        make_example(row, prefix)
        for _, _, row in selected
    ]

    manifest = {
        "dataset": SOURCE,
        "seed": SEED,
        "sampling": "up to 40 tickets per event",
        "total_samples": len(records),
        "event_groups": len(groups),
        "excluded_exact_overlaps_or_duplicates": excluded_overlap,
        "excluded_invalid": excluded_invalid,
        "event_counts": event_counts,
        "priority_distribution": dict(Counter(
            row["priority"]
            for _, _, row in selected
        )),
        "category_distribution": dict(Counter(
            row["topic"]
            for _, _, row in selected
        )),
        "sample_metadata": [
            {
                "output_index": i,
                "source_index": index,
                "event_id": event_id,
            }
            for i, (event_id, index, _) in enumerate(selected)
        ],
    }

    # Do not overwrite an existing frozen evaluation set.
    output = DATA / "event_shift_test.jsonl"
    manifest_path = DATA / "event_shift_manifest.json"

    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            "Event-shift files already exist. "
            "Do not overwrite the frozen evaluation set."
        )

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                record, ensure_ascii=False
            ) + "\n")

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Event groups:", len(groups))
    print("Selected samples:", len(records))
    print("Priority distribution:",
          manifest["priority_distribution"])
    print("Category distribution:",
          manifest["category_distribution"])
    print("Exact overlaps / duplicates excluded:",
          excluded_overlap)

    print("\nSamples per event:")
    for event_id, info in event_counts.items():
        print(
            f"  {event_id}: "
            f"{info['selected']} / {info['eligible']}"
        )

    print("\nSaved:", output)
    print("Saved:", manifest_path)


if __name__ == "__main__":
    main()