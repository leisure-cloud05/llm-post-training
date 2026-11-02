import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/sft/original"
SOURCE = "VladislavMarinovich/polaris-support-tickets-v2"

SEED = 42
THRESHOLD = 0.70
PER_CLASS_PER_SPLIT = 25
CLASSES = ("critical", "high")


def norm(text):
    return re.sub(
        r"\s+", " ", str(text or "").strip().casefold()
    )


def key(subject, body):
    return (norm(subject), norm(body))


def parse_prompt(prompt):
    match = re.search(
        r"\n\n(?=Subject: |Body: )", prompt
    )
    if match is None:
        raise ValueError("Cannot find ticket block")

    prefix = prompt[:match.end()]
    ticket = prompt[match.end():]

    if ticket.startswith("Subject: "):
        subject_part, sep, body = ticket.partition("\nBody: ")
        if not sep:
            raise ValueError("Body marker missing")
        subject = subject_part[len("Subject: "):]
    else:
        subject = ""
        body = ticket[len("Body: "):]

    return prefix, key(subject, body)


def load_existing():
    rows = []
    prompt_prefix = None

    for split in ("train", "valid", "test"):
        with (DATA / f"{split}.jsonl").open(
            encoding="utf-8"
        ) as f:
            for line in f:
                example = json.loads(line)
                prompt = example["messages"][0]["content"]
                prefix, ticket_key = parse_prompt(prompt)

                if prompt_prefix is None:
                    prompt_prefix = prefix
                elif prefix != prompt_prefix:
                    raise ValueError(
                        "Inconsistent instruction prompt detected"
                    )

                rows.append(ticket_key)

    return rows, prompt_prefix


def ticket_text(ticket_key):
    subject, body = ticket_key
    return f"{subject} {body}".strip()


def build_example(row, prefix):
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
            {"role": "user", "content": prefix + ticket},
            {
                "role": "assistant",
                "content": json.dumps(
                    answer, ensure_ascii=False
                ),
            },
        ]
    }


def main():
    rng = random.Random(SEED)

    used_keys, prefix = load_existing()
    used_set = set(used_keys)

    source = load_dataset(SOURCE, split="train")

    # Collect unused, non-event High/Critical tickets.
    candidates = []
    seen = set()

    for index, row in enumerate(source):
        if str(row.get("event_id") or "").strip():
            continue

        if row["priority"] not in CLASSES:
            continue

        if not all(
            str(row.get(field) or "").strip()
            for field in ("body", "topic", "sentiment")
        ):
            continue

        ticket_key = key(row["subject"], row["body"])

        if ticket_key in used_set or ticket_key in seen:
            continue

        seen.add(ticket_key)
        candidates.append({
            "source_index": index,
            "key": ticket_key,
            "row": row,
        })

    print("Candidate counts:", dict(
        Counter(c["row"]["priority"] for c in candidates)
    ))

    all_texts = (
        [ticket_text(k) for k in used_keys]
        + [ticket_text(c["key"]) for c in candidates]
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    features = vectorizer.fit_transform(all_texts)

    n_used = len(used_keys)
    used_matrix = features[:n_used]
    candidate_matrix = features[n_used:]

    # Maximum similarity to any previously used ticket.
    similarities_to_used = cosine_similarity(
        candidate_matrix,
        used_matrix,
    )
    max_to_used = similarities_to_used.max(axis=1)

    # Pairwise similarity among unused candidates.
    candidate_sim = cosine_similarity(candidate_matrix)

    eligible = [
        i for i, score in enumerate(max_to_used)
        if score < THRESHOLD
    ]

    rng.shuffle(eligible)

    selected = {"dev": [], "test": []}
    chosen_indices = []

    # Alternate between classes and splits to avoid
    # allocating every rare sample to one set first.
    targets = [
        ("dev", "critical"),
        ("test", "critical"),
        ("dev", "high"),
        ("test", "high"),
    ]

    while any(
        sum(
            candidates[i]["row"]["priority"] == cls
            for i in selected[split]
        ) < PER_CLASS_PER_SPLIT
        for split, cls in targets
    ):
        made_progress = False

        for split, cls in targets:
            count = sum(
                candidates[i]["row"]["priority"] == cls
                for i in selected[split]
            )
            if count >= PER_CLASS_PER_SPLIT:
                continue

            for i in eligible:
                if candidates[i]["row"]["priority"] != cls:
                    continue

                if i in chosen_indices:
                    continue

                if chosen_indices and np.any(
                    candidate_sim[i, chosen_indices]
                    >= THRESHOLD
                ):
                    continue

                selected[split].append(i)
                chosen_indices.append(i)
                made_progress = True
                break

        if not made_progress:
            raise RuntimeError(
                "Not enough independent candidates at "
                f"threshold {THRESHOLD}. No files written."
            )

    # Verify full separation before writing.
    assert len(selected["dev"]) == 50
    assert len(selected["test"]) == 50

    chosen_sim = candidate_sim[
        np.ix_(chosen_indices, chosen_indices)
    ].copy()
    np.fill_diagonal(chosen_sim, 0)

    assert chosen_sim.max() < THRESHOLD
    assert max(max_to_used[i] for i in chosen_indices) < THRESHOLD

    manifest = {
        "seed": SEED,
        "similarity_threshold": THRESHOLD,
        "method": "TF-IDF character 3-5gram cosine similarity",
        "splits": {},
    }

    for split in ("dev", "test"):
        records = []
        for i in selected[split]:
            candidate = candidates[i]
            records.append(
                build_example(candidate["row"], prefix)
            )

        path = DATA / f"diagnostic_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(
                    record, ensure_ascii=False
                ) + "\n")

        counts = Counter(
            candidates[i]["row"]["priority"]
            for i in selected[split]
        )

        manifest["splits"][split] = {
            "count": len(records),
            "priority_counts": dict(counts),
            "source_indices": [
                candidates[i]["source_index"]
                for i in selected[split]
            ],
        }

        print(f"\n{split.upper()}: {dict(counts)}")
        print("Saved:", path)

    manifest_path = DATA / "diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nMax similarity to existing data:",
          round(float(max_to_used[chosen_indices].max()), 4))
    print("Max similarity within diagnostic sets:",
          round(float(chosen_sim.max()), 4))
    print("Saved:", manifest_path)
    print("\nDiagnostic sets built successfully.")


if __name__ == "__main__":
    main()