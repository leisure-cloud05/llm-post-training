import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/sft/original"
RESULTS_DIR = ROOT / "results/sft/sft"
RESULTS_DIR.mkdir(exist_ok=True)

DATASET = "VladislavMarinovich/polaris-support-tickets-v2"
THRESHOLD = 0.85


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def make_key(subject, body):
    return normalize(subject), normalize(body)


def extract_ticket(example):
    prompt = example["messages"][0]["content"]

    start = prompt.find("\n\nSubject: ")
    if start == -1:
        start = prompt.find("\n\nBody: ")

    if start == -1:
        raise ValueError("Ticket block not found")

    ticket = prompt[start + 2:]

    if ticket.startswith("Subject: "):
        subject_part, sep, body = ticket.partition("\nBody: ")
        if not sep:
            raise ValueError("Body marker not found")
        subject = subject_part[len("Subject: "):]
    else:
        subject = ""
        body = ticket[len("Body: "):]

    return make_key(subject, body)


def read_split(name):
    path = DATA_DIR / f"{name}.jsonl"

    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    result = []
    for i, row in enumerate(rows):
        subject, body = extract_ticket(row)
        result.append({
            "split": name,
            "index": i,
            "subject": subject,
            "body": body,
            "text": f"{subject} {body}".strip(),
        })
    return result


def source_record(row, index):
    subject, body = make_key(row["subject"], row["body"])
    return {
        "split": "unused",
        "index": index,
        "priority": row["priority"],
        "subject": subject,
        "body": body,
        "text": f"{subject} {body}".strip(),
    }


def main():
    used = []
    for name in ("train", "valid", "test"):
        used.extend(read_split(name))

    used_keys = {
        (r["subject"], r["body"]) for r in used
    }

    source = load_dataset(DATASET, split="train")
    candidates = []

    for i, row in enumerate(source):
        if str(row.get("event_id") or "").strip():
            continue

        record = source_record(row, i)
        key = (record["subject"], record["body"])

        if not record["body"] or key in used_keys:
            continue

        if record["priority"] in ("critical", "high"):
            candidates.append(record)

    print("Used records:", len(used))
    print("Unused high/critical records:", len(candidates))
    print("Priority counts:", dict(
        Counter(r["priority"] for r in candidates)
    ))

    # Fit vocabulary on both used and candidate tickets.
    texts = [r["text"] for r in used + candidates]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)

    used_matrix = matrix[:len(used)]
    candidate_matrix = matrix[len(used):]

    # Search nearest used ticket for every unused candidate.
    report = []

    for start in range(0, len(candidates), 64):
        end = min(start + 64, len(candidates))

        similarities = cosine_similarity(
            candidate_matrix[start:end],
            used_matrix,
            dense_output=True,
        )

        for offset, scores in enumerate(similarities):
            candidate = candidates[start + offset]
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            nearest = used[best_index]

            report.append({
                "priority": candidate["priority"],
                "source_index": candidate["index"],
                "max_similarity": round(best_score, 4),
                "near_duplicate": best_score >= THRESHOLD,
                "nearest_split": nearest["split"],
                "nearest_index": nearest["index"],
                "candidate_text": candidate["text"][:500],
                "nearest_text": nearest["text"][:500],
            })

    # Also check repeated text among unused critical tickets.
    critical_indices = [
        i for i, r in enumerate(candidates)
        if r["priority"] == "critical"
    ]

    print("\n=== UNUSED CRITICAL INTERNAL SIMILARITY ===")

    if len(critical_indices) > 1:
        critical_matrix = candidate_matrix[critical_indices]
        similarities = cosine_similarity(critical_matrix)
        np.fill_diagonal(similarities, 0)

        pairs = np.argwhere(
            np.triu(similarities >= THRESHOLD, k=1)
        )

        print("Pairs above threshold:", len(pairs))
        for a, b in pairs[:5]:
            print(
                f"  source {candidates[critical_indices[a]]['index']}"
                f" <-> source {candidates[critical_indices[b]]['index']}"
                f": {similarities[a, b]:.3f}"
            )

    report.sort(
        key=lambda r: (
            r["priority"] != "critical",
            -r["max_similarity"],
        )
    )

    output_path = RESULTS_DIR / "near_duplicate_audit.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    print("\n=== CROSS-SPLIT SIMILARITY AUDIT ===")

    for priority in ("critical", "high"):
        subset = [r for r in report if r["priority"] == priority]
        flagged = [
            r for r in subset if r["near_duplicate"]
        ]

        print(f"\n{priority.upper()}")
        print("Candidates:", len(subset))
        print(f"Similarity >= {THRESHOLD}:", len(flagged))

        if subset:
            scores = [r["max_similarity"] for r in subset]
            print("Median max similarity:", round(
                float(np.median(scores)), 3
            ))
            print("90th percentile:", round(
                float(np.percentile(scores, 90)), 3
            ))

        print("Top 3 nearest matches:")
        for row in subset[:3]:
            print(
                f"  similarity={row['max_similarity']:.3f}, "
                f"split={row['nearest_split']}"
            )
            print("  Candidate:", row["candidate_text"][:160])
            print("  Nearest:  ", row["nearest_text"][:160])

    print(f"\nSaved: {output_path}")
    print("Audit finished. No dataset files were modified.")


if __name__ == "__main__":
    main()