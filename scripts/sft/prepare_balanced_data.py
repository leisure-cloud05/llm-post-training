import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/sft/original"
TARGET = ROOT / "data/sft/balanced"

SEED = 42
TOTAL = 4000
CRITICAL_RATIO = 0.10


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def urgency(row):
    return json.loads(
        row["messages"][-1]["content"]
    )["urgency"]


def main():
    rng = random.Random(SEED)

    train = load_jsonl(SOURCE / "train.jsonl")

    if len(train) != TOTAL:
        raise ValueError("Expected 4000 training examples.")

    critical = [
        (i, row) for i, row in enumerate(train)
        if urgency(row) == "critical"
    ]

    others = [
        (i, row) for i, row in enumerate(train)
        if urgency(row) != "critical"
    ]

    n_critical = int(TOTAL * CRITICAL_RATIO)
    n_others = TOTAL - n_critical

    if not critical or len(others) < n_others:
        raise ValueError("Insufficient source examples.")

    # Oversample critical examples with replacement.
    selected_critical = [
        rng.choice(critical)
        for _ in range(n_critical)
    ]

    # Sample other classes without replacement.
    selected_others = rng.sample(others, n_others)

    selected = selected_critical + selected_others
    rng.shuffle(selected)

    TARGET.mkdir(exist_ok=True)

    with (TARGET / "train.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for _, row in selected:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    # Keep validation and standard test identical.
    for name in ("valid.jsonl", "test.jsonl"):
        shutil.copyfile(
            SOURCE / name,
            TARGET / name,
        )

    counts = Counter(urgency(row) for _, row in selected)
    exposure = Counter(i for i, _ in selected)

    manifest = {
        "seed": SEED,
        "sampling_method": "critical oversampling",
        "critical_target_ratio": CRITICAL_RATIO,
        "total_records": len(selected),
        "class_distribution": dict(counts),
        "unique_source_examples": len(exposure),
        "critical_unique_examples": len(
            {i for i, row in selected
             if urgency(row) == "critical"}
        ),
        "max_repetitions": max(exposure.values()),
        "validation_test_unchanged": True,
    }

    (TARGET / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    print("\nBalanced dataset prepared.")


if __name__ == "__main__":
    main()