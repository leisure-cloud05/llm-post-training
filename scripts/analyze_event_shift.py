import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODELS = {
    "base": "event_shift_base",
    "strong": "event_shift_strong",
    "lora": "event_shift_lora",
    "balanced": "event_shift_balanced",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_urgency(row):
    if not row["label_compliant"]:
        return None
    return row["prediction"]["urgency"]


def analyze_group(rows):
    n = len(rows)
    exact = sum(
        row["label_compliant"]
        and row["prediction"] == row["gold"]
        for row in rows
    )

    urgency_correct = sum(
        get_urgency(row) == row["gold"]["urgency"]
        for row in rows
    )

    critical_rows = [
        row for row in rows
        if row["gold"]["urgency"] == "critical"
    ]

    critical_correct = sum(
        get_urgency(row) == "critical"
        for row in critical_rows
    )

    return {
        "samples": n,
        "exact_match": exact / n,
        "urgency_accuracy": urgency_correct / n,
        "critical_support": len(critical_rows),
        "critical_recall": (
            critical_correct / len(critical_rows)
            if critical_rows else None
        ),
    }


def main():
    manifest = json.loads(
        (DATA / "event_shift_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    metadata = manifest["sample_metadata"]
    predictions = {
        model: load_jsonl(
            RESULTS / f"{filename}_predictions.jsonl"
        )
        for model, filename in MODELS.items()
    }

    total = len(metadata)

    for model, rows in predictions.items():
        if len(rows) != total:
            raise ValueError(
                f"{model}: expected {total}, got {len(rows)}"
            )

    # Validate row alignment using gold labels.
    reference = predictions["lora"]

    for model, rows in predictions.items():
        for i, (a, b) in enumerate(zip(reference, rows)):
            if a["gold"] != b["gold"]:
                raise ValueError(
                    f"Gold mismatch: model={model}, index={i}"
                )

    groups = defaultdict(list)

    for i, item in enumerate(metadata):
        groups[item["event_id"]].append(i)

    output = []

    for event_id in sorted(groups):
        indices = groups[event_id]

        record = {
            "event_id": event_id,
            "samples": len(indices),
        }

        for model, rows in predictions.items():
            group_rows = [rows[i] for i in indices]
            metrics = analyze_group(group_rows)

            for metric, value in metrics.items():
                if metric != "samples":
                    record[f"{model}_{metric}"] = value

        record["exact_match_delta"] = (
            record["balanced_exact_match"]
            - record["lora_exact_match"]
        )
        record["urgency_delta"] = (
            record["balanced_urgency_accuracy"]
            - record["lora_urgency_accuracy"]
        )

        output.append(record)

    output_path = RESULTS / "event_level_analysis.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(output[0].keys())
        )
        writer.writeheader()
        writer.writerows(output)

    print("=== EVENT-LEVEL ANALYSIS ===")
    print(
        f"{'Event':38} "
        f"{'N':>4} "
        f"{'LoRA EM':>9} "
        f"{'Bal EM':>9} "
        f"{'LoRA Urg':>9} "
        f"{'Bal Urg':>9}"
    )

    for row in output:
        print(
            f"{row['event_id'][:38]:38} "
            f"{row['samples']:4d} "
            f"{row['lora_exact_match']:9.3f} "
            f"{row['balanced_exact_match']:9.3f} "
            f"{row['lora_urgency_accuracy']:9.3f} "
            f"{row['balanced_urgency_accuracy']:9.3f}"
        )

    print("\n=== LARGEST URGENCY DECLINES ===")
    for row in sorted(
        output, key=lambda r: r["urgency_delta"]
    )[:5]:
        print(
            row["event_id"],
            f"delta={row['urgency_delta']:+.3f}",
            f"critical_n={row['lora_critical_support']}",
            f"critical_recall_lora={row['lora_critical_recall']}",
            f"critical_recall_balanced={row['balanced_critical_recall']}",
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()