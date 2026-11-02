import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/sft/sft"

LABELS = {
    "category": [
        "alerts", "attribution", "billing", "connectors",
        "dashboards", "northstar", "reports", "users_workspace"
    ],
    "sentiment": [
        "angry", "anxious", "confused", "frustrated",
        "neutral", "overwhelmed"
    ],
    "urgency": ["critical", "high", "low", "medium"],
}

INVALID = "__invalid__"


def load_predictions(name):
    path = RESULTS / f"{name}_predictions.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predicted_label(row, field):
    pred = row.get("prediction")
    if not row.get("label_compliant") or not isinstance(pred, dict):
        return INVALID
    return pred.get(field, INVALID)


def exact_match(row):
    if not row.get("label_compliant"):
        return False
    return all(
        row["prediction"].get(field) == row["gold"][field]
        for field in LABELS
    )


def analyze(name, rows):
    result = {
        "samples": len(rows),
        "exact_match": sum(exact_match(r) for r in rows) / len(rows),
        "fields": {},
    }

    print(f"\n===== {name.upper()} =====")
    print(f"Exact match: {result['exact_match']:.3f}")

    for field, labels in LABELS.items():
        y_true = [r["gold"][field] for r in rows]
        y_pred = [predicted_label(r, field) for r in rows]

        report = classification_report(
            y_true, y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        accuracy = sum(
            a == b for a, b in zip(y_true, y_pred)
        ) / len(rows)

        macro_f1 = report["macro avg"]["f1-score"]

        result["fields"][field] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": {
                label: report[label] for label in labels
            },
        }

        print(f"\n{field}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Macro-F1: {macro_f1:.3f}")

        for label in labels:
            s = report[label]
            print(
                f"  {label:18} "
                f"support={int(s['support']):3} "
                f"precision={s['precision']:.3f} "
                f"recall={s['recall']:.3f} "
                f"f1={s['f1-score']:.3f}"
            )

        matrix_labels = labels + [INVALID]
        matrix = confusion_matrix(
            y_true, y_pred, labels=matrix_labels
        )

        path = RESULTS / f"{name}_{field}_confusion.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true/pred"] + matrix_labels)
            for label, counts in zip(matrix_labels, matrix):
                writer.writerow([label] + counts.tolist())

    return result


def main():
    base = load_predictions("base")
    lora = load_predictions("lora")

    if len(base) != len(lora) or not base:
        raise ValueError("Prediction files are missing or incompatible.")

    for i, (a, b) in enumerate(zip(base, lora)):
        if a["gold"] != b["gold"]:
            raise ValueError(f"Gold label mismatch at index {i}")

    base_result = analyze("base", base)
    lora_result = analyze("lora", lora)

    transitions = Counter()
    error_cases = []

    for index, (a, b) in enumerate(zip(base, lora)):
        before = exact_match(a)
        after = exact_match(b)

        if not before and after:
            status = "fixed_by_lora"
        elif before and not after:
            status = "regressed"
        elif not before and not after:
            status = "still_wrong"
        else:
            status = "both_correct"

        transitions[status] += 1

        if status != "both_correct":
            error_cases.append({
                "index": index,
                "status": status,
                "gold": a["gold"],
                "base_prediction": a["prediction"],
                "lora_prediction": b["prediction"],
            })

    print("\n===== ERROR TRANSITIONS =====")
    for key, value in sorted(transitions.items()):
        print(f"{key}: {value}")

    output = {
        "base": base_result,
        "lora": lora_result,
        "error_transitions": dict(transitions),
    }

    (RESULTS / "error_analysis.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (RESULTS / "error_cases.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in error_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nError analysis completed.")


if __name__ == "__main__":
    main()
