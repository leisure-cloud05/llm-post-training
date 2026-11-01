import argparse
import json
from pathlib import Path

from mlx_lm import load, generate

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

LABELS = {
    "category": {
        "connectors", "dashboards", "alerts", "reports",
        "attribution", "billing", "users_workspace", "northstar"
    },
    "sentiment": {
        "neutral", "frustrated", "confused",
        "angry", "overwhelmed", "anxious"
    },
    "urgency": {"low", "medium", "high", "critical"},
}


def parse_prediction(text):
    try:
        prediction = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(prediction, dict):
        return None

    return prediction


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=None)

    # NEW: Select the evaluation dataset.
    parser.add_argument(
        "--data-file",
        default="data/test.jsonl",
    )

    # NEW: Give this experiment a separate output name.
    parser.add_argument(
        "--output-name",
        default=None,
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    test_path = Path(args.data_file)
    if not test_path.is_absolute():
        test_path = root / test_path

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    name = args.output_name or (
        "lora" if args.adapter else "base"
    )

    model, tokenizer = load(
        MODEL,
        adapter_path=args.adapter,
    )

    with test_path.open(encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if args.limit is not None:
        records = records[:args.limit]

    if not records:
        raise ValueError("No evaluation records found.")

    keys = list(LABELS)
    correct = {key: 0 for key in keys}
    valid_json = 0
    compliant = 0
    exact_match = 0
    predictions = []

    for index, record in enumerate(records, start=1):
        messages = record["messages"]

        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=128,
            verbose=False,
        )

        gold = json.loads(messages[-1]["content"])
        pred = parse_prediction(output)

        is_json = pred is not None

        is_compliant = (
            is_json
            and set(pred) == set(keys)
            and all(
                isinstance(pred[key], str)
                and pred[key] in LABELS[key]
                for key in keys
            )
        )

        valid_json += int(is_json)
        compliant += int(is_compliant)

        if is_compliant:
            for key in keys:
                correct[key] += int(
                    pred[key] == gold[key]
                )

            exact_match += int(
                all(
                    pred[key] == gold[key]
                    for key in keys
                )
            )

        predictions.append({
            "gold": gold,
            "prediction": pred,
            "raw_output": output,
            "valid_json": is_json,
            "label_compliant": is_compliant,
        })

        if index % 25 == 0 or index == len(records):
            print(
                f"Evaluated {index}/{len(records)}",
                flush=True,
            )

    n = len(records)

    metrics = {
        "model": MODEL,
        "adapter": args.adapter,
        "data_file": str(test_path),
        "samples": n,
        "json_validity": valid_json / n,
        "label_compliance": compliant / n,
        "exact_match_accuracy": exact_match / n,
        "field_accuracy": {
            key: correct[key] / n
            for key in keys
        },
    }

    suffix = f"_{n}" if args.limit is not None else ""

    output_path = (
        results_dir / f"{name}_metrics{suffix}.json"
    )
    predictions_path = (
        results_dir / f"{name}_predictions{suffix}.jsonl"
    )

    # Prevent accidental loss of previous experiments.
    for path in (output_path, predictions_path):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Choose another --output-name."
            )

    output_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with predictions_path.open(
        "w", encoding="utf-8"
    ) as f:
        for row in predictions:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                ) + "\n"
            )

    print("\nEvaluation results:")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved metrics to {output_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()