import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "argilla/ultrafeedback-binarized-preferences-cleaned"
TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def normalized_prompt(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def assistant_text(messages):
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if messages[-1].get("role") != "assistant":
        return None
    text = messages[-1].get("content")
    return text if isinstance(text, str) and text.strip() else None


def token_count(tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    if not isinstance(ids, list) or not ids or not isinstance(ids[0], int):
        raise TypeError("Unexpected tokenization result: expected list of integer token IDs")
    return len(ids)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/dpo")
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--valid", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-rating-gap", type=float, default=1.0)
    parser.add_argument("--max-scanned", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = args.train + args.valid + args.test
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    source = load_dataset(DATASET, split="train", streaming=True)
    counters = Counter()
    eligible = []
    seen = set()

    for row in source.take(args.max_scanned):
        counters["scanned"] += 1
        prompt = row.get("prompt")
        chosen_msgs = row.get("chosen")
        rejected_msgs = row.get("rejected")
        if not isinstance(prompt, str) or not prompt.strip():
            counters["bad_prompt"] += 1
            continue
        chosen = assistant_text(chosen_msgs)
        rejected = assistant_text(rejected_msgs)
        if chosen is None or rejected is None or chosen.strip() == rejected.strip():
            counters["bad_responses"] += 1
            continue
        # UltraFeedback uses message-formatted chosen/rejected; confirm same user prompt.
        if (not isinstance(chosen_msgs[0].get("content"), str)
            or not isinstance(rejected_msgs[0].get("content"), str)
            or normalized_prompt(chosen_msgs[0]["content"]) != normalized_prompt(prompt)
            or normalized_prompt(rejected_msgs[0]["content"]) != normalized_prompt(prompt)):
            counters["prompt_mismatch"] += 1
            continue
        try:
            gap = float(row["chosen-rating"]) - float(row["rejected-rating"])
        except (TypeError, ValueError, KeyError):
            counters["invalid_rating"] += 1
            continue
        if not (gap >= args.min_rating_gap):
            counters["low_gap"] += 1
            continue
        key = normalized_prompt(prompt)
        if key in seen:
            counters["duplicate_prompt"] += 1
            continue
        try:
            max_len = max(token_count(tokenizer, chosen_msgs), token_count(tokenizer, rejected_msgs))
        except (TypeError, ValueError, KeyError):
            counters["tokenization_failed"] += 1
            continue
        if max_len > args.max_tokens:
            counters["too_long"] += 1
            continue
        seen.add(key)
        # Keep a single-turn plain-text pair; training format will be determined separately.
        eligible.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_rating": float(row["chosen-rating"]),
            "rejected_rating": float(row["rejected-rating"]),
            "rating_gap": gap,
            "max_chat_tokens": max_len,
            "source": row.get("source"),
        })
        counters["eligible"] += 1

    if len(eligible) < target:
        raise RuntimeError(
            f"Only {len(eligible)} eligible records; need {target}. "
            "Increase --max-scanned or relax the screening limits. "
            f"Counts: {dict(counters)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[:target]
    splits = {
        "train": selected[:args.train],
        "valid": selected[args.train:args.train + args.valid],
        "test": selected[args.train + args.valid:],
    }

    assert len({normalized_prompt(r["prompt"]) for r in selected}) == target
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        write_jsonl(out / f"{name}.jsonl", records)

    manifest = {
        "dataset": DATASET,
        "tokenizer": TOKENIZER,
        "seed": args.seed,
        "max_scanned": args.max_scanned,
        "max_tokens": args.max_tokens,
        "min_rating_gap": args.min_rating_gap,
        "counts": dict(counters),
        "splits": {k: len(v) for k, v in splits.items()},
        "selection": "First max_scanned source records; filter; seeded shuffle; take requested counts",
        "split_prompt_sha256": {
            k: hashlib.sha256("\n".join(sorted(normalized_prompt(r["prompt"]) for r in v)).encode()).hexdigest()
            for k, v in splits.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("\nSaved preference pairs to:", out.resolve())
    print("Note: These are canonical preference records, not yet MLX training files.")


if __name__ == "__main__":
    main()