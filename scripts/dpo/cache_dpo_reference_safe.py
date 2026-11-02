import argparse
import gc
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from check_dpo_loss import (
    assistant_response,
    make_pair,
    answer_logprob,
)

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER = "models/dpo_sft/best"
GB = 1024 ** 3


def memory_stats():
    active = mx.get_active_memory() / GB
    cached = mx.get_cache_memory() / GB
    peak = mx.get_peak_memory() / GB
    return active, cached, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--memory-warning-gb", type=float, default=8)
    parser.add_argument("--data-file", default="data/dpo/train.jsonl")
    parser.add_argument(
        "--output",
        default="data/dpo/reference_safe_smoke.jsonl",
    )
    args = parser.parse_args()

    assert args.start >= 0
    assert 1 <= args.limit <= 20

    # Restrict reusable MLX allocation cache.
    mx.set_cache_limit(256 * 1024 * 1024)

    print("Loading frozen SFT Reference...", flush=True)
    model, tokenizer = load(MODEL, adapter_path=ADAPTER)
    model.freeze()
    model.eval()

    active, cached, peak = memory_stats()
    print(
        f"Initial memory: active={active:.2f}GB "
        f"cache={cached:.2f}GB peak={peak:.2f}GB",
        flush=True,
    )

    if active + cached > args.memory_warning_gb:
        raise RuntimeError("Memory threshold exceeded after loading model")

    results = []
    processed = 0

    with open(args.data_file, encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index < args.start:
                continue
            if processed >= args.limit:
                break

            row = json.loads(line)
            prompt = row["prompt"].strip()
            chosen = assistant_response(row["chosen"])
            rejected = assistant_response(row["rejected"])

            chosen_pair = make_pair(
                tokenizer, prompt, chosen, args.max_tokens
            )
            rejected_pair = make_pair(
                tokenizer, prompt, rejected, args.max_tokens
            )

            logp_c = answer_logprob(model, chosen_pair)
            mx.eval(logp_c)
            lc = float(logp_c)

            del logp_c, chosen_pair
            gc.collect()
            mx.clear_cache()

            logp_r = answer_logprob(model, rejected_pair)
            mx.eval(logp_r)
            lr = float(logp_r)

            del logp_r, rejected_pair
            gc.collect()
            mx.clear_cache()

            assert math.isfinite(lc) and math.isfinite(lr)

            results.append({
                "index": index,
                "ref_chosen_logp": lc,
                "ref_rejected_logp": lr,
            })

            processed += 1
            active, cached, peak = memory_stats()

            print(
                f"{processed}/{args.limit} "
                f"index={index} "
                f"logp=({lc:.2f}, {lr:.2f}) "
                f"active={active:.2f}GB "
                f"cache={cached:.2f}GB "
                f"peak={peak:.2f}GB",
                flush=True,
            )

            if active + cached > args.memory_warning_gb:
                raise RuntimeError(
                    "Memory warning threshold reached; "
                    "stopping before next sample"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(results)} records to {output}")
    print("Process complete.")


if __name__ == "__main__":
    main()
