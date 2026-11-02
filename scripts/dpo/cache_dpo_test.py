import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

DATA = Path("data/dpo/test.jsonl")
ADAPTER = Path("models/dpo_sft/best")
OUT = Path("cache/dpo/ref_cache/test")
MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
BATCH = 20
TOTAL = 400
GB = 1024 ** 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows():
    with DATA.open(encoding="utf-8") as f:
        result = [json.loads(line) for line in f if line.strip()]
    if len(result) != TOTAL:
        raise RuntimeError(f"Expected {TOTAL} test rows, got {len(result)}")
    return result


def validate_batch(path, start):
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if len(records) != BATCH:
            return False
        for i, record in enumerate(records):
            if record["index"] != start + i:
                return False
            for key in ("ref_chosen_logp", "ref_rejected_logp"):
                if not math.isfinite(float(record[key])):
                    return False
        return True
    except (ValueError, KeyError, OSError, TypeError):
        return False


def run_worker(start):
    import mlx.core as mx
    from mlx_lm import load
    from check_dpo_loss import (
        assistant_response,
        make_pair,
        answer_logprob,
    )

    mx.set_memory_limit(10 * GB)
    mx.set_cache_limit(256 * 1024 ** 2)

    data = rows()
    model, tokenizer = load(MODEL, adapter_path=str(ADAPTER))
    model.freeze()
    model.eval()

    output = OUT / f"batch_{start:04d}.jsonl"
    temporary = OUT / f"batch_{start:04d}.partial.jsonl"

    with temporary.open("w", encoding="utf-8") as f:
        for idx in range(start, start + BATCH):
            row = data[idx]
            result = {"index": idx}

            for name in ("chosen", "rejected"):
                pair = make_pair(
                    tokenizer,
                    row["prompt"].strip(),
                    assistant_response(row[name]),
                    1024,
                )

                logp = answer_logprob(model, pair)
                mx.eval(logp)

                result[f"ref_{name}_logp"] = float(logp)

                del pair, logp
                gc.collect()
                mx.clear_cache()

            f.write(json.dumps(result) + "\n")

            active = mx.get_active_memory() / GB
            peak = mx.get_peak_memory() / GB

            if active > 8 or peak > 9.5:
                raise RuntimeError(
                    f"Memory safety stop: active={active:.2f}, "
                    f"peak={peak:.2f} GiB"
                )

    if not validate_batch(temporary, start):
        raise RuntimeError(f"Batch validation failed: {start}")

    os.replace(temporary, output)
    print(
        f"PASS batch {start:04d}-{start + BATCH - 1:04d} "
        f"peak={mx.get_peak_memory() / GB:.2f}GiB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-start", type=int)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT / "source_hashes.json"
    current = {
        "test_sha256": sha256(DATA),
        "sft_adapter_sha256": sha256(
            ADAPTER / "adapters.safetensors"
        ),
        "model": MODEL,
        "max_tokens": 1024,
    }

    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved != current:
            raise RuntimeError(
                "Test data or SFT adapter changed. "
                "Existing cache cannot be reused."
            )
    else:
        if list(OUT.glob("batch_*.jsonl")):
            raise RuntimeError(
                "Existing cache has no source manifest."
            )
        manifest_path.write_text(
            json.dumps(current, indent=2),
            encoding="utf-8",
        )

    if args.worker_start is not None:
        start = args.worker_start
        if start < 0 or start % BATCH or start + BATCH > TOTAL:
            raise ValueError("Invalid worker start")
        run_worker(start)
        return

    rows()

    for start in range(0, TOTAL, BATCH):
        output = OUT / f"batch_{start:04d}.jsonl"

        if validate_batch(output, start):
            print(f"SKIP {start:04d}-{start + BATCH - 1:04d}")
            continue

        print(
            f"RUN  {start:04d}-{start + BATCH - 1:04d}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-start",
                str(start),
            ],
            check=True,
        )

    records = []
    for start in range(0, TOTAL, BATCH):
        path = OUT / f"batch_{start:04d}.jsonl"
        if not validate_batch(path, start):
            raise RuntimeError(f"Invalid batch: {path}")
        with path.open(encoding="utf-8") as f:
            records.extend(
                json.loads(line) for line in f if line.strip()
            )

    assert len(records) == TOTAL
    assert [r["index"] for r in records] == list(range(TOTAL))
    assert json.loads(manifest_path.read_text()) == current

    print("\nTest Reference Cache: 400/400 PASS")
    print("Test data / SFT adapter SHA-256: PASS")


if __name__ == "__main__":
    main()
