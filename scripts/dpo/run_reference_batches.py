import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

BATCH_SIZE = 20
EXPECTED = {"train": 4000, "valid": 400}


def check_batch(path, start, count):
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return (
            len(rows) == count
            and all(
                row["index"] == start + i
                and math.isfinite(row["ref_chosen_logp"])
                and math.isfinite(row["ref_rejected_logp"])
                for i, row in enumerate(rows)
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    split = args.split
    total = EXPECTED[split]
    end = total if args.end is None else args.end

    if not 0 < end <= total or end % BATCH_SIZE:
        parser.error(f"--end must be a multiple of 20, up to {total}")

    source = Path(f"data/dpo/{split}.jsonl")
    output_dir = Path(f"cache/dpo/ref_cache/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with source.open(encoding="utf-8") as f:
        actual = sum(bool(line.strip()) for line in f)
    if actual != total:
        raise RuntimeError(f"{source}: expected {total}, got {actual}")

    completed = 0

    for start in range(0, end, BATCH_SIZE):
        output = output_dir / f"batch_{start:04d}.jsonl"

        if check_batch(output, start, BATCH_SIZE):
            completed += BATCH_SIZE
            print(f"SKIP {split} {start}-{start + 19}", flush=True)
            continue

        temporary = output_dir / f"batch_{start:04d}.partial.jsonl"

        command = [
            sys.executable,
            "scripts/dpo/cache_dpo_reference_safe.py",
            "--data-file", str(source),
            "--start", str(start),
            "--limit", str(BATCH_SIZE),
            "--output", str(temporary),
            "--memory-warning-gb", "8",
        ]

        print(f"RUN  {split} {start}-{start + 19}", flush=True)

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if result.returncode != 0:
            print(result.stdout[-5000:], flush=True)
            raise RuntimeError(f"Batch {start} failed; stopped safely")

        if not check_batch(temporary, start, BATCH_SIZE):
            raise RuntimeError(f"Batch {start} failed validation")

        os.replace(temporary, output)
        completed += BATCH_SIZE

        memory_lines = [
            line for line in result.stdout.splitlines()
            if "active=" in line and "peak=" in line
        ]
        if memory_lines:
            print(memory_lines[-1], flush=True)

        print(f"Progress: {completed}/{end}", flush=True)

    print(f"\nPASS: {split} first {end} records cached and verified.")


if __name__ == "__main__":
    main()
