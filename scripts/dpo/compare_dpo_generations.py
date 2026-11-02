import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTERS = {
    "sft": Path("models/dpo_sft/best"),
    "dpo": Path("models/dpo/best"),
}

DATA = Path("data/dpo/test.jsonl")
OUT = Path("results/dpo/generation_comparison")

SEED = 2026
N = 20
BATCH = 5
MAX_NEW_TOKENS = 256
GB = 1024 ** 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def get_manifest():
    rows = read_jsonl(DATA)

    assert len(rows) == 400

    # Avoid very long prompts in the first qualitative test.
    # Selection is deterministic and independent of model outputs.
    candidates = [
        i for i, row in enumerate(rows)
        if 30 <= len(row["prompt"].strip()) <= 700
    ]

    if len(candidates) < N:
        raise RuntimeError("Not enough eligible prompts")

    indices = random.Random(SEED).sample(
        candidates, N
    )

    return {
        "model": MODEL,
        "test_sha256": sha256(DATA),
        "sft_sha256": sha256(
            ADAPTERS["sft"] / "adapters.safetensors"
        ),
        "dpo_sha256": sha256(
            ADAPTERS["dpo"] / "adapters.safetensors"
        ),
        "seed": SEED,
        "indices": indices,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": "greedy",
    }


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT / "manifest.json"
    current = get_manifest()

    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved != current:
            raise RuntimeError(
                "Model, data, or sampling settings changed. "
                "Do not mix comparison results."
            )
    else:
        if list(OUT.glob("*.jsonl")):
            raise RuntimeError(
                "Existing results have no matching manifest"
            )

        manifest_path.write_text(
            json.dumps(current, indent=2),
            encoding="utf-8",
        )

    return current


def valid_batch(path, indices):
    if not path.exists():
        return False

    try:
        records = read_jsonl(path)

        return (
            len(records) == len(indices)
            and [r["index"] for r in records] == indices
            and all(
                isinstance(r["response"], str)
                and isinstance(r["response_tokens"], int)
                for r in records
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def worker(kind, start):
    import mlx.core as mx
    from mlx_lm import load, generate

    mx.set_memory_limit(10 * GB)
    mx.set_cache_limit(256 * 1024 ** 2)

    manifest = prepare()
    indices = manifest["indices"][start:start + BATCH]
    rows = read_jsonl(DATA)

    model, tokenizer = load(
        MODEL,
        adapter_path=str(ADAPTERS[kind]),
    )

    model.freeze()
    model.eval()

    results = []

    for idx in indices:
        prompt = rows[idx]["prompt"].strip()

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Default MLX-LM sampler uses greedy decoding.
        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=MAX_NEW_TOKENS,
            verbose=False,
        )

        token_count = len(
            tokenizer.encode(
                response,
                add_special_tokens=False,
            )
        )

        results.append({
            "index": idx,
            "model": kind,
            "prompt": prompt,
            "response": response,
            "response_tokens": token_count,
        })

        gc.collect()
        mx.clear_cache()

        active = mx.get_active_memory() / GB
        peak = mx.get_peak_memory() / GB

        print(
            f"{kind.upper()} index={idx} "
            f"tokens={token_count} "
            f"active={active:.2f}GiB "
            f"peak={peak:.2f}GiB",
            flush=True,
        )

        if active > 8 or peak >= 9.5:
            raise RuntimeError(
                "Memory safety threshold reached"
            )

    target = OUT / f"{kind}_{start:02d}.jsonl"
    temporary = OUT / f"{kind}_{start:02d}.partial.jsonl"

    with temporary.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    if not valid_batch(temporary, indices):
        raise RuntimeError("Batch verification failed")

    os.replace(temporary, target)


def write_report(manifest):
    indices = manifest["indices"]
    combined = {}

    for kind in ("sft", "dpo"):
        records = []

        for start in range(0, N, BATCH):
            path = OUT / f"{kind}_{start:02d}.jsonl"

            expected = indices[start:start + BATCH]
            if not valid_batch(path, expected):
                raise RuntimeError(
                    f"Incomplete results: {path}"
                )

            records.extend(read_jsonl(path))

        combined[kind] = records

    sft_tokens = [
        x["response_tokens"] for x in combined["sft"]
    ]
    dpo_tokens = [
        x["response_tokens"] for x in combined["dpo"]
    ]

    summary = {
        "samples": N,
        "sft_mean_response_tokens": sum(sft_tokens) / N,
        "dpo_mean_response_tokens": sum(dpo_tokens) / N,
        "sft_max_token_count": max(sft_tokens),
        "dpo_max_token_count": max(dpo_tokens),
        "sft_hit_token_limit": sum(
            x >= MAX_NEW_TOKENS for x in sft_tokens
        ),
        "dpo_hit_token_limit": sum(
            x >= MAX_NEW_TOKENS for x in dpo_tokens
        ),
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# SFT vs DPO: Generation Comparison",
        "",
        f"Test samples: {N}",
        f"Random seed: {SEED}",
        "Decoding: Greedy",
        f"Max new tokens: {MAX_NEW_TOKENS}",
        "",
        "## Length statistics",
        "",
        f"- SFT mean output tokens: "
        f"{summary['sft_mean_response_tokens']:.1f}",
        f"- DPO mean output tokens: "
        f"{summary['dpo_mean_response_tokens']:.1f}",
        f"- SFT outputs hitting limit: "
        f"{summary['sft_hit_token_limit']}",
        f"- DPO outputs hitting limit: "
        f"{summary['dpo_hit_token_limit']}",
        "",
    ]

    for i, idx in enumerate(indices):
        sft = combined["sft"][i]
        dpo = combined["dpo"][i]

        assert sft["index"] == dpo["index"] == idx
        assert sft["prompt"] == dpo["prompt"]

        lines.extend([
            f"## Example {i + 1} (Test index {idx})",
            "",
            "### Prompt",
            "",
            "````text",
            sft["prompt"],
            "````",
            "",
            f"### SFT ({sft['response_tokens']} tokens)",
            "",
            "````text",
            sft["response"],
            "````",
            "",
            f"### DPO ({dpo['response_tokens']} tokens)",
            "",
            "````text",
            dpo["response"],
            "````",
            "",
        ])

    report = OUT / "comparison_report.md"
    report.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n=== GENERATION COMPARISON ===")
    print(json.dumps(summary, indent=2))
    print("\nReport:", report)
    print("PASS: SFT and DPO generations completed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker-kind",
        choices=["sft", "dpo"],
    )
    parser.add_argument(
        "--worker-start",
        type=int,
    )

    args = parser.parse_args()
    manifest = prepare()

    if args.worker_kind is not None:
        start = args.worker_start

        if (
            start is None
            or start < 0
            or start >= N
            or start % BATCH != 0
        ):
            raise ValueError("Invalid worker start")

        worker(args.worker_kind, start)
        return

    for kind in ("sft", "dpo"):
        for start in range(0, N, BATCH):
            indices = manifest["indices"][
                start:start + BATCH
            ]

            output = OUT / f"{kind}_{start:02d}.jsonl"

            if valid_batch(output, indices):
                print(f"SKIP {kind} {start:02d}")
                continue

            print(f"RUN {kind} {start:02d}", flush=True)

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-kind",
                    kind,
                    "--worker-start",
                    str(start),
                ],
                check=True,
            )

    write_report(manifest)


if __name__ == "__main__":
    main()
