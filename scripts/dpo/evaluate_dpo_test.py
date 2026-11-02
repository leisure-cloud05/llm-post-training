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
REF = Path("cache/dpo/ref_cache/test")
DPO = Path("models/dpo/best")
OUT = Path("results/dpo/eval_test_dpo")

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
TOTAL = 400
BATCH = 20
BETA = 0.1
GB = 1024 ** 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def stable_dpo_loss(margin):
    return (
        math.log1p(math.exp(-abs(margin)))
        + max(0.0, -margin)
    )


def validate_batch(path, start):
    if not path.exists():
        return False

    try:
        records = read_jsonl(path)
        if len(records) != BATCH:
            return False

        for i, row in enumerate(records):
            if row["index"] != start + i:
                return False

            for key in (
                "policy_chosen_logp",
                "policy_rejected_logp",
                "reward_margin",
                "dpo_loss",
            ):
                if not math.isfinite(float(row[key])):
                    return False

        return True

    except (KeyError, ValueError, TypeError, OSError):
        return False


def get_manifest():
    if not DATA.exists():
        raise RuntimeError("Test dataset not found")

    if not (DPO / "adapters.safetensors").exists():
        raise RuntimeError("Best DPO adapter not found")

    saved_ref = json.loads(
        (REF / "source_hashes.json").read_text()
    )

    test_hash = sha256(DATA)
    sft_hash = sha256(
        "models/dpo_sft/best/adapters.safetensors"
    )

    if saved_ref["test_sha256"] != test_hash:
        raise RuntimeError("Test data hash mismatch")

    if saved_ref["sft_adapter_sha256"] != sft_hash:
        raise RuntimeError("SFT reference hash mismatch")

    if saved_ref["model"] != MODEL:
        raise RuntimeError("Base model mismatch")

    if saved_ref["max_tokens"] != 1024:
        raise RuntimeError("Token limit mismatch")

    best_info = json.loads(
        (DPO / "best.json").read_text()
    )

    if best_info["step"] != 1000:
        raise RuntimeError(
            "Expected selected DPO checkpoint at Step 1000"
        )

    checkpoint_meta = json.loads(
        Path(
            "models/dpo/best/training_metadata.json"
        ).read_text()
    )

    if checkpoint_meta["sft_adapter_sha256"] != sft_hash:
        raise RuntimeError("Training SFT adapter hash mismatch")

    if checkpoint_meta["beta"] != BETA:
        raise RuntimeError("DPO beta mismatch")

    return {
        "test_sha256": test_hash,
        "sft_sha256": sft_hash,
        "dpo_sha256": sha256(
            DPO / "adapters.safetensors"
        ),
        "model": MODEL,
        "beta": BETA,
        "max_tokens": 1024,
        "total": TOTAL,
    }


def get_reference():
    records = []

    for start in range(0, TOTAL, BATCH):
        path = REF / f"batch_{start:04d}.jsonl"
        records.extend(read_jsonl(path))

    if len(records) != TOTAL:
        raise RuntimeError("Reference cache count mismatch")

    for i, record in enumerate(records):
        if record["index"] != i:
            raise RuntimeError("Reference cache order mismatch")

        for key in (
            "ref_chosen_logp",
            "ref_rejected_logp",
        ):
            if not math.isfinite(float(record[key])):
                raise RuntimeError(
                    f"Invalid reference logp at index {i}"
                )

    return records


def worker(start):
    import mlx.core as mx
    from mlx_lm import load

    from check_dpo_loss import (
        assistant_response,
        make_pair,
        answer_logprob,
    )

    mx.set_memory_limit(10 * GB)
    mx.set_cache_limit(256 * 1024 ** 2)

    data = read_jsonl(DATA)
    refs = get_reference()

    assert len(data) == TOTAL

    model, tokenizer = load(
        MODEL,
        adapter_path=str(DPO),
    )

    model.freeze()
    model.eval()

    temporary = OUT / f"batch_{start:04d}.partial.jsonl"
    final = OUT / f"batch_{start:04d}.jsonl"

    with temporary.open("w", encoding="utf-8") as f:
        for idx in range(start, start + BATCH):
            row = data[idx]
            result = {"index": idx}

            for kind in ("chosen", "rejected"):
                pair = make_pair(
                    tokenizer,
                    row["prompt"].strip(),
                    assistant_response(row[kind]),
                    1024,
                )

                lp = answer_logprob(model, pair)
                mx.eval(lp)

                result[
                    f"policy_{kind}_logp"
                ] = float(lp)

                del pair, lp
                gc.collect()
                mx.clear_cache()

            policy_delta = (
                result["policy_chosen_logp"]
                - result["policy_rejected_logp"]
            )

            ref_delta = (
                float(refs[idx]["ref_chosen_logp"])
                - float(refs[idx]["ref_rejected_logp"])
            )

            margin = BETA * (
                policy_delta - ref_delta
            )

            result["reward_margin"] = margin
            result["dpo_loss"] = stable_dpo_loss(margin)

            f.write(json.dumps(result) + "\n")

            active = mx.get_active_memory() / GB
            peak = mx.get_peak_memory() / GB

            if active > 8 or peak >= 9.5:
                raise RuntimeError(
                    f"Memory safety stop: "
                    f"active={active:.2f}GiB, "
                    f"peak={peak:.2f}GiB"
                )

    if not validate_batch(temporary, start):
        raise RuntimeError(
            f"Failed batch validation: {start}"
        )

    os.replace(temporary, final)

    print(
        f"PASS {start:04d}-{start+BATCH-1:04d} "
        f"peak={mx.get_peak_memory()/GB:.2f}GiB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-start", type=int)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    current = get_manifest()
    manifest_file = OUT / "manifest.json"

    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != current:
            raise RuntimeError(
                "Model or dataset changed. "
                "Existing evaluation cannot be reused."
            )
    else:
        if list(OUT.glob("batch_*.jsonl")):
            raise RuntimeError(
                "Existing evaluation has no manifest"
            )

        manifest_file.write_text(
            json.dumps(current, indent=2),
            encoding="utf-8",
        )

    if args.worker_start is not None:
        start = args.worker_start
        if start < 0 or start % BATCH or start + BATCH > TOTAL:
            raise ValueError("Invalid worker start")

        worker(start)
        return

    for start in range(0, TOTAL, BATCH):
        path = OUT / f"batch_{start:04d}.jsonl"

        if validate_batch(path, start):
            print(f"SKIP {start:04d}-{start+BATCH-1:04d}")
            continue

        print(
            f"RUN  {start:04d}-{start+BATCH-1:04d}",
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

    results = []

    for start in range(0, TOTAL, BATCH):
        path = OUT / f"batch_{start:04d}.jsonl"

        if not validate_batch(path, start):
            raise RuntimeError(f"Invalid batch: {start}")

        results.extend(read_jsonl(path))

    if len(results) != TOTAL:
        raise RuntimeError("Incomplete test evaluation")

    if [r["index"] for r in results] != list(range(TOTAL)):
        raise RuntimeError("Test index order mismatch")

    margins = [r["reward_margin"] for r in results]
    losses = [r["dpo_loss"] for r in results]

    # Treat numerical near-ties separately.
    tolerance = 1e-5

    metrics = {
        "samples": TOTAL,
        "sft_reference_dpo_loss": math.log(2),
        "dpo_test_loss": sum(losses) / TOTAL,
        "dpo_reward_accuracy": sum(
            x > tolerance for x in margins
        ) / TOTAL,
        "dpo_tie_rate": sum(
            abs(x) <= tolerance for x in margins
        ) / TOTAL,
        "dpo_mean_reward_margin": sum(margins) / TOTAL,
    }

    if json.loads(manifest_file.read_text()) != current:
        raise RuntimeError("Evaluation manifest changed")

    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "source": current,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== INDEPENDENT TEST RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("\nPASS: 400/400 test examples evaluated.")
    print("Saved to:", OUT / "summary.json")


if __name__ == "__main__":
    main()
