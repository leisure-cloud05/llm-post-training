import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("results/dpo/generation_comparison")
OUT = Path("results/dpo/generation_comparison_512")
DATA = Path("data/dpo/test.jsonl")

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTERS = {
    "sft": Path("models/dpo_sft/best"),
    "dpo": Path("models/dpo/best"),
}

LIMIT = 512
GB = 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)

    original = json.loads((ROOT / "manifest.json").read_text())
    indices = original["indices"]

    assert len(indices) == 20
    assert digest(DATA) == original["test_sha256"]

    selected = set()
    original_outputs = {}

    for kind in ("sft", "dpo"):
        assert digest(
            ADAPTERS[kind] / "adapters.safetensors"
        ) == original[f"{kind}_sha256"]

        records = []
        for start in range(0, 20, 5):
            records.extend(
                read_jsonl(ROOT / f"{kind}_{start:02d}.jsonl")
            )

        assert [r["index"] for r in records] == indices

        original_outputs[kind] = {
            r["index"]: r for r in records
        }

        for record in records:
            if record["response_tokens"] >= 256:
                selected.add(record["index"])

    # Preserve original sample order.
    selected = [idx for idx in indices if idx in selected]

    manifest = {
        "test_sha256": digest(DATA),
        "sft_sha256": original["sft_sha256"],
        "dpo_sha256": original["dpo_sha256"],
        "selected_indices": selected,
        "selection": "either original output >= 256 tokens",
        "max_new_tokens": LIMIT,
        "decoding": "greedy",
    }

    path = OUT / "manifest.json"

    if path.exists():
        assert json.loads(path.read_text()) == manifest, (
            "Manifest mismatch: refusing to mix experiments"
        )
    else:
        assert not list(OUT.glob("*.json")), (
            "Existing outputs without manifest"
        )
        path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    return manifest, original_outputs


def worker(kind, idx):
    import mlx.core as mx
    from mlx_lm import load, generate

    mx.set_memory_limit(10 * GB)
    mx.set_cache_limit(256 * 1024 ** 2)

    data = read_jsonl(DATA)
    prompt = data[idx]["prompt"].strip()

    model, tokenizer = load(
        MODEL,
        adapter_path=str(ADAPTERS[kind]),
    )
    model.freeze()
    model.eval()

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Same greedy settings as the original comparison.
    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=LIMIT,
        verbose=False,
    )

    token_count = len(
        tokenizer.encode(
            response,
            add_special_tokens=False,
        )
    )

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
        raise RuntimeError("Memory safety stop")

    result = {
        "index": idx,
        "model": kind,
        "prompt": prompt,
        "response": response,
        "response_tokens": token_count,
    }

    target = OUT / f"{kind}_{idx:03d}.json"
    temporary = OUT / f"{kind}_{idx:03d}.partial.json"

    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary, target)


def main():
    manifest, original = prepare()
    selected = manifest["selected_indices"]

    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        kind = sys.argv[2]
        idx = int(sys.argv[3])

        assert kind in ADAPTERS
        assert idx in selected

        worker(kind, idx)
        return

    print("Selected test samples:", len(selected))
    print("Test indices:", selected, flush=True)

    for kind in ("sft", "dpo"):
        for idx in selected:
            target = OUT / f"{kind}_{idx:03d}.json"

            if target.exists():
                record = json.loads(target.read_text())
                if (
                    record.get("index") == idx
                    and record.get("model") == kind
                    and isinstance(record.get("response"), str)
                    and isinstance(record.get("response_tokens"), int)
                ):
                    print(f"SKIP {kind} {idx}")
                    continue

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    kind,
                    str(idx),
                ],
                check=True,
            )

    lines = [
        "# SFT vs DPO: 512-Token Follow-up",
        "",
        "Exploratory evaluation of previously length-limited samples.",
        "Same test prompts and greedy decoding; no retraining.",
        "",
    ]

    summary = {
        "samples": len(selected),
        "sft_mean_tokens": 0,
        "dpo_mean_tokens": 0,
        "sft_hit_512": 0,
        "dpo_hit_512": 0,
    }

    for kind in ("sft", "dpo"):
        outputs = [
            json.loads(
                (OUT / f"{kind}_{idx:03d}.json").read_text()
            )
            for idx in selected
        ]

        summary[f"{kind}_mean_tokens"] = (
            sum(x["response_tokens"] for x in outputs)
            / len(outputs)
        )

        summary[f"{kind}_hit_512"] = sum(
            x["response_tokens"] >= LIMIT
            for x in outputs
        )

    for idx in selected:
        lines.extend([
            f"## Test index {idx}",
            "",
            "### Prompt",
            "",
            "````text",
            original["sft"][idx]["prompt"],
            "````",
            "",
        ])

        for kind in ("sft", "dpo"):
            before = original[kind][idx]
            after = json.loads(
                (OUT / f"{kind}_{idx:03d}.json").read_text()
            )

            lines.extend([
                f"### {kind.upper()}",
                "",
                f"Original tokens: {before['response_tokens']}",
                f"New tokens: {after['response_tokens']}",
                "",
                "````text",
                after["response"],
                "````",
                "",
            ])

    (OUT / "comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== 512-TOKEN FOLLOW-UP ===")
    print(json.dumps(summary, indent=2))
    print("PASS: comparison completed.")
    print("Report:", OUT / "comparison_report.md")


if __name__ == "__main__":
    main()
