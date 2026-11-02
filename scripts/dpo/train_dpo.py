import argparse
import gc
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

from check_dpo_loss import (
    assistant_response,
    make_pair,
    answer_logprob,
)

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SFT_DIR = Path("models/dpo_sft/best")
DATA_DIR = Path("data/dpo")

GB = 1024 ** 3
MEMORY_LIMIT = 10 * GB
CACHE_LIMIT = 256 * 1024 ** 2
PEAK_STOP = 9.5 * GB

BETA = 0.1
LR = 1e-6
SEED = 42
MAX_TOKENS = 1024
EVAL_EVERY = 100
MAX_STEPS = 1000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def verify_hashes():
    path = DATA_DIR / "reference_source_hashes.txt"
    expected = {}

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            expected[parts[-1]] = parts[0]

    for split in ("train", "valid"):
        name = f"data_dpo/{split}.jsonl"
        actual_path = f"data/dpo/{split}.jsonl"
        if sha256(actual_path) != expected.get(name):
            raise RuntimeError(
                f"Dataset hash mismatch: {name}"
            )

    print("Source SHA-256: PASS", flush=True)


def read_cache(split, count):
    folder = DATA_DIR / "ref_cache" / split
    records = []

    for path in sorted(folder.glob("batch_*.jsonl")):
        records.extend(read_jsonl(path))

    if len(records) != count:
        raise RuntimeError(
            f"{split}: expected {count} cache records, "
            f"got {len(records)}"
        )

    for i, record in enumerate(records):
        if record["index"] != i:
            raise RuntimeError(
                f"{split}: cache index mismatch at {i}"
            )

        for key in (
            "ref_chosen_logp",
            "ref_rejected_logp",
        ):
            if not math.isfinite(float(record[key])):
                raise RuntimeError(
                    f"{split}: invalid cached logp at {i}"
                )

    return records


def softplus_negative(x):
    # Numerically stable log(1 + exp(-x))
    return math.log1p(math.exp(-abs(x))) + max(0.0, -x)


def memory_report(label):
    active = mx.get_active_memory() / GB
    cached = mx.get_cache_memory() / GB
    peak = mx.get_peak_memory() / GB

    print(
        f"{label} "
        f"active={active:.2f}GiB "
        f"cached={cached:.2f}GiB "
        f"peak={peak:.2f}GiB",
        flush=True,
    )

    return peak


def clean_memory():
    gc.collect()
    mx.clear_cache()


def pair_inputs(tokenizer, row):
    prompt = row["prompt"].strip()

    chosen = make_pair(
        tokenizer,
        prompt,
        assistant_response(row["chosen"]),
        MAX_TOKENS,
    )

    rejected = make_pair(
        tokenizer,
        prompt,
        assistant_response(row["rejected"]),
        MAX_TOKENS,
    )

    return chosen, rejected


def validate(model, tokenizer, rows, refs):
    model.eval()

    total_loss = 0.0
    total_margin = 0.0
    correct = 0
    ties = 0

    # A small tolerance avoids counting numerical noise
    # around the initial SFT/reference equality.
    tolerance = 1e-5

    for i, row in enumerate(rows):
        chosen, rejected = pair_inputs(
            tokenizer, row
        )

        chosen_lp = answer_logprob(model, chosen)
        rejected_lp = answer_logprob(model, rejected)

        mx.eval(chosen_lp, rejected_lp)

        policy_margin = (
            float(chosen_lp) - float(rejected_lp)
        )

        reference_margin = (
            float(refs[i]["ref_chosen_logp"])
            - float(refs[i]["ref_rejected_logp"])
        )

        reward_margin = BETA * (
            policy_margin - reference_margin
        )

        total_loss += softplus_negative(reward_margin)
        total_margin += reward_margin

        if reward_margin > tolerance:
            correct += 1
        elif abs(reward_margin) <= tolerance:
            ties += 1

        del chosen, rejected
        del chosen_lp, rejected_lp
        clean_memory()

        if (i + 1) % 100 == 0:
            peak = memory_report(
                f"  validation {i + 1}/{len(rows)}"
            )
            if peak >= PEAK_STOP / GB:
                raise RuntimeError(
                    "MLX peak reached safety threshold "
                    "during validation"
                )

    model.train()

    n = len(rows)

    return {
        "val_loss": total_loss / n,
        "reward_accuracy": correct / n,
        "tie_rate": ties / n,
        "mean_reward_margin": total_margin / n,
        "samples": n,
    }


def save_checkpoint(
    model,
    optimizer,
    output_dir,
    step,
    metrics,
    run_info,
):
    final = output_dir / f"step_{step:04d}"
    temporary = output_dir / f".step_{step:04d}.tmp"

    if final.exists():
        raise RuntimeError(
            f"Checkpoint already exists: {final}"
        )

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir(parents=True)

    weights = dict(
        tree_flatten(model.trainable_parameters())
    )
    state = tree_flatten(
        optimizer.state,
        destination={},
    )

    mx.eval(*weights.values())
    mx.eval(*state.values())

    mx.save_safetensors(
        str(temporary / "adapters.safetensors"),
        weights,
    )

    mx.save_safetensors(
        str(temporary / "optimizer.safetensors"),
        state,
    )

    shutil.copy2(
        SFT_DIR / "adapter_config.json",
        temporary / "adapter_config.json",
    )

    metadata = {
        **run_info,
        "step": step,
        "metrics": metrics,
    }

    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    temporary.rename(final)

    del weights, state
    clean_memory()

    print(f"CHECKPOINT SAVED: {final}", flush=True)

    return final


def update_best(output_dir, checkpoint, step, metrics):
    best_dir = output_dir / "best"
    temp_dir = output_dir / ".best.tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    shutil.copy2(
        checkpoint / "adapters.safetensors",
        temp_dir / "adapters.safetensors",
    )

    shutil.copy2(
        checkpoint / "adapter_config.json",
        temp_dir / "adapter_config.json",
    )

    (temp_dir / "best.json").write_text(
        json.dumps(
            {
                "step": step,
                **metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if best_dir.exists():
        shutil.rmtree(best_dir)

    temp_dir.rename(best_dir)

    print(
        f"NEW BEST: step={step}, "
        f"val_loss={metrics['val_loss']:.6f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Total target step, not extra steps",
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/dpo",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing step_XXXX checkpoint",
    )

    args = parser.parse_args()

    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(
            f"--steps must be between 1 and {MAX_STEPS}"
        )

    if args.steps % EVAL_EVERY != 0:
        raise ValueError(
            "--steps must be a multiple of 100"
        )

    # Set memory limits BEFORE model loading.
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(CACHE_LIMIT)

    print("MLX memory limit: 10 GiB", flush=True)
    print("MLX cache limit: 256 MiB", flush=True)

    verify_hashes()

    train_rows = read_jsonl(
        DATA_DIR / "train.jsonl"
    )
    valid_rows = read_jsonl(
        DATA_DIR / "valid.jsonl"
    )

    assert len(train_rows) == 4000
    assert len(valid_rows) == 400

    train_refs = read_cache("train", 4000)
    valid_refs = read_cache("valid", 400)

    # Fixed, deterministic order.
    order = list(range(len(train_rows)))
    random.Random(SEED).shuffle(order)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_hash = sha256(
        SFT_DIR / "adapters.safetensors"
    )

    run_info = {
        "model": MODEL,
        "sft_adapter_sha256": sft_hash,
        "train_sha256": sha256(DATA_DIR / "train.jsonl"),
        "valid_sha256": sha256(DATA_DIR / "valid.jsonl"),
        "seed": SEED,
        "beta": BETA,
        "learning_rate": LR,
        "max_tokens": MAX_TOKENS,
        "eval_every": EVAL_EVERY,
    }

    start_step = 0

    if args.resume is not None:
        checkpoint = Path(args.resume)
        metadata = json.loads(
            (checkpoint / "metadata.json").read_text()
        )

        for key, value in run_info.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"Resume configuration mismatch: {key}"
                )

        start_step = int(metadata["step"])

        if args.steps <= start_step:
            raise ValueError(
                "--steps must exceed checkpoint step"
            )

    print("Loading SFT-initialized policy...", flush=True)

    model, tokenizer = load(
        MODEL,
        adapter_path=str(SFT_DIR),
    )

    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])
    model.train()

    optimizer = optim.Adam(
        learning_rate=LR
    )

    if args.resume is not None:
        # Restore LoRA weights and Adam momentum.
        model.load_weights(
            str(checkpoint / "adapters.safetensors"),
            strict=False,
        )

        optimizer.state = tree_unflatten(
            mx.load(
                str(checkpoint / "optimizer.safetensors")
            )
        )

        mx.eval(model.parameters(), optimizer.state)

        print(
            f"RESUMED FROM STEP {start_step}",
            flush=True,
        )

    trainable = dict(
        tree_flatten(model.trainable_parameters())
    )

    print(
        "Trainable parameters:",
        sum(x.size for x in trainable.values()),
        flush=True,
    )

    memory_report("After model loading")

    best_file = output_dir / "best" / "best.json"
    best_loss = float("inf")

    if best_file.exists():
        best = json.loads(best_file.read_text())
        best_loss = float(best["val_loss"])

    loss_history = []

    try:
        for step in range(start_step, args.steps):
            idx = order[step]

            chosen, rejected = pair_inputs(
                tokenizer,
                train_rows[idx],
            )

            reference_margin = mx.array(
                float(train_refs[idx]["ref_chosen_logp"])
                - float(train_refs[idx]["ref_rejected_logp"]),
                dtype=mx.float32,
            )

            def loss_fn():
                chosen_lp = answer_logprob(
                    model, chosen
                )
                rejected_lp = answer_logprob(
                    model, rejected
                )

                score = BETA * (
                    chosen_lp
                    - rejected_lp
                    - reference_margin
                )

                return mx.logaddexp(
                    mx.array(0.0, dtype=mx.float32),
                    -score,
                )
            # Reset the peak counter for this training step.
            mx.reset_peak_memory()

            loss, grads = nn.value_and_grad(
                model, loss_fn
            )()

            loss, grads = nn.value_and_grad(
                model, loss_fn
            )()

            mx.eval(loss, grads)
            value = float(loss)

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite loss at step {step + 1}"
                )

            optimizer.update(model, grads)

            mx.eval(
                model.parameters(),
                optimizer.state,
            )

            loss_history.append(value)

            del loss, grads, chosen, rejected
            del reference_margin
            clean_memory()

            current_step = step + 1
            step_peak = mx.get_peak_memory() / GB

            print(
                f"step={current_step} "
                f"individual_peak={step_peak:.2f}GiB",
                flush=True,
            )

            if (
                current_step % 10 == 0
                or current_step == start_step + 1
            ):
                recent = loss_history[-10:]
                mean_loss = sum(recent) / len(recent)

                peak = memory_report(
                    f"step={current_step} "
                    f"train_loss_10={mean_loss:.6f}"
                )

                if peak >= PEAK_STOP / GB:
                    print(
                        "Memory safety threshold reached.",
                        flush=True,
                    )

                    save_checkpoint(
                        model,
                        optimizer,
                        output_dir,
                        current_step,
                        {"emergency": True},
                        run_info,
                    )

                    raise RuntimeError(
                        "Stopped to protect memory"
                    )

            if current_step % EVAL_EVERY == 0:
                print(
                    f"\nVALIDATING STEP {current_step}",
                    flush=True,
                )

                metrics = validate(
                    model,
                    tokenizer,
                    valid_rows,
                    valid_refs,
                )

                print(
                    json.dumps(
                        metrics,
                        indent=2,
                    ),
                    flush=True,
                )

                checkpoint_path = save_checkpoint(
                    model,
                    optimizer,
                    output_dir,
                    current_step,
                    metrics,
                    run_info,
                )

                if metrics["val_loss"] < best_loss:
                    best_loss = metrics["val_loss"]

                    update_best(
                        output_dir,
                        checkpoint_path,
                        current_step,
                        metrics,
                    )

                loss_history.clear()
                clean_memory()

        print(
            f"\nPASS: training reached step {args.steps}",
            flush=True,
        )
        print(
            f"Best validation DPO loss: {best_loss:.6f}",
            flush=True,
        )

    except Exception:
        print(
            "\nTraining interrupted. "
            "Previously completed checkpoints are preserved.",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
