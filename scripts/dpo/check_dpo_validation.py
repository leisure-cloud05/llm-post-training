import gc
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from check_dpo_loss import assistant_response, make_pair, answer_logprob

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SFT = "models/dpo_sft/best"
DPO = "models/dpo/best"
BETA = 0.1
LIMIT = 20

mx.set_cache_limit(256 * 1024 * 1024)

rows = [
    json.loads(line)
    for line in Path("data/dpo/valid.jsonl").open(encoding="utf-8")
    if line.strip()
][:LIMIT]

refs = []
for path in sorted(Path("cache/dpo/ref_cache/valid").glob("batch_*.jsonl")):
    with path.open(encoding="utf-8") as f:
        refs.extend(json.loads(line) for line in f if line.strip())

assert len(refs) >= LIMIT
assert all(refs[i]["index"] == i for i in range(LIMIT))

def evaluate(adapter_path, name):
    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    model.freeze()
    model.eval()

    losses = []
    margins = []

    for i, row in enumerate(rows):
        prompt = row["prompt"].strip()

        chosen = make_pair(
            tokenizer,
            prompt,
            assistant_response(row["chosen"]),
            1024,
        )
        rejected = make_pair(
            tokenizer,
            prompt,
            assistant_response(row["rejected"]),
            1024,
        )

        chosen_lp = answer_logprob(model, chosen)
        rejected_lp = answer_logprob(model, rejected)
        mx.eval(chosen_lp, rejected_lp)

        policy_margin = float(chosen_lp - rejected_lp)
        ref_margin = (
            float(refs[i]["ref_chosen_logp"])
            - float(refs[i]["ref_rejected_logp"])
        )

        reward_margin = BETA * (policy_margin - ref_margin)
        loss = math.log1p(math.exp(-abs(reward_margin))) + max(0.0, -reward_margin)

        losses.append(loss)
        margins.append(reward_margin)

        del chosen_lp, rejected_lp, chosen, rejected
        gc.collect()
        mx.clear_cache()

    print(f"\n{name}")
    print(f"Samples: {len(losses)}")
    print(f"Mean DPO loss: {sum(losses)/len(losses):.6f}")
    print(
        f"Reward accuracy: "
        f"{sum(m > 0 for m in margins)/len(margins):.2%}"
    )
    print(f"Mean reward margin: {sum(margins)/len(margins):.6f}")

    del model
    gc.collect()
    mx.clear_cache()

print("Evaluating SFT baseline...")
evaluate(SFT, "SFT")

print("\nEvaluating final DPO...")
evaluate(DPO, "DPO Final")

print("\nPASS: validation evaluation completed.")
