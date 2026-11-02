import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load


def assistant_response(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        messages = [m for m in value if isinstance(m, dict) and m.get("role") == "assistant"]
        if messages and isinstance(messages[-1].get("content"), str):
            return messages[-1]["content"].strip()
    raise ValueError("No assistant response found")


def make_pair(tokenizer, prompt, answer, max_tokens):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": answer}],
        tokenize=True, add_generation_prompt=False, return_dict=False,
    )
    if not isinstance(full, list) or not full or not isinstance(full[0], int):
        raise ValueError("Expected a flat token list from the tokenizer")
    if full[:len(prefix)] != prefix:
        raise ValueError("Prompt prefix differs in full chat template: check tokenizer formatting")
    if len(full) > max_tokens:
        raise ValueError(f"Sequence length {len(full)} exceeds max_tokens={max_tokens}")
    if len(full) <= len(prefix):
        raise ValueError("Empty answer after template formatting")
    # In a causal LM, target token i is predicted from input token i-1.
    # Thus answer targets begin at index len(prefix) - 1 in shifted logits.
    return mx.array(full[:-1], dtype=mx.int32)[None, :], mx.array(full[1:], dtype=mx.int32), len(prefix) - 1


def answer_logprob(model, encoded):
    inputs, targets, start = encoded
    logits = model(inputs)[0].astype(mx.float32)
    # Select only assistant target positions (including assistant EOS, excluding prompt).
    logits = logits[start:]
    targets = targets[start:]
    chosen_logits = mx.take_along_axis(logits, targets[:, None], axis=-1)[:, 0]
    token_logprobs = chosen_logits - mx.logsumexp(logits, axis=-1)
    return mx.sum(token_logprobs)


def dpo_loss(policy, chosen, rejected, ref_chosen, ref_rejected, beta):
    policy_margin = answer_logprob(policy, chosen) - answer_logprob(policy, rejected)
    reference_margin = ref_chosen - ref_rejected
    score = beta * (policy_margin - reference_margin)
    # Stable -log(sigmoid(score))
    return mx.logaddexp(mx.array(0.0, dtype=mx.float32), -score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--adapter", default="models/dpo_sft/best")
    parser.add_argument("--data-file", default="data/dpo/valid.jsonl")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    with Path(args.data_file).open(encoding="utf-8") as stream:
        row = json.loads(next(line for line in stream if line.strip()))
    prompt = row["prompt"].strip()
    chosen_answer = assistant_response(row["chosen"])
    rejected_answer = assistant_response(row["rejected"])
    if not prompt or not chosen_answer or not rejected_answer:
        raise ValueError("Empty prompt or response")

    print("Loading Policy and Reference from the SAME SFT checkpoint...")
    policy, tokenizer = load(args.model, adapter_path=args.adapter)
    reference, _ = load(args.model, adapter_path=args.adapter)
    policy.freeze()
    policy.unfreeze(keys=["lora_a", "lora_b"])
    reference.freeze()
    policy.eval()
    reference.eval()
    trainable = dict(tree_flatten(policy.trainable_parameters()))
    assert trainable and not tree_flatten(reference.trainable_parameters())

    chosen = make_pair(tokenizer, prompt, chosen_answer, args.max_tokens)
    rejected = make_pair(tokenizer, prompt, rejected_answer, args.max_tokens)
    print("Sequence lengths (chosen/rejected):", chosen[0].shape[1]+1, rejected[0].shape[1]+1)
    print("Answer target counts (chosen/rejected):", chosen[1].shape[0]-chosen[2], rejected[1].shape[0]-rejected[2])

    ref_c = answer_logprob(reference, chosen)
    ref_r = answer_logprob(reference, rejected)
    pol_c = answer_logprob(policy, chosen)
    pol_r = answer_logprob(policy, rejected)
    mx.eval(ref_c, ref_r, pol_c, pol_r)
    print("Reference chosen/rejected logp:", float(ref_c), float(ref_r))
    print("Policy chosen/rejected logp:   ", float(pol_c), float(pol_r))
    print("Max logp discrepancy:", max(abs(float(ref_c-pol_c)), abs(float(ref_r-pol_r))))
    assert max(abs(float(ref_c-pol_c)), abs(float(ref_r-pol_r))) < 1e-3

    loss_fn = lambda: dpo_loss(policy, chosen, rejected, ref_c, ref_r, args.beta)
    loss, grads = nn.value_and_grad(policy, loss_fn)()
    flat_grads = tree_flatten(grads)
    mx.eval(loss, *[g for _, g in flat_grads])
    print("Initial DPO loss:", float(loss), "expected:", math.log(2))
    assert abs(float(loss) - math.log(2)) < 1e-3
    grad_sum = sum(float(mx.sum(mx.abs(g.astype(mx.float32)))) for _, g in flat_grads)
    print("Gradient tensors:", len(flat_grads))
    print("Gradient abs sum:", grad_sum)
    assert math.isfinite(grad_sum) and grad_sum > 0
    print("PASS: conditional logp, reference alignment, DPO loss, and LoRA gradients")
    print("No optimizer step performed; model files unchanged.")

if __name__ == "__main__":
    main()
