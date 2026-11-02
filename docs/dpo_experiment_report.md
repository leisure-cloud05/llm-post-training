# Direct Preference Optimization: Training and Evaluation

## 1. Overview

This experiment investigates Direct Preference Optimization (DPO) as a post-training method for the 4-bit quantized `Qwen2.5-1.5B-Instruct` model. DPO was implemented directly with MLX and MLX-LM, without using a third-party DPO training framework. The goals were to verify a custom DPO implementation, train a LoRA policy on memory-constrained local hardware, evaluate preference learning on held-out data, and assess whether improvements in preference-pair scores transfer to free-form generation.

All experiments ran locally on a 24 GB Apple Silicon MacBook Air.

## 2. Models and Training Pipeline

The base model was `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. A general instruction-following SFT adapter was trained on chosen responses from the preference dataset, then used both to initialize the DPO policy and as the frozen DPO reference. The SFT stage used LoRA rank 8 over 16 layers (5,275,648 trainable parameters), learning rate `1e-5`, micro-batch size 1, gradient accumulation 4, and maximum sequence length 1,024. Training ran for 8,000 steps, with the Step 6,500 adapter selected based on validation-loss behavior (reported validation loss approximately 0.962).

This general-response SFT adapter is distinct from the earlier customer-support classification LoRA experiments in the same project.

## 3. Preference Data

Preference pairs were drawn from `argilla/ultrafeedback-binarized-preferences-cleaned`. The preparation pipeline processed a fixed source subset, required a chosen/rejected rating gap of at least 1, filtered for fully formatted sequences of at most 1,024 tokens, deduplicated normalized prompts, and created prompt-disjoint splits using seed 42.

| Split | Preference pairs | Use |
| --- | ---: | --- |
| Train | 4,000 | Policy optimization |
| Validation | 400 | Checkpoint selection |
| Test | 400 | Final held-out evaluation |
| **Total** | **4,800** | |

Dataset SHA-256 hashes were retained to detect changes or stale caches. The test split was not used during optimization or checkpoint selection.

## 4. Custom DPO Implementation

For a prompt \(x\), preferred answer \(y_w\), rejected answer \(y_l\), policy \(\pi_\theta\), and frozen SFT reference \(\pi_{\mathrm{ref}}\), the loss is

\[
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\!\left(\beta\left[\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x)+\log\pi_{\mathrm{ref}}(y_l\mid x)\right]\right).
\]

The implementation used \(\beta=0.1\), applied the same Qwen chat template to policy and reference, and summed token log-probabilities **only over assistant completion tokens**. Prompt tokens were masked out. Sequence log-probabilities were not length-normalized, consistent with the standard sequence-level DPO objective.

To reduce unified-memory requirements, the frozen reference policy's chosen and rejected log-probabilities were computed ahead of time and cached on disk, keyed by dataset row index. Reference caching ran in independent Python subprocesses of 20 pairs, with SHA-256 and index-consistency checks. During training, only the trainable LoRA policy was loaded.

### Implementation checks

- Before updates, policy and reference logits matched, and DPO loss equaled \(\log 2\approx 0.693147\).
- Gradients flowed to the intended LoRA tensors.
- Ten-step and twenty-step smoke tests completed.
- All 224 LoRA weight tensors changed after ten optimizer steps; the maximum absolute observed weight change was approximately \(5.50\times10^{-5}\).
- Validation scoring and reference-cache consistency checks passed.

## 5. DPO Training and Memory Management

| Setting | Value |
| --- | --- |
| Initialization | General-response SFT adapter |
| Optimizer | Adam |
| Trainable parameters | LoRA only, 5,275,648 |
| Learning rate | `1e-6` |
| DPO beta | `0.1` |
| Micro-batch size | 1 |
| Gradient accumulation | 1 |
| Maximum sequence length | 1,024 |
| Random seed | 42 |
| Total optimizer updates | 1,000 |
| Validation and checkpoint interval | 100 steps |
| Selection criterion | Lowest validation DPO loss |

The MLX allocator budget was set to 10 GiB, with a cache limit of 256 MiB and a safety-stop threshold of 9.5 GiB. Intermediate tensors were released and caches cleared after updates. Checkpoints stored LoRA weights, Adam optimizer state, and run metadata to support resuming training.

One attempt was automatically stopped at a reported cumulative MLX peak of 9.68 GiB, and the run subsequently resumed from a saved Step 120 checkpoint. Training ultimately completed all 1,000 updates. These allocator readings are **not** measures of total macOS memory consumption, so the configured MLX budget does not guarantee a system-wide hard limit.

## 6. Validation Results

All checkpoints were evaluated on the same 400 validation preference pairs.

| Step | Validation DPO loss ↓ | Reward accuracy ↑ | Mean reward margin |
| ---: | ---: | ---: | ---: |
| 100 | 0.565913 | 79.00% | 0.350431 |
| 200 | 0.522939 | 76.25% | 0.509688 |
| 300 | 0.506606 | 77.75% | 0.607040 |
| 400 | 0.497127 | 78.50% | 0.699177 |
| 500 | 0.492498 | 77.25% | 0.757031 |
| 600 | 0.487429 | 78.00% | 0.819130 |
| 700 | 0.488609 | 78.00% | 0.824191 |
| 800 | 0.476294 | 79.50% | 0.887904 |
| 900 | 0.468790 | 79.75% | 0.931102 |
| **1,000** | **0.466386** | **79.75%** | **0.971659** |

Validation DPO loss generally decreased across training, with a small increase at Step 700. Mean reward margin increased steadily, while reward accuracy fluctuated between approximately 76% and 80%. Step 1,000 was selected using the predetermined minimum-validation-loss criterion.

## 7. Independent Test Evaluation

The selected Step 1,000 adapter was evaluated once on 400 held-out preference pairs, using cached SFT reference log-probabilities and identical completion scoring.

| Metric | Test result |
| --- | ---: |
| Preference pairs | 400 |
| Same-policy/reference initialization loss | 0.693147 |
| **DPO test loss** | **0.426000** |
| **DPO reward accuracy** | **81.50%** |
| Mean reward margin | 1.080624 |
| Tie rate | 0.00% |

The test DPO loss was approximately **38.5% lower** than the \(\log 2\) initialization baseline. Reward accuracy reports the fraction of supplied preference pairs on which the DPO policy increased the chosen-versus-rejected log-probability difference **relative to the frozen SFT reference**. It must not be interpreted as the percentage of correct free-form generations, nor is \(\log 2\) a standalone SFT generation-quality score.

## 8. Free-Form Generation Evaluation

### 8.1 Initial 256-token comparison

A separate qualitative evaluation used 20 sampled test prompts, identical chat formatting and greedy decoding, and a maximum generation length of 256 tokens.

| Metric | SFT | DPO |
| --- | ---: | ---: |
| Sampled prompts | 20 | 20 |
| Mean output tokens | 176.35 | 205.75 |
| Outputs reaching 256 tokens | 9 | 9 |

DPO outputs were approximately 16.7% longer on average in this sample, but both systems reached the generation cap in 9 of 20 cases. The cap limits conclusions about natural completion length and answer completeness.

### 8.2 Exploratory 512-token follow-up

Eleven prompts for which at least one model reached the original cap were regenerated using a 512-token maximum, again with greedy decoding and unchanged adapters.

| Metric | SFT | DPO |
| --- | ---: | ---: |
| Selected prompts | 11 | 11 |
| Mean output tokens | 403.00 | 405.91 |
| Outputs reaching 512 tokens | 5 | 4 |

The mean output lengths were much closer in this **truncation-selected subset**. This does not overturn the initial 20-prompt length observation, but it shows that conclusions about verbosity depend on the generation budget and sample selection. Multiple responses still hit the expanded cap.

### 8.3 Qualitative case findings

| Example | Observed comparison |
| --- | --- |
| Excel loan payment | DPO produced a more appropriate `PMT` formula than SFT, but still included inaccurate supporting explanations. |
| Human toes vs. fish toes | SFT gave a concise, broadly correct response; DPO introduced incorrect biological claims and excess elaboration. |
| Python sorting and error handling | Neither model fulfilled the complete ordering and input-validation requirements, even when given more tokens. |
| Scala homophone correction | Both produced unreliable technical content; the longer generations included malformed or implausible code. |
| Scala/Akka parallelism | Both had technical weaknesses; DPO repeated import lines extensively in the 512-token follow-up. |
| Corporate zero-waste presentation | SFT provided more concrete employee-engagement practices, while DPO tended toward a broader, more outline-like slide structure. |
| Breastfeeding advice | DPO included inappropriate advice about mixing expressed milk with water, demonstrating a concrete safety-related generation failure. |

These are **qualitative observations**, not blinded human-preference scores. The case review demonstrates task-dependent outcomes, including some improvements, unchanged weaknesses, and clear regressions.

## 9. Interpretation and Limitations

The held-out preference metrics demonstrate that the custom DPO optimizer learned to increase the model's relative preference for supplied chosen answers. However, **preference-pair scoring and free-form generation assess different behaviors**. A favorable relative log-probability margin does not establish factual correctness, executable code, safety, or reliable instruction adherence during generation.

The study has several limitations: one main DPO configuration was tested; free-form generation was manually reviewed on only 20 prompts; 11 follow-up prompts were selected because of earlier truncation; no blinded multi-rater human preference study was performed; and sequence-level log-probabilities may reflect response-length effects. Findings are specific to this model, dataset, and local training setup.

## 10. Conclusion

A custom MLX/MLX-LM DPO implementation successfully trained a LoRA policy on constrained local hardware and achieved **0.4260 held-out DPO loss and 81.5% reward accuracy** on 400 independent preference pairs. Reference caching, per-process evaluation, memory monitoring, and resumable checkpoints enabled the 1,000-step training run.

Nevertheless, qualitative generations did **not** show consistent improvements in factual accuracy, code correctness, concise communication, or instruction following; several examples exhibited repetition and potentially unsafe content. The principal finding is therefore that **stronger relative preference alignment does not automatically translate into higher-quality free-form responses**.

Potential follow-up work includes blinded response-level evaluation, targeted code correctness tests, controlled output-length analysis, and additional training configurations. None of these follow-ups were performed in the present experiment.
