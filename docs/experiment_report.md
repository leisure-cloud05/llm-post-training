# Experimental Report: Parameter-Efficient Instruction Fine-Tuning and Class-Aware Evaluation

## 1. Research Objective

This project studies task-specific supervised fine-tuning (SFT) of a 4-bit quantized Qwen2.5-1.5B-Instruct model for structured classification of synthetic English customer-support tickets. Given a ticket subject and body, the model must produce a JSON object with exactly three fields: `category`, `sentiment`, and `urgency`.

The investigation addresses four questions:

1. Does task-specific LoRA fine-tuning improve classification and strict JSON compliance over an instruction-tuned base model?
2. Can class-aware resampling improve recognition of the rare `critical` urgency class?
3. What trade-offs arise in other urgency classes, especially `high`?
4. How do these methods behave on event-associated records from the same synthetic source?

## 2. Dataset and Protocol

**Source:** [Polaris Support Tickets v2](https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2), a synthetic SaaS support-ticket dataset (23,994 source records). The model inputs are `subject` and `body`; the output targets map `topic` to `category`, `sentiment` to `sentiment`, and `priority` to `urgency`.

The initial experiment excluded 6,000 event-linked records and selected 5,000 cleaned ordinary tickets, split into **4,000 training**, **500 validation**, and **500 standard test** examples. A normalized subject-and-body audit found zero exact overlaps between these three splits, and all 5,000 inputs were matched to records in the original source. Exact matching does not rule out template-level or semantic duplication.

Additional analyses used:

- **High/Critical diagnostic development:** 25 `high` and 25 `critical` tickets.
- **High/Critical diagnostic test:** 25 `high` and 25 `critical` tickets, held aside from checkpoint selection.
- **Event-associated test:** 520 tickets, sampled as 40 per event from 13 event groups. These records were excluded from ordinary-ticket training.

Diagnostic candidates were selected from unused ordinary tickets after normalized exact-overlap screening and character 3–5-gram TF-IDF cosine-similarity screening (selection threshold: 0.70). Maximum observed cosine similarity to the existing 5,000 examples among selected diagnostics was 0.6932; maximum within the two selected diagnostic splits was 0.606. These heuristic thresholds do not establish semantic independence. The event-associated set was checked for exact overlaps but not comprehensively screened for near-duplicates.

The diagnostic sets are intentionally class-balanced; their aggregate accuracy must not be interpreted as prevalence-weighted production accuracy. The event-associated set is group-balanced, not representative of the original event-group frequency distribution.

## 3. Model and Training Configuration

| Item | Configuration |
|---|---|
| Backbone | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Framework | MLX-LM on Apple Silicon (24 GB unified memory) |
| Adaptation | LoRA on a 4-bit quantized backbone (QLoRA-style configuration) |
| Training objective | Response-only supervised fine-tuning (`--mask-prompt`) |
| LoRA rank | 8 |
| Adapted layers | 16 |
| Trainable parameters | 5.276 million / 1,543.714 million (0.342%) |
| Batch size | 2 |
| Learning rate | `1e-5` |
| Iterations | 500 |
| Checkpoint rule | Minimum validation loss among evaluated checkpoints |

The original LoRA experiment and the class-aware LoRA experiment both selected the **iteration 400** checkpoint by validation loss. Validation losses at iteration 400 were 0.019 and 0.015, respectively. The second training run started from the same base model rather than continuing from the first adapter. The first run relied on the framework's default seed; the second explicitly used seed 0.

**Class-aware intervention:** The training corpus remained 4,000 occurrences, but `critical` exposure increased from **49/4,000 (1.225%)** to **400/4,000 (10%)** by sampling the same 49 unique critical examples with replacement. The resampled corpus contained 3,649 distinct source examples. To maintain 4,000 occurrences, some non-critical examples were omitted. Consequently, this experiment changes both class exposure and non-critical example diversity; it does not isolate a pure class-weighting effect.

## 4. Evaluation Methods

The primary evaluation pipeline scores a response as label-compliant only if its raw output parses as a JSON object with exactly the required three keys and valid labels. A noncompliant response counts as incorrect on every field. Reported metrics include strict JSON validity, label compliance, three-field exact match, per-field accuracy, per-class precision/recall/F1, and macro-F1.

An additional *extracted JSON* analysis allows a complete Markdown-fenced JSON object to be parsed. This is reported separately rather than retroactively relaxing strict-output requirements for only one method.

The four compared methods are:

1. **Base:** original instruction-tuned model, original task prompt, no adapter.
2. **Strong Prompt:** same base model, manually expanded label definitions and decision rules, no adapter.
3. **Original QLoRA:** original task prompt and standard-distribution SFT adapter.
4. **Class-aware QLoRA:** original task prompt and critical-oversampled SFT adapter.

The strong prompt was a single manually designed zero-shot prompt, not a comprehensive prompt-search or few-shot baseline. Its definitions may not match all synthetic dataset labeling conventions.

## 5. Standard Test Results (n = 500)

| Metric | Base | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|---:|
| Strict JSON validity | 99.6% | 90.4% | 100.0% | 100.0% |
| Three-field exact match | 20.2% | 13.4% | 76.2% | **79.8%** |
| Category accuracy | 72.0% | 76.6% | 97.8% | **99.0%** |
| Sentiment accuracy | 45.8% | 39.6% | 86.0% | **86.8%** |
| Urgency accuracy | 39.6% | 25.0% | 90.2% | **92.8%** |

For the strong prompt, extracted JSON validity was 100.0%, and extracted exact match was **14.2%** (versus 13.4% under strict parsing). Thus, Markdown fences explained some format failures but only a small portion of its classification errors.

### Urgency-specific results

| Metric | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|
| Urgency macro-F1 | 0.635 | **0.838** |
| Critical precision | 0.0% | 62.5% |
| Critical recall | 0/6 (0.0%) | **5/6 (83.3%)** |
| High recall | 78.6% | 61.9% |
| Medium recall | 83.2% | 100.0% |
| Low recall | 98.5% | 93.0% |

The standard test set contains only six critical tickets. Its critical-class estimate is therefore highly uncertain. Furthermore, the standard test set informed the earlier error analysis and subsequent research question, so it is a historical benchmark rather than a pristine, untouched final holdout.

## 6. High/Critical Diagnostic Evaluation

### Diagnostic development (n = 50)

| Metric | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|
| Three-field exact match | 30.0% | 56.0% |
| Urgency accuracy | 44.0% | 76.0% |
| Critical recall | 2/25 (8.0%) | **21/25 (84.0%)** |
| High recall | 20/25 (80.0%) | 17/25 (68.0%) |
| High/Critical macro-F1 | 0.373 | **0.843** |

### Diagnostic test (n = 50)

| Metric | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|
| Three-field exact match | 16.0% | 28.0% | **54.0%** |
| Urgency accuracy | 52.0% | 48.0% | **70.0%** |
| Critical recall | 1/25 (4.0%) | 1/25 (4.0%) | **22/25 (88.0%)** |
| High recall | 25/25 (100.0%) | 23/25 (92.0%) | 13/25 (52.0%) |
| High/Critical macro-F1 | 0.376 | 0.358 | **0.740** |

Here High/Critical macro-F1 averages F1 over **only the two represented classes**. The diagnostic model predicts among all four urgency labels; `medium` and `low` predictions remain errors but are not added as zero-support classes to the two-class macro-average.

On the diagnostic test, the original adapter classified 24/25 critical tickets as `high`, whereas the class-aware adapter correctly classified 22/25 as `critical`. However, the class-aware adapter classified only 13/25 high tickets correctly, including five high-to-critical errors. This illustrates a substantial class-specific recall trade-off. The balanced diagnostic sample is too small to establish production-level precision or recall.

## 7. Event-Associated Distribution-Shift Evaluation (n = 520)

The event-associated evaluation spans 13 named events, with 40 tickets sampled from each. Only two categories occur (`connectors` and `dashboards`), and the urgency distribution differs substantially from ordinary tickets. The selected event set contains 71 critical, 173 high, 76 medium, and 200 low tickets.

### Overall results

| Metric | Base | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|---:|
| Strict JSON validity | 99.6% | 95.2% | 100.0% | 100.0% |
| Label compliance | 99.6% | 95.0% | 100.0% | 100.0% |
| Three-field exact match | 26.9% | 28.5% | 56.7% | **59.8%** |
| Category accuracy | 75.8% | 86.5% | 95.2% | **99.0%** |
| Sentiment accuracy | 44.8% | 40.0% | 72.9% | **78.3%** |
| Urgency accuracy | 49.8% | 44.0% | **80.8%** | 75.8% |

### Urgency class-level results

| Metric | Base | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|---:|
| Urgency macro-F1 | 0.305 | 0.290 | 0.649 | **0.736** |
| Critical recall | 0.0% | 0.0% | 1/71 (1.4%) | **63/71 (88.7%)** |
| High recall | 98.8% | 100.0% | **98.3%** | 46.2% |
| Medium recall | 2.6% | 14.5% | 82.9% | **98.7%** |
| Low recall | 43.0% | 22.5% | **93.0%** | 88.0% |

The class-aware adapter improves event-set exact match by approximately 3.1 percentage points and urgency macro-F1 by 0.087, but reduces urgency accuracy by 5.0 percentage points. Its high recall falls from 170/173 (98.3%) to 80/173 (46.2%); among the 173 high tickets, it predicts `critical` for 30 and `medium` for 63. Thus, minority-class improvement is accompanied by consequential errors on an adjacent class.

### Event-level patterns

The event-level analysis shows heterogeneous effects, rather than a uniform advantage for either adapter:

| Event | Original urgency accuracy | Class-aware urgency accuracy | Change |
|---|---:|---:|---:|
| `outage_dashboards_2026q1` | 52.5% | 80.0% | +27.5 pp |
| `outage_dashboards_2025q1` | 50.0% | 67.5% | +17.5 pp |
| `outage_ga4_sync_2025q3` | 90.0% | 70.0% | -20.0 pp |
| `outage_hubspot_sync_2024q4` | 97.5% | 60.0% | -37.5 pp |

All four examples contain 40 tickets. The largest observed urgency declines occurred in event groups with **zero critical support**, suggesting that improved detection of critical tickets does not compensate for increased mistakes on other priority classes within those groups. Event subgroup results are exploratory and susceptible to small-sample variation. Event type is inferred from source event identifiers rather than independently validated annotations.

**Interpretation caution:** Comparison of standard-test and event-associated accuracy mixes genuine changes in scenario wording with large changes in label distributions. This is a *within-source, event-associated distribution-shift evaluation*, not proof of cross-domain or real-world robustness.

## 8. Main Findings

1. **Task-specific SFT is effective in the evaluated setting.** The original QLoRA adapter improves standard-test exact match from 20.2% to 76.2% over the base instruction model.
2. **A longer zero-shot prompt is not a substitute for task adaptation here.** The manually designed strong prompt achieves 13.4% strict exact match (14.2% after fenced-JSON extraction) on the standard test. This finding applies only to the tested prompt, not to all prompt-engineering methods.
3. **Class-aware resampling changes minority-class behavior substantially.** Critical recall rises from 0% to 83.3% on the standard test, 4% to 88% on diagnostic test, and 1.4% to 88.7% on event-associated test.
4. **The improvement has meaningful trade-offs.** High recall drops from 92% to 52% on diagnostic test and from 98.3% to 46.2% on event-associated test. Event-associated urgency accuracy decreases even as urgency macro-F1 increases.
5. **Distribution matters.** Aggregate accuracy alone conceals failures on rare classes and can reverse the apparent comparison between training strategies when class proportions change.

## 9. Limitations and Validity Threats

- **Synthetic and potentially templated data:** All results derive from one synthetic source; neither ecological validity nor real-world transfer has been established.
- **No exhaustive near-duplicate guarantee:** Exact matches and character n-gram similarity checks do not eliminate semantic or template-level overlap, particularly for the event-associated set.
- **Small critical-class evaluation subsets:** The standard test includes only six critical cases; each diagnostic split has 25.
- **Development-driven inquiry:** The original test's errors informed the research question. The diagnostic test was reserved from checkpoint selection, but candidates and task design were developed after inspecting other test results.
- **Single training run per strategy:** Reported improvements have not been replicated across several seeds and should not be interpreted as statistically robust causal estimates.
- **Intervention confounding:** Critical oversampling also reduced diversity among non-critical training examples.
- **Prompt baseline coverage:** The strong-prompt experiment tested one handcrafted prompt, without few-shot examples or automatic prompt optimization.
- **Quantization terminology:** The work uses LoRA on a 4-bit model via MLX-LM; it should not be represented as a verified reproduction of every algorithmic detail in the original QLoRA paper.
- **No general-capability retention evaluation:** Changes to the model's general instruction-following abilities outside the ticket task were not measured.

## 10. Reproducibility and Artifacts

Repository scripts include dataset inspection and preparation, baseline/model evaluation, class-aware data preparation, exact and near-duplicate audits, diagnostic-set construction, event-set construction, error analysis, and event-level analysis. The source datasets are retrievable from Hugging Face; local JSONL records and model adapter weights are intentionally excluded from the Git repository.

Representative training command for the standard run:

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data data --mask-prompt \
  --batch-size 2 --num-layers 16 \
  --learning-rate 1e-5 --iters 500 \
  --steps-per-report 10 --steps-per-eval 100 \
  --save-every 100 --adapter-path adapters
```

The class-aware run uses the same settings with `--data data_balanced --adapter-path adapters_balanced --seed 0`. Both analyses use the saved iteration-400 adapters, selected by validation loss. The evaluation script accepts `--adapter`, `--data-file`, `--output-name`, and `--prompt-mode` arguments. Exact training and file-generation commands should be maintained in a separate reproducibility guide and checked against the committed scripts.

## 11. Planned Extensions (Not Yet Implemented)

- Consolidate experiment metadata, generation settings, and summary tables for easier replication.
- Investigate SFT loss masking and LoRA parameterization in a standalone technical note.
- Where resources permit, repeat training with additional seeds and test alternative resampling strategies.
- Explore preference optimization (e.g., DPO) on a **separate preference-suitable customer-support response task**, using justified chosen/rejected pairs and an independent evaluation protocol. DPO is not part of the reported results.

## Dataset Attribution

Marinovich, V. (2026). *Polaris Support Tickets — Synthetic Dataset (v2).* [Dataset on Hugging Face](https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2). License reported on the dataset card: CC BY-SA 4.0.