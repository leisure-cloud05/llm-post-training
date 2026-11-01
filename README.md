# LLM Instruction Fine-Tuning

**Parameter-efficient instruction fine-tuning and class-aware optimization of Qwen2.5 for structured customer support ticket classification on Apple Silicon.**

## Overview

This project investigates supervised fine-tuning (SFT) with LoRA on a 4-bit quantized Qwen2.5-1.5B-Instruct model using MLX-LM.

The task is to predict three attributes of English customer support tickets in a structured JSON format:

- **Category:** Primary support topic
- **Sentiment:** Customer sentiment
- **Urgency:** Ticket priority

Example output:

```json
{
  "category": "billing",
  "sentiment": "frustrated",
  "urgency": "high"
}
```

The project focuses on three research questions:

1. How much does task-specific SFT improve structured prediction compared with the original instruction-following model?
2. Can class-aware training improve recognition of underrepresented critical-priority tickets?
3. What trade-offs arise between minority-class recall and performance on other priority classes?

## Experimental Setup

| Component | Configuration |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct (4-bit) |
| Model repository | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Training framework | MLX-LM |
| Training method | Response-only SFT with LoRA on a quantized model |
| Hardware | Apple Silicon MacBook Air, 24GB unified memory |
| Dataset | Polaris Support Tickets v2 |
| Language | English |
| Output format | Three-field JSON |
| LoRA rank | 8 |
| LoRA layers | 16 |
| Trainable parameters | 5.276M (0.342% of model parameters) |
| Learning rate | 1e-5 |
| Batch size | 2 |
| Training iterations | 500 |
| Checkpoint selection | Lowest validation loss among evaluated checkpoints |

Both fine-tuning experiments used the same model architecture and primary training hyperparameters. The main experimental change was the training-data sampling strategy.

## Dataset

**Source:** [Polaris Support Tickets v2](https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2)

The source dataset contains 23,994 synthetic customer support tickets for a fictional SaaS platform.

Input fields:

- `subject`
- `body`

Target mappings:

| Original field | Output field |
|---|---|
| `topic` | `category` |
| `sentiment` | `sentiment` |
| `priority` | `urgency` |

Event-linked records were excluded from the initial in-distribution experiment. Eligible ordinary tickets were cleaned, checked for exact duplicates, and sampled using a fixed seed.

The resulting experimental splits contain:

| Split | Samples |
|---|---:|
| Train | 4,000 |
| Validation | 500 |
| Standard test | 500 |

The standard train, validation, and test splits contain no exact cross-split duplicates under the normalized subject-and-body matching procedure.

Additional near-duplicate audits were conducted using character-level TF-IDF with cosine similarity. These heuristic checks do not guarantee the absence of semantically equivalent or template-derived examples.

Label definitions are documented in `docs/label_schema.md`.

## Methodology

### 1. Baseline Evaluation

The original Qwen2.5-1.5B-Instruct model was evaluated without task-specific fine-tuning.

The model received an instruction to classify each ticket into the permitted category, sentiment, and urgency labels and return a valid JSON object.

### 2. Supervised Fine-Tuning with LoRA

The quantized model was fine-tuned using instruction-response examples, with loss masking applied to the prompt portion.

The first experiment used 4,000 training examples with their naturally occurring class distribution.

Training ran for 500 iterations, and the checkpoint from iteration 400 was selected based on the lowest validation loss.

### 3. Error Analysis

Predictions from the original model and the fine-tuned model were compared using:

- Per-field accuracy and Macro-F1
- Class-level precision, recall, and F1
- Confusion matrices
- Exact-match error transitions

Although standard SFT substantially improved overall classification performance, the original fine-tuned model failed to correctly identify any of the six critical-priority tickets in the standard test set.

All six were misclassified as `high`.

### 4. Class-Aware Resampling

To investigate minority-class recognition, a second LoRA experiment increased the exposure of `critical` tickets in the training data.

| Training property | Original SFT | Class-aware SFT |
|---|---:|---:|
| Training records | 4,000 | 4,000 |
| Critical training occurrences | 49 | 400 |
| Critical exposure ratio | 1.23% | 10.0% |
| Unique training examples | 4,000 | 3,649 |
| Unique critical examples | 49 | 49 |

Critical examples were oversampled with replacement. Other priority classes were sampled without replacement to preserve the total training-set size.

The experiment did not introduce new critical examples. Some original training examples from other classes were omitted, so the intervention changed both the class exposure distribution and the diversity of non-critical examples.

The second model was trained from the same quantized base model, using the same primary hyperparameters and validation-loss checkpoint selection rule. Its iteration-400 checkpoint was also selected.

### 5. High/Critical Diagnostic Evaluation

A targeted diagnostic benchmark was constructed to examine the boundary between `high` and `critical`.

| Diagnostic split | High | Critical | Total |
|---|---:|---:|---:|
| Development | 25 | 25 | 50 |
| Test | 25 | 25 | 50 |

Candidate tickets were selected from previously unused, non-event source records, with normalized exact-match exclusions and character-level TF-IDF similarity screening.

The diagnostic sets were constructed before the second fine-tuning experiment. The development split was used for analysis, while the diagnostic test split was reserved for final comparison.

Because both diagnostic splits intentionally balance High and Critical, their aggregate accuracy does not estimate performance under the original ticket-priority distribution.

## Results

### Standard Test Set — 500 Samples

| Metric | Base model | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|
| JSON validity | 99.6% | 100.0% | 100.0% |
| Three-field exact match | 20.2% | 76.2% | **79.8%** |
| Category accuracy | 72.0% | 97.8% | **99.0%** |
| Sentiment accuracy | 45.8% | 86.0% | **86.8%** |
| Urgency accuracy | 39.6% | 90.2% | **92.8%** |

For urgency classification:

| Metric | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|
| Urgency Macro-F1 | 0.635 | **0.838** |
| Critical precision | 0.0% | 62.5% |
| Critical recall | 0.0% (0/6) | **83.3% (5/6)** |
| High recall | 78.6% | 61.9% |
| Medium recall | 83.2% | 100.0% |
| Low recall | 98.5% | 93.0% |

The class-aware model improved three-field exact-match accuracy from 76.2% to 79.8% on the standard test set. However, the critical-priority subset contains only six examples, making its class-specific metrics highly sensitive to individual predictions.

### High/Critical Diagnostic Test — 50 Samples

| Metric | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|
| JSON validity | 100.0% | 100.0% |
| Three-field exact match | 28.0% | **54.0%** |
| Category accuracy | 94.0% | **100.0%** |
| Sentiment accuracy | 68.0% | **74.0%** |
| Urgency accuracy | 48.0% | **70.0%** |
| High/Critical Macro-F1 | 0.358 | **0.740** |
| Critical precision | 100.0% (1/1) | 81.5% (22/27) |
| Critical recall | 4.0% (1/25) | **88.0% (22/25)** |
| High recall | 92.0% (23/25) | 52.0% (13/25) |

The original QLoRA model identified only one of the 25 critical tickets in the diagnostic test set, misclassifying 24 as high priority.

The class-aware model correctly identified 22 of the 25 critical tickets. However, it also incorrectly classified five high-priority tickets as critical and showed lower High recall.

**Key finding:** Increasing critical-class exposure substantially improved minority-class recall and High/Critical Macro-F1 in this experiment, while introducing a measurable trade-off in High recall.

These findings support the usefulness of class-aware training under the evaluated conditions, rather than showing that every class benefits equally.

## Evaluation Metrics

The evaluation pipeline reports:

- JSON validity
- Label-set compliance
- Per-field classification accuracy
- Three-field exact-match accuracy
- Per-class precision, recall, and F1
- Macro-F1
- Confusion matrices
- Error transitions between model variants

For the targeted High/Critical benchmark, Macro-F1 is calculated over the two represented priority classes, not over all four possible urgency labels.

## Project Status

**Completed**

- Apple Silicon / MLX-LM training environment
- Polaris v2 dataset inspection and preprocessing
- Reproducible train, validation, and test generation
- Original Qwen baseline evaluation
- Response-only LoRA fine-tuning
- Classification error analysis
- Exact-match and heuristic near-duplicate audits
- High/Critical diagnostic dataset construction
- Class-aware training-data resampling
- Second LoRA training experiment
- Standard and diagnostic test comparisons

**Planned**

- Strong-prompt baseline comparison
- Event-shift evaluation on event-linked source tickets
- Additional analysis of SFT loss masking and LoRA parameterization
- Consolidated experiment report and reproducibility documentation
- Exploration of preference optimization (DPO) on a suitable preference-based task

## Limitations

1. **Synthetic dataset:** Polaris v2 consists of generated support scenarios. Results may not transfer to real customer-service workloads.

2. **Limited minority-class samples:** The original training split contains only 49 unique critical examples. The standard test set contains six critical examples, while the diagnostic test set contains 25.

3. **Template and semantic overlap:** Exact duplicate checks and TF-IDF similarity screening reduce certain forms of overlap but cannot rule out semantically equivalent or template-derived records.

4. **Resampling trade-offs:** Oversampling repeated critical examples while subsampling other classes changes both class exposure and training-set diversity.

5. **Single-run comparisons:** Each fine-tuning strategy was evaluated from one training run. The observed differences have not been verified across multiple random seeds.

6. **Evaluation reuse:** The standard test set was inspected during error analysis and is best treated as a historical benchmark rather than an untouched final holdout.

7. **Targeted evaluation distribution:** The balanced High/Critical diagnostic benchmark evaluates minority-class discrimination, not overall performance under naturally occurring ticket frequencies.

## Dataset Attribution

Marinovich, V. (2026). *Polaris Support Tickets — Synthetic Dataset (v2).*

https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2

Dataset license: CC BY-SA 4.0, as stated on the Hugging Face dataset card.