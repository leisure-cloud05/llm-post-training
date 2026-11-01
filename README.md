# LLM Instruction Fine-Tuning

**Parameter-Efficient Fine-Tuning, Class-Aware Optimization, and Generalization Analysis of Qwen2.5 on Apple Silicon**

## Overview

This project investigates task-specific supervised fine-tuning (SFT) of a 4-bit quantized Qwen2.5-1.5B-Instruct model using LoRA and MLX-LM.

The task is to classify English customer support tickets and generate structured JSON predictions for three attributes:

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

Beyond basic fine-tuning, the project investigates class imbalance, minority-class recognition, prompt-based baselines, and performance under event-associated distribution shift.

### Research Questions

1. How much does task-specific SFT improve structured classification compared with an instruction-following base model?
2. Can class-aware resampling improve critical-priority recognition without sacrificing performance on other classes?
3. Can a stronger zero-shot prompt achieve similar gains without parameter updates?
4. How do fine-tuned models behave under event-associated distribution shift?

See [Experiment Report](docs/experiment_report.md) for the complete experimental methodology, results, and limitations.

## Experimental Setup

| Component | Configuration |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct (4-bit) |
| Model repository | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Framework | MLX-LM |
| Fine-tuning | Response-only SFT with LoRA on a quantized model |
| Hardware | Apple Silicon MacBook Air, 24GB unified memory |
| Dataset | Polaris Support Tickets v2 |
| Language | English |
| Output format | Three-field JSON |
| LoRA rank | 8 |
| LoRA layers | 16 |
| Trainable parameters | 5.276M (0.342%) |
| Learning rate | 1e-5 |
| Batch size | 2 |
| Training iterations | 500 |
| Checkpoint selection | Minimum validation loss |

Both SFT experiments selected the checkpoint at iteration 400 using validation loss.

## Dataset

**Source:** [Polaris Support Tickets v2](https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2)

The dataset contains 23,994 synthetic tickets from a fictional SaaS support platform.

The task uses `subject` and `body` as inputs, with the following target mappings:

| Source field | Prediction field |
|---|---|
| `topic` | `category` |
| `sentiment` | `sentiment` |
| `priority` | `urgency` |

### Data Splits

Event-linked tickets were excluded from the initial training and standard evaluation pipeline.

| Split | Samples |
|---|---:|
| Train | 4,000 |
| Validation | 500 |
| Standard Test | 500 |
| Diagnostic Dev | 50 |
| Diagnostic Test | 50 |
| Event-shift Test | 520 |

The diagnostic sets contain 25 High and 25 Critical tickets each. The event-shift set contains 40 tickets from each of 13 event groups.

Data preparation includes exact duplicate checking and heuristic character-level TF-IDF similarity screening. These checks do not guarantee the absence of semantic or template-level overlap.

See [Label Schema](docs/label_schema.md) for the target labels.

## Methods

### 1. Base Model and Strong Prompt

Two zero-shot baselines were evaluated without parameter updates:

- **Base:** Original classification prompt specifying the allowed labels and output format.
- **Strong Prompt:** Expanded label descriptions, severity criteria, and JSON output instructions.

The Strong Prompt experiment also distinguishes strictly valid JSON from JSON that can be recovered by removing Markdown code fences.

### 2. Original SFT with LoRA

The quantized Qwen model was fine-tuned on 4,000 support tickets using supervised instruction-response examples.

Prompt tokens were excluded from the supervised loss using response-only loss masking.

The resulting model substantially improved structured classification accuracy compared with the untuned instruction model.

### 3. Class-Aware SFT

Error analysis revealed that the original SFT model frequently confused Critical with High priority.

To investigate this issue, a second training run increased the Critical exposure ratio.

| Training property | Original SFT | Class-aware SFT |
|---|---:|---:|
| Training records | 4,000 | 4,000 |
| Critical occurrences | 49 | 400 |
| Critical ratio | 1.23% | 10.0% |
| Unique training examples | 4,000 | 3,649 |
| Unique Critical examples | 49 | 49 |

Critical examples were oversampled with replacement, while other classes were subsampled without replacement.

Both models used the same base model, principal hyperparameters, and checkpoint selection rule.

This intervention changes both minority-class exposure and non-critical sample diversity, which limits causal attribution to the sampling ratio alone.

### 4. Error and Generalization Analysis

The evaluation pipeline examines:

- Structured output validity and label compliance
- Per-field accuracy and three-field exact match
- Per-class precision, recall, F1, and Macro-F1
- Confusion matrices and error transitions
- High/Critical diagnostic classification
- Performance under event-associated distribution shift
- Event-level performance differences across 13 event groups

## Results

### Standard Test — 500 Samples

| Metric | Base | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|---:|
| Strict JSON validity | 99.6% | 90.4% | 100.0% | 100.0% |
| Exact match | 20.2% | 13.4% | 76.2% | **79.8%** |
| Category accuracy | 72.0% | 76.6% | 97.8% | **99.0%** |
| Sentiment accuracy | 45.8% | 39.6% | 86.0% | **86.8%** |
| Urgency accuracy | 39.6% | 25.0% | 90.2% | **92.8%** |

The Strong Prompt achieved 100% extractable JSON validity and 14.2% extracted exact match, but did not outperform the original prompt on overall exact-match accuracy.

The fine-tuned models substantially outperformed both zero-shot prompting baselines in this evaluation.

### Minority-Class Performance — Standard Test

| Metric | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|
| Urgency Macro-F1 | 0.635 | **0.838** |
| Critical recall | 0.0% (0/6) | **83.3% (5/6)** |
| Critical precision | 0.0% | 62.5% |
| High recall | 78.6% | 61.9% |
| Medium recall | 83.2% | 100.0% |
| Low recall | 98.5% | 93.0% |

The standard test set contains only six Critical examples, so its minority-class results should be interpreted cautiously.

### High/Critical Diagnostic Test — 50 Samples

| Metric | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|
| Exact match | 16.0% | 28.0% | **54.0%** |
| Urgency accuracy | 52.0% | 48.0% | **70.0%** |
| Critical recall | 4.0% | 4.0% | **88.0%** |
| High recall | 100.0% | 92.0% | 52.0% |
| High/Critical Macro-F1 | 0.376 | 0.358 | **0.740** |

Class-aware SFT improved Critical recognition from 1/25 to 22/25 compared with original SFT, while reducing High recall.

The diagnostic set uses a balanced High/Critical distribution and is not representative of ordinary ticket frequencies.

### Event-Shift Test — 520 Samples

Event-linked records were excluded from initial SFT training and evaluated separately using 40 selected tickets from each of 13 event groups.

| Metric | Base | Strong Prompt | Original QLoRA | Class-aware QLoRA |
|---|---:|---:|---:|---:|
| Exact match | 26.9% | 28.5% | 56.7% | **59.8%** |
| Category accuracy | 75.8% | 86.5% | 95.2% | **99.0%** |
| Sentiment accuracy | 44.8% | 40.0% | 72.9% | **78.3%** |
| Urgency accuracy | 49.8% | 44.0% | **80.8%** | 75.8% |
| Urgency Macro-F1 | 0.305 | 0.290 | 0.649 | **0.736** |
| Critical recall | 0.0% | 0.0% | 1.4% | **88.7%** |
| High recall | 98.8% | 100.0% | **98.3%** | 46.2% |

**Key finding:** Class-aware resampling consistently improved Critical recall across the evaluated datasets, but reduced High recall. Under event-associated distribution shift, Urgency Macro-F1 increased while overall Urgency Accuracy decreased.

This illustrates an important trade-off between minority-class sensitivity and majority-class performance.

### Event-Level Analysis

The effect of class-aware training varied across event types.

- **Connector launches:** Original QLoRA generally retained higher urgency accuracy.
- **Dashboard outages:** Class-aware QLoRA showed better average urgency accuracy.
- **Connector synchronization outages:** Class-aware QLoRA exhibited substantial urgency-accuracy declines in several event groups.

For example, `outage_dashboards_2026q1` improved from 52.5% to 80.0% urgency accuracy, whereas `outage_hubspot_sync_2024q4` decreased from 97.5% to 60.0%.

These are exploratory subgroup findings based on 40 selected tickets per event, not evidence of a general causal relationship between event type and model behavior.

## Reproduction

### Environment

The experiments were conducted on Apple Silicon using Python 3.11 and MLX-LM.

Install the required packages in a compatible Python environment:

```bash
pip install "mlx-lm[train]" datasets scikit-learn numpy pandas
```

### Data Preparation

```bash
python scripts/inspect_data.py
python scripts/prepare_data.py
```

The exact training data preparation, label schema, and audit procedures are described in the experiment report.

### Original LoRA Training

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data data \
  --mask-prompt \
  --batch-size 2 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --iters 500 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --save-every 100 \
  --adapter-path adapters \
  --seed 0
```

### Class-Aware Training

```bash
python scripts/prepare_balanced_data.py
```

```bash
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data data_balanced \
  --mask-prompt \
  --batch-size 2 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --iters 500 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --save-every 100 \
  --adapter-path adapters_balanced \
  --seed 0
```

Both experiments selected their iteration-400 adapters according to minimum validation loss.

### Evaluation

The evaluation script supports choosing the dataset, an optional adapter, and a prompt mode.

Example zero-shot evaluation:

```bash
python scripts/evaluate.py \
  --data-file data/test.jsonl \
  --output-name base
```

Strong Prompt evaluation:

```bash
python scripts/evaluate.py \
  --data-file data/test.jsonl \
  --prompt-mode strong \
  --output-name strong_prompt_test
```

Fine-tuned model evaluation, after placing the selected adapter and its configuration in `adapters_best/`:

```bash
python scripts/evaluate.py \
  --adapter adapters_best \
  --data-file data/test.jsonl \
  --output-name original_lora_reproduction
```

Additional experiment scripts provide duplicate auditing, diagnostic dataset generation, resampling, and event-level analysis.

Detailed metrics and aggregate confusion matrices are stored in `results/`. Model weights, generated datasets, and per-example outputs are not required for browsing the repository.

## Project Status

**Completed**

- Qwen2.5 baseline and Strong Prompt evaluation
- Response-only SFT with quantized-model LoRA
- Class-aware resampling and second fine-tuning run
- Exact duplicate and near-duplicate data audits
- Minority-class diagnostic evaluation
- Confusion matrix and error-transition analysis
- Event-associated distribution-shift evaluation
- Event-level subgroup analysis
- Consolidated experimental report

**Future Work**

- Further reproducibility and repeated-seed experiments
- Additional analysis of SFT loss masking and LoRA parameterization
- Preference optimization with DPO on a suitable response-quality task

DPO is a proposed extension and is not part of the completed experiments reported above.

## Limitations

1. **Synthetic data:** Results are based on scenario-generated tickets from a fictional platform, not real customer-service workloads.
2. **Limited independent minority examples:** Only 49 unique Critical examples were present in the original training split.
3. **Heuristic duplicate screening:** Exact matching and character-level TF-IDF do not eliminate all semantic or template overlap.
4. **Resampling confounding:** The class-aware intervention increased Critical exposure while reducing the diversity of other classes.
5. **Single-run comparisons:** The fine-tuning results have not been replicated across multiple training seeds.
6. **Evaluation reuse:** The standard test set was used during early error analysis and should be treated as a historical benchmark.
7. **Distribution differences:** The diagnostic benchmark is class-balanced, and the event-shift benchmark is sampled by event group. Their aggregate scores are not directly comparable to the standard test distribution.
8. **Scope of generalization:** Event-shift evaluation measures changes within the same synthetic data source, not real-world or cross-domain generalization.

## Documentation

- [Full Experiment Report](docs/experiment_report.md)
- [Label Schema](docs/label_schema.md)

## Dataset Attribution

Marinovich, V. (2026). *Polaris Support Tickets — Synthetic Dataset (v2).*

[Dataset on Hugging Face](https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2)

Dataset license: CC BY-SA 4.0, as stated on the Hugging Face dataset card.