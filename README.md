# LLM Instruction Fine-Tuning

Parameter-efficient fine-tuning of Qwen for structured customer support ticket classification using MLX-LM on Apple Silicon.

## Overview

This project studies supervised instruction fine-tuning with LoRA on a quantized language model.

The target task is to generate structured JSON predictions for three attributes of English customer support tickets:

- **Category:** Primary support topic
- **Sentiment:** Customer sentiment
- **Urgency:** Ticket priority

The objective is to compare a pretrained instruction-following model with its task-specific fine-tuned version under consistent evaluation conditions.

## Planned Experiment

| Component | Configuration |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct (4-bit) |
| Training framework | MLX-LM |
| Training method | Supervised Fine-Tuning with LoRA on a quantized model |
| Hardware | Apple Silicon MacBook Air, 24GB unified memory |
| Dataset | Polaris Support Tickets v2 |
| Data language | English |
| Output format | Three-field JSON |

## Dataset

**Source:** https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2

The dataset contains 23,994 synthetic support tickets for a fictional SaaS platform.

This project uses:

- `subject` and `body` as model inputs.
- `topic` as the category target.
- `sentiment` as the sentiment target.
- `priority` as the urgency target.

The initial experiment targets approximately 4,000 training, 500 validation, and 500 test samples, subject to data-quality checks.

Dataset labels and preprocessing rules are documented in `docs/label_schema.md`.

## Methodology

1. Inspect source fields, label distributions, and duplicate records.
2. Construct reproducible training, validation, and test subsets.
3. Convert records into instruction-following JSONL examples.
4. Evaluate the base model.
5. Fine-tune the quantized Qwen model using LoRA.
6. Evaluate the fine-tuned model under identical test conditions.
7. Analyze output validity and classification errors.

## Evaluation Metrics

- JSON validity
- Label-set compliance
- Per-field classification accuracy
- Three-field exact-match accuracy

## Project Status

**Completed:**
- Conda environment setup with Python 3.11
- MLX-LM installation and import verification
- Initial Qwen2.5-0.5B inference test
- Git repository initialization

**Pending:**
- Polaris v2 dataset inspection and preparation
- Baseline evaluation
- QLoRA fine-tuning
- Post-training evaluation
- Results and error analysis

## Limitations

The dataset is synthetic and scenario-generated. High accuracy on held-out synthetic records does not guarantee real-world generalization.

## Dataset Attribution

Marinovich, V. (2026). *Polaris Support Tickets — Synthetic Dataset (v2).*

https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2

Dataset license: CC BY-SA 4.0, according to the Hugging Face dataset card.