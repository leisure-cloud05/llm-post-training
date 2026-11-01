# Label Schema: Structured Support Ticket Classification

## 1. Task Definition

This project investigates supervised instruction fine-tuning for structured classification of English customer support tickets.

Given a ticket subject and body, the model generates a JSON object with exactly three fields:

- `category`: Primary support topic
- `sentiment`: Customer sentiment
- `urgency`: Ticket priority

The model must predict labels from the ticket text without accessing ground-truth labels or unrelated metadata.

## 2. Dataset

**Name:** Polaris Support Tickets — Synthetic Dataset (v2)

**Source:** https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2

**Size:** 23,994 records

**Language:** English

**Type:** Synthetic customer support tickets for a fictional SaaS platform.

The source dataset contains a single original train split. This project will create its own training, validation, and test subsets.

## 3. Label Mapping

| Output field | Source column | Number of classes |
|---|---|---|
| `category` | `topic` | 8 |
| `sentiment` | `sentiment` | 6 |
| `urgency` | `priority` | 4 |

The output field `urgency` is a project-specific name for the source dataset's `priority` attribute.

**Priority labels:**

- `low`
- `medium`
- `high`
- `critical`

The complete `topic` and `sentiment` label lists will be extracted programmatically from the downloaded dataset to avoid introducing unsupported labels.

Original labels will be preserved without manually merging or renaming classes.

## 4. Model Input

The model receives only the ticket subject and body.

Example input:

**Subject:** Scheduled report never came

**Body:** My scheduled report didn't arrive this morning. We need it for our daily performance review.

The source fields `topic`, `sentiment`, and `priority` must not appear in the input prompt.

## 5. Output Format

The model must return a valid JSON object containing exactly:

- `category`
- `sentiment`
- `urgency`

Example output format:

`{"category": "<topic label>", "sentiment": "<sentiment label>", "urgency": "medium"}`

All predictions must belong to the allowed source label sets.

No additional text, explanations, or Markdown formatting should be generated.

## 6. Dataset Preparation

The initial experiment targets approximately:

| Split | Target size |
|---|---:|
| Training | 4,000 |
| Validation | 500 |
| Test | 500 |

These are planned sample counts, not completed dataset statistics.

Preparation requirements:

- Remove missing or unusable records.
- Validate source labels.
- Check exact and near-duplicate texts.
- Inspect class distributions.
- Identify event-driven tickets using `event_id` and related metadata.
- Reduce leakage from related incidents when constructing dataset splits.
- Use a fixed random seed and record final sample counts.

Test samples must remain excluded from training and model selection.

## 7. Evaluation

Evaluate the base model and fine-tuned model using the same held-out test set, prompt template, and decoding settings.

Metrics:

- JSON validity rate
- Label-set compliance rate
- Category accuracy
- Sentiment accuracy
- Urgency accuracy
- Three-field exact-match accuracy

Report invalid outputs and representative classification errors.

## 8. Dataset Limitations

- The dataset is synthetic and does not contain real customer interactions.
- Labels are generated from predefined scenarios and are not independently human-verified.
- Some records share incident or scenario context.
- Evaluation performance may be optimistic relative to real-world customer support.
- This experiment demonstrates a reproducible fine-tuning and evaluation workflow, not production-level generalization.

## 9. Dataset Attribution

Marinovich, V. (2026). *Polaris Support Tickets — Synthetic Dataset (v2).*

Dataset: https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2

License: CC BY-SA 4.0, as indicated on the Hugging Face dataset page.