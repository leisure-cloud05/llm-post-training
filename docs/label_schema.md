# Label Schema: Chinese Customer Feedback Classification

## 1. Task Definition

Given a Chinese customer feedback message, produce a valid JSON object containing exactly three fields:

- `category`: Primary issue category
- `sentiment`: Overall sentiment
- `urgency`: Priority of the reported issue

The model must use only the predefined labels and must not add explanations or extra fields.

## 2. Category Labels

| Label | Definition |
|---|---|
| 产品质量 | Product defects, damage, or functional failures |
| 物流配送 | Shipping delays, delivery problems, or lost packages |
| 售后服务 | Customer support, refunds, returns, or exchange requests |
| 其他 | Feedback outside the categories above |

For messages involving multiple issues, assign the category representing the primary complaint.

If the primary issue is ambiguous, apply the following tie-breaking order:

产品质量 > 物流配送 > 售后服务 > 其他

## 3. Sentiment Labels

| Label | Definition |
|---|---|
| 正面 | Clear satisfaction, praise, or positive experience |
| 中性 | Factual statements without strong positive or negative sentiment |
| 负面 | Complaints, disappointment, or dissatisfaction |

## 4. Urgency Labels

| Label | Definition |
|---|---|
| 高 | Safety risks, critical failures preventing normal use, or explicit requests for immediate action |
| 中 | Problems requiring resolution without immediate risk or critical impact |
| 低 | General inquiries, suggestions, praise, or minor inconvenience |

Urgency must be determined from the reported impact, not sentiment alone.

## 5. Output Format

Return only a valid JSON object with exactly three keys:

```json
{
  "category": "产品质量",
  "sentiment": "负面",
  "urgency": "高"
}
```

No Markdown fences, additional keys, or explanations are allowed in model outputs.

## 6. Example

**Input:**

耳机刚买两天就坏了，客服一直不回复。

**Expected output:**

```json
{
  "category": "产品质量",
  "sentiment": "负面",
  "urgency": "高"
}
```

## 7. Dataset Split Rules

- Training set: Used to update model parameters.
- Validation set: Used for development and model selection.
- Test set: Reserved for final evaluation.
- Identical or near-duplicate messages must not appear across different splits.
- All splits must follow the same labeling rules.
- Dataset generation methods, limitations, and split sizes must be documented.

## 8. Evaluation

Primary metrics:

- Exact-match accuracy across all three fields
- Per-field classification accuracy
- JSON validity rate
- Label-set compliance rate

The base model and fine-tuned model must be evaluated using the same held-out test set, prompt format, and decoding configuration.

## 9. Scope and Limitations

This is a task-specific experimental labeling scheme, not a universal industry standard.

If synthetic data is used, its origin must be disclosed. Performance on synthetic test data does not establish real-world generalization.