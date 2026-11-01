# LLM Instruction Fine-Tuning

Parameter-efficient fine-tuning of Qwen for structured customer support ticket classification using MLX-LM and QLoRA on Apple Silicon.

## Project Overview

This project investigates supervised instruction fine-tuning for generating structured JSON predictions from English customer support tickets.

The model predicts three fields:

- **Category:** billing, technical, account, feature_request, other
- **Sentiment:** positive, neutral, negative
- **Urgency:** low, medium, high

## Planned Setup

- **Base model:** Qwen2.5-1.5B-Instruct (4-bit)
- **Training:** Supervised Fine-Tuning with LoRA on a quantized base model
- **Framework:** MLX-LM
- **Hardware:** Apple Silicon MacBook Air (24GB unified memory)
- **Dataset:** TriageIQ (2,000 synthetic labeled tickets)

## Evaluation Plan

Compare the base and fine-tuned models on the held-out test set using JSON validity, label compliance, per-field accuracy, and exact-match accuracy.

## Dataset Attribution

Dataset: https://huggingface.co/datasets/coldstart88/triageiq-dataset

The dataset is synthetic, English-only, and not independently human-verified. Results must be interpreted within these limitations.

## Status

Environment setup and initial Qwen inference completed. Dataset preparation, fine-tuning, and evaluation are pending.