# LLM Post-Training: LoRA SFT and Custom DPO on Apple Silicon

A hands-on LLM post-training project built with **MLX** on Apple Silicon. It combines supervised fine-tuning (SFT) for structured customer-support classification with a separate instruction-following SFT → Direct Preference Optimization (DPO) pipeline. The emphasis is on **implementing and evaluating training methods**, not just calling a high-level trainer.

## Highlights

- **LoRA SFT for structured prediction:** compared the base model, standard fine-tuning, and class-balanced fine-tuning on customer-support classification.
- **Custom DPO implementation:** implemented the pairwise preference objective using policy and frozen-reference response log probabilities, without a third-party DPO trainer.
- **Memory-aware experimentation:** built batched reference-log-probability caching and checkpointed workflows for an Apple Silicon laptop.
- **Two evaluation perspectives:** reported held-out preference metrics and inspected generated responses qualitatively, including cases where improvements in the DPO objective did **not** translate into better answers.

## Results at a Glance

### Structured customer-support SFT

| Model / setup | Exact-match accuracy |
|---|---:|
| Base model | 20.2% |
| Standard LoRA SFT | 76.2% |
| Class-balanced LoRA SFT | **79.8%** |

The percentages above are from the project's recorded classification experiments. See [`docs/experiment_report.md`](docs/experiment_report.md) and the detailed metrics in [`results/sft/`](results/sft/).

### Instruction SFT → DPO

| Metric | Result |
|---|---:|
| Held-out preference pairs | 400 |
| DPO test loss | **0.4260** |
| Reward accuracy | **81.5%** |
| Mean reward margin | **+1.0806** |
| Selected DPO checkpoint | Step 1,000 |
| DPO temperature parameter β | 0.1 |

The training data comprised **4,000** preference pairs, with **400** validation and **400** test pairs. The preference dataset was prepared from `argilla/ultrafeedback-binarized-preferences-cleaned`. The policy initialization was a separately trained instruction-SFT LoRA adapter, rather than the customer-support classification adapter.

**Important interpretation:** Reward accuracy measures whether the learned policy gives a larger *relative* preference margin to chosen than rejected responses under the DPO formulation. It is **not** a measure of factual correctness or a human win rate. Qualitative generation comparisons found some improvements, but also verbosity, truncation at generation limits, and failures to follow instructions. See [`docs/dpo_experiment_report.md`](docs/dpo_experiment_report.md) for the full discussion.

## Datasets

The two training tracks use **different datasets and objectives**. The processed datasets are kept locally under `data/` and are not included in this repository.

| Stage | Source | Experimental splits | Purpose |
|---|---|---|---|
| Customer-support classification SFT | [Polaris Support Tickets v2](https://huggingface.co/datasets/VladislavMarinovich/polaris-support-tickets-v2) (synthetic SaaS support tickets) | 4,000 train / 500 validation / 500 standard test | Predict `category`, `sentiment`, and `urgency` as a strict JSON object from ticket `subject` and `body` |
| Instruction SFT for DPO initialization | Chosen responses from the [Argilla UltraFeedback preference data](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 4,000 train / 400 validation | Train a separate general instruction-following reference/initial policy |
| DPO preference training and evaluation | [Argilla UltraFeedback binarized preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) | 4,000 train / 400 validation / 400 held-out test preference pairs | Optimize and evaluate relative preference for chosen versus rejected responses |

For classification, the original experiment excluded 6,000 event-linked records before selecting 5,000 ordinary tickets. Additional high/critical diagnostic splits (50 examples each) and an event-associated test set (520 examples) were used for targeted evaluation. The **class-balanced** SFT variant resampled the original training split to increase representation of `critical` tickets; it did not introduce an independent dataset.

For DPO, preparation filtered preference pairs by rating gap and sequence length, deduplicated normalized prompts, and created prompt-disjoint splits. The instruction-SFT track used the chosen answers from the training and validation preference splits; the held-out DPO test pairs were not used for training or checkpoint selection.

The local preparation scripts are in [`scripts/sft/`](scripts/sft/) and [`scripts/dpo/`](scripts/dpo/). Recreating a split from an upstream dataset does **not** automatically guarantee byte-identical data to the recorded experiments; use the saved metrics, reports, manifests, and hashes to assess comparability. See the [SFT report](docs/experiment_report.md) and [DPO report](docs/dpo_experiment_report.md) for dataset selection and evaluation details.

## Model and Training Approach

**Base model:** [`mlx-community/Qwen2.5-1.5B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit). Training and evaluation use the MLX ecosystem on Apple Silicon.

### Track A — Structured-classification SFT

1. Prepare structured customer-support examples and label schema.
2. Evaluate the unmodified instruction model as a baseline.
3. Train and evaluate a LoRA adapter.
4. Prepare a class-balanced training variant and compare results.
5. Analyze confusion matrices, diagnostic splits, and distribution shifts.

Label definitions: [`docs/label_schema.md`](docs/label_schema.md).

### Track B — Instruction SFT and DPO

1. Prepare disjoint preference training, validation, and test splits.
2. Fine-tune a separate instruction-SFT policy on the chosen responses.
3. Cache frozen reference-policy response log probabilities in batches.
4. Train a LoRA policy using the DPO objective:

   \[
   \mathcal{L}_{\mathrm{DPO}} =
   -\log \sigma\left(\beta\left[
   \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
   -\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
   \right]\right).
   \]

5. Select a checkpoint using validation measurements; evaluate once on the held-out test split.
6. Compare SFT and DPO generations at token limits of 256 and 512, with qualitative error analysis.

The custom implementation and helper routines are in [`scripts/dpo/`](scripts/dpo/). Frozen-reference caching reduces repeated reference-model computation, while smaller subprocess batches help bound memory use.

## Repository Layout

```text
.
├── requirements.txt          # Pinned direct Python dependencies
├── scripts/
│   ├── sft/                  # Data preparation, classification evaluation, error analysis
│   └── dpo/                  # Preference preparation, reference caching, DPO training/evaluation
├── data/                     # Local datasets; not committed
│   ├── sft/{original,balanced}/
│   ├── dpo_sft/
│   └── dpo/
├── models/                   # Four selected LoRA adapters and their configurations
│   ├── sft/{original_best,balanced_best}/
│   ├── dpo_sft/best/
│   └── dpo/best/
├── results/
│   ├── sft/                  # Classification metrics and error analyses
│   └── dpo/                  # DPO metrics and comparison summaries
├── docs/                     # Experiment reports and label schema
├── cache/                    # Local reference-log-probability cache; not committed
├── checkpoints/              # Optional new training runs; not committed
└── archive/                  # Local training checkpoints; not committed
```

`archive/` preserves earlier checkpoints and optimizer states locally but is excluded from the published Git repository. The four selected adapters under `models/` are the release-facing weights.

## Getting Started

### Environment

- Apple Silicon Mac with a working Python environment and MLX support.
- Python 3.11.18 was used for the experiments.
- Core dependencies include `mlx`, `mlx-lm`, `datasets`, `numpy`, and `scikit-learn` (plus other imports required by individual scripts).
- The base model is fetched through the Hugging Face / MLX ecosystem when needed.

Install the declared project dependencies in a fresh Python 3.11 environment on an Apple Silicon Mac:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` pins the direct dependencies used in the local experiment environment (`mlx==0.32.4`, `mlx-lm==0.32.2`, `datasets==5.1.0`, `numpy==2.4.6`, and `scikit-learn==1.9.1`). These pins do not lock all transitive dependencies, and a fresh-environment installation and end-to-end reproduction have not yet been verified.

### Lightweight repository checks

From the project root, check Python syntax and the release-facing model artifacts without triggering training:

```bash
python -m compileall -q scripts/sft scripts/dpo
python - <<'PYCODE'
from pathlib import Path
adapters = [
    'models/sft/original_best',
    'models/sft/balanced_best',
    'models/dpo_sft/best',
    'models/dpo/best',
]
for folder in adapters:
    for filename in ('adapters.safetensors', 'adapter_config.json'):
        assert (Path(folder) / filename).is_file(), f'Missing: {folder}/{filename}'
print('PASS: syntax and four adapter artifact paths')
PYCODE
```

These are **static checks**. Separately, all four released adapters were successfully loaded and used to generate short responses on the original Apple Silicon development machine after directory restructuring. That smoke test does not establish clean-clone reproducibility or reproduce evaluation metrics.

### Load a published LoRA adapter

With the core MLX dependencies installed and access to the upstream base model, the DPO adapter can be loaded with:

```python
from mlx_lm import load

model, tokenizer = load(
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    adapter_path="models/dpo/best",
)
```

The same API can be used with `models/dpo_sft/best`, `models/sft/original_best`, or `models/sft/balanced_best`. All four adapters passed sequential loading and short-generation smoke tests in the development environment; these checks have **not** been repeated on a clean clone.

### Reproducing specific stages

Run commands from the **repository root** on Apple Silicon. Commands that process or train on data require datasets that are **not included in this Git repository**. The following are based on recorded experiment parameters and current CLI interfaces; they are **not** a verified clean-clone, end-to-end reproduction.

**Customer-support SFT — prepare data and evaluate a published adapter**

```bash
python scripts/sft/prepare_data.py
python scripts/sft/prepare_balanced_data.py

python scripts/sft/evaluate.py \
  --adapter models/sft/original_best \
  --data-file data/sft/original/test.jsonl \
  --output-name original_lora_recheck

python scripts/sft/evaluate.py \
  --adapter models/sft/balanced_best \
  --data-file data/sft/original/test.jsonl \
  --output-name balanced_lora_recheck
```

The data-preparation commands may download or reconstruct local samples and should **not** be used to assume that regenerated splits are identical to the recorded experiments without verifying source selection, seeds, and manifests. The evaluation script should write to `results/sft/` after the repository's output-path fix; the commands above are intended for a separate recheck, not to reproduce metrics automatically. Use `--prompt-mode strong` when evaluating the strong-prompt variant.

**Recorded customer-support LoRA training settings**

The SFT experiment report records a 4-bit MLX base model, LoRA rank 8, 16 adapted layers, batch size 2, learning rate `1e-5`, response-only masking, 500 training iterations, and selection of the iteration-400 checkpoint. The original log's command targets the old directory layout; the following *adapted command* uses the new directories, but has not been rerun after restructuring:

```bash
python -m mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data data/sft/original --mask-prompt \
  --batch-size 2 --num-layers 16 \
  --learning-rate 1e-5 --iters 500 \
  --adapter-path checkpoints/sft_original
```

The class-aware experiment used the same primary settings with `--data data/sft/balanced`, a separate adapter destination, and `--seed 0`. **Do not overwrite the published best adapters**: checkpoint selection and copying the best iteration are separate steps. The exact installed MLX-LM CLI option compatibility should be confirmed before retraining.

**DPO preference split preparation and reference caching**

The *recorded* split sizes were 4,000 training / 400 validation / 400 test pairs. The preparation CLI exposes these arguments:

```bash
python scripts/dpo/prepare_dpo_data.py \
  --output-dir data/dpo \
  --train 4000 --valid 400 --test 400 \
  --max-tokens 1024 --seed 42

python scripts/dpo/run_reference_batches.py --split train
python scripts/dpo/run_reference_batches.py --split valid
```

Preparation can require downloading the upstream preference dataset; regenerating it must be checked against the recorded `manifest.json` and hash files. Reference caching requires a locally available instruction-SFT adapter at `models/dpo_sft/best`. Its cache is written under `cache/dpo/ref_cache/` and is computed in batches of 20. Test-cache creation uses the separate `scripts/dpo/cache_dpo_test.py` workflow (not `run_reference_batches.py`, which accepts only `train` or `valid`).

**Custom DPO training**

```bash
python scripts/dpo/train_dpo.py \
  --steps 1000 \
  --output-dir checkpoints/dpo
```

`--steps` is a *total target step*, not an additional-step count; steps must be multiples of 100. The run expects matching prepared datasets, frozen-reference caches, and source hashes. Use `--resume PATH_TO_STEP_CHECKPOINT` only with a compatible checkpoint. This command is documented for reproducibility and **should not be run merely to inspect the repository**. Its output directory is intentionally separate from published `models/` weights and retained historical `archive/` checkpoints.

**DPO evaluation**

The held-out test evaluator `scripts/dpo/evaluate_dpo_test.py` uses 400 test pairs, a cached frozen-reference split, the selected `models/dpo/best` adapter, and model/hash/metadata checks. Its full evaluation involves worker subprocesses and is not a lightweight smoke test. The complete invocation and regeneration of its test reference cache have **not** been verified from a clean clone. For the reproducible reported results, consult `results/dpo/test_metrics.json` and `docs/dpo_experiment_report.md` rather than claiming the test can be rerun with one dependency-free command.

### Data and evaluation artifacts

The Git repository intentionally excludes the local `data/` directory and `cache/` reference-log-probability files. To run training or reproduce the held-out DPO evaluation, the appropriate processed splits and reference cache must first be prepared. The relevant data-preparation and reference-cache scripts include (their CLI defaults and complete execution order have not yet been verified on a clean clone):

- [`scripts/sft/prepare_data.py`](scripts/sft/prepare_data.py) and [`scripts/sft/prepare_balanced_data.py`](scripts/sft/prepare_balanced_data.py)
- [`scripts/dpo/prepare_dpo_data.py`](scripts/dpo/prepare_dpo_data.py)
- [`scripts/dpo/run_reference_batches.py`](scripts/dpo/run_reference_batches.py) and [`scripts/dpo/cache_dpo_reference_safe.py`](scripts/dpo/cache_dpo_reference_safe.py)

The published metrics and reports can be inspected **without** downloading datasets or loading a model:

- [`results/sft/`](results/sft/) and [`docs/experiment_report.md`](docs/experiment_report.md)
- [`results/dpo/test_metrics.json`](results/dpo/test_metrics.json)
- [`results/dpo/checkpoint_metrics.json`](results/dpo/checkpoint_metrics.json)
- [`results/dpo/generation_metrics.json`](results/dpo/generation_metrics.json)
- [`results/dpo/generation_512_metrics.json`](results/dpo/generation_512_metrics.json)
- [`docs/dpo_experiment_report.md`](docs/dpo_experiment_report.md)

**Reproduction caveat:** The code was developed around local experiment artifacts and has undergone path, syntax, hash, and manifest checks after repository restructuring, but a complete clean-clone end-to-end reproduction has **not** yet been verified. Do not assume that all training/evaluation scripts run without additional setup or cached inputs.

## Limitations

- Experiments use a relatively small, quantized 1.5B-parameter base model and laptop-scale compute.
- Classification scores describe the project's evaluation setup and should not be generalized to other label schemes or datasets.
- DPO's preference-based test metrics do not guarantee better response quality. The qualitative review identified lengthy responses, instruction-following errors, and generation-cap effects.
- The released adapters do not include the original full base-model weights.
- Processed training/evaluation data and historical checkpoints are not part of the Git repository.

## Documentation

- [SFT experiment report](docs/experiment_report.md)
- [DPO experiment report](docs/dpo_experiment_report.md)
- [Classification label schema](docs/label_schema.md)

## Acknowledgments

This project builds upon the Qwen2.5-Instruct model, the MLX and MLX-LM frameworks, and the Polaris Support Tickets and Argilla UltraFeedback datasets. Thanks to the respective authors and open-source communities for making these resources publicly available.
