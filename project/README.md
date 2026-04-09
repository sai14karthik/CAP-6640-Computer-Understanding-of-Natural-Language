# Hallucination Detection and Measurement in Large Language Models

CAP 6640 - Computer Understanding of Natural Language  
**Topic #3**: Benchmarking and Evaluation of NLP Systems

## Overview

This project evaluates hallucination detection and measurement across open-source LLMs on factual question-answering datasets. It uses **5 models** and **5 datasets** by default (configurable in `src/config.py`).

## Setup

```bash
cd project
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, Hugging Face `transformers` and `datasets`. For 7B+ models, a GPU with ~8GB+ VRAM is recommended; 8-bit quantization is used by default to reduce memory.

**No GPU?** Use the `--cpu` flag with small models (gpt2, distilgpt2). See "Running without a GPU" below.

## Project Structure

```
project/
├── src/
│   ├── config.py        # Datasets, models, and hyperparameters
│   ├── load_datasets.py # Load TruthfulQA, WikiQA, SQuAD v2, etc.
│   ├── load_models.py   # Load Hugging Face LLMs
│   ├── evaluate.py      # Metrics and evaluation loop
│   ├── run_experiments.py # Main experiment runner
│   └── utils.py         # Helpers
├── results/             # Output JSON results (created on first run)
├── scripts/
│   ├── test_setup.py    # Test data loading (no GPU)
│   └── analyze_results.py # Aggregate results table/CSV
├── requirements.txt
├── run.py               # Entry point
└── README.md
```

## Usage

**Quick run (5 models × 5 datasets; use small sample first):**

```bash
python run.py --max_samples 10
```

Then full run (up to 500 samples per dataset):

```bash
python run.py --max_samples 500
```

**Custom models and datasets:**

```bash
python run.py --models phi-2 mistral-7b --datasets truthfulqa wiki_qa --max_samples 100
```

**Few-shot prompting:**

```bash
python run.py --models phi-2 --datasets truthfulqa --prompt_type few_shot --max_samples 50
```

**All models and datasets (slow):**

```bash
python run.py --all --max_samples 100
```

**Running without a GPU (CPU only):**

```bash
python run.py --cpu --models gpt2 --datasets truthfulqa --max_samples 30
```

Uses small models (`gpt2`, `distilgpt2`) that run on CPU. Slower but no GPU needed.

**One model, one dataset, full size on CPU (recommended first full run):**

```bash
python run.py --cpu --models gpt2 --datasets truthfulqa --full
```

Runs `gpt2` on the full TruthfulQA subset (500 samples per config). For other datasets use e.g. `--datasets wiki_qa`.

**Future: run everything on CPU (all CPU models × all 5 datasets):**

```bash
python run.py --cpu --datasets truthfulqa wiki_qa natural_questions fever squad_v2 --full
```

Runs both `gpt2` and `distilgpt2` on all 5 datasets at full size (500 samples each). Total: 2 models × 5 datasets × 500 = 5,000 evaluations. Run overnight or in batches if needed.

**Options:**

- `--models`: Model keys from config (e.g. `phi-2`, `mistral-7b`, `llama2-7b`)
- `--datasets`: Dataset names (e.g. `truthfulqa`, `wiki_qa`, `squad_v2`)
- `--max_samples`: Max samples per dataset (default 50)
- `--prompt_type`: `zero_shot` or `few_shot`
- `--no_8bit`: Disable 8-bit quantization (needs more VRAM)
- `--output_dir`: Directory for result JSONs (default: `results/`)
- `--cpu`: CPU only; use small models (gpt2, distilgpt2). No GPU required.
- `--full`: Use full dataset size per config (500 samples per dataset). Use with one model + one dataset for a full CPU run.

## Output

- Per run: `results/<model>_<dataset>_<prompt_type>.json` with accuracy, precision, recall, hallucination rate, and metadata.
- Summary: `results/summary.json` with one row per model–dataset pair.
- **Aggregate table:** `python scripts/analyze_results.py` prints a summary table; `--format csv` for CSV.

## Datasets and Models (5 + 5)

- **5 Datasets:** TruthfulQA, WikiQA, Natural Questions, FEVER, SQuAD 2.0 (see `PROJECT_DATASETS` in `src/config.py`).
- **5 Models:** GPT-2, Phi-2, Mistral 7B, Llama 2 7B, Qwen 7B (see `PROJECT_MODELS` in `src/config.py`). GPU recommended; for CPU-only use `--cpu` with small models (gpt2, distilgpt2).

## Metrics

- **Accuracy (contain):** Fraction of answers that contain the ground-truth answer (normalized).
- **Accuracy (exact):** Exact match after normalization.
- **F1 (token):** Mean token-overlap F1 against the best matching reference (SQuAD-style).
- **Precision / Recall:** For single-answer QA, both equal accuracy (correct vs incorrect).
- **Hallucination rate:** 1 − accuracy (contain).

## License

For academic use as part of CAP 6640, UCF.
