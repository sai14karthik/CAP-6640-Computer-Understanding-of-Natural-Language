"""
Configuration for Hallucination Detection and Measurement in LLMs.
CAP 6640 - Computer Understanding of Natural Language
"""

# Dataset names and Hugging Face IDs
DATASETS = {
    "truthfulqa": {
        "hf_id": "truthful_qa",
        "config": "generation",
        "max_samples": 500,  # subset for faster runs; set None for full
        "question_key": "question",
        "answer_key": "best_answer",
    },
    "wiki_qa": {
        "hf_id": "wiki_qa",
        "config": None,
        "max_samples": 500,
        "question_key": "question",
        "answer_key": "answer",
    },
    "natural_questions": {
        "hf_id": "natural_questions",
        "config": None,
        "split": "train",
        "streaming": True,
        "max_samples": 500,
        "question_key": "question",
        "answer_key": "short_answers",
    },
    "fever": {
        "hf_id": "fever",
        "config": "v1.0",
        "split": "labelled_dev",
        "max_samples": 500,
        "question_key": "claim",
        "answer_key": "label",
    },
    "squad_v2": {
        "hf_id": "squad_v2",
        "config": None,
        "max_samples": 500,
        "question_key": "question",
        "answer_key": "answers",
    },
}

# Open-source models (Hugging Face model IDs)
MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "phi-2": "microsoft/phi-2",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-7b": "google/gemma-7b-it",
    "qwen-7b": "Qwen/Qwen-7B-Chat",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    # CPU-only (no GPU needed); small models for testing the pipeline
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
}

# Models that run on CPU in reasonable time (small; use for --cpu)
CPU_MODELS = ["gpt2", "distilgpt2"]

# Project: exactly 5 models, 5 datasets (aligned with report: GPT-2 + four 7B-class models)
PROJECT_MODELS = [
    "gpt2",
    "phi-2",
    "mistral-7b",
    "llama2-7b",
    "qwen-7b",
]
PROJECT_DATASETS = [
    "truthfulqa",
    "wiki_qa",
    "natural_questions",
    "fever",
    "squad_v2",
]

# Default for run.py: use project 5 models, 5 datasets
DEFAULT_MODELS = PROJECT_MODELS
DEFAULT_DATASETS = PROJECT_DATASETS

# Generation settings
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.1,
    "do_sample": False,
    "pad_token_id": None,
}

# Prompt templates
ZERO_SHOT_TEMPLATE = "Question: {question}\nAnswer:"
FEW_SHOT_TEMPLATE = "Question: {example_q}\nAnswer: {example_a}\n\nQuestion: {question}\nAnswer:"

# Paths
RESULTS_DIR = "results"
DATA_CACHE_DIR = "data_cache"
