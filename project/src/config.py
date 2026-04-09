DATASETS = {
    "truthfulqa": {
        "hf_id": "truthful_qa",
        "config": "generation",
        "max_samples": 500,
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

MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "phi-2": "microsoft/phi-2",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-7b": "google/gemma-7b-it",
    "qwen-7b": "Qwen/Qwen-7B-Chat",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
}

CPU_MODELS = ["gpt2", "distilgpt2"]

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

DEFAULT_MODELS = PROJECT_MODELS
DEFAULT_DATASETS = PROJECT_DATASETS

GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.1,
    "do_sample": False,
    "pad_token_id": None,
}

ZERO_SHOT_TEMPLATE = "Question: {question}\nAnswer:"
FEW_SHOT_TEMPLATE = "Question: {example_q}\nAnswer: {example_a}\n\nQuestion: {question}\nAnswer:"

RESULTS_DIR = "results"
DATA_CACHE_DIR = "data_cache"
