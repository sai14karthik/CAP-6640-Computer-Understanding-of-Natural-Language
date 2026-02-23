"""
Utility functions for hallucination detection project.
"""

import os
import json
from pathlib import Path
from typing import List


def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, filepath: str) -> None:
    """Save dict to JSON file."""
    ensure_dir(os.path.dirname(filepath) or ".")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath) as f:
        return json.load(f)


def normalize_answer(text: str) -> str:
    """Normalize for comparison: lowercase, strip, collapse whitespace. Empty/non-string -> ''."""
    if not text or not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Correct iff normalize(prediction) == normalize(ground_truth)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_answer(prediction: str, ground_truth: str) -> bool:
    """Correct iff normalized ground_truth is a substring of normalized prediction."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if not gt_norm:
        return False
    return gt_norm in pred_norm


def contains_any_answer(prediction: str, ground_truths: List[str]) -> bool:
    """Correct iff prediction contains (after normalization) at least one of the ground truths."""
    if not ground_truths:
        return False
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        gt_norm = normalize_answer(gt) if isinstance(gt, str) else ""
        if gt_norm and gt_norm in pred_norm:
            return True
    return False
