#!/usr/bin/env python3
"""Test dataset loading and config without running models. Run from project root."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATASETS, MODELS
from src.load_datasets import load_dataset_by_name, get_all_dataset_names


def main():
    print("Config: Datasets =", list(DATASETS.keys()))
    print("Config: Models =", list(MODELS.keys()))
    for name in ["truthfulqa", "wiki_qa", "natural_questions", "fever", "squad_v2"]:
        if name not in get_all_dataset_names():
            continue
        try:
            data = load_dataset_by_name(name, max_samples=5)
            print(f"  {name}: loaded {len(data)} samples")
            if data:
                print(f"    Example Q: {data[0].get('question', '')[:60]}...")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    print("Setup check done.")


if __name__ == "__main__":
    main()
