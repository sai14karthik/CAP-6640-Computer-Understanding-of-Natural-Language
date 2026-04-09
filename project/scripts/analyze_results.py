#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"No directory: {results_dir}")
        return

    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
            if "error" in data:
                continue
            rows.append({
                "model": data.get("model", ""),
                "dataset": data.get("dataset", ""),
                "prompt_type": data.get("prompt_type", ""),
                "accuracy_contain": data.get("accuracy_contain"),
                "accuracy_exact": data.get("accuracy_exact"),
                "f1": data.get("f1"),
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "hallucination_rate": data.get("hallucination_rate"),
                "num_samples": data.get("num_samples"),
            })
        except Exception as e:
            print(f"Skip {f.name}: {e}")

    if not rows:
        print("No result files found.")
        return

    if args.format == "csv":
        print("model,dataset,prompt_type,accuracy_contain,accuracy_exact,f1,precision,recall,hallucination_rate,num_samples")
        for r in rows:
            print(",".join(str(r.get(k, "")) for k in ["model", "dataset", "prompt_type", "accuracy_contain", "accuracy_exact", "f1", "precision", "recall", "hallucination_rate", "num_samples"]))
        return

    print("\nHallucination Detection - Results Summary")
    print("=" * 80)
    for r in rows:
        acc = r.get("accuracy_contain")
        em = r.get("accuracy_exact")
        f1 = r.get("f1")
        hall = r.get("hallucination_rate")
        p = r.get("precision")
        q = r.get("recall")
        n = r.get("num_samples")
        acc_s = f"{acc:.4f}" if acc is not None else "N/A"
        em_s = f"{em:.4f}" if em is not None else "N/A"
        f1_s = f"{f1:.4f}" if f1 is not None else "N/A"
        hall_s = f"{hall:.4f}" if hall is not None else "N/A"
        p_s = f"{p:.4f}" if p is not None else "N/A"
        q_s = f"{q:.4f}" if q is not None else "N/A"
        print(f"  {r['model']:15} | {r['dataset']:18} | {r['prompt_type']:10} | acc={acc_s} em={em_s} f1={f1_s} | P={p_s} R={q_s} | hall={hall_s} | n={n}")
    print("=" * 80)
