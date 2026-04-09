from typing import Any, Dict, List, Optional

from .utils import exact_match, contains_any_answer, max_f1_over_refs
from .config import ZERO_SHOT_TEMPLATE, FEW_SHOT_TEMPLATE, GENERATION_CONFIG
from .load_models import generate_answer


def _ref_list_from_item(d: Dict[str, Any], primary_ref: str) -> List[str]:
    cand = (
        d.get("correct_answers")
        or d.get("acceptable_answers")
        or (d.get("answers") if isinstance(d.get("answers"), list) else None)
    )
    if not isinstance(cand, list) or not cand:
        return [primary_ref] if primary_ref else []
    out = []
    for a in cand:
        if a is None:
            continue
        if isinstance(a, str) and a.strip():
            out.append(a.strip())
        elif isinstance(a, dict) and a.get("text"):
            out.append(str(a["text"]).strip())
        elif isinstance(a, (list, tuple)) and a:
            out.append(str(a[0]).strip())
        else:
            out.append(str(a).strip())
    return out


def _get_ref_list_for_index(
    i: int,
    references: List[Any],
    refs_per_item: Optional[List[List[str]]],
) -> List[str]:
    if refs_per_item and i < len(refs_per_item) and refs_per_item[i]:
        return [r.strip() for r in refs_per_item[i] if isinstance(r, str) and (r or "").strip()]
    if i < len(references) and references[i] is not None:
        r = references[i]
        if isinstance(r, str) and r.strip():
            return [r.strip()]
    return []


def compute_accuracy(
    predictions: List[str],
    references: List[Any],
    match: str = "contain",
    refs_per_item: Optional[List[List[str]]] = None,
) -> float:
    if not predictions or not references:
        return 0.0
    n = min(len(predictions), len(references))
    correct = 0
    for i in range(n):
        pred = (predictions[i] or "").strip()
        ref_list = _get_ref_list_for_index(i, references, refs_per_item)
        if not ref_list:
            continue
        if match == "exact":
            correct += 1 if any(exact_match(pred, r) for r in ref_list) else 0
        else:
            correct += 1 if contains_any_answer(pred, ref_list) else 0
    return correct / n if n else 0.0


def compute_hallucination_rate(
    predictions: List[str],
    references: List[Any],
    match: str = "contain",
    refs_per_item: Optional[List[List[str]]] = None,
) -> float:
    acc = compute_accuracy(predictions, references, match=match, refs_per_item=refs_per_item)
    return 1.0 - acc


def compute_mean_f1(
    predictions: List[str],
    references: List[Any],
    refs_per_item: Optional[List[List[str]]] = None,
) -> float:
    if not predictions or not references:
        return 0.0
    n = min(len(predictions), len(references))
    total = 0.0
    counted = 0
    for i in range(n):
        pred = (predictions[i] or "").strip()
        ref_list = _get_ref_list_for_index(i, references, refs_per_item)
        if not ref_list:
            continue
        total += max_f1_over_refs(pred, ref_list)
        counted += 1
    return total / counted if counted else 0.0


def compute_precision_recall(
    predictions: List[str],
    references: List[Any],
    match: str = "contain",
    refs_per_item: Optional[List[List[str]]] = None,
) -> tuple:
    if not predictions or not references:
        return 0.0, 0.0
    n = min(len(predictions), len(references))
    tp = 0
    for i in range(n):
        pred = (predictions[i] or "").strip()
        ref_list = _get_ref_list_for_index(i, references, refs_per_item)
        if not ref_list:
            continue
        if match == "exact":
            tp += 1 if any(exact_match(pred, r) for r in ref_list) else 0
        else:
            tp += 1 if contains_any_answer(pred, ref_list) else 0
    return (tp / n, tp / n) if n else (0.0, 0.0)


def run_evaluation(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    prompt_type: str = "zero_shot",
    num_few_shot: int = 3,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    from tqdm import tqdm

    data = dataset[: max_samples or len(dataset)]
    predictions = []
    references = [str(d.get("answer", "") or "").strip() for d in data]
    refs_per_item = [_ref_list_from_item(d, references[i]) for i, d in enumerate(data)]

    for i, item in enumerate(tqdm(data, desc="Evaluating", disable=not verbose)):
        q = item.get("question", "")

        if prompt_type == "few_shot" and num_few_shot > 0 and i >= num_few_shot:
            examples = data[:num_few_shot]
            ex_text = ""
            for ex in examples:
                eq = ex.get("question", "")
                ea = ex.get("answer", "")
                ex_text += f"Question: {eq}\nAnswer: {ea}\n\n"
            prompt = ex_text + ZERO_SHOT_TEMPLATE.format(question=q)
        else:
            prompt = ZERO_SHOT_TEMPLATE.format(question=q)

        pred = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            do_sample=GENERATION_CONFIG["do_sample"],
        )
        predictions.append(pred)

    acc_contain = compute_accuracy(predictions, references, match="contain", refs_per_item=refs_per_item)
    acc_exact = compute_accuracy(predictions, references, match="exact", refs_per_item=refs_per_item)
    mean_f1 = compute_mean_f1(predictions, references, refs_per_item=refs_per_item)
    hall_rate = 1.0 - acc_contain
    precision, recall = compute_precision_recall(predictions, references, match="contain", refs_per_item=refs_per_item)
    assert abs(precision - acc_contain) < 1e-9 and abs(recall - acc_contain) < 1e-9

    return {
        "accuracy_contain": acc_contain,
        "accuracy_exact": acc_exact,
        "f1": mean_f1,
        "precision": precision,
        "recall": recall,
        "hallucination_rate": hall_rate,
        "num_samples": len(data),
        "predictions": predictions,
        "references": references,
    }
