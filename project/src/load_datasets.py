from typing import Any, Dict, List, Optional
from datasets import load_dataset
from .config import DATASETS


def _get_answer_text(item: Any, key: str, dataset_name: str) -> str:
    val = item.get(key)
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        if not val:
            return ""
        if isinstance(val[0], dict) and "text" in val[0]:
            return val[0]["text"].strip()
        if isinstance(val[0], str):
            return val[0].strip()
    if isinstance(val, dict) and "text" in val:
        return val["text"].strip()
    return str(val).strip()


def load_truthfulqa(max_samples: Optional[int] = 500) -> List[Dict[str, Any]]:
    ds = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
    rows = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        q = item.get("question", "")
        best = item.get("best_answer")
        correct_list = item.get("correct_answers") or []
        if isinstance(correct_list, str):
            correct_list = [correct_list]
        all_answers = []
        if best:
            all_answers.append(best.strip() if isinstance(best, str) else str(best).strip())
        for a in correct_list:
            if isinstance(a, str) and a.strip() and a.strip() not in all_answers:
                all_answers.append(a.strip())
            elif isinstance(a, list) and a:
                all_answers.append(str(a[0]).strip())
        if not all_answers:
            all_answers = [""]
        rows.append({
            "question": q,
            "answer": all_answers[0],
            "correct_answers": all_answers,
            "dataset": "truthfulqa",
        })
    return rows


def load_wiki_qa(max_samples: Optional[int] = 500) -> List[Dict[str, str]]:
    ds = load_dataset("wiki_qa", split="test", trust_remote_code=True)
    rows = []
    seen_questions = set()
    for i, item in enumerate(ds):
        if max_samples and len(rows) >= max_samples:
            break
        q = item.get("question", "").strip()
        if q in seen_questions:
            continue
        seen_questions.add(q)
        ans = item.get("answer", "")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        rows.append({"question": q, "answer": str(ans).strip(), "dataset": "wiki_qa"})
    return rows


def _natural_questions_question_text(item: Dict[str, Any]) -> str:
    q = item.get("question")
    if isinstance(q, dict):
        return str(q.get("text") or "").strip()
    return str(q or "").strip()


def _natural_questions_short_answer_texts(item: Dict[str, Any]) -> List[str]:
    ann = item.get("annotations") or {}
    sa = ann.get("short_answers")
    texts: List[str] = []
    if isinstance(sa, list):
        for x in sa:
            if isinstance(x, dict) and x.get("text") is not None:
                t = str(x["text"]).strip()
                if t:
                    texts.append(t)
            elif isinstance(x, str) and x.strip():
                texts.append(x.strip())
    elif isinstance(sa, dict) and sa.get("text") is not None:
        t = str(sa["text"]).strip()
        if t:
            texts.append(t)
    return texts


def load_natural_questions(max_samples: Optional[int] = 500) -> List[Dict[str, Any]]:
    n = max_samples if max_samples is not None else 500
    ds = load_dataset("natural_questions", split="train", streaming=True, trust_remote_code=True)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        q_text = _natural_questions_question_text(item)
        texts = _natural_questions_short_answer_texts(item)
        if not texts:
            texts = [""]
        rows.append({
            "question": q_text,
            "answer": texts[0],
            "correct_answers": texts,
            "dataset": "natural_questions",
        })
    return rows


def load_fever(max_samples: Optional[int] = 500) -> List[Dict[str, Any]]:
    ds = load_dataset("fever", "v1.0", split="labelled_dev", trust_remote_code=True)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        claim = str(item.get("claim", "") or "").strip()
        lab = str(item.get("label", "") or "").strip()
        alts = [lab]
        low = lab.lower()
        if low and low not in alts:
            alts.append(low)
        rows.append({
            "question": claim,
            "answer": lab,
            "correct_answers": alts,
            "dataset": "fever",
        })
    return rows


def load_squad_v2(max_samples: Optional[int] = 500) -> List[Dict[str, str]]:
    ds = load_dataset("squad_v2", split="validation", trust_remote_code=True)
    rows = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        q = item.get("question", "")
        answers = item.get("answers", {})
        if answers and answers.get("text"):
            ans = answers["text"][0]
        else:
            ans = ""
        rows.append({"question": q, "answer": ans, "dataset": "squad_v2", "context": item.get("context", "")})
    return rows


def load_dataset_by_name(name: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    cfg = DATASETS.get(name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    n = max_samples if max_samples is not None else cfg.get("max_samples")

    if name == "truthfulqa":
        return load_truthfulqa(n)
    if name == "wiki_qa":
        return load_wiki_qa(n)
    if name == "squad_v2":
        return load_squad_v2(n)
    if name == "natural_questions":
        return load_natural_questions(n)
    if name == "fever":
        return load_fever(n)

    hf_id = cfg["hf_id"]
    config = cfg.get("config")
    q_key = cfg["question_key"]
    a_key = cfg["answer_key"]
    split = cfg.get("split", "validation")
    try:
        if config is not None:
            ds = load_dataset(hf_id, config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset(hf_id, config, split="train", trust_remote_code=True) if config is not None else load_dataset(hf_id, split="train", trust_remote_code=True)
        except Exception:
            ds = load_dataset(hf_id, split="train", trust_remote_code=True)
    rows = []
    for i, item in enumerate(ds):
        if n and i >= n:
            break
        q = item.get(q_key, "")
        ans = _get_answer_text(item, a_key, name)
        rows.append({"question": str(q).strip(), "answer": ans, "dataset": name})
    return rows


def get_all_dataset_names() -> List[str]:
    return list(DATASETS.keys())
