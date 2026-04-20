"""
Dataset preparation pipeline.

Downloads and samples:
  - TriviaQA (Wikipedia): 150 Qs, stratified by difficulty, single-entity answers only
  - PopQA (long-tail):    150 Qs, Wikipedia views < 10k
  - MedQA (USMLE):        100 Qs, closed-book

Run this script once:
    python data_prep.py

Output: data/triviaqa.json, data/popqa.json, data/medqa.json
        data/all_questions.json  (merged, with dataset label)

Requirements: pip install datasets
"""

import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

from config import DATA_DIR, DATASET_SIZES

SEED = 42
random.seed(SEED)


def _fingerprint(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ══════════════════════════════════════════════════════════════
#  TriviaQA
# ══════════════════════════════════════════════════════════════

def prepare_triviaqa(n: int = DATASET_SIZES["triviaqa"]) -> list[dict]:
    """
    Source: mandarjoshi/trivia_qa (rc.wikipedia subset)
    Sampling: stratified by difficulty proxy (answer-string length as
    rough proxy — short = easy entity, long = harder).
    Filter: only unambiguous single-entity answers.
    """
    from datasets import load_dataset

    print("[TriviaQA] Loading dataset …")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")

    # Filter: single-entity answers (one alias, no list-style answers)
    filtered = []
    for row in ds:
        answer_obj = row["answer"]
        aliases = answer_obj.get("aliases", [])
        value = answer_obj.get("value", "").strip()
        # Keep only questions with a clear single-entity answer
        if not value:
            continue
        # Skip list-type answers (contain "and", ";", or are very long)
        if len(value) > 80 or ";" in value:
            continue
        filtered.append({
            "question": row["question"].strip(),
            "gold_answer": value,
            "aliases": aliases,
            "answer_length": len(value),
        })

    print(f"[TriviaQA] {len(filtered)} candidates after filtering")

    # Stratified sampling by answer length terciles (proxy for difficulty)
    filtered.sort(key=lambda x: x["answer_length"])
    third = len(filtered) // 3
    buckets = [filtered[:third], filtered[third:2*third], filtered[2*third:]]
    per_bucket = n // 3

    sampled = []
    for i, bucket in enumerate(buckets):
        k = per_bucket if i < 2 else (n - 2 * per_bucket)
        sampled.extend(random.sample(bucket, min(k, len(bucket))))

    # Standardize
    questions = []
    for s in sampled:
        questions.append({
            "qid": f"tqa_{_fingerprint(s['question'])}",
            "dataset": "triviaqa",
            "question": s["question"],
            "gold_answer": s["gold_answer"],
            "aliases": s["aliases"],
        })

    print(f"[TriviaQA] Sampled {len(questions)} questions")
    return questions


# ══════════════════════════════════════════════════════════════
#  PopQA
# ══════════════════════════════════════════════════════════════

def prepare_popqa(n: int = DATASET_SIZES["popqa"]) -> list[dict]:
    """
    Source: akariasai/PopQA
    Sampling: entities with Wikipedia page views < 10,000 (long-tail),
    targeting baseline hallucination rate ~30-60%.
    """
    from datasets import load_dataset

    print("[PopQA] Loading dataset …")
    ds = load_dataset("akariasai/PopQA", split="test")

    # Filter by popularity
    filtered = []
    for row in ds:
        views = row.get("s_wiki_views") or row.get("wiki_views") or 0
        answer = (row.get("possible_answers") or [""])[0] if isinstance(row.get("possible_answers"), list) else str(row.get("possible_answers", ""))
        # Try obj_label as primary answer
        obj = row.get("obj", "").strip()
        if not obj:
            continue
        if views >= 10_000:
            continue
        filtered.append({
            "question": row["question"].strip(),
            "gold_answer": obj,
            "aliases": [obj],       # PopQA usually has one canonical answer
            "wiki_views": views,
        })

    print(f"[PopQA] {len(filtered)} candidates (views < 10k)")
    sampled = random.sample(filtered, min(n, len(filtered)))

    questions = []
    for s in sampled:
        questions.append({
            "qid": f"pop_{_fingerprint(s['question'])}",
            "dataset": "popqa",
            "question": s["question"],
            "gold_answer": s["gold_answer"],
            "aliases": s["aliases"],
        })

    print(f"[PopQA] Sampled {len(questions)} questions")
    return questions


# ══════════════════════════════════════════════════════════════
#  MedQA
# ══════════════════════════════════════════════════════════════

def prepare_medqa(n: int = DATASET_SIZES["medqa"]) -> list[dict]:
    """
    Source: bigbio/med_qa  (USMLE-style, English, 4-option MCQ)
    We convert to open-ended by removing options — the model must
    answer without choices (closed-book).  Gold answer is the correct
    option text.
    """
    from datasets import load_dataset

    print("[MedQA] Loading dataset …")
    # Try the bigbio variant first; fall back to GBaker/MedQA-USMLE-4-options
    try:
        ds = load_dataset("bigbio/med_qa", name="med_qa_en_source", split="test")
        q_key, ans_key, options_key = "question", "answer", "options"
    except Exception:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        q_key, ans_key, options_key = "question", "answer", "options"

    # Collect and sample
    all_items = []
    for row in ds:
        question_text = row[q_key].strip()
        # Extract gold answer
        if isinstance(row[ans_key], dict):
            gold = row[ans_key].get("text", str(row[ans_key]))
        else:
            gold = str(row[ans_key]).strip()

        # For MCQ-style datasets, the answer might be a letter index.
        # If options are provided as a dict/list, resolve to text.
        options = row.get(options_key, {})
        if isinstance(options, dict) and len(gold) == 1 and gold.isalpha():
            gold = options.get(gold, gold)

        if not gold or not question_text:
            continue

        all_items.append({
            "question": question_text,
            "gold_answer": gold,
        })

    print(f"[MedQA] {len(all_items)} candidates")
    sampled = random.sample(all_items, min(n, len(all_items)))

    questions = []
    for s in sampled:
        questions.append({
            "qid": f"med_{_fingerprint(s['question'])}",
            "dataset": "medqa",
            "question": s["question"],
            "gold_answer": s["gold_answer"],
            "aliases": [s["gold_answer"]],
        })

    print(f"[MedQA] Sampled {len(questions)} questions")
    return questions


# ══════════════════════════════════════════════════════════════
#  Merge & Save
# ══════════════════════════════════════════════════════════════

def save_dataset(questions: list[dict], name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"  → Saved {len(questions)} questions to {path}")


def main():
    print("=" * 60)
    print("  Dataset Preparation Pipeline")
    print("=" * 60)

    tqa = prepare_triviaqa()
    save_dataset(tqa, "triviaqa")

    popqa = prepare_popqa()
    save_dataset(popqa, "popqa")

    medqa = prepare_medqa()
    save_dataset(medqa, "medqa")

    # Merged file
    all_qs = tqa + popqa + medqa
    save_dataset(all_qs, "all_questions")

    print(f"\nTotal: {len(all_qs)} questions across 3 datasets.")
    print("Done.")


if __name__ == "__main__":
    main()
