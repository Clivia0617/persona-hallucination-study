"""
Experiment runner for RQ3 — Persona Switching & Residual Effects.

Output:  results/rq3_responses.csv

Three conditions per (persona_pair, warmup_k, model):
  (i)   clean_start  — fresh conversation with Persona B
  (ii)  post_switch  — k turns Persona A, then switch to B (history kept)
  (iii) null_history — k turns neutral filler, then switch to B (length control)

Features: incremental CSV append + resume from checkpoint.
"""

import json
import copy
import logging
import pandas as pd

from config import MODELS, RESULTS_DIR, DATA_DIR
from prompts import (
    PERSONA_CONDITIONS,
    USER_PROMPT_TEMPLATE,
    RQ3_PERSONA_PAIRS,
    RQ3_WARMUP_LENGTHS,
    RQ3_WARMUP_QUESTIONS,
)
from api_client import query_with_history, ExperimentLogger
from judge import judge_factual_accuracy, judge_persona_purity
from metrics import certainty_score, count_hedge_words, response_word_count

logger = logging.getLogger(__name__)

CSV_PATH = RESULTS_DIR / "rq3_responses.csv"

CSV_COLUMNS = [
    "condition", "pair_label", "distance_label",
    "persona_a_id", "persona_b_id", "warmup_k",
    "model_key", "turn_index",
    "qid", "dataset", "question", "gold_answer",
    "content",
    "certainty_score", "hedge_count", "word_count",
    "judge_verdict", "judge_reasoning",
    "purity_verdict", "purity_reasoning",
    "model_actual", "response_id", "timestamp",
]


# ═══════════════════ Resume helpers ═══════════════════════════

def _load_completed_keys() -> set:
    """(condition, pair_label, warmup_k, model_key, qid, turn_index)"""
    if not CSV_PATH.exists():
        return set()
    try:
        cols = ["condition","pair_label","warmup_k","model_key","qid","turn_index"]
        df = pd.read_csv(CSV_PATH, usecols=cols)
        return set(zip(df["condition"], df["pair_label"], df["warmup_k"],
                       df["model_key"], df["qid"], df["turn_index"]))
    except Exception:
        return set()


def _append_csv(record: dict):
    df = pd.DataFrame([record], columns=CSV_COLUMNS)
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False, encoding="utf-8-sig")


# ═══════════════════ Conversation builders ════════════════════

def _build_warmup(system_prompt: str, warmup_qs: list[str], model_key: str) -> list[dict]:
    """Execute k warm-up turns, return full message history."""
    messages = [{"role": "system", "content": system_prompt}]
    for wq in warmup_qs:
        messages.append({"role": "user", "content": wq})
        result = query_with_history(model_key, messages)
        messages.append({"role": "assistant", "content": result["content"]})
    return messages


def _switch_system(messages: list[dict], new_system: str) -> list[dict]:
    msgs = copy.deepcopy(messages)
    msgs[0] = {"role": "system", "content": new_system}
    return msgs


# ═══════════════════ Per-condition runners ════════════════════

def _compute_metrics(content: str) -> dict:
    return {
        "certainty_score": round(certainty_score(content), 4),
        "hedge_count": count_hedge_words(content),
        "word_count": response_word_count(content),
    }


def _run_clean_start(model_key, pb_id, test_qs, pair_label, dist_label, wk, completed):
    """Condition (i): fresh conversation, Persona B only."""
    pb = PERSONA_CONDITIONS[pb_id]
    records = []
    for turn_i, q in enumerate(test_qs, 1):
        key = ("clean_start", pair_label, wk, model_key, q["qid"], turn_i)
        if key in completed:
            continue

        msgs = [
            {"role": "system", "content": pb.system_prompt},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=q["question"])},
        ]
        result = query_with_history(model_key, msgs)
        content = result["content"]

        j = judge_factual_accuracy(q["question"], q["gold_answer"], content)

        rec = {
            "condition": "clean_start", "pair_label": pair_label,
            "distance_label": dist_label,
            "persona_a_id": "", "persona_b_id": pb_id,
            "warmup_k": wk, "model_key": model_key, "turn_index": turn_i,
            "qid": q["qid"], "dataset": q["dataset"],
            "question": q["question"], "gold_answer": q["gold_answer"],
            "content": content, **_compute_metrics(content),
            "judge_verdict": j["verdict"], "judge_reasoning": j["reasoning"],
            "purity_verdict": "", "purity_reasoning": "",
            "model_actual": result["model_actual"],
            "response_id": result["response_id"], "timestamp": result["timestamp"],
        }
        _append_csv(rec)
        records.append(rec)
    return records


def _run_post_switch(model_key, pa_id, pb_id, wk, test_qs, pair_label, dist_label, completed):
    """Condition (ii): k warm-up with A, then switch to B, ask test Qs sequentially."""
    pa = PERSONA_CONDITIONS[pa_id]
    pb = PERSONA_CONDITIONS[pb_id]
    warmup_qs = RQ3_WARMUP_QUESTIONS[:wk]

    # Check if ALL turns already done — skip warm-up build if so
    all_done = all(
        ("post_switch", pair_label, wk, model_key, q["qid"], t)
        in completed
        for t, q in enumerate(test_qs, 1)
    )
    if all_done:
        return []

    logger.info(f"  构建 warm-up ({wk} 轮) Persona A={pa_id} ...")
    history = _build_warmup(pa.system_prompt, warmup_qs, model_key)
    switched = _switch_system(history, pb.system_prompt)

    records = []
    for turn_i, q in enumerate(test_qs, 1):
        key = ("post_switch", pair_label, wk, model_key, q["qid"], turn_i)
        user_prompt = USER_PROMPT_TEMPLATE.format(question=q["question"])
        switched.append({"role": "user", "content": user_prompt})
        result = query_with_history(model_key, switched)
        content = result["content"]
        switched.append({"role": "assistant", "content": content})

        if key in completed:
            continue

        j = judge_factual_accuracy(q["question"], q["gold_answer"], content)
        pj = judge_persona_purity(pa_id, pb_id, content)

        rec = {
            "condition": "post_switch", "pair_label": pair_label,
            "distance_label": dist_label,
            "persona_a_id": pa_id, "persona_b_id": pb_id,
            "warmup_k": wk, "model_key": model_key, "turn_index": turn_i,
            "qid": q["qid"], "dataset": q["dataset"],
            "question": q["question"], "gold_answer": q["gold_answer"],
            "content": content, **_compute_metrics(content),
            "judge_verdict": j["verdict"], "judge_reasoning": j["reasoning"],
            "purity_verdict": pj["verdict"], "purity_reasoning": pj["reasoning"],
            "model_actual": result["model_actual"],
            "response_id": result["response_id"], "timestamp": result["timestamp"],
        }
        _append_csv(rec)
        records.append(rec)
    return records


def _run_null_history(model_key, pb_id, wk, test_qs, pair_label, dist_label, completed):
    """Condition (iii): k neutral filler warm-up, then switch to B."""
    neutral = PERSONA_CONDITIONS["neutral_none"]
    pb = PERSONA_CONDITIONS[pb_id]
    warmup_qs = RQ3_WARMUP_QUESTIONS[:wk]

    all_done = all(
        ("null_history", pair_label, wk, model_key, q["qid"], t)
        in completed
        for t, q in enumerate(test_qs, 1)
    )
    if all_done:
        return []

    logger.info(f"  构建 null-history warm-up ({wk} 轮) ...")
    history = _build_warmup(neutral.system_prompt, warmup_qs, model_key)
    switched = _switch_system(history, pb.system_prompt)

    records = []
    for turn_i, q in enumerate(test_qs, 1):
        key = ("null_history", pair_label, wk, model_key, q["qid"], turn_i)
        user_prompt = USER_PROMPT_TEMPLATE.format(question=q["question"])
        switched.append({"role": "user", "content": user_prompt})
        result = query_with_history(model_key, switched)
        content = result["content"]
        switched.append({"role": "assistant", "content": content})

        if key in completed:
            continue

        j = judge_factual_accuracy(q["question"], q["gold_answer"], content)

        rec = {
            "condition": "null_history", "pair_label": pair_label,
            "distance_label": dist_label,
            "persona_a_id": "", "persona_b_id": pb_id,
            "warmup_k": wk, "model_key": model_key, "turn_index": turn_i,
            "qid": q["qid"], "dataset": q["dataset"],
            "question": q["question"], "gold_answer": q["gold_answer"],
            "content": content, **_compute_metrics(content),
            "judge_verdict": j["verdict"], "judge_reasoning": j["reasoning"],
            "purity_verdict": "", "purity_reasoning": "",
            "model_actual": result["model_actual"],
            "response_id": result["response_id"], "timestamp": result["timestamp"],
        }
        _append_csv(rec)
        records.append(rec)
    return records


# ═══════════════════ Full RQ3 orchestrator ════════════════════

def run_rq3_pipeline(
    questions: list[dict],
    pair_indices: list[int] | None = None,
    warmup_lengths: list[int] | None = None,
    model_keys: list[str] | None = None,
    n_post_switch_turns: int = 20,
    dry_run: bool = False,
):
    pairs = [RQ3_PERSONA_PAIRS[i] for i in (pair_indices or range(len(RQ3_PERSONA_PAIRS)))]
    wk_list = warmup_lengths or RQ3_WARMUP_LENGTHS
    mk_list = model_keys or list(MODELS.keys())
    test_qs = questions[:n_post_switch_turns]
    completed = _load_completed_keys()

    logger.info(
        f"RQ3 计划: {len(pairs)} 对 × {len(wk_list)} warmup × {len(mk_list)} 模型 × 3 条件"
    )
    logger.info(f"已完成: {len(completed)} 条")

    if dry_run:
        logger.info("DRY RUN.")
        return

    for pa_id, pb_id, dist in pairs:
        pair_label = f"{pa_id}->{pb_id}"
        for wk in wk_list:
            for mk in mk_list:
                logger.info(f">>> {pair_label}, k={wk}, model={mk}")
                _run_clean_start(mk, pb_id, test_qs, pair_label, dist, wk, completed)
                _run_post_switch(mk, pa_id, pb_id, wk, test_qs, pair_label, dist, completed)
                _run_null_history(mk, pb_id, wk, test_qs, pair_label, dist, completed)

    logger.info(f"RQ3 完成。数据: {CSV_PATH}")


# ═══════════════════ Data loader ══════════════════════════════

def load_questions(datasets=None):
    if datasets is None:
        p = DATA_DIR / "all_questions.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    questions = []
    for ds in (datasets or ["triviaqa","popqa","medqa"]):
        p = DATA_DIR / f"{ds}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                questions.extend(json.load(f))
    return questions
