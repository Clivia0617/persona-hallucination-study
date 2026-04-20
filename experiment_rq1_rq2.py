"""
Experiment runner for RQ1/RQ2 — Static Persona Experiment.

Output:   results/rq1_rq2_responses.csv
          results/human_spotcheck_sample.csv

Features:
  - Incremental CSV append (each record saved immediately, crash-safe)
  - Resume from checkpoint (skips already-completed calls on restart)
"""

import json
import logging
import pandas as pd
from pathlib import Path

from config import MODELS, REPEAT_PER_CONDITION, RESULTS_DIR, DATA_DIR
from prompts import (
    PERSONA_CONDITIONS,
    ALL_CONDITION_IDS,
    CONFIDENCE_ORDINAL,
    USER_PROMPT_TEMPLATE,
)
from api_client import query_model, ExperimentLogger
from judge import judge_factual_accuracy
from metrics import certainty_score, count_hedge_words, response_word_count

logger = logging.getLogger(__name__)

CSV_PATH = RESULTS_DIR / "rq1_rq2_responses.csv"

CSV_COLUMNS = [
    "qid", "dataset", "question", "gold_answer",
    "condition_id", "persona_category", "confidence_level", "confidence_ordinal",
    "model_key", "repeat_index",
    "content",
    "certainty_score", "hedge_count", "word_count",
    "judge_verdict", "judge_reasoning",
    "model_actual", "response_id", "latency_s",
    "prompt_tokens", "completion_tokens", "timestamp",
]


# ═══════════════════ Resume helpers ═══════════════════════════

def _load_completed_keys() -> set:
    """Load set of (qid, condition_id, model_key, repeat_index) already done."""
    if not CSV_PATH.exists():
        return set()
    try:
        df = pd.read_csv(CSV_PATH, usecols=["qid","condition_id","model_key","repeat_index"])
        keys = set(zip(df["qid"], df["condition_id"], df["model_key"], df["repeat_index"]))
        logger.info(f"断点续跑: 已完成 {len(keys)} 条记录")
        return keys
    except Exception:
        return set()


def _append_csv(record: dict):
    """Append one record to CSV (create with header if new)."""
    df = pd.DataFrame([record], columns=CSV_COLUMNS)
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False, encoding="utf-8-sig")


# ═══════════════════ Core experiment ══════════════════════════

def run_rq12_pipeline(
    questions: list[dict],
    condition_ids: list[str] = ALL_CONDITION_IDS,
    model_keys: list[str] | None = None,
    repeats: int = REPEAT_PER_CONDITION,
    skip_judge: bool = False,
    dry_run: bool = False,
):
    """
    Main entry point. For each (question, condition, model, repeat):
      1) Call model via OpenRouter
      2) Call LLM-as-judge
      3) Compute style metrics
      4) Append to CSV immediately
    Automatically skips records that already exist in the CSV.
    """
    model_keys = model_keys or list(MODELS.keys())
    completed = _load_completed_keys()

    total = len(questions) * len(condition_ids) * len(model_keys) * repeats
    remaining = total - len(completed)
    logger.info(
        f"实验计划: {len(questions)} 题 × {len(condition_ids)} 条件 "
        f"× {len(model_keys)} 模型 × {repeats} 重复 = {total} 总调用"
    )
    logger.info(f"已完成: {len(completed)}, 剩余: {remaining}")

    if dry_run:
        logger.info("DRY RUN — 不会发起 API 调用。")
        return

    exp_logger = ExperimentLogger("rq1_rq2")
    done = 0

    for q in questions:
        user_prompt = USER_PROMPT_TEMPLATE.format(question=q["question"])
        for cond_id in condition_ids:
            cond = PERSONA_CONDITIONS[cond_id]
            for mk in model_keys:
                for rep in range(repeats):
                    key = (q["qid"], cond_id, mk, rep)
                    if key in completed:
                        continue

                    done += 1
                    if done % 50 == 0 or done == 1:
                        logger.info(f"进度: {done}/{remaining}")

                    # Model call
                    try:
                        result = query_model(mk, cond.system_prompt, user_prompt)
                    except Exception as e:
                        logger.error(f"API 错误 {key}: {e}")
                        continue

                    content = result["content"]

                    # Judge call
                    verdict, reasoning = "", ""
                    if not skip_judge:
                        try:
                            j = judge_factual_accuracy(q["question"], q["gold_answer"], content)
                            verdict, reasoning = j["verdict"], j["reasoning"]
                        except Exception as e:
                            logger.error(f"Judge 错误 {key}: {e}")
                            verdict = "judge_error"

                    record = {
                        "qid": q["qid"], "dataset": q["dataset"],
                        "question": q["question"], "gold_answer": q["gold_answer"],
                        "condition_id": cond_id,
                        "persona_category": cond.category,
                        "confidence_level": cond.confidence,
                        "confidence_ordinal": CONFIDENCE_ORDINAL.get(cond.confidence, 0),
                        "model_key": mk, "repeat_index": rep,
                        "content": content,
                        "certainty_score": round(certainty_score(content), 4),
                        "hedge_count": count_hedge_words(content),
                        "word_count": response_word_count(content),
                        "judge_verdict": verdict, "judge_reasoning": reasoning,
                        "model_actual": result["model_actual"],
                        "response_id": result["response_id"],
                        "latency_s": result["latency_s"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "timestamp": result["timestamp"],
                    }

                    _append_csv(record)
                    exp_logger.log(record)

    logger.info(f"RQ1/RQ2 完成。新增 {done} 条。数据: {CSV_PATH}")


# ═══════════════════ Human spot-check sampler ═════════════════

def generate_spotcheck_sample(n: int = 80, seed: int = 42):
    """
    Stratified sample (by persona_category × model_key) for human verification.
    Outputs CSV with blank human_verdict column.
    """
    df = pd.read_csv(CSV_PATH)
    groups = df.groupby(["persona_category", "model_key"])
    n_groups = groups.ngroups
    per_group = max(1, n // n_groups)

    sampled = groups.apply(
        lambda g: g.sample(n=min(per_group, len(g)), random_state=seed)
    ).reset_index(drop=True)

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)

    out = sampled[["qid","dataset","question","gold_answer",
                    "condition_id","model_key","content","judge_verdict"]].copy()
    out["human_verdict"] = ""

    path = RESULTS_DIR / "human_spotcheck_sample.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"人工抽检样本 ({len(out)} 条): {path}")
    return path


# ═══════════════════ Data loader ══════════════════════════════

def load_questions(datasets: list[str] | None = None) -> list[dict]:
    if datasets is None:
        path = DATA_DIR / "all_questions.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        datasets = ["triviaqa", "popqa", "medqa"]
    questions = []
    for ds in datasets:
        p = DATA_DIR / f"{ds}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                questions.extend(json.load(f))
    return questions
