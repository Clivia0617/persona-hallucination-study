"""
LLM-as-Judge evaluation pipeline.

Two judges:
  1. Factual accuracy judge  → correct / incorrect / abstain / cannot_determine
  2. Persona purity judge    → A / B / NEUTRAL  (RQ3 only)
"""

import re
import logging

from api_client import call_openrouter
from config import JUDGE_MODEL, GEN_PARAMS
from prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    PURITY_JUDGE_SYSTEM_PROMPT,
    PURITY_JUDGE_USER_TEMPLATE,
    PERSONA_CONDITIONS,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  Verdict extraction
# ══════════════════════════════════════════════════════════════

_VERDICT_RE = re.compile(
    r"VERDICT:\s*(correct|incorrect|abstain|cannot_determine)",
    re.IGNORECASE,
)

_PURITY_RE = re.compile(
    r"VERDICT:\s*(A|B|NEUTRAL)",
    re.IGNORECASE,
)


def _extract(pattern, text: str, default: str = "cannot_determine") -> str:
    """Extract last match of pattern from judge response."""
    matches = pattern.findall(text)
    if not matches:
        logger.warning(f"Could not parse verdict from judge output: {text[:200]}")
        return default
    return matches[-1].lower()


# ══════════════════════════════════════════════════════════════
#  Factual accuracy judge
# ══════════════════════════════════════════════════════════════

def judge_factual_accuracy(
    question: str,
    gold_answer: str,
    model_response: str,
) -> dict:
    """
    Call the LLM judge and return structured result.

    Returns
    -------
    dict with keys:
        verdict    : "correct" | "incorrect" | "abstain" | "cannot_determine"
        reasoning  : full judge response text
        judge_meta : API metadata (model, latency, tokens, etc.)
    """
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        model_response=model_response,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = call_openrouter(
        model=JUDGE_MODEL,
        messages=messages,
        gen_params={**GEN_PARAMS, "max_tokens": 512},
    )

    verdict = _extract(_VERDICT_RE, result["content"])

    return {
        "verdict": verdict,
        "reasoning": result["content"],
        "judge_meta": {
            "model_actual": result["model_actual"],
            "latency_s": result["latency_s"],
            "tokens": result["total_tokens"],
            "response_id": result["response_id"],
        },
    }


# ══════════════════════════════════════════════════════════════
#  Persona purity judge (RQ3)
# ══════════════════════════════════════════════════════════════

def judge_persona_purity(
    persona_a_id: str,
    persona_b_id: str,
    model_response: str,
) -> dict:
    """
    Classify whether a post-switch response is more consistent
    with Persona A or Persona B.

    Returns
    -------
    dict with keys:
        verdict   : "a" | "b" | "neutral"
        reasoning : full judge text
        judge_meta
    """
    pa = PERSONA_CONDITIONS[persona_a_id]
    pb = PERSONA_CONDITIONS[persona_b_id]

    user_prompt = PURITY_JUDGE_USER_TEMPLATE.format(
        persona_a_desc=f"[{pa.category}, {pa.confidence}] {pa.system_prompt}",
        persona_b_desc=f"[{pb.category}, {pb.confidence}] {pb.system_prompt}",
        model_response=model_response,
    )

    messages = [
        {"role": "system", "content": PURITY_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = call_openrouter(
        model=JUDGE_MODEL,
        messages=messages,
        gen_params={**GEN_PARAMS, "max_tokens": 512},
    )

    verdict = _extract(_PURITY_RE, result["content"], default="neutral")

    return {
        "verdict": verdict,
        "reasoning": result["content"],
        "judge_meta": {
            "model_actual": result["model_actual"],
            "latency_s": result["latency_s"],
            "tokens": result["total_tokens"],
            "response_id": result["response_id"],
        },
    }


# ══════════════════════════════════════════════════════════════
#  Batch judge helper
# ══════════════════════════════════════════════════════════════

def batch_judge_accuracy(records: list[dict]) -> list[dict]:
    """
    Judge a batch of experiment records.

    Each record must have keys: question, gold_answer, content (model response).
    Returns the same records augmented with judge_verdict, judge_reasoning,
    and judge_meta.
    """
    results = []
    for i, rec in enumerate(records):
        if i % 50 == 0:
            logger.info(f"Judging record {i}/{len(records)} …")
        j = judge_factual_accuracy(
            question=rec["question"],
            gold_answer=rec["gold_answer"],
            model_response=rec["content"],
        )
        results.append({
            **rec,
            "judge_verdict": j["verdict"],
            "judge_reasoning": j["reasoning"],
            "judge_meta": j["judge_meta"],
        })
    return results
