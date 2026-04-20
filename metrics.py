"""
Metrics computation for all three RQs.

  - Certainty Score (CS)
  - Hallucination Rate (HR)
  - Abstention Rate (AR)
  - Residual Hallucination Excess (RHE)
  - Persona Purity Score (PPS)
  - Bootstrap confidence intervals
"""

import re
import numpy as np
from collections import Counter

from config import BOOTSTRAP_ITERATIONS

# ══════════════════════════════════════════════════════════════
#  Hedge-word lexicon (~50 expressions)
# ══════════════════════════════════════════════════════════════

HEDGE_WORDS = [
    # Epistemic hedges
    "maybe", "perhaps", "possibly", "probably", "likely",
    "might", "could be", "may be", "conceivably",
    # Uncertainty markers
    "i think", "i believe", "i suppose", "i guess",
    "it seems", "it appears", "it is possible", "it is likely",
    "as far as i know", "to my knowledge", "if i recall",
    "i'm not sure", "i am not sure", "i'm not certain", "i am not certain",
    "not entirely sure", "not entirely certain",
    # Approximators
    "approximately", "roughly", "around", "about",
    "more or less", "in the ballpark",
    # Softeners
    "somewhat", "fairly", "rather", "relatively",
    "to some extent", "in a sense", "sort of", "kind of",
    # Disclaimers
    "however", "although", "that said", "on the other hand",
    "it depends", "it varies", "not necessarily",
    "there is debate", "opinions differ", "some argue",
    # Explicit uncertainty
    "uncertain", "unclear", "unknown", "debatable",
    "i don't know", "i do not know",
]

# Pre-compile for speed
_HEDGE_PATTERNS = [re.compile(r'\b' + re.escape(h) + r'\b', re.IGNORECASE)
                   for h in HEDGE_WORDS]


def count_hedge_words(text: str) -> int:
    """Count the number of hedge expressions found in text."""
    count = 0
    for pat in _HEDGE_PATTERNS:
        count += len(pat.findall(text))
    return count


def certainty_score(text: str) -> float:
    """
    CS = 1 - (hedge_count / word_count)
    Returns value in [0, 1]; higher = more certain.
    """
    words = text.split()
    if not words:
        return 0.0
    hedge_count = count_hedge_words(text)
    return max(0.0, 1.0 - hedge_count / len(words))


# ══════════════════════════════════════════════════════════════
#  Hallucination Rate
# ══════════════════════════════════════════════════════════════

def hallucination_rate(verdicts: list[str]) -> float:
    """
    HR = #incorrect / (total - #abstentions)

    verdicts: list of "correct", "incorrect", "abstain", "cannot_determine"
    cannot_determine is excluded from both numerator and denominator.
    """
    counter = Counter(verdicts)
    incorrect = counter.get("incorrect", 0)
    correct = counter.get("correct", 0)
    # denominator = total minus abstentions minus cannot_determine
    denom = correct + incorrect
    if denom == 0:
        return float("nan")
    return incorrect / denom


def abstention_rate(verdicts: list[str]) -> float:
    """AR = #abstentions / total"""
    counter = Counter(verdicts)
    total = sum(counter.values())
    if total == 0:
        return float("nan")
    return counter.get("abstain", 0) / total


# ══════════════════════════════════════════════════════════════
#  RQ3 Metrics
# ══════════════════════════════════════════════════════════════

def residual_hallucination_excess(
    hr_post_switch_t: float,
    hr_clean_start: float,
) -> float:
    """RHE_t = HR_{B|A,t} - HR_B"""
    return hr_post_switch_t - hr_clean_start


def persona_purity_score(purity_verdicts: list[str], target_persona: str = "B") -> float:
    """
    PPS = P(response classified as Persona B) among post-switch responses.
    purity_verdicts: list of "A", "B", or "NEUTRAL"
    """
    if not purity_verdicts:
        return float("nan")
    target_count = sum(1 for v in purity_verdicts if v.upper() == target_persona.upper())
    return target_count / len(purity_verdicts)


# ══════════════════════════════════════════════════════════════
#  Bootstrap Confidence Intervals
# ══════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: list | np.ndarray,
    stat_fn=np.mean,
    n_iter: int = BOOTSTRAP_ITERATIONS,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Returns (point_estimate, ci_lower, ci_upper).

    Parameters
    ----------
    values : array-like of raw observations
    stat_fn : callable applied to each bootstrap sample (default: mean)
    n_iter : number of bootstrap iterations
    ci : confidence level
    """
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    boot_stats = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats.append(stat_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, 100 * alpha)
    upper = np.percentile(boot_stats, 100 * (1 - alpha))
    point = stat_fn(arr)
    return (float(point), float(lower), float(upper))


def bootstrap_hr(verdicts: list[str], n_iter: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float, float]:
    """
    Bootstrap CI specifically for hallucination rate.
    Resamples the verdict list, computes HR each time.
    """
    rng = np.random.RandomState(42)
    arr = np.array(verdicts)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    boot_hrs = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=n, replace=True).tolist()
        boot_hrs.append(hallucination_rate(sample))

    boot_hrs = np.array(boot_hrs)
    point = hallucination_rate(verdicts)
    lower = float(np.nanpercentile(boot_hrs, 2.5))
    upper = float(np.nanpercentile(boot_hrs, 97.5))
    return (point, lower, upper)


# ══════════════════════════════════════════════════════════════
#  Response length (token-approximate)
# ══════════════════════════════════════════════════════════════

def response_word_count(text: str) -> int:
    return len(text.split())
