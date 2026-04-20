"""
Central configuration for the Persona Injection & Hallucination study.
All experiment-wide constants live here.
"""

import os
from pathlib import Path

# ─────────────────────────── Paths ───────────────────────────
# Works both in Jupyter (no __file__) and command-line (has __file__)
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    # Running inside Jupyter — use current working directory
    PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [DATA_DIR, LOG_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# ─────────────────────────── API ─────────────────────────────
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "0000000000000000000"          # ← replace or set env var
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ─────────────────────────── Models ──────────────────────────
MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gemini-1.5-flash": "google/gemini-2.0-flash-001",
}

# Judge model (also via OpenRouter)
JUDGE_MODEL = "openai/gpt-4o-mini-2024-07-18"

# ─────────────────────────── Generation params ───────────────
GEN_PARAMS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 1024,
}

# ─────────────────────────── Experiment sizing ───────────────
REPEAT_PER_CONDITION = 10        # queries per (question, condition) pair
HUMAN_SPOT_CHECK_N = 80          # stratified sample for human verification
BOOTSTRAP_ITERATIONS = 1000

# ─────────────────────────── Dataset sizes ───────────────────
DATASET_SIZES = {
    "triviaqa": 150,
    "popqa": 150,
    "medqa": 100,
}

# ─────────────────────────── Rate limiting ───────────────────
API_MAX_RETRIES = 5
API_RETRY_BASE_DELAY = 2.0       # seconds; exponential back-off
API_CALLS_PER_MINUTE = 60        # conservative default
