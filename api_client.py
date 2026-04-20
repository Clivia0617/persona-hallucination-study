"""
OpenRouter API client with:
  - exponential back-off retries
  - per-minute rate limiting
  - full metadata logging (model version, latency, token usage, response id)
"""

import time
import json
import uuid
import logging
import requests
from datetime import datetime
from pathlib import Path
from threading import Lock

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    GEN_PARAMS,
    LOG_DIR,
    API_MAX_RETRIES,
    API_RETRY_BASE_DELAY,
    API_CALLS_PER_MINUTE,
)

logger = logging.getLogger(__name__)

# ─────────────────── Simple token-bucket rate limiter ────────
class RateLimiter:
    """Enforces max N calls per 60-second sliding window."""

    def __init__(self, calls_per_minute: int = API_CALLS_PER_MINUTE):
        self.calls_per_minute = calls_per_minute
        self.timestamps: list[float] = []
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            # purge timestamps older than 60 s
            self.timestamps = [t for t in self.timestamps if now - t < 60]
            if len(self.timestamps) >= self.calls_per_minute:
                sleep_for = 60 - (now - self.timestamps[0]) + 0.1
                logger.info(f"Rate limit reached, sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
            self.timestamps.append(time.time())


_limiter = RateLimiter()


# ─────────────────── Core call function ──────────────────────
def call_openrouter(
    model: str,
    messages: list[dict],
    gen_params: dict | None = None,
    extra_headers: dict | None = None,
) -> dict:
    """
    Send a chat-completion request to OpenRouter and return a rich
    result dict including the response text **and** full metadata.

    Parameters
    ----------
    model : str
        OpenRouter model identifier, e.g. "openai/gpt-4o-mini-2024-07-18".
    messages : list[dict]
        Conversation messages in OpenAI format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    gen_params : dict, optional
        Override default generation parameters from config.
    extra_headers : dict, optional
        Extra HTTP headers (e.g. HTTP-Referer for OpenRouter rankings).

    Returns
    -------
    dict with keys:
        request_id      – our UUID for this call
        model_requested – the model string we sent
        model_actual    – the model string OpenRouter actually routed to
        timestamp       – ISO 8601 UTC
        latency_s       – wall-clock seconds
        prompt_tokens   – from usage block
        completion_tokens
        total_tokens
        response_id     – OpenRouter's own id
        finish_reason
        content         – the assistant's text reply
        raw_response    – full JSON for archival
    """
    params = {**GEN_PARAMS, **(gen_params or {})}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/persona-hallucination-study",
        "X-Title": "Persona-Hallucination-Study",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        **params,
    }

    request_id = str(uuid.uuid4())
    last_err = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        _limiter.wait()
        t0 = time.time()
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            latency = time.time() - t0

            if resp.status_code == 429:
                wait = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"429 rate-limited, retry {attempt}, wait {wait:.1f}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Extract metadata
            choice = data["choices"][0]
            usage = data.get("usage", {})

            result = {
                "request_id": request_id,
                "model_requested": model,
                "model_actual": data.get("model", model),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "latency_s": round(latency, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "response_id": data.get("id", ""),
                "finish_reason": choice.get("finish_reason", ""),
                "content": choice["message"]["content"],
                "raw_response": data,
            }
            return result

        except requests.exceptions.HTTPError as e:
            last_err = e
            if resp.status_code >= 500:
                wait = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Server error {resp.status_code}, retry {attempt}, wait {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            wait = API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Connection issue, retry {attempt}, wait {wait:.1f}s")
            time.sleep(wait)
            continue

    raise RuntimeError(
        f"OpenRouter call failed after {API_MAX_RETRIES} retries. Last error: {last_err}"
    )


# ─────────────────── Logging helper ──────────────────────────
class ExperimentLogger:
    """
    Append-only JSONL logger for every API call.
    One file per experiment run; each line is a full result dict.
    """

    def __init__(self, experiment_name: str):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"{experiment_name}_{timestamp}.jsonl"
        self.path.touch()
        logger.info(f"Logging to {self.path}")

    def log(self, result: dict, extra: dict | None = None):
        """Write one record. `extra` merges in experiment-specific fields."""
        record = {**result}
        if extra:
            record.update(extra)
        # Strip raw_response to save disk (keep id only)
        record.pop("raw_response", None)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict]:
        """Read back all logged records."""
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records


# ─────────────────── Convenience wrappers ────────────────────
def query_model(
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    gen_params: dict | None = None,
) -> dict:
    """
    High-level wrapper: model_key is a short name from config.MODELS.
    Returns the full metadata dict from call_openrouter.
    """
    from config import MODELS
    model_id = MODELS[model_key]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return call_openrouter(model_id, messages, gen_params)


def query_with_history(
    model_key: str,
    messages: list[dict],
    gen_params: dict | None = None,
) -> dict:
    """
    For multi-turn conversations (RQ3).
    `messages` is a full conversation list including system message.
    """
    from config import MODELS
    model_id = MODELS[model_key]
    return call_openrouter(model_id, messages, gen_params)
