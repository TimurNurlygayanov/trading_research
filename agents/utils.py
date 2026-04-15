"""Shared utilities for all agents."""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-local token-per-minute rate limiter
# ---------------------------------------------------------------------------
# Anthropic limit: 8 000 output tokens / minute for claude-sonnet-4-6.
# We use 7 000 as the soft ceiling to leave a 1 000-token safety buffer.
# All LLM calls go through call_claude(), which acquires a slot here before
# hitting the API and settles it with the actual token count afterwards.
# This serialises calls within a single process (Modal container or orchestrator).

_OUTPUT_TPM_LIMIT = 7_000   # soft cap (hard limit is 8 000)
_MIN_CALL_GAP_SECS = 2.0    # minimum seconds between consecutive calls

_tpm_lock = threading.Lock()
_tpm_window: list[tuple[float, int]] = []   # (monotonic_ts, output_tokens)
_last_call_at: float = 0.0


def _expire_tpm_window() -> None:
    """Drop entries older than 60 s.  Must be called inside _tpm_lock."""
    cutoff = time.monotonic() - 60.0
    while _tpm_window and _tpm_window[0][0] < cutoff:
        _tpm_window.pop(0)


def _acquire_tpm_slot(max_tokens: int) -> None:
    """
    Block until the sliding window has room for max_tokens output tokens,
    then reserve the slot.  Releases the lock before returning.
    """
    global _last_call_at
    with _tpm_lock:
        # Enforce minimum inter-call gap
        gap = time.monotonic() - _last_call_at
        if gap < _MIN_CALL_GAP_SECS:
            time.sleep(_MIN_CALL_GAP_SECS - gap)

        _expire_tpm_window()
        used = sum(t for _, t in _tpm_window)

        while used + max_tokens > _OUTPUT_TPM_LIMIT:
            if _tpm_window:
                sleep_for = (_tpm_window[0][0] + 60.1) - time.monotonic()
                if sleep_for > 0:
                    log.info(
                        "token_budget_wait max_tokens=%d used=%d limit=%d sleep=%.1fs",
                        max_tokens, used, _OUTPUT_TPM_LIMIT, sleep_for,
                    )
                    time.sleep(sleep_for)
            else:
                time.sleep(5)
            _expire_tpm_window()
            used = sum(t for _, t in _tpm_window)

        _tpm_window.append((time.monotonic(), max_tokens))
        _last_call_at = time.monotonic()


def _settle_tpm_slot(actual_tokens: int) -> None:
    """Replace the last reservation with the call's real output token count."""
    with _tpm_lock:
        if _tpm_window:
            ts = _tpm_window[-1][0]
            _tpm_window[-1] = (ts, actual_tokens)


# ---------------------------------------------------------------------------
# Central LLM call wrapper
# ---------------------------------------------------------------------------

def call_claude(**kwargs):
    """
    Single entry point for all Anthropic API calls.

    Before each attempt:
      - Acquires a slot in the process-local token-per-minute budget
      - Enforces a minimum 2-second gap between consecutive calls

    On success:  settles the reservation with the actual output token count.
    On 429:      zeroes the reservation, waits, and re-acquires before retry.
    Retries up to 5 times with 30 → 60 → 120 s backoff.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    max_tokens = kwargs.get("max_tokens", 1024)

    _acquire_tpm_slot(max_tokens)

    wait = 30
    for attempt in range(5):
        try:
            response = client.messages.create(**kwargs)
            _settle_tpm_slot(response.usage.output_tokens)
            return response
        except anthropic.RateLimitError as exc:
            _settle_tpm_slot(0)   # free the reservation we burned
            if attempt == 4:
                raise
            log.warning(
                "rate_limit_retry wait=%ds attempt=%d/5: %s",
                wait, attempt + 1, exc,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120)
            _acquire_tpm_slot(max_tokens)   # re-acquire before next try


def add_pipeline_note(strategy_id: str, text: str) -> None:
    """Append an automated pipeline note to the strategy's comments."""
    from db import supabase_client as db
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        return
    existing = strategy.get("comments") or []
    if isinstance(existing, str):
        try:
            existing = json.loads(existing)
        except (ValueError, TypeError):
            existing = []
    ts = datetime.datetime.utcnow().isoformat()[:19]
    existing.append({"text": f"[pipeline] {text}", "ts": ts})
    db.update_strategy(strategy_id, {"comments": existing})


def full_description(strategy: dict) -> str:
    """
    Return the strategy's full description as seen by agents:
    original hypothesis + any user comments appended below.

    This ensures every agent in the pipeline has the same complete context
    regardless of when the user added comments.
    """
    parts: list[str] = []

    hypothesis = (strategy.get("hypothesis") or "").strip()
    if hypothesis:
        parts.append(hypothesis)

    comments = strategy.get("comments") or []
    if isinstance(comments, str):
        try:
            comments = json.loads(comments)
        except (ValueError, TypeError):
            comments = []

    if comments:
        lines = ["--- User comments ---"]
        for c in comments:
            ts = (c.get("ts") or "")[:10]
            text = (c.get("text") or "").strip()
            if text:
                lines.append(f"[{ts}] {text}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
