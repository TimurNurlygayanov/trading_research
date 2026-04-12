"""
Agent 5: Summariser
Writes a human-readable markdown report for a completed strategy backtest.
Uploads the report to Cloudflare R2. No Linear — results are visible in the dashboard.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
import boto3
from botocore.config import Config
from dotenv import load_dotenv

from db import supabase_client as db
from agents.prompts import SUMMARISER_SYSTEM, SUMMARISER_USER_TEMPLATE

load_dotenv()
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


def run_summariser(strategy_id: str) -> dict[str, Any]:
    """
    Write a report and upload to R2.
    Updates strategy.report_url + status="done" in DB.
    """
    strategy = db.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    user_msg = _build_user_message(strategy)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SUMMARISER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    usage = response.usage
    cost = _estimate_cost(MODEL, usage.input_tokens, usage.output_tokens)
    db.log_spend("summariser", MODEL, usage.input_tokens, usage.output_tokens, cost, strategy_id)

    report_text: str = response.content[0].text.strip()
    r2_url = _upload_to_r2(strategy_id, report_text)

    db.update_strategy(strategy_id, {
        "report_url": r2_url,       # None if R2 not configured — that's fine
        "report_text": report_text,  # Always saved to DB for dashboard display
        "status": "done",
    })

    log.info(f"Summariser: strategy={strategy_id} done, report_url={r2_url}")
    return {"report_text": report_text, "r2_url": r2_url}


# ── Report content builder ────────────────────────────────────────────────────

def _build_user_message(strategy: dict) -> str:
    wf_scores = strategy.get("walk_forward_scores", "[]")
    if isinstance(wf_scores, str):
        try:
            wf_scores = json.loads(wf_scores)
        except (json.JSONDecodeError, TypeError):
            pass

    hyperparams = strategy.get("hyperparams", "{}")
    if isinstance(hyperparams, str):
        try:
            hyperparams = json.loads(hyperparams)
        except (json.JSONDecodeError, TypeError):
            pass

    risk_params = {
        "sl_atr": (hyperparams or {}).get("sl_atr", "N/A"),
        "tp_atr": (hyperparams or {}).get("tp_atr", "N/A"),
        "max_daily_losses": (hyperparams or {}).get("max_daily_losses", "N/A"),
    }

    return SUMMARISER_USER_TEMPLATE.format(
        strategy_name=strategy.get("name", strategy.get("id", "Unknown")),
        hypothesis=strategy.get("hypothesis", "N/A"),
        sharpe=_fmt(strategy.get("backtest_sharpe")),
        calmar=_fmt(strategy.get("backtest_calmar")),
        max_drawdown=_fmt(strategy.get("max_drawdown")),
        win_rate=_fmt(strategy.get("win_rate")),
        total_trades=strategy.get("total_signals", "N/A"),
        signals_per_year=_fmt(strategy.get("signals_per_year")),
        profit_factor=_fmt(strategy.get("profit_factor")),
        oos_sharpe=_fmt(strategy.get("oos_sharpe")),
        oos_win_rate=_fmt(strategy.get("oos_win_rate")),
        oos_total_trades=strategy.get("oos_total_trades", "N/A"),
        walk_forward_scores=wf_scores,
        monte_carlo_pvalue=_fmt(strategy.get("monte_carlo_pvalue")),
        leakage_score=_fmt(strategy.get("leakage_score")),
        hyperparams=hyperparams,
        best_session_hours=strategy.get("best_session_hours", "N/A"),
        risk_params=risk_params,
    )


def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


# ── Cloudflare R2 upload ──────────────────────────────────────────────────────

def _upload_to_r2(strategy_id: str, report_text: str) -> str | None:
    """
    Upload markdown report to Cloudflare R2.

    Cloudflare R2 API Token credentials:
      R2_ACCOUNT_ID        → your Cloudflare account ID (32-char hex)
      R2_ACCESS_KEY_ID     → "Access Key ID" from the R2 API token page
      R2_SECRET_ACCESS_KEY → "Secret Access Key" shown once on token creation
                             (also accepted as R2_API_KEY for convenience)
      R2_BUCKET_NAME       → bucket name, e.g. "trading-research"
    """
    account_id = os.environ.get("R2_ACCOUNT_ID")
    bucket     = os.environ.get("R2_BUCKET_NAME")
    # Support both naming conventions
    access_key = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_API_KEY")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_API_KEY")

    if not all([account_id, access_key, secret_key, bucket]):
        log.warning("R2 credentials incomplete — skipping report upload. "
                    "Need: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME")
        return None

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    file_key = f"reports/{strategy_id}/report.md"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        s3.put_object(
            Bucket=bucket,
            Key=file_key,
            Body=report_text.encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )
        r2_url = f"{endpoint_url}/{bucket}/{file_key}"
        log.info(f"Summariser: uploaded report to R2: {r2_url}")
        return r2_url
    except Exception as e:
        log.error(f"Summariser: R2 upload failed for {strategy_id}: {e}")
        return None


# ── Cost helper ───────────────────────────────────────────────────────────────

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
        "claude-sonnet-4-6":         (0.003,   0.015),
        "claude-opus-4-6":           (0.015,   0.075),
    }
    in_price, out_price = prices.get(model, (0.003, 0.015))
    return (input_tokens * in_price + output_tokens * out_price) / 1000
