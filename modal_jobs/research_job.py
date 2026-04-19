"""
Modal job: runs standalone research tasks.

A research task is a statistical investigation of market data or an indicator.
It can be:
  - Spawned by the implementer agent when it needs to understand something
    before writing strategy code (e.g., "does 1m micro-structure predict 1H?")
  - Submitted directly by the user as a standalone research question
  - Part of the article / indicator analysis pipeline (future)

Pipeline:
1. Load research_task from DB
2. Researcher LLM agent writes Python analysis code for the question
3. Execute the code in Modal's isolated env (data access via fetch_ohlcv)
4. Capture output: summary, key_findings, HTML report (optional)
5. Store results in research_tasks + knowledge_base
6. Any strategy waiting for this task will be unblocked by the queue worker
"""
import os as _os
import modal

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

app = modal.App("trading-research-research")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "pandas>=2.3.2",
        "pandas_ta==0.4.71b0",
        "numpy",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "requests==2.32.3",
        "supabase==2.9.1",
        "python-dotenv==1.0.1",
        "pyarrow",
        "massive",
        "anthropic>=0.25.0",
    )
    .add_local_dir(_os.path.join(_ROOT, "db"),       remote_path="/root/db")
    .add_local_dir(_os.path.join(_ROOT, "agents"),   remote_path="/root/agents")
    .add_local_dir(_os.path.join(_ROOT, "backtest"), remote_path="/root/backtest")
)

# Reuse the same OHLCV cache volume as the backtest job
ohlcv_cache = modal.Volume.from_name("trading-research-ohlcv-cache", create_if_missing=True)
CACHE_DIR = "/ohlcv_cache"


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
)
def run_research_task(task_id: str) -> dict:
    """
    Execute a research task:
    1. LLM writes Python analysis code for the research question
    2. Code is executed with access to fetch_ohlcv + standard data science libs
    3. Results (summary, findings, report) stored back to DB
    """
    import traceback
    import pandas as pd

    try:
        from db import supabase_client as db
        from agents.researcher import generate_research_code
        from backtest.data_fetcher import fetch_ohlcv

        task = db.get_research_task(task_id)
        if not task:
            raise ValueError(f"Research task {task_id} not found")

        db.update_research_task(task_id, {"status": "running"})

        question = task["question"]
        title = task["title"]
        data_req = task.get("data_requirements") or {}

        # 1. LLM generates analysis code
        code = generate_research_code(
            task_id=task_id,
            title=title,
            question=question,
            data_requirements=data_req,
        )

        # 2. Load data if specified
        df = None
        if data_req.get("symbol") and data_req.get("timeframe"):
            symbol = data_req["symbol"]
            tf = data_req["timeframe"]
            start = data_req.get("start", "2018-01-01")
            end = data_req.get("end", "2026-12-31")

            import os
            cache_file = f"{CACHE_DIR}/{symbol}_{tf}.parquet"
            if os.path.exists(cache_file):
                df = pd.read_parquet(cache_file)
            else:
                df = fetch_ohlcv(symbol, tf, start=start, end=end)
                os.makedirs(CACHE_DIR, exist_ok=True)
                df.to_parquet(cache_file)
                ohlcv_cache.commit()

        # 3. Validate syntax before executing (catches LLM truncation early)
        try:
            compile(code, "<research_analysis>", "exec")
        except SyntaxError as se:
            raise ValueError(
                f"Generated analysis code has a syntax error (likely truncated by token limit): "
                f"{se}"
            ) from se

        result = _execute_analysis(code, df=df, question=question)

        # 4. Persist results
        summary = result.get("summary", "No summary provided.")
        key_findings = result.get("key_findings", [])
        report_text = result.get("report_text") or result.get("html_report") or summary

        db.update_research_task(task_id, {
            "status": "done",
            "modal_job_id": None,
            "result_summary": summary,
            "key_findings": key_findings,
            "report_text": report_text,
            "generated_code": code,
            "error_log": None,
        })

        # 5. Insert key findings into knowledge_base for future agent context
        if key_findings:
            for finding in key_findings[:5]:  # cap at 5 per task
                finding_text = finding if isinstance(finding, str) else finding.get("finding", "")
                if finding_text:
                    db.insert_knowledge({
                        "category": "edge_case",
                        "indicator": task.get("type", "research"),
                        "timeframe": data_req.get("timeframe"),
                        "asset": data_req.get("symbol"),
                        "session": None,
                        "summary": f"[Research: {title}] {finding_text}",
                        "sharpe_ref": None,
                        "strategy_id": task.get("created_by_strategy_id"),
                    })

        return {
            "passed": True,
            "task_id": task_id,
            "summary": summary,
            "findings_count": len(key_findings),
        }

    except Exception as e:
        tb = traceback.format_exc()
        try:
            from db import supabase_client as db2
            db2.update_research_task(task_id, {
                "status": "failed",
                "error_log": f"{type(e).__name__}: {e}\n{tb[:500]}",
            })
        except Exception:
            pass
        raise


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("trading-research-secrets")],
    volumes={CACHE_DIR: ohlcv_cache},
)
def run_indicator_research_task(task_id: str) -> dict:
    """
    Run a systematic indicator forward-return analysis task.
    Called for research_tasks with type='indicator_research'.
    Falls back to run_research_task logic if the task has no research_spec
    (handles tasks that were misrouted due to a stale type field).
    """
    import sys
    sys.path.insert(0, "/root")

    from db import supabase_client as _db
    task = _db.get_research_task(task_id)
    if not task or not task.get("research_spec"):
        # Not a proper indicator_research task — delegate to the general researcher.
        # .local() runs the function body in-process (no new Modal spawn).
        return run_research_task.local(task_id)

    from agents.indicator_researcher import run_indicator_research
    return run_indicator_research(task_id, cache_dir=CACHE_DIR)


def _execute_analysis(code: str, df, question: str) -> dict:
    """
    Execute LLM-generated analysis code in a restricted namespace.
    The code must define run_analysis(data) and return a dict with
    'summary', 'key_findings', and optionally 'report_text'.
    """
    import io
    import sys
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats

    # Build a namespace with data science tools.
    # __import__ is required for any `import` statement inside the executed code;
    # without it every `import pandas_ta`, `import warnings`, etc. raises ImportError.
    # The code runs inside Modal's isolated container so this is safe.
    namespace = {
        "pd": pd,
        "np": np,
        "scipy_stats": scipy_stats,
        "__builtins__": __builtins__,
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    result = {}
    try:
        exec(compile(code, "<research_analysis>", "exec"), namespace)

        if "run_analysis" not in namespace:
            raise ValueError("Generated code must define run_analysis(data)")

        data = {"df": df, "question": question}
        result = namespace["run_analysis"](data)

        if not isinstance(result, dict):
            result = {"summary": str(result), "key_findings": []}

    finally:
        sys.stdout = old_stdout
        stdout_output = captured.getvalue()

    # If no summary but there was stdout output, use that
    if not result.get("summary") and stdout_output:
        result["summary"] = stdout_output[:1000]

    result.setdefault("key_findings", [])
    result.setdefault("summary", "Analysis completed — no summary returned.")
    return result
