"""
Re-queue all completed indicator_research tasks so they re-run with the
updated _TEST_ASSETS (now includes EURUSD 5m) and the new regime_filter
requirement in the code-gen prompt.

Existing knowledge_base entries are NOT deleted — new results will be appended.
After re-running, each task will have additional 5m entries in the knowledge_base
plus regime-filtered variants of the 1h/4h results.

Usage:
    python scripts/requeue_research.py [--dry-run]
    python scripts/requeue_research.py --type indicator_research  # default
    python scripts/requeue_research.py --type all                 # includes general research
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from db.supabase_client import get_client


def requeue(task_type: str = "indicator_research", dry_run: bool = False) -> int:
    sb = get_client()

    # Fetch all done tasks of the given type
    q = sb.table("research_tasks").select("id, title, type, status")
    if task_type != "all":
        q = q.eq("type", task_type)
    q = q.eq("status", "done")
    tasks = q.limit(2000).execute().data or []

    print(f"Found {len(tasks)} done {task_type!r} tasks to re-queue")
    if not tasks:
        return 0

    if dry_run:
        print("[DRY RUN] Would reset these tasks to 'pending':")
        for t in tasks[:20]:
            title = t['title'][:70].encode('ascii', errors='replace').decode('ascii')
            print(f"  {t['id']}  {title}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        return len(tasks)

    # Batch reset in chunks of 100
    reset_count = 0
    chunk_size = 100
    task_ids = [t["id"] for t in tasks]

    for i in range(0, len(task_ids), chunk_size):
        chunk = task_ids[i : i + chunk_size]
        sb.table("research_tasks").update({
            "status": "pending",
            "modal_job_id": None,
            "error_log": None,
        }).in_("id", chunk).execute()
        reset_count += len(chunk)
        print(f"  Reset {reset_count}/{len(task_ids)} ...")

    print(f"\nDone. {reset_count} tasks reset to 'pending'.")
    print("The queue worker will pick them up and re-run with EURUSD 5m + regime_filter.")
    return reset_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without making changes")
    parser.add_argument("--type", default="indicator_research",
                        help="Task type to re-queue: indicator_research | all")
    args = parser.parse_args()

    n = requeue(task_type=args.type, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nTotal re-queued: {n}")
