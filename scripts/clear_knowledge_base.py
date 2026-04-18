"""
Delete all knowledge_base entries and reset done/running research tasks to pending.

Use this when the knowledge_base contains results from an outdated research pipeline
(e.g., before 5m timeframe and regime_filter were added). After running this:
  1. Merge + deploy the updated indicator_researcher.py to Modal
  2. The pipeline will re-populate the knowledge_base with correct results

Usage:
    python scripts/clear_knowledge_base.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from db.supabase_client import get_client


def main(dry_run: bool = False) -> None:
    sb = get_client()

    # ── Count knowledge_base entries ─────────────────────────────────────────
    kb_all: list[dict] = []
    offset = 0
    while True:
        batch = (
            sb.table("knowledge_base")
            .select("id, category, indicator, timeframe, asset")
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        kb_all.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    from collections import Counter
    cats = Counter(e.get("category", "?") for e in kb_all)
    print(f"Knowledge base: {len(kb_all)} entries")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:<10} {n:>5}")

    # ── Research tasks to reset ───────────────────────────────────────────────
    tasks_all: list[dict] = []
    offset = 0
    while True:
        batch = (
            sb.table("research_tasks")
            .select("id, title, status")
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        tasks_all.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    task_counts = Counter(t["status"] for t in tasks_all)
    to_reset = [t for t in tasks_all if t["status"] in ("done", "running")]
    print(f"\nResearch tasks: {sum(task_counts.values())} total")
    for s, n in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:<12} {n:>4}")
    print(f"Tasks to reset to pending: {len(to_reset)}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        print("Would delete all knowledge_base entries and reset done/running tasks to pending.")
        return

    # ── Delete all knowledge_base entries ────────────────────────────────────
    kb_ids = [e["id"] for e in kb_all]
    deleted = 0
    chunk_size = 100
    for i in range(0, len(kb_ids), chunk_size):
        chunk = kb_ids[i : i + chunk_size]
        sb.table("knowledge_base").delete().in_("id", chunk).execute()
        deleted += len(chunk)
        print(f"  KB deleted {deleted}/{len(kb_ids)} ...")

    # ── Reset done/running tasks to pending ───────────────────────────────────
    reset_ids = [t["id"] for t in to_reset]
    requeued = 0
    for i in range(0, len(reset_ids), chunk_size):
        chunk = reset_ids[i : i + chunk_size]
        sb.table("research_tasks").update({
            "status": "pending",
            "modal_job_id": None,
            "error_log": None,
        }).in_("id", chunk).execute()
        requeued += len(chunk)

    print(f"\nDone.")
    print(f"  Deleted {deleted} knowledge_base entries")
    print(f"  Reset {requeued} tasks to pending")
    print(f"\nNext steps:")
    print(f"  1. git add agents/indicator_researcher.py agents/prompts.py && git commit && git push")
    print(f"  2. Deploy updated code to Modal")
    print(f"  3. Start the queue worker — pipeline will re-run all {len(tasks_all)} tasks fresh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making changes")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
