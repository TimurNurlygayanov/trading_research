"""
Cleanup research task queue.

Problems this fixes:
  1. LLM combo generator ran repeatedly and flooded the queue with hundreds of
     near-identical variations (203 RSI+BB+OBV, 138 RSI+EMA, etc.).
  2. Failed tasks need to be re-queued so the pipeline retries them.

Strategy:
  - KEEP pending tasks whose title matches the static INDICATOR_SPECS catalogue
    (these are precise, deduplicated, cover all indicator families and parameter sweeps).
  - DELETE pending tasks not in the catalogue (LLM-generated excess / near-duplicates).
  - RE-QUEUE all 'failed' tasks back to 'pending'.
  - Leave 'done' and 'running' tasks untouched.

Usage:
    python scripts/cleanup_research_tasks.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from db.supabase_client import get_client
from agents.indicator_researcher import INDICATOR_SPECS


def main(dry_run: bool = False) -> None:
    sb = get_client()

    # ── Build the canonical title set from the static catalogue ──────────────
    canonical_titles: set[str] = {
        f"[Indicator] {spec['title']}" for spec in INDICATOR_SPECS
    }
    print(f"Static catalogue size: {len(canonical_titles)} unique titles")

    # ── Fetch all tasks (paginated) ───────────────────────────────────────────
    all_tasks: list[dict] = []
    offset = 0
    while True:
        batch = (
            sb.table("research_tasks")
            .select("id, title, status, type, created_at")
            .order("created_at", desc=False)
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        all_tasks.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    print(f"Total tasks in DB: {len(all_tasks)}")

    pending  = [t for t in all_tasks if t["status"] == "pending"]
    failed   = [t for t in all_tasks if t["status"] == "failed"]
    done     = [t for t in all_tasks if t["status"] == "done"]
    running  = [t for t in all_tasks if t["status"] == "running"]

    print(f"  pending={len(pending)}  failed={len(failed)}  done={len(done)}  running={len(running)}")

    # ── Identify pending tasks to delete ─────────────────────────────────────
    to_delete = [t for t in pending if t["title"] not in canonical_titles]
    to_keep   = [t for t in pending if t["title"] in canonical_titles]

    print(f"\nPending breakdown:")
    print(f"  Matches static catalogue -> KEEP  : {len(to_keep)}")
    print(f"  LLM-generated / not in catalogue -> DELETE: {len(to_delete)}")

    if to_delete:
        print(f"\nSample of tasks to delete (first 10):")
        for t in to_delete[:10]:
            print(f"  {t['title'][:80].encode('ascii', errors='replace').decode()}")
        if len(to_delete) > 10:
            print(f"  ... and {len(to_delete) - 10} more")

    # ── Identify failed tasks to re-queue ────────────────────────────────────
    print(f"\nFailed tasks to re-queue: {len(failed)}")
    for t in failed[:5]:
        print(f"  {t['title'][:70].encode('ascii', errors='replace').decode()}")
    if len(failed) > 5:
        print(f"  ... and {len(failed) - 5} more")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # ── Delete LLM-generated pending tasks ───────────────────────────────────
    delete_ids = [t["id"] for t in to_delete]
    deleted = 0
    chunk_size = 100
    for i in range(0, len(delete_ids), chunk_size):
        chunk = delete_ids[i : i + chunk_size]
        sb.table("research_tasks").delete().in_("id", chunk).execute()
        deleted += len(chunk)
        print(f"  Deleted {deleted}/{len(delete_ids)} ...")

    # ── Re-queue failed tasks ─────────────────────────────────────────────────
    failed_ids = [t["id"] for t in failed]
    requeued = 0
    for i in range(0, len(failed_ids), chunk_size):
        chunk = failed_ids[i : i + chunk_size]
        sb.table("research_tasks").update({
            "status": "pending",
            "modal_job_id": None,
            "error_log": None,
        }).in_("id", chunk).execute()
        requeued += len(chunk)

    print(f"\nDone.")
    print(f"  Deleted  : {deleted} LLM-generated pending tasks")
    print(f"  Re-queued: {requeued} failed tasks to pending")
    print(f"  Remaining pending (catalogue tasks): {len(to_keep)}")
    print(f"  Done tasks untouched: {len(done)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making changes")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
