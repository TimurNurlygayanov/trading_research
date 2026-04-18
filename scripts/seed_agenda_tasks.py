"""
Pre-seed agenda-driven research tasks immediately.

Generates up to `--per-agenda` tasks per agenda (default: 5) and inserts them
into the research_tasks table. Safe to run multiple times — skips duplicates.

Usage:
    python scripts/seed_agenda_tasks.py
    python scripts/seed_agenda_tasks.py --per-agenda 10
    python scripts/seed_agenda_tasks.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-seed agenda research tasks")
    parser.add_argument("--per-agenda", type=int, default=5,
                        help="Max tasks to generate per agenda (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without inserting")
    args = parser.parse_args()

    from agents.research_agenda import RESEARCH_AGENDAS, generate_agenda_tasks
    from db import supabase_client as db

    existing = db.get_research_tasks(status="all", limit=5000, task_type="indicator_research")
    existing_titles: set[str] = {t["title"].lower() for t in existing}

    # Show current counts per agenda
    print("Current agenda task counts:")
    for agenda in RESEARCH_AGENDAS:
        agenda_id = agenda["agenda_id"]
        count = sum(
            1 for t in existing
            if (t.get("research_spec") or {}).get("agenda_id") == agenda_id
        )
        print(f"  {agenda_id}: {count}/{agenda['n_tasks']} tasks")
    print()

    total_created = 0

    for agenda in RESEARCH_AGENDAS:
        agenda_id = agenda["agenda_id"]
        spec_type = agenda["spec_type"]
        target_n = agenda["n_tasks"]

        already = sum(
            1 for t in existing
            if (t.get("research_spec") or {}).get("agenda_id") == agenda_id
        )
        remaining = min(args.per_agenda, target_n - already)
        if remaining <= 0:
            print(f"[{agenda_id}] already at target ({already}/{target_n}), skipping")
            continue

        print(f"[{agenda_id}] generating {remaining} tasks (have {already}/{target_n})...")

        if args.dry_run:
            print(f"  (dry-run) would call Claude to generate {remaining} tasks")
            continue

        proposals = generate_agenda_tasks(agenda, n=remaining, existing_titles=existing_titles)
        print(f"  Claude returned {len(proposals)} proposals")

        created = 0
        for p in proposals:
            if not isinstance(p, dict) or not p.get("title"):
                continue
            title = f"[{agenda_id}] {p['title']}"
            if title.lower() in existing_titles:
                print(f"  skip (duplicate): {title}")
                continue
            spec = {
                "agenda_id":   agenda_id,
                "spec_id":     p.get("spec_id", f"{agenda_id}_{created}"),
                "spec_type":   spec_type,
                "indicator":   p.get("indicator", "custom"),
                "category":    p.get("category", "custom"),
                "title":       p.get("title", ""),
                "description": p.get("description", ""),
            }
            try:
                db.insert_research_task({
                    "type":          "indicator_research",
                    "title":         title,
                    "question":      spec["description"],
                    "research_spec": spec,
                    "status":        "pending",
                })
                existing_titles.add(title.lower())
                print(f"  + {title}")
                created += 1
            except Exception as exc:
                print(f"  ERROR inserting {title}: {exc}")

        print(f"  [{agenda_id}] created {created} tasks")
        total_created += created

    print()
    if args.dry_run:
        print("Dry-run complete. Re-run without --dry-run to insert.")
    else:
        print(f"Done. Total tasks created: {total_created}")
