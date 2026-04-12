"""
submit_idea.py — submit a trading strategy idea to the research pipeline.

Interactive usage (prompts for each field):
    python submit_idea.py

CLI usage (non-interactive):
    python submit_idea.py --title "RSI mean reversion on 4H" \
                          --description "Buy when RSI < 30 on 4H, exit at 50" \
                          --notes "Works well in ranging markets" \
                          --priority 3

The idea is inserted into the Supabase user_ideas table and the pipeline
will pick it up within the next queue worker cycle (every 10 minutes).
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


def _get_supabase_client():
    """Build a Supabase client from environment variables."""
    try:
        from supabase import create_client, Client
    except ImportError:
        print("ERROR: supabase package not installed. Run: pip install supabase")
        sys.exit(1)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        print(
            "ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must be set in your environment "
            "or .env file."
        )
        sys.exit(1)

    return create_client(url, key)


def _prompt(prompt_text: str, default: str | None = None, required: bool = True) -> str:
    """Prompt the user for a value, optionally with a default."""
    if default is not None:
        display = f"{prompt_text} [{default}]: "
    else:
        display = f"{prompt_text}: "

    while True:
        value = input(display).strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("  This field is required. Please enter a value.")


def _prompt_priority() -> int:
    """Prompt for an integer priority between 1 and 10."""
    while True:
        raw = input("Priority (1=highest, 10=lowest) [5]: ").strip()
        if not raw:
            return 5
        try:
            value = int(raw)
            if 1 <= value <= 10:
                return value
            print("  Please enter a number between 1 and 10.")
        except ValueError:
            print("  Please enter a whole number.")


def submit_idea(
    title: str,
    description: str,
    notes: str = "",
    priority: int = 5,
) -> str:
    """
    Insert a new idea into the user_ideas table.
    Returns the generated UUID of the new row.
    """
    supabase = _get_supabase_client()

    idea_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    payload = {
        "id": idea_id,
        "title": title,
        "description": description,
        "notes": notes,
        "priority": priority,
        "status": "pending",
        "created_at": now,
    }

    response = supabase.table("user_ideas").insert(payload).execute()

    if hasattr(response, "error") and response.error:
        print(f"ERROR: Supabase insert failed: {response.error}")
        sys.exit(1)

    return idea_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a trading strategy idea to the research pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--title", type=str, default=None, help="Short name for the strategy idea")
    parser.add_argument("--description", type=str, default=None, help="Hypothesis / description of the idea")
    parser.add_argument("--notes", type=str, default="", help="Optional extra notes or context")
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        choices=range(1, 11),
        metavar="1-10",
        help="Priority: 1=highest, 10=lowest (default: 5)",
    )

    args = parser.parse_args()

    # Determine whether we are in interactive or CLI mode.
    # CLI mode: all required fields supplied as arguments.
    interactive = args.title is None or args.description is None

    if interactive:
        print()
        print("=== Submit a Trading Strategy Idea ===")
        print("This will be added to the research pipeline queue.")
        print()
        title = args.title or _prompt("Strategy title")
        description = args.description or _prompt("Hypothesis / description")
        notes = args.notes or _prompt("Additional notes (optional)", default="", required=False)
        priority = args.priority if args.priority is not None else _prompt_priority()
        print()
    else:
        title = args.title
        description = args.description
        notes = args.notes
        priority = args.priority if args.priority is not None else 5

    # Submit
    print(f"Submitting idea: '{title}' (priority {priority}) ...")
    idea_id = submit_idea(title=title, description=description, notes=notes, priority=priority)

    print()
    print("Idea submitted successfully.")
    print(f"  ID       : {idea_id}")
    print(f"  Title    : {title}")
    print(f"  Priority : {priority}")
    print()
    print("The pipeline will pick it up within the next 10 minutes.")


if __name__ == "__main__":
    main()
