"""
Database migration runner.

On every startup, runs any SQL files in db/migrations/ that have not yet
been applied, in filename order. Applied migrations are recorded in the
_migrations table so they are never run twice.

Migration files must be named:  NNNN_description.sql  (e.g. 0003_add_foo.sql)
They should be written to be safe if run on an existing DB (IF NOT EXISTS, etc.)
but the tracking table ensures each one only runs once.

Usage (called from orchestrator/main.py lifespan):
    from db.migrate import run_migrations
    run_migrations()
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# We use psycopg2 with the direct Postgres connection string (Supabase provides this
# as DATABASE_URL in Settings → Database → Connection string → URI format).
# Format: postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres


def _get_conn():
    """Return a psycopg2 connection to Supabase Postgres."""
    import psycopg2
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL env var not set. "
            "Find it in Supabase → Settings → Database → Connection string (URI)."
        )
    return psycopg2.connect(db_url, sslmode="require")


def run_migrations() -> list[str]:
    """
    Apply all pending migrations in order. Returns list of applied file names.
    Logs each step; raises on hard error so startup fails loudly.
    """
    try:
        conn = _get_conn()
    except Exception as exc:
        log.error("migration_db_connect_failed", error=str(exc))
        # Don't crash the server over a missing DATABASE_URL — just skip.
        # Migrations can still be run manually via the Supabase SQL editor.
        return []

    applied: list[str] = []
    try:
        with conn:
            with conn.cursor() as cur:
                # Ensure tracking table exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS _migrations (
                        id         serial      PRIMARY KEY,
                        filename   text        NOT NULL UNIQUE,
                        applied_at timestamptz NOT NULL DEFAULT now()
                    )
                """)

                # Find already-applied migrations
                cur.execute("SELECT filename FROM _migrations ORDER BY filename")
                done = {row[0] for row in cur.fetchall()}

                # Collect pending files
                migration_files = sorted(
                    f for f in MIGRATIONS_DIR.glob("*.sql")
                    if re.match(r"^\d{4}_", f.name)
                )

                for path in migration_files:
                    if path.name in done:
                        log.debug("migration_already_applied", file=path.name)
                        continue

                    log.info("migration_applying", file=path.name)
                    sql = path.read_text(encoding="utf-8")
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO _migrations (filename) VALUES (%s)", (path.name,)
                    )
                    applied.append(path.name)
                    log.info("migration_applied", file=path.name)

    except Exception as exc:
        log.error("migration_failed", error=str(exc))
        raise
    finally:
        conn.close()

    if applied:
        log.info("migrations_complete", applied=applied)
    else:
        log.info("migrations_up_to_date")
    return applied
