"""Numbered SQL migrations + the runner that applies them.

Production state lives in PostgreSQL; the schema evolves as numbered
SQL files under ``migrations/`` (``0001_...sql``, ``0002_...sql``, …),
applied in filename order by :func:`run_migrations`. A
``schema_migrations`` bookkeeping table records what has run, so
applying is idempotent and safe on every startup — the store and the
vector index both call it.

Each file runs in its own transaction together with its bookkeeping
insert: a file is either fully applied and recorded or not applied at
all. Files are plain SQL (no down-migrations — this is a reference
prototype; a production team adds reversibility when it needs it).

Migrations also run standalone: ``python -m plant_loop.migrations``
applies everything pending against ``DATABASE_URL`` — the step a
docker-compose init service, a CI job, or an operator runs before the
app starts (the app also applies them lazily on first use, so the
demo path needs no ceremony).

The directory is overridable via ``MIGRATIONS_DIR`` (tests, packaged
installs); by default it is the repo-root ``migrations/`` folder.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def migrations_dir() -> Path:
    override = os.environ.get("MIGRATIONS_DIR")
    if override:
        return Path(override)
    # src/plant_loop/migrations.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2] / "migrations"


def migration_files() -> list[Path]:
    directory = migrations_dir()
    if not directory.is_dir():
        return []
    return sorted(directory.glob("[0-9][0-9][0-9][0-9]_*.sql"))


def run_migrations(conn) -> list[str]:
    """Apply pending migrations on an open psycopg connection.

    Returns the versions applied during this call (empty when the
    schema is already current).
    """
    with conn.cursor() as cur:
        cur.execute(_ENSURE_TABLE_SQL)
        cur.execute("SELECT version FROM schema_migrations")
        applied = {row[0] for row in cur.fetchall()}
    conn.commit()

    newly_applied: list[str] = []
    for path in migration_files():
        version = path.name
        if version in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
        conn.commit()
        newly_applied.append(version)
    return newly_applied


def main() -> int:
    """``python -m plant_loop.migrations`` — apply pending migrations."""
    import psycopg

    from .config import load_dotenv

    load_dotenv()
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("error: DATABASE_URL is not set — nothing to migrate.")
        return 1
    with psycopg.connect(url) as conn:
        applied = run_migrations(conn)
    if applied:
        print(f"applied {len(applied)} migration(s): {', '.join(applied)}")
    else:
        print("schema is already current; no migrations applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
