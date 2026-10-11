"""PostgreSQL-backed paths, exercised only when a database is offered.

Set TEST_DATABASE_URL (or DATABASE_URL) to a scratch database —
ideally the compose stack's — and these run: migrations apply
cleanly and idempotently, the Postgres store round-trips records,
board runs and errors, and the Postgres checkpointer pauses and
resumes a gate. Without one they skip, and the rest of the suite
proves the same behaviours against the in-memory doubles. The
production claim for these paths rests on this file having run
against a real database (see the README's honest-status note).
"""

from __future__ import annotations

import os

import pytest

DATABASE_URL = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="no TEST_DATABASE_URL/DATABASE_URL — Postgres paths unverified here"
)


@pytest.fixture()
def pg_url():
    from plant_loop.migrations import run_migrations

    run_migrations(DATABASE_URL)
    return DATABASE_URL


def test_postgres_store_roundtrip(pg_url):
    from plant_loop.samples import load_sample_cases
    from plant_loop.store import PostgresStore

    from helpers import make_service

    store = PostgresStore(pg_url)
    assert store.ping() is True
    service = make_service()
    service.store = store
    case = next(c for c in load_sample_cases() if c.case_id == "EXC-103")
    result = service.run_case(case).result
    fetched = store.get_case("EXC-103")
    assert fetched is not None and fetched.exit_reason == result.exit_reason
    assert any(row["case_id"] == "EXC-103" for row in store.board())
    assert store.metrics()["cases"] >= 1


def test_postgres_checkpointer_pauses_and_resumes(pg_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    from plant_loop.checkpoints import default_checkpointer
    from plant_loop.samples import load_sample_cases

    from helpers import make_service

    checkpointer = default_checkpointer()
    assert checkpointer is not None
    service = make_service()
    case = next(c for c in load_sample_cases() if c.case_id == "EXC-108")
    run = service.run_case(case)
    assert run.awaiting_supervisor is True
    record = service.record_decision("EXC-108", "approved", by="PG test")
    assert record.status == "approved"
