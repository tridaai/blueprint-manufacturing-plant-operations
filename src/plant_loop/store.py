"""Record store: case records and board runs, behind the Store port.

Two implementations, one behaviour:

- :class:`PostgresStore` — production. One row per case in
  ``case_records`` (migration 0001): the full loop record as a JSONB
  payload (the record the supervisor reviewed is exactly the record
  on file), with promoted columns (status, exit reason, priority)
  for board queries. Board runs live in ``board_runs``; failed runs
  leave an entry in ``run_errors`` so a dead provider call is counted
  even though it produced no record.
- :class:`InMemoryStore` — the test double. Same semantics, no
  database; it exists so tests and offline demos stay hermetic, and
  the API's readiness probe says so out loud when it is in use.

The service depends on this module's interface, never on SQL.
"""

from __future__ import annotations

import json

from . import economics as economics_module
from .config import env_str, load_dotenv
from .schemas import BoardItem, CaseResult, HistoryEntry, PartClaim, PmObligation, PmProposal

# The record type the store carries (the name ports.py refers to).
CaseRecord = CaseResult

# PM records (pm.py) are two models on one table, told apart by
# their kind: a proposal, or the obligation its approval creates.
PmRecord = PmProposal | PmObligation


def _pm_from_payload(payload: dict) -> PmRecord:
    if payload.get("kind") == "obligation":
        return PmObligation.model_validate(payload)
    return PmProposal.model_validate(payload)


class StoreBase:
    """Shared read models over the stored records."""

    def save_case(self, record: CaseResult) -> None:  # pragma: no cover - port stub
        raise NotImplementedError

    def get_case(self, case_id: str) -> CaseResult | None:  # pragma: no cover
        raise NotImplementedError

    def all_cases(self) -> list[CaseResult]:  # pragma: no cover
        raise NotImplementedError

    def save_board_run(self, run: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_board_run(self, run_id: str) -> dict | None:  # pragma: no cover
        raise NotImplementedError

    def all_board_runs(self) -> list[dict]:
        return []

    def save_board_item(self, item: BoardItem) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_board_item(self, item_id: str) -> BoardItem | None:  # pragma: no cover
        raise NotImplementedError

    def all_board_items(self) -> list[BoardItem]:
        return []

    def save_history_entry(self, entry: HistoryEntry) -> None:  # pragma: no cover
        raise NotImplementedError

    def all_history_entries(self) -> list[HistoryEntry]:
        return []

    def save_part_claim(self, claim: PartClaim) -> None:  # pragma: no cover
        raise NotImplementedError

    def all_part_claims(self) -> list[PartClaim]:
        return []

    def save_pm_record(self, record: PmRecord) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_pm_record(self, record_id: str) -> PmRecord | None:  # pragma: no cover
        raise NotImplementedError

    def all_pm_records(self) -> list[PmRecord]:
        return []

    def save_error(self, entry: dict) -> None:
        pass  # doubles may legitimately keep no error log

    def all_errors(self) -> list[dict]:
        return []

    def board(self) -> list[dict]:
        """Cases waiting at the supervisor gate, in board order.

        Each row also carries the case's downtime economics
        (economics.py), computed here from the record so the board
        shows the same figure whatever stamped it: the estimated
        minutes, and the dollar exposure — None on a line the
        plant model does not rate, and on a safety stop, which is
        never priced.
        """
        waiting = [r for r in self.all_cases() if r.status == "awaiting_supervisor"]
        waiting.sort(key=lambda r: (-r.priority, r.reported_at or "", r.case_id))
        rows = []
        for r in waiting:
            econ = economics_module.for_record(r)
            rows.append(
                {
                    "case_id": r.case_id,
                    "type": r.type,
                    "title": r.title,
                    "machine_id": r.machine_id,
                    "line_id": r.line_id,
                    "priority": r.priority,
                    "exit_reason": r.exit_reason,
                    "rounds": len(r.rounds),
                    "status": r.status,
                    "reported_at": r.reported_at,
                    "estimated_minutes": econ.estimated_minutes if econ else None,
                    "exposure_usd": econ.estimated_usd if econ else None,
                    "rate_usd_per_hour": econ.rate_usd_per_hour if econ else None,
                }
            )
        return rows

    def metrics(self) -> dict:
        records = self.all_cases()
        by_status: dict[str, int] = {}
        by_exit: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_backend: dict[str, int] = {}
        tokens_in = tokens_out = 0
        cost = 0.0
        rounds_total = 0
        for r in records:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            by_exit[r.exit_reason] = by_exit.get(r.exit_reason, 0) + 1
            by_type[r.type] = by_type.get(r.type, 0) + 1
            by_backend[r.telemetry.backend] = by_backend.get(r.telemetry.backend, 0) + 1
            tokens_in += r.telemetry.input_tokens
            tokens_out += r.telemetry.output_tokens
            cost += r.telemetry.estimated_cost_usd or 0.0
            rounds_total += len(r.rounds)
        return {
            "cases": len(records),
            "by_status": by_status,
            "by_exit": by_exit,
            "by_type": by_type,
            "by_backend": by_backend,
            "awaiting_supervisor": by_status.get("awaiting_supervisor", 0),
            "safety_stops": by_exit.get("safety_stop", 0),
            "total_rounds": rounds_total,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "estimated_cost_usd": round(cost, 6),
            "failed_runs": len(self.all_errors()),
        }


class InMemoryStore(StoreBase):
    """The test double: records in dicts, insertion order kept."""

    def __init__(self) -> None:
        self._cases: dict[str, CaseResult] = {}
        self._board_runs: dict[str, dict] = {}
        self._board_items: dict[str, BoardItem] = {}
        self._history_entries: dict[str, HistoryEntry] = {}
        self._part_claims: dict[str, PartClaim] = {}
        self._pm_records: dict[str, PmRecord] = {}
        self._errors: list[dict] = []

    def save_case(self, record: CaseResult) -> None:
        self._cases[record.case_id] = record

    def get_case(self, case_id: str) -> CaseResult | None:
        return self._cases.get(case_id)

    def all_cases(self) -> list[CaseResult]:
        return list(self._cases.values())

    def save_board_run(self, run: dict) -> None:
        self._board_runs[run["run_id"]] = run

    def get_board_run(self, run_id: str) -> dict | None:
        return self._board_runs.get(run_id)

    def all_board_runs(self) -> list[dict]:
        return list(self._board_runs.values())

    def save_board_item(self, item: BoardItem) -> None:
        self._board_items[item.item_id] = item

    def get_board_item(self, item_id: str) -> BoardItem | None:
        return self._board_items.get(item_id)

    def all_board_items(self) -> list[BoardItem]:
        return list(self._board_items.values())

    def save_history_entry(self, entry: HistoryEntry) -> None:
        # Upsert by entry id: a confirmation is one per case, and
        # re-saving the same case's entry is the same fact, not a
        # second memory.
        self._history_entries[entry.entry_id] = entry

    def all_history_entries(self) -> list[HistoryEntry]:
        return list(self._history_entries.values())

    def save_part_claim(self, claim: PartClaim) -> None:
        # Upsert by claim id: a claim is one per convergence, and
        # re-saving it (its release, stamped later) is the same
        # ledger entry changing standing, not a second claim.
        self._part_claims[claim.claim_id] = claim

    def all_part_claims(self) -> list[PartClaim]:
        return list(self._part_claims.values())

    def save_pm_record(self, record: PmRecord) -> None:
        # Upsert by record id: a proposal's decision and an
        # obligation's cycle stamps are the same record changing
        # state, not new records.
        record_id = record.proposal_id if isinstance(record, PmProposal) else record.obligation_id
        self._pm_records[record_id] = record

    def get_pm_record(self, record_id: str) -> PmRecord | None:
        return self._pm_records.get(record_id)

    def all_pm_records(self) -> list[PmRecord]:
        return list(self._pm_records.values())

    def save_error(self, entry: dict) -> None:
        self._errors.append(entry)

    def all_errors(self) -> list[dict]:
        return list(self._errors)


class PostgresStore(StoreBase):
    """Production store over psycopg; migrations applied lazily."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._migrated = False

    def _connect(self):
        import psycopg

        conn = psycopg.connect(self._url)
        if not self._migrated:
            from .migrations import run_migrations

            run_migrations(conn)
            self._migrated = True
        return conn

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def save_case(self, record: CaseResult) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO case_records (case_id, status, exit_reason, priority, payload) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (case_id) DO UPDATE SET status = EXCLUDED.status, "
                    "exit_reason = EXCLUDED.exit_reason, priority = EXCLUDED.priority, "
                    "payload = EXCLUDED.payload, updated_at = now()",
                    (
                        record.case_id,
                        record.status,
                        record.exit_reason,
                        record.priority,
                        record.model_dump_json(),
                    ),
                )
            conn.commit()

    def get_case(self, case_id: str) -> CaseResult | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM case_records WHERE case_id = %s", (case_id,))
                row = cur.fetchone()
        return CaseResult.model_validate(row[0]) if row else None

    def all_cases(self) -> list[CaseResult]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM case_records ORDER BY created_at, case_id")
                return [CaseResult.model_validate(row[0]) for row in cur.fetchall()]

    def save_board_run(self, run: dict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO board_runs (run_id, payload) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (run_id) DO UPDATE SET payload = EXCLUDED.payload",
                    (run["run_id"], json.dumps(run)),
                )
            conn.commit()

    def get_board_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM board_runs WHERE run_id = %s", (run_id,))
                row = cur.fetchone()
        return row[0] if row else None

    def all_board_runs(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM board_runs ORDER BY created_at, run_id")
                return [row[0] for row in cur.fetchall()]

    def save_board_item(self, item: BoardItem) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO board_items (item_id, kind, status, payload) "
                    "VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (item_id) DO UPDATE SET kind = EXCLUDED.kind, "
                    "status = EXCLUDED.status, payload = EXCLUDED.payload",
                    (item.item_id, item.kind, item.status, item.model_dump_json()),
                )
            conn.commit()

    def get_board_item(self, item_id: str) -> BoardItem | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM board_items WHERE item_id = %s", (item_id,))
                row = cur.fetchone()
        return BoardItem.model_validate(row[0]) if row else None

    def all_board_items(self) -> list[BoardItem]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM board_items ORDER BY created_at, item_id")
                return [BoardItem.model_validate(row[0]) for row in cur.fetchall()]

    def save_history_entry(self, entry: HistoryEntry) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO history_entries (entry_id, machine_id, fault_class, payload) "
                    "VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (entry_id) DO UPDATE SET machine_id = EXCLUDED.machine_id, "
                    "fault_class = EXCLUDED.fault_class, payload = EXCLUDED.payload",
                    (entry.entry_id, entry.machine_id, entry.fault_class, entry.model_dump_json()),
                )
            conn.commit()

    def all_history_entries(self) -> list[HistoryEntry]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM history_entries ORDER BY created_at, entry_id")
                return [HistoryEntry.model_validate(row[0]) for row in cur.fetchall()]

    def save_part_claim(self, claim: PartClaim) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO part_claims (claim_id, case_id, part_id, status, payload) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (claim_id) DO UPDATE SET case_id = EXCLUDED.case_id, "
                    "part_id = EXCLUDED.part_id, status = EXCLUDED.status, "
                    "payload = EXCLUDED.payload",
                    (claim.claim_id, claim.case_id, claim.part_id, claim.status,
                     claim.model_dump_json()),
                )
            conn.commit()

    def all_part_claims(self) -> list[PartClaim]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM part_claims ORDER BY created_at, claim_id")
                return [PartClaim.model_validate(row[0]) for row in cur.fetchall()]

    def save_pm_record(self, record: PmRecord) -> None:
        record_id = record.proposal_id if isinstance(record, PmProposal) else record.obligation_id
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO pm_records (record_id, kind, status, machine_id, fault_class, payload) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (record_id) DO UPDATE SET kind = EXCLUDED.kind, "
                    "status = EXCLUDED.status, machine_id = EXCLUDED.machine_id, "
                    "fault_class = EXCLUDED.fault_class, payload = EXCLUDED.payload",
                    (record_id, record.kind, record.status, record.machine_id,
                     record.fault_class, record.model_dump_json()),
                )
            conn.commit()

    def get_pm_record(self, record_id: str) -> PmRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM pm_records WHERE record_id = %s", (record_id,))
                row = cur.fetchone()
        return _pm_from_payload(row[0]) if row else None

    def all_pm_records(self) -> list[PmRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM pm_records ORDER BY created_at, record_id")
                return [_pm_from_payload(row[0]) for row in cur.fetchall()]

    def save_error(self, entry: dict) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO run_errors (detail) VALUES (%s::jsonb)",
                    (json.dumps(entry),),
                )
            conn.commit()

    def all_errors(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT detail FROM run_errors ORDER BY id")
                return [row[0] for row in cur.fetchall()]


def default_store() -> StoreBase:
    load_dotenv()
    url = env_str("DATABASE_URL", "")
    if url:
        return PostgresStore(url)
    return InMemoryStore()
