-- Machine records: a machine's recorded lifecycle state.
--
-- The plant model (kb.json) is authored data: it says which
-- machines exist, and it never changes at runtime. Whether a
-- machine still stands on the floor is a runtime fact with a
-- lifecycle of its own — a supervisor decommissions a machine
-- (lifecycle.py), the act retires the machine's standing PM
-- obligations and lists its open cases for disposition, and the
-- record must be findable by every later request: the intake
-- guard that refuses new work for a dead machine, the board run
-- past a retired obligation's due point, the handoff that keeps
-- the retirement readable. Like the PM records (0006), this is
-- cross-record runtime state: riding it on a case record or a
-- board item would scatter one lifecycle across other records'
-- payloads. So machine records follow the same precedent: their
-- own table, the full record as a JSONB payload, promoted
-- columns for the queries the surfaces run.
--
-- One row per machine, ever: a machine is decommissioned once,
-- and the row is the permanent record of that act (its payload
-- also accumulates the case dispositions as they are chosen).
CREATE TABLE IF NOT EXISTS machine_records (
    record_id   TEXT PRIMARY KEY,
    machine_id  TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'decommissioned',
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_machine_records_status ON machine_records (status);
