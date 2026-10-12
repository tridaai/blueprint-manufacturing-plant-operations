-- Window records: planned downtime windows and the plans
-- computed for them.
--
-- A downtime window is declared by a supervisor (when it starts,
-- how long it runs, the lines and/or machines it covers), its
-- plan is computed in code over the board's records, and a
-- supervisor confirms it — possibly days after the declaration,
-- in a different request than the one that computed the plan
-- (windows.py). The plan's confirmation re-reads the state it
-- was computed over, and a second window's planning must find
-- the first window's confirmed bookings to refuse a
-- double-booking. Like the PM records (0006) and the machine
-- records (0007), this is cross-record runtime state with a
-- lifecycle of its own: riding it on a case record or a board
-- run would scatter one window across other records' payloads.
-- So window records follow the same precedent: their own table,
-- the full record as a JSONB payload, promoted columns for the
-- queries the surfaces run.
CREATE TABLE IF NOT EXISTS window_records (
    window_id   TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'proposed',
    starts_at   TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_window_records_status ON window_records (status);
