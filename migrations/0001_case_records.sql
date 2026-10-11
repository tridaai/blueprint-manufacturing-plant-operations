-- Case records, one row per exception case, plus board runs and the
-- failed-run log.
--
-- The full loop record (diagnosis, plan, rounds, gaps, telemetry) is
-- stored as one JSONB payload so the record the supervisor reviewed
-- is exactly the record on file; a row is re-saved in place when the
-- supervisor decides, so the payload always carries the current
-- status. Promoted columns (status, exit, priority) exist for board
-- queries and metrics without unpacking JSON on every read.
CREATE TABLE IF NOT EXISTS case_records (
    case_id     TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'awaiting_supervisor',
    exit_reason TEXT NOT NULL DEFAULT '',
    priority    DOUBLE PRECISION NOT NULL DEFAULT 0,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_case_records_status ON case_records (status);
CREATE INDEX IF NOT EXISTS idx_case_records_exit ON case_records (exit_reason);

-- One row per board-loop run: order worked, exits, remainder, handoff.
CREATE TABLE IF NOT EXISTS board_runs (
    run_id      TEXT PRIMARY KEY,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Failed runs leave no case record; they still get counted.
CREATE TABLE IF NOT EXISTS run_errors (
    id          BIGSERIAL PRIMARY KEY,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    detail      JSONB NOT NULL
);
