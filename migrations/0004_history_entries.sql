-- History entries: the plant's memory of supervisor-confirmed faults.
--
-- The knowledge base (data/kb.json) is a packaged artefact: its
-- seeded maintenance history is authored ground truth and is never
-- appended to at runtime. What supervisors confirm on the floor is
-- runtime state, so it lives with the other runtime state — its own
-- table, same shape as board_items: the full entry as a JSONB
-- payload, promoted machine/fault columns for history queries.
--
-- One row per confirming case (entry_id derives from the case id,
-- and an outcome is recorded once per case), so a confirmation can
-- never be written twice. Seeded KB history stays in the KB; the
-- merged view the diagnoser reads is assembled in code
-- (history.py), confirmed entries first, each entry carrying its
-- source so the two kinds are never confused.
CREATE TABLE IF NOT EXISTS history_entries (
    entry_id    TEXT PRIMARY KEY,
    machine_id  TEXT NOT NULL DEFAULT '',
    fault_class TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_history_entries_machine ON history_entries (machine_id);
CREATE INDEX IF NOT EXISTS idx_history_entries_fault ON history_entries (fault_class);
