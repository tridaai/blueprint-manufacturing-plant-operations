-- PM records: preventive-maintenance proposals and obligations.
--
-- When the recurring-fault watch's pattern (reliability.py) earns
-- a preventive-maintenance proposal (pm.py), the proposal is a
-- record a supervisor decides on — possibly days later, from a
-- different request — and approval creates a standing obligation
-- that board runs weeks later must find when its due point
-- arrives. Like the part claims (0005), this is cross-record
-- runtime state with a lifecycle of its own: riding proposals on
-- the reliability board item or the obligation on a case record
-- would scatter one lifecycle across other records' payloads and
-- make "what is due?" a scan of things that are not the answer.
-- So PM records follow the board_items / history_entries /
-- part_claims precedent: their own table, the full record as a
-- JSONB payload, promoted columns for the queries the surfaces
-- run (by kind, by status, by machine + fault class).
--
-- One table, two kinds, distinguished by the kind column the way
-- board_items distinguishes its kinds: "proposal" (PROPOSED until
-- a supervisor approves or rejects it) and "obligation" (the
-- standing PM duty an approval creates). The proposal an
-- obligation came from is named inside its payload.
CREATE TABLE IF NOT EXISTS pm_records (
    record_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'proposal',
    status      TEXT NOT NULL DEFAULT 'proposed',
    machine_id  TEXT NOT NULL DEFAULT '',
    fault_class TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pm_records_kind ON pm_records (kind);
CREATE INDEX IF NOT EXISTS idx_pm_records_status ON pm_records (status);
CREATE INDEX IF NOT EXISTS idx_pm_records_machine_fault ON pm_records (machine_id, fault_class);
