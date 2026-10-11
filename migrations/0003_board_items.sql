-- Board items: the non-case entries the board carries.
--
-- A reliability_review item (raised by the recurring-fault watch in
-- reliability.py) is not an exception case and does not belong in
-- case_records: it has no loop record, no diagnosis, no plan. It gets
-- its own table with the same shape of storage — the full item as a
-- JSONB payload (the evidence list included), promoted kind/status
-- columns for board queries.
--
-- Supervisor outcomes need no table of their own: an outcome rides
-- inside the case record's JSONB payload, so the record on file is
-- still exactly the record the supervisor reviewed — now including
-- what happened next.
CREATE TABLE IF NOT EXISTS board_items (
    item_id     TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'reliability_review',
    status      TEXT NOT NULL DEFAULT 'open',
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_board_items_status ON board_items (status);
CREATE INDEX IF NOT EXISTS idx_board_items_kind ON board_items (kind);
