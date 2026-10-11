-- Part claims: the board's allocation ledger.
--
-- A claim is runtime state the whole plant shares — a later board
-- run must measure the shelf net of the claims an earlier run left
-- standing, and a gate rejection (a different request, possibly
-- days later) must find and release its case's claims. Riding the
-- claims on the case records would scatter one ledger across every
-- record that ever converged and make "what is outstanding?" a
-- full scan with the answer mutating inside other records'
-- payloads. So claims follow the board_items / history_entries
-- precedent: their own table, the full claim as a JSONB payload,
-- promoted columns for the outstanding-claims query.
--
-- One row per claim (claim_id derives from the case id + the part
-- drawn, and a case reaches the gate once per run), so a claim can
-- never be written twice for the same convergence.
CREATE TABLE IF NOT EXISTS part_claims (
    claim_id TEXT PRIMARY KEY,
    case_id  TEXT NOT NULL DEFAULT '',
    part_id  TEXT NOT NULL DEFAULT '',
    status   TEXT NOT NULL DEFAULT 'claimed',
    payload  JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_part_claims_part ON part_claims (part_id);
CREATE INDEX IF NOT EXISTS idx_part_claims_case ON part_claims (case_id);
CREATE INDEX IF NOT EXISTS idx_part_claims_status ON part_claims (status);
