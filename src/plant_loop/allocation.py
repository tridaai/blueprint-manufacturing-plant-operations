"""The allocation ledger: the board shares one shelf.

Rounds 1–4 measured every case against the parts shelf as if the
case were alone on it. On a real board that is a lie the floor
pays for: two cases plan around the same last unit of a part, both
converge, and the conflict surfaces at the workbench with two work
orders in hand. This module is the ledger that ends the lie.

**Claims.** When a case's plan converges, the quantities it will
draw are claimed for that case — a claim per satisfied requirement,
on the part actually drawn (the substitute, when the plan runs on
one). Claims are made as cases reach the gate, and cases reach the
gate in board order, so claims follow priority by construction: a
later case can only ever lose a part to an earlier, higher-priority
one, and the record says exactly that.

**Effective stock.** Every parts measurement in the verify stage —
the shelf truth and the substitute search alike — reads the shelf
*net of outstanding claims* through this ledger. A case that loses
a part to a claim takes the ordinary paths (substitute search over
effective stock, else ``blocked_external``), with the gap naming
the claiming case instead of reporting a bare stockout that hides
the board's own decision.

**Release.** A claim stands while its case waits and when the
supervisor approves it (the approved work order will draw the
part). A gate rejection releases it — stamped with the reason
(:data:`RELEASE_REASON`), when, and by whom — back into the pool
for *future* runs. A release never reaches back into the finished
run it came from: runs are records, and a case that blocked on a
claim keeps that record until a fresh run re-plans it.

A claim is a planning reservation on the record. It is not a
warehouse transaction, it draws no stock, and it never marks work
performed — the gate notice on every record still governs.

Storage: claims are their own store collection (migration 0005),
the board_items / history_entries precedent — one ledger the whole
plant reads, not fragments inside case payloads. The ledger reads
through the store on every call, so a claim made by one case in a
board run is already standing when the next case in the same run
measures the shelf (the board graph persists each case before the
next is worked), and a release recorded by a later gate decision
is visible to the next run that starts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import kb
from .events import format_timestamp
from .ports import Store
from .schemas import EXIT_CONVERGED, CaseResult, PartClaim

RELEASE_REASON = "released_by_gate_rejection"


class AllocationLedger:
    """Claims over the store: read the shelf net, claim, release."""

    def __init__(self, store: Store) -> None:
        self.store = store

    # -- reads ---------------------------------------------------------------

    def all_claims(self) -> list[PartClaim]:
        return self.store.all_part_claims()

    def outstanding(self, part_id: str = "") -> list[PartClaim]:
        """Claims still standing, in the order they were recorded."""
        claims = [c for c in self.store.all_part_claims() if c.status == "claimed"]
        if part_id:
            claims = [c for c in claims if c.part_id == part_id]
        return claims

    def claims_for_case(self, case_id: str) -> list[PartClaim]:
        return [c for c in self.store.all_part_claims() if c.case_id == case_id]

    def claims_for_run(self, run_id: str) -> list[PartClaim]:
        return [c for c in self.store.all_part_claims() if c.run_id == run_id]

    def claimed_qty(self, part_id: str) -> int:
        return sum(c.qty for c in self.outstanding(part_id))

    def effective_qty(self, part_id: str) -> int | None:
        """Shelf qty net of outstanding claims (None: not stocked)."""
        on_hand = kb.part_qty(part_id)
        if on_hand is None:
            return None
        return on_hand - self.claimed_qty(part_id)

    def note_for(self, part_id: str, needed: int, *, current_run_id: str = "") -> str:
        """Why claims make this part short — or "" when they don't.

        The note names the one claiming case whose claim took the
        units this requirement needs: claims are walked in the
        order they were recorded, and the claim at which the
        remaining shelf first drops below ``needed`` is the claim
        that decided it. When the bare shelf cannot satisfy the
        requirement either, the stockout is the shelf's own fact
        and there is no claim to name — the note stays empty rather
        than blaming a claim for a shortage it did not cause.
        """
        on_hand = kb.part_qty(part_id)
        if on_hand is None or on_hand < needed:
            return ""
        remaining = on_hand
        for claim in self.outstanding(part_id):
            remaining -= claim.qty
            if remaining < needed:
                if current_run_id and claim.run_id == current_run_id:
                    return f"claimed by {claim.case_id}, higher priority on this board"
                if claim.run_id == "single":
                    return f"claimed by {claim.case_id} (single case run)"
                return f"claimed by {claim.case_id} (board run {claim.run_id})"
        return ""

    # -- writes ----------------------------------------------------------------

    def claim_from_record(self, record: CaseResult, run_id: str) -> list[PartClaim]:
        """Claim a converged record's drawn parts. Idempotent.

        Only a converged plan claims — every other exit (safety
        stops above all: they escalate at round 0 and must never
        hold a part, or a claim) leaves the ledger untouched. The
        claim id derives from the case + the part drawn, so
        persisting the same record twice claims once.
        """
        if record.exit_reason != EXIT_CONVERGED:
            return []
        existing = {c.claim_id for c in self.store.all_part_claims()}
        made: list[PartClaim] = []
        stamp = format_timestamp(datetime.now(timezone.utc))
        for req in record.plan.required_parts:
            if not req.satisfied:
                continue
            drawn = req.substitute_part_id or req.part_id
            claim_id = f"claim-{record.case_id}-{drawn}"
            if claim_id in existing:
                continue
            claim = PartClaim(
                claim_id=claim_id,
                run_id=run_id,
                case_id=record.case_id,
                part_id=drawn,
                original_part_id=req.part_id if req.substitute_part_id else "",
                qty=req.qty,
                status="claimed",
                claimed_at=stamp,
            )
            self.store.save_part_claim(claim)
            existing.add(claim_id)
            made.append(claim)
        return made

    def release_for_case(self, case_id: str, by: str = "") -> list[PartClaim]:
        """Release a rejected case's standing claims, stamped.

        Called when the supervisor rejects a case at the gate. The
        released claims return their quantities to the effective
        pool for future runs; the finished run that produced the
        blocked neighbour is a record and does not change.
        """
        stamp = format_timestamp(datetime.now(timezone.utc))
        released: list[PartClaim] = []
        for claim in self.claims_for_case(case_id):
            if claim.status != "claimed":
                continue
            claim.status = "released"
            claim.released_at = stamp
            claim.released_by = by
            claim.release_reason = RELEASE_REASON
            self.store.save_part_claim(claim)
            released.append(claim)
        return released
