"""Carry-over: the next shift inherits the board, as a mechanic.

Until this round a board run ended and its unfinished business
existed only as records plus handoff prose — the next shift's
board was assembled fresh, by hand, from the prose. This module
makes the handover a mechanism with the same standing as the
loops themselves:

**The manifest.** When a board run ends, its unfinished business
is computed in code (:func:`compute_manifest`) and stored on the
run record — the offer the next shift is made:

- ``not_worked`` — cases the shift budget cut before the loop
  reached them. They have no record; the run record carries their
  intake payloads (``case_payloads``), so the offer is
  self-contained.
- ``blocked`` — cases that exited ``blocked_external`` and still
  wait at the gate. The gap that blocked them is a part the loop
  could not source *then*; the world (stock, claims) may have
  moved, so these are the re-work candidates.
- ``at_gate`` — cases that reached the supervisor gate and still
  wait for a decision: converged plans, escalations, and safety
  stops. They are inherited as standing gate items — never
  re-worked, because a human is already the next step.

**Acceptance.** A new board run names its predecessor
(``carry_over_from``) or it starts clean — there is no implicit
inheritance. :func:`settle` then settles the offer against the
store *as it stands at acceptance* (decisions land between runs;
the manifest is the offer, the settlement is the truth):

- a ``blocked`` case is queued for **re-work with its loop state
  intact**: the resume seed carries its diagnosis, plan, open
  gaps, its round records and its cumulative token spend, so the
  loop re-enters at its loop-back point — the next round is round
  N+1, ``max_rounds`` bounds the case's total rounds across
  shifts, and the first resumed round re-diagnoses focused on the
  carried gaps, exactly as a loop-back round does. Nothing is
  re-derived cold; nothing is waved through either — the parts
  pass re-measures the shelf as it stands now, which is the whole
  point of re-working a stockout.
- an ``at_gate`` case is inherited standing: it consumes no budget
  in the new run, appears in the run's settlement, and its record
  is untouched.
- a ``not_worked`` case is queued as a fresh intake — it never
  ran, so there is no state to resume.
- a case decided since the run ended (approved / rejected, or
  dropped already) is **closed**: it does not carry, and the
  settlement says so.

**Claims across the boundary.** Claims are run-scoped planning
reservations (allocation.py), and the turnover adjudicates every
standing claim in the ledger — a claim is a shelf fact, not a
board fact, so a case that finished on an earlier board (or a
single run) leaves the same stale paperwork as one that finished
on this one:

- claims of accepted cases are **renewed** into the accepting
  run (``carried_from_run`` stamps the hop; still releasable by a
  gate rejection, as before);
- claims whose case's business is finished — decided and done,
  dropped at this acceptance, superseded by a fresh submission
  of the same case — are **released** with reason
  ``released_run_ended``: a part a finished board held cannot
  stay silently claimed forever;
- two exemptions: a claim whose case is **approved and still
  awaiting its outcome** keeps standing (round 5's rule — the
  approved work order will draw the part; once the outcome is
  recorded the case is done, and the next turnover releases the
  claim), and a claim whose case still waits at a gate *outside
  this board's scope* keeps standing — its own board's turnover,
  or its gate decision, will adjudicate it.

**Drops.** At acceptance the supervisor may drop a carried case
(actor + reason, audited): the case leaves the board — its record
is stamped (``dropped_from_board``, status ``dropped``) — and its
claims are released. Dropping is the only way a case leaves the
board without a gate decision, and it is never silent.

**Safety.** A safety case never re-enters the loop through
carry-over. A safety stop that is still waiting carries as an
``at_gate`` escalation record for the human; an unworked safety
case carries as a fresh intake and stops at round 0 when the new
board reaches it, exactly as it would have on the old one. The
re-work subset is ``blocked_external`` only, and the settlement
refuses — in code, not by convention — to build a resume seed
for a safety case.

**Records, not recounts.** Case records are one per case id
(store upsert), so a re-worked case's new record *replaces* its
old one: the stats and economics read each case once, at its
latest record — a case worked on two boards is one case in the
plant's numbers, never two.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .allocation import RELEASE_RUN_ENDED, AllocationLedger
from .events import format_timestamp
from .ports import ObjectStore, Store
from .schemas import (
    EXIT_BLOCKED_EXTERNAL,
    EXIT_SAFETY_STOP,
    CaseResult,
)

KIND_NOT_WORKED = "not_worked"
KIND_BLOCKED = "blocked"
KIND_AT_GATE = "at_gate"


# ---------------------------------------------------------------------------
# The manifest: a finished run's unfinished business, computed in code
# ---------------------------------------------------------------------------


def _waiting_on(record: CaseResult) -> str:
    """What this gate case is waiting on, in plain words."""
    if record.exit_reason == EXIT_SAFETY_STOP:
        return (
            "supervisor decision — SAFETY escalation; carried as a "
            "record for the human, never re-worked automatically"
        )
    if record.exit_reason == EXIT_BLOCKED_EXTERNAL:
        missing = [
            p.part_id for p in record.plan.required_parts if not p.satisfied
        ]
        parts = ", ".join(missing) if missing else "an external part"
        return (
            f"a part the shelf could not source ({parts}) — the loop "
            "re-measures the shelf when the next board accepts it"
        )
    if record.exit_reason == "converged":
        return "supervisor decision on the drafted work order"
    kinds = ", ".join(sorted({g.kind for g in record.open_gaps})) or "review"
    return f"supervisor decision on the escalation (open gaps: {kinds})"


def compute_manifest(store: Store, run_payload: dict) -> list[dict]:
    """The carry-over offer of one finished board run.

    Reads the run payload (its ``processed`` / ``remaining`` /
    ``case_payloads``) against the store's *current* records:
    a processed case enters the manifest only while it still
    waits at the gate, so a manifest computed at run end and one
    recomputed later agree on the cases that are still the
    run's unfinished business.
    """
    manifest: list[dict] = []
    payloads = run_payload.get("case_payloads") or {}
    claims_by_case: dict[str, list[str]] = {}
    for claim in store.all_part_claims():
        if claim.status == "claimed":
            claims_by_case.setdefault(claim.case_id, []).append(claim.claim_id)

    seen: set[str] = set()
    for case_id in run_payload.get("processed") or []:
        record = store.get_case(case_id)
        if record is None or record.status != "awaiting_supervisor":
            continue
        kind = (
            KIND_BLOCKED
            if record.exit_reason == EXIT_BLOCKED_EXTERNAL
            else KIND_AT_GATE
        )
        manifest.append(
            {
                "case_id": case_id,
                "kind": kind,
                "exit_reason": record.exit_reason,
                "status": record.status,
                "rounds_used": len(record.rounds),
                "waiting_on": _waiting_on(record),
                "claim_ids": claims_by_case.get(case_id, []),
                "has_payload": case_id in payloads,
            }
        )
        seen.add(case_id)
    # Inheritance chains: a gate case this run itself inherited
    # and left standing is re-offered in this run's manifest, so
    # a case nobody has decided yet is never dropped from the
    # chain just because one shift inherited it without working it.
    settled = run_payload.get("carry_over_settled") or {}
    for case_id in settled.get("inherited_at_gate") or []:
        if case_id in seen:
            continue
        record = store.get_case(case_id)
        if record is None or record.status != "awaiting_supervisor":
            continue
        manifest.append(
            {
                "case_id": case_id,
                "kind": KIND_AT_GATE,
                "exit_reason": record.exit_reason,
                "status": record.status,
                "rounds_used": len(record.rounds),
                "waiting_on": _waiting_on(record),
                "claim_ids": claims_by_case.get(case_id, []),
                "has_payload": case_id in payloads,
            }
        )
        seen.add(case_id)
    remaining = {item["case_id"]: item for item in run_payload.get("remaining") or []}
    for case_id, item in remaining.items():
        manifest.append(
            {
                "case_id": case_id,
                "kind": KIND_NOT_WORKED,
                "exit_reason": "",
                "status": "",
                "rounds_used": 0,
                "waiting_on": f"never worked — {item.get('reason', 'shift budget')}",
                "claim_ids": [],
                "has_payload": case_id in payloads,
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# Acceptance: settle the offer against the store as it stands now
# ---------------------------------------------------------------------------


def _resume_seed(record: CaseResult) -> dict:
    """The loop state a blocked case carries into its next run.

    Rounds and token spend are cumulative across shifts: the seed
    holds every round record the case has ever run, and the loop
    continues at round N+1 against the same ``max_rounds`` bound.
    """
    return {
        "diagnosis": record.diagnosis.model_dump(),
        "plan": record.plan.model_dump(),
        "gaps": [g.model_dump() for g in record.open_gaps],
        "rounds": [r.model_dump() for r in record.rounds],
        "input_tokens": sum(r.input_tokens for r in record.rounds),
        "output_tokens": sum(r.output_tokens for r in record.rounds),
        "from_exit": record.exit_reason,
    }


def _case_payload(
    store: Store,
    objects: ObjectStore | None,
    run_payload: dict,
    case_id: str,
) -> dict | None:
    """The intake payload for a carried case: the predecessor run
    record's own copy first (the contract is self-contained), the
    intake archive second (runs saved before manifests existed)."""
    payloads = run_payload.get("case_payloads") or {}
    if case_id in payloads:
        return payloads[case_id]
    if objects is not None:
        try:
            body = objects.get(f"cases/{case_id}.json")
        except Exception:
            body = None
        if body:
            try:
                return json.loads(body)
            except (ValueError, TypeError):
                return None
    return None


def settle(
    store: Store,
    objects: ObjectStore | None,
    allocation: AllocationLedger,
    run_payload: dict,
    *,
    into_run_id: str,
    drops: list[dict] | None = None,
    new_case_ids: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Settle a predecessor run's manifest for an accepting run.

    ``drops`` are ``{case_id, by, reason}`` dicts; a drop naming a
    case outside the manifest is a ``ValueError``. ``new_case_ids``
    are the accepting run's own fresh submissions — an explicit
    re-submission supersedes the carried entry for the same case.

    Returns ``(queue, settled)``: board-graph queue entries
    (``{"case": ..., "resume": ...}`` wrappers) for the cases the
    accepting run will work, and the settlement record — every
    case's disposition, and the claims renewed and released at
    the turnover.
    """
    pred_id = run_payload.get("run_id", "")
    manifest = compute_manifest(store, run_payload)
    by_id = {entry["case_id"]: entry for entry in manifest}
    drop_map = {d["case_id"]: d for d in drops or []}
    unknown_drops = sorted(set(drop_map) - set(by_id))
    if unknown_drops:
        raise ValueError(
            "cannot drop case(s) not on the carry-over manifest of "
            f"{pred_id}: {', '.join(unknown_drops)}"
        )
    new_case_ids = new_case_ids or set()
    stamp = format_timestamp(datetime.now(timezone.utc))

    settled: dict = {
        "from_run": pred_id,
        "inherited_at_gate": [],
        "reworked": [],
        "fresh": [],
        "dropped": [],
        "closed": [],
        "superseded": [],
        "unavailable": [],
        "renewed_claims": [],
        "released_claims": [],
    }
    queue: list[dict] = []
    renewed_cases: set[str] = set()

    def _renew(case_id: str) -> None:
        renewed = allocation.renew_for_case(case_id, into_run_id)
        if renewed:
            renewed_cases.add(case_id)
            settled["renewed_claims"].extend(c.claim_id for c in renewed)

    def _release(claim, by: str) -> None:
        allocation.release_claim(claim, by=by, reason=RELEASE_RUN_ENDED)
        settled["released_claims"].append(claim.claim_id)

    for entry in manifest:
        case_id = entry["case_id"]
        record = store.get_case(case_id)
        if case_id in drop_map:
            drop = drop_map[case_id]
            if record is not None:
                record.status = "dropped"
                record.dropped_from_board = {
                    "from_run": pred_id,
                    "into_run": into_run_id,
                    "by": drop.get("by", ""),
                    "reason": drop.get("reason", ""),
                    "at": stamp,
                }
                store.save_case(record)
            for claim in allocation.claims_for_case(case_id):
                if claim.status == "claimed":
                    _release(claim, by=drop.get("by", ""))
            settled["dropped"].append(case_id)
            continue
        if case_id in new_case_ids:
            # A fresh submission of the same case supersedes the
            # carried entry; its standing claims fall to the sweep.
            settled["superseded"].append(case_id)
            continue
        kind = entry["kind"]
        if kind in (KIND_BLOCKED, KIND_AT_GATE):
            if record is None or record.status != "awaiting_supervisor":
                settled["closed"].append(case_id)
                continue
            if kind == KIND_BLOCKED:
                if record.type == "safety_observation" or (
                    record.exit_reason == EXIT_SAFETY_STOP
                ):
                    # Refused in code, not by convention: a safety
                    # case is never resumed into the loop. (By
                    # construction a blocked entry cannot be one —
                    # the guard keeps the invariant local.)
                    settled["inherited_at_gate"].append(case_id)
                    _renew(case_id)
                    continue
                payload = _case_payload(store, objects, run_payload, case_id)
                if payload is None:
                    settled["unavailable"].append(case_id)
                    continue
                queue.append({"case": payload, "resume": _resume_seed(record)})
                settled["reworked"].append(case_id)
                _renew(case_id)
            else:
                settled["inherited_at_gate"].append(case_id)
                _renew(case_id)
        elif kind == KIND_NOT_WORKED:
            payload = _case_payload(store, objects, run_payload, case_id)
            if payload is None:
                settled["unavailable"].append(case_id)
                continue
            queue.append({"case": payload, "resume": None})
            settled["fresh"].append(case_id)

    # The turnover sweep: every standing claim in the ledger is
    # adjudicated, because a claim is a shelf fact, not a board
    # fact — a case that finished on an earlier board (or a
    # single run) leaves the same stale paperwork as one that
    # finished on this one. A claim survives the sweep only when
    # its case still has live business with the shelf:
    #
    # - renewed at this turnover (accepted carry-over), or
    # - approved with its outcome still outstanding — the
    #   approved work order will draw the part (round 5's rule), or
    # - still waiting at the gate and *not* part of this board's
    #   turnover (a single run's case, another board's case) —
    #   its own board's turnover, or its gate decision, will
    #   adjudicate it.
    #
    # Everything else — decided and done, rejected, dropped,
    # superseded, orphaned — is released: the case's business
    # with the shelf is over, and the part goes back in the pool.
    scope = set(by_id) | set(run_payload.get("processed") or [])
    for claim in allocation.outstanding():
        if claim.case_id in renewed_cases or claim.case_id in settled["dropped"]:
            continue  # renewed above, or released with its drop
        record = store.get_case(claim.case_id)
        if record is not None:
            if record.status == "approved" and record.outcome is None:
                continue
            if (
                record.status == "awaiting_supervisor"
                and claim.case_id not in scope
            ):
                continue
        _release(claim, by=f"board run {into_run_id}")

    return queue, settled


def settled_lines(settled: dict) -> list[str]:
    """The turnover as handoff-summary lines (accepting runs)."""
    total = (
        len(settled["inherited_at_gate"])
        + len(settled["reworked"])
        + len(settled["fresh"])
    )
    lines = [
        f"Carry-over from {settled['from_run']}: {total} case(s) inherited "
        f"({len(settled['reworked'])} re-worked with their loop history, "
        f"{len(settled['inherited_at_gate'])} standing at the gate, "
        f"{len(settled['fresh'])} not previously worked), "
        f"{len(settled['renewed_claims'])} claim(s) renewed, "
        f"{len(settled['released_claims'])} claim(s) released — the board turned over."
    ]
    if settled["dropped"]:
        lines.append(
            f"  dropped at acceptance: {', '.join(settled['dropped'])}"
        )
    if settled["closed"]:
        lines.append(
            "  closed since the previous run ended (decided at the "
            f"gate): {', '.join(settled['closed'])}"
        )
    return lines


def manifest_lines(manifest: list[dict]) -> list[str]:
    """The outgoing offer as handoff-summary lines (producing runs)."""
    if not manifest:
        return []
    ids = ", ".join(
        f"{e['case_id']} ({e['kind'].replace('_', ' ')})" for e in manifest
    )
    return [
        f"Carry-over to the next shift: {len(manifest)} case(s) — {ids}. "
        "Their claims stand until a successor run accepts the board."
    ]
