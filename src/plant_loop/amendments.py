"""Gate amendments: the supervisor edits, the machine re-checks.

The supervisor gate used to offer two moves: approve the plan as
proposed, or reject it and send the case back around the loop.
The common case on a real floor is smaller than either — swap
this part for the one on the shelf, drop the step that does not
apply here. This module is that third path, built so the small
fix cannot become a quiet weakening of the whole discipline:

- **An amendment is a record, not an edit.** It names its kind
  (``part_swap`` / ``step_strike``), its supervisor and its
  reason, and it appends to the case's record. The loop's
  original plan is never modified — the amended plan every
  surface prints is *derived* by replaying the passing
  amendments over it (:func:`apply_passing_amendments`), so the
  record always shows both what the loop proposed and what the
  supervisor changed, by whom, and why.
- **Re-verification is code, run before any approval.** Recording
  an amendment immediately re-runs the verify stage's code
  checks over the amended plan: :func:`guardrails.verify_plan`
  (mandatory safety steps complete, parts real, no banned
  actions, no SOP contradictions) with each requirement
  re-measured against the shelf net of *other* cases' claims —
  plus, for a swap, the same SOP-constraint test the substitute
  search applies (:func:`substitution.constraint_violations`).
  No model is consulted: the critic already spoke in the loop;
  this is arithmetic and rules. A plan that fails sits at the
  gate as AMENDMENT_FAILED with the failing checks on its face,
  and the approve path refuses it — the supervisor may amend
  again, or reject.
- **Claims follow the verified plan.** A passing swap moves the
  case's allocation in the same act (allocation.py): the old
  part's claim is released stamped ``released_by_amendment``,
  the new part is claimed under the case's run. A failed
  amendment moves nothing.
- **The vocabulary is deliberately small.** A swap names a part
  from the plant register *by id* — a free-text part is refused
  at recording, because a part the register does not carry
  cannot be checked against a shelf, a spec sheet, or an SOP. A
  strike names a step that is not mandatory — striking a
  mandatory safety step is refused, naming the step and its SOP:
  an amendment may redirect a plan, it may never thin its
  safety steps. And there is no way to *add* a step: an added
  step has no diagnosis behind it. The supervisor directs the
  work; the planner proposes it; new steps come from the loop
  on a fresh run, not from the gate.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import guardrails, kb, substitution
from .allocation import AllocationLedger
from .events import format_timestamp
from .schemas import (
    EXIT_SAFETY_STOP,
    ActionPlan,
    AmendmentVerification,
    CaseResult,
    ExceptionCase,
    Gap,
    PlanAmendment,
    SpecDelta,
)

PART_SWAP = "part_swap"
STEP_STRIKE = "step_strike"
AMENDMENT_KINDS = (PART_SWAP, STEP_STRIKE)

VERIFIED = "passed"
FAILED = "failed"


def _now() -> str:
    return format_timestamp(datetime.now(timezone.utc))


def drawn_part(req) -> str:
    """The part a requirement actually draws (its substitute, when
    the plan runs on one) — the id a swap names as its ``from``."""
    return req.substitute_part_id or req.part_id


def amendment_state(record: CaseResult) -> str:
    """The case's amendment face: ``none`` until one is recorded,
    then the latest amendment's verification verdict — ``verified``
    (the amended plan is the plan at the gate) or
    ``amendment_failed`` (the latest attempt failed re-verification
    and approval is refused until it is resolved)."""
    if not record.amendments:
        return "none"
    latest = record.amendments[-1]
    return "verified" if latest.verification.status == VERIFIED else "amendment_failed"


def latest_failed_amendment(record: CaseResult) -> PlanAmendment | None:
    """The amendment blocking approval, when one is: the latest
    recorded amendment, if its re-verification failed."""
    if record.amendments and record.amendments[-1].verification.status == FAILED:
        return record.amendments[-1]
    return None


# -- the amended plan --------------------------------------------------------


def _apply_one(plan: ActionPlan, amendment: PlanAmendment) -> None:
    """Apply one verified amendment to a plan copy, marked as made."""
    if amendment.kind == PART_SWAP:
        for req in plan.required_parts:
            if req.part_id == amendment.slot_part_id:
                req.substitute_part_id = (
                    "" if amendment.to_part_id == amendment.slot_part_id
                    else amendment.to_part_id
                )
                req.available_qty = amendment.to_available_qty
                req.satisfied = True  # the re-verification measured it so
        for step in plan.steps:
            if step.part_id == amendment.from_part_id:
                step.part_id = amendment.to_part_id
                step.text += (
                    f" [AMENDED AT THE GATE: {amendment.to_part_id} in place of "
                    f"{amendment.from_part_id} — {amendment.amendment_id} "
                    f"by {amendment.by}]"
                )
    elif amendment.kind == STEP_STRIKE:
        plan.steps = [s for s in plan.steps if s.step_id != amendment.step_id]


def apply_passing_amendments(record: CaseResult) -> ActionPlan:
    """The plan as it stands at the gate: the loop's original plan
    with every *passing* amendment replayed over it, in order.

    A failed amendment contributes nothing — it was tried, the
    checks refused it, and the plan the supervisor is looking at
    is the last one that verified. The original on the record is
    never touched; this returns a deep copy.
    """
    plan = record.plan.model_copy(deep=True)
    for amendment in record.amendments:
        if amendment.verification.status == VERIFIED:
            _apply_one(plan, amendment)
    return plan


# -- spec deltas ---------------------------------------------------------------


def _spec_deltas(slot_part_id: str, to_part_id: str) -> list[SpecDelta]:
    """The spec differences between the slot's own part and the
    swapped-in part, computed from the two spec sheets — the
    deltas for a pairing the KB's compatibility relation does not
    author. An attribute one side does not carry reads as None:
    an unverifiable rating is shown as absent, never as equal."""
    original = kb.part_specs(slot_part_id)
    candidate = kb.part_specs(to_part_id)
    deltas: list[SpecDelta] = []
    for attribute in sorted(set(original) | set(candidate)):
        before, after = original.get(attribute), candidate.get(attribute)
        if before != after:
            deltas.append(
                SpecDelta(attribute=attribute, original=before, substitute=after)
            )
    return deltas


def _swap_paperwork(record: CaseResult, slot_part_id: str, to_part_id: str):
    """A swap's compatibility basis + spec deltas, from the KB
    when the relation authors them, computed when it does not.

    Three honest shapes: the swap returns the plan to the slot's
    own part (the part the SOP names — its basis is itself); the
    pairing is KB-listed for the slot (the authored basis and
    deltas ride, exactly as a verify-stage substitution's do);
    or the pairing is the supervisor's own — stated plainly as
    unlisted, with the deltas computed from the spec sheets, so
    the record never dresses a human's judgment as the
    catalogue's.
    """
    if to_part_id == slot_part_id:
        return (
            "the slot's own part — the amendment returns the plan to the part "
            "the SOP names for this slot",
            _spec_deltas(slot_part_id, to_part_id),
        )
    top = record.diagnosis.top()
    sop = kb.sop_for(record.machine_id, top.fault_class) if top else None
    for entry in kb.substitute_entries(sop, slot_part_id):
        if entry["part_id"] == to_part_id:
            return (
                entry["basis"],
                [SpecDelta.model_validate(d) for d in entry["spec_deltas"]],
            )
    return (
        f"supervisor-specified pairing — {to_part_id} is not listed in the KB "
        f"compatibility relation for the {slot_part_id} slot; compatibility is "
        "judged on the SOP's constraints for the slot and the shelf, and the "
        "recorded supervisor carries the pairing",
        _spec_deltas(slot_part_id, to_part_id),
    )


# -- re-verification -----------------------------------------------------------


def _remeasure(plan: ActionPlan, ledger: AllocationLedger, case_id: str) -> None:
    """Shelf truth for an amended plan, in place: every
    requirement re-measured against the shelf net of other cases'
    claims — the same effective shelf the verify stage measures,
    with the case's own claims set aside (they are the plan being
    re-measured, not contention against it). No substitute search
    runs: the supervisor's swap is the proposal being checked,
    and the machine does not improve it into a different one."""
    for req in plan.required_parts:
        effective = ledger.effective_qty_excluding(drawn_part(req), case_id)
        req.available_qty = effective if effective is not None else 0
        req.satisfied = effective is not None and effective >= req.qty


def reverify(
    record: CaseResult,
    candidate: ActionPlan,
    ledger: AllocationLedger,
    *,
    swap: PlanAmendment | None = None,
) -> tuple[list[Gap], str]:
    """The code checks over an amended plan. Returns (gaps, detail).

    ``guardrails.verify_plan`` is the whole jury for the plan's
    shape; a swap adds the SOP-constraint test for its slot, with
    failures recorded as ``incompatible_part`` gaps carrying the
    constraint's own words, and a shortfall a claim caused naming
    the claiming case (allocation.py's note) on the gap.
    """
    case = ExceptionCase(
        case_id=record.case_id, type=record.type, title=record.title,
        machine_id=record.machine_id, line_id=record.line_id,
    )
    gaps = guardrails.verify_plan(case, record.diagnosis, candidate)
    if swap is not None and swap.kind == PART_SWAP:
        top = record.diagnosis.top()
        sop = kb.sop_for(record.machine_id, top.fault_class) if top else None
        for violation in substitution.constraint_violations(
            sop, swap.slot_part_id, swap.to_part_id
        ):
            gaps.append(
                Gap(
                    kind="incompatible_part",
                    detail=violation,
                    ref=swap.to_part_id,
                    blocking=True,
                )
            )
        note = ledger.note_for(swap.to_part_id, swap.qty)
        for gap in gaps:
            if gap.kind != "part_unavailable":
                continue
            if gap.ref == swap.slot_part_id and swap.to_part_id != swap.slot_part_id:
                # verify_plan words the gap for the slot's part, but
                # the availability it measured is the drawn part's —
                # say which part the amended plan actually draws, so
                # the shortage is read against the right shelf.
                gap.detail = (
                    f"{gap.detail} (the amended plan draws "
                    f"{swap.to_part_id} for the {swap.slot_part_id} slot)"
                )
            if (
                note
                and gap.ref in (swap.slot_part_id, swap.to_part_id)
                and "claimed by" not in gap.detail
            ):
                gap.detail = f"{gap.detail} — {note}"
    blocking = [g for g in gaps if g.blocking]
    if blocking:
        detail = (
            f"re-verification FAILED — {len(blocking)} blocking check(s): "
            + "; ".join(g.detail for g in blocking)
        )
    else:
        detail = (
            "re-verification passed — the verify stage's code checks find "
            "no gaps in the amended plan"
        )
    return gaps, detail


# -- recording -------------------------------------------------------------------


def build_amendment(
    record: CaseResult,
    ledger: AllocationLedger,
    *,
    kind: str,
    by: str = "",
    reason: str = "",
    from_part_id: str = "",
    to_part_id: str = "",
    step_id: str = "",
) -> PlanAmendment:
    """Validate, build and re-verify one amendment for a gated case.

    Recording refusals raise :class:`ValueError` and write
    nothing: the case is not at the gate, the actor or the reason
    is missing, the part is not on the register, the slot or step
    named is not in the plan, or the step is a mandatory safety
    step. A built amendment always carries its re-verification —
    passed or failed, the attempt and the verdict both land on
    the record (the caller appends and persists).
    """
    if record.status != "awaiting_supervisor":
        raise ValueError(
            f"case {record.case_id} is {record.status} — amendments are "
            "recorded at the supervisor gate, on a case still awaiting "
            "its decision"
        )
    if record.exit_reason == EXIT_SAFETY_STOP or (
        not record.plan.steps and not record.plan.required_parts
    ):
        raise ValueError(
            f"case {record.case_id} is a safety stop — it has no plan to "
            "amend; it waits for the supervisor's decision itself"
        )
    if not by or not by.strip():
        raise ValueError("an amendment must name its supervisor")
    if not reason or not reason.strip():
        raise ValueError(
            "an amendment must carry the supervisor's reason — the record "
            "is how the next shift knows why the plan changed"
        )
    if kind not in AMENDMENT_KINDS:
        raise ValueError(
            f"amendment kind must be one of {', '.join(AMENDMENT_KINDS)}, "
            f"got {kind!r}"
        )

    amendment = PlanAmendment(
        amendment_id=f"amd-{record.case_id}-{len(record.amendments) + 1}",
        case_id=record.case_id,
        kind=kind,
        by=by,
        reason=reason,
        recorded_at=_now(),
    )
    current = apply_passing_amendments(record)
    candidate = current.model_copy(deep=True)

    if kind == PART_SWAP:
        if not to_part_id or kb.part_qty(to_part_id) is None:
            raise ValueError(
                f"part {to_part_id!r} is not on the plant's parts register — "
                "an amendment names a KB part by id; a free-text part cannot "
                "be checked against the shelf, the spec sheets or the SOP, "
                "and is refused"
            )
        req = next(
            (r for r in current.required_parts if drawn_part(r) == from_part_id),
            None,
        )
        if req is None:
            drawn = sorted({drawn_part(r) for r in current.required_parts})
            raise ValueError(
                f"the plan for case {record.case_id} draws no part "
                f"{from_part_id!r} — it draws: {', '.join(drawn) or 'no parts'}"
            )
        if to_part_id == from_part_id:
            raise ValueError(
                f"an amendment swapping {from_part_id} for itself changes "
                "nothing and records nothing"
            )
        amendment.slot_part_id = req.part_id
        amendment.from_part_id = from_part_id
        amendment.to_part_id = to_part_id
        amendment.qty = req.qty
        amendment.basis, amendment.spec_deltas = _swap_paperwork(
            record, req.part_id, to_part_id
        )
        _apply_one(
            candidate,
            PlanAmendment(
                amendment_id=amendment.amendment_id,
                case_id=record.case_id,
                kind=PART_SWAP,
                slot_part_id=req.part_id,
                from_part_id=from_part_id,
                to_part_id=to_part_id,
                qty=req.qty,
                verification={"status": VERIFIED},
            ),
        )
        _remeasure(candidate, ledger, record.case_id)
        swapped_req = next(
            r for r in candidate.required_parts if r.part_id == req.part_id
        )
        amendment.to_available_qty = swapped_req.available_qty
        gaps, detail = reverify(record, candidate, ledger, swap=amendment)
    else:  # step_strike
        step = next((s for s in current.steps if s.step_id == step_id), None)
        if step is None:
            raise ValueError(
                f"the plan for case {record.case_id} has no step {step_id!r} "
                f"— its steps are: "
                f"{', '.join(s.step_id for s in current.steps) or 'none'}"
            )
        top = record.diagnosis.top()
        sop = kb.sop_for(record.machine_id, top.fault_class) if top else None
        mandatory = kb.mandatory_safety_steps(sop)
        if step.mandatory or any(
            step.step_id.endswith(m["step_id"]) for m in mandatory
        ):
            raise ValueError(
                f"step {step.step_id!r} ({step.text}) is a mandatory safety "
                f"step of {sop['sop_id'] if sop else 'the diagnosed SOP'} — "
                "mandatory safety steps cannot be struck at the gate; an "
                "amendment may redirect a plan, never thin its safety "
                "steps. The supervisor's paths are approval or rejection"
            )
        amendment.step_id = step.step_id
        amendment.step_text = step.text
        candidate.steps = [
            s for s in candidate.steps if s.step_id != step.step_id
        ]
        _remeasure(candidate, ledger, record.case_id)
        gaps, detail = reverify(record, candidate, ledger)

    amendment.verification = AmendmentVerification(
        status=FAILED if any(g.blocking for g in gaps) else VERIFIED,
        checked_at=_now(),
        detail=detail,
        gaps=gaps,
    )
    return amendment
