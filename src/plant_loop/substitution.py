"""The substitute search: verify-stage replanning around a stockout.

Substitution used to be a habit of one planner: the offline rules
engine quietly swapped a stocked-out part for the SOP's approved
substitute while drafting, and every other backend was on its own.
Round 3 moves the search into the loop's verify stage, where it
belongs — it now runs for *every* plan, whoever drafted it, and it
leaves paperwork:

1. **Shelf truth first.** A plan's part requirements are claims; the
   verify stage re-measures each one against inventory in code
   (``satisfied`` / ``available_qty`` are set from the shelf, never
   taken from the planner's word).
2. **The search.** For a requirement the shelf cannot satisfy, the
   KB's compatibility relation for that part slot is walked in
   order. Each candidate is checked against the shelf (it must
   cover the required quantity) and against the SOP's constraints
   for the slot (``part_constraints`` — a fan that moves less air
   than the SOP's physics requires is not a substitute, whatever
   the catalogue calls it).
3. **Replanning.** The first candidate that clears both is applied:
   plan steps are revised onto it, the requirement is satisfied by
   substitute, and a :class:`~plant_loop.schemas.Substitution`
   record (original, substitute, compatibility basis, spec deltas)
   rides on the plan — so the record, the work order and the packet
   all flag the substitution openly for the supervisor.
4. **Rejection is documented.** A candidate that fails is recorded
   as ``rejected`` with the reason named; when no candidate passes,
   the surviving ``part_unavailable`` gap is rewritten to document
   the search, and the case exits ``blocked_external`` — the loop
   cannot fix a shelf, but it can prove it looked.
"""

from __future__ import annotations

import operator

from . import kb
from .schemas import ActionPlan, SpecDelta, Substitution

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def constraint_violations(sop: dict | None, part_id: str, candidate_id: str) -> list[str]:
    """Why a candidate fails the SOP's constraints for the slot.

    An empty list means the candidate clears every constraint. A
    candidate whose spec sheet does not even carry the constrained
    attribute fails closed: an unverifiable rating is not a rating.
    """
    specs = kb.part_specs(candidate_id)
    violations: list[str] = []
    for constraint in kb.part_constraints(sop, part_id):
        attribute = constraint["attribute"]
        op = constraint["op"]
        value = constraint["value"]
        reason = constraint.get("reason", "")
        actual = specs.get(attribute)
        if actual is None:
            violations.append(
                f"{candidate_id} carries no {attribute} spec to check against the "
                f"{sop['sop_id'] if sop else 'SOP'} constraint ({attribute} {op} {value})"
            )
            continue
        compare = _OPS.get(op)
        if compare is None or not compare(actual, value):
            violations.append(
                f"{attribute} {actual} fails the {sop['sop_id'] if sop else 'SOP'} "
                f"requirement {attribute} {op} {value} — {reason}"
            )
    return violations


def evaluate(
    sop: dict | None, part_id: str, qty: int
) -> tuple[Substitution | None, list[Substitution]]:
    """Walk the slot's compatibility relation for one requirement.

    Returns ``(winner, records)``: the first candidate that is in
    stock and clears the SOP's constraints, plus one record per
    candidate considered — rejections first (each with its reason),
    the winner last when there is one. Candidates after a winner are
    not considered; the search stops at the first part that works.
    """
    records: list[Substitution] = []
    winner: Substitution | None = None
    for entry in kb.substitute_entries(sop, part_id):
        candidate = entry["part_id"]
        record = Substitution(
            original_part_id=part_id,
            substitute_part_id=candidate,
            qty=qty,
            basis=entry["basis"],
            spec_deltas=[SpecDelta.model_validate(d) for d in entry["spec_deltas"]],
            status="rejected",
        )
        on_hand = kb.part_qty(candidate)
        if on_hand is None:
            record.reason = f"{candidate} is listed as compatible but is not on the parts register"
        elif on_hand < qty:
            record.reason = f"shelf holds {on_hand} of {candidate}; the plan needs {qty}"
        else:
            violations = constraint_violations(sop, part_id, candidate)
            if violations:
                record.reason = "; ".join(violations)
            else:
                record.status = "applied"
                record.reason = (
                    f"in stock ({on_hand} on hand) and clears every "
                    f"{sop['sop_id'] if sop else 'SOP'} constraint for the {part_id} slot"
                )
                winner = record
        records.append(record)
        if winner is not None:
            break
    return winner, records


def apply_to_plan(plan: ActionPlan, record: Substitution) -> None:
    """Revise a plan onto its substitute, in place.

    Steps that named the original part now name the substitute (and
    say so in their text); the requirement is satisfied *by
    substitute* — ``available_qty`` keeps the original part's shelf
    count, which is the number that explains why the substitution
    happened at all.
    """
    original, substitute = record.original_part_id, record.substitute_part_id
    for step in plan.steps:
        if step.part_id == original:
            step.part_id = substitute
            step.text += f" [SUBSTITUTION: {substitute} fitted in place of {original}]"
    for req in plan.required_parts:
        if req.part_id == original:
            req.substitute_part_id = substitute
            req.satisfied = True
    note = (
        f"Verify-stage substitution: {substitute} applied for {original} "
        f"(KB compatibility: {record.basis or 'listed in the SOP'})."
    )
    plan.notes = f"{plan.notes} {note}".strip()


def search_summary(part_id: str, records: list[Substitution]) -> str:
    """One sentence documenting a search that found nothing usable —
    the text a surviving part gap carries onto the record."""
    if not records:
        return f"no substitute is listed in the KB compatibility relation for {part_id}"
    considered = "; ".join(
        f"{r.substitute_part_id} rejected — {r.reason}" for r in records
    )
    return f"substitute search for {part_id}: {considered}"
