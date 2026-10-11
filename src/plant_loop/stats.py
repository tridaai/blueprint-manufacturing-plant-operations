"""Live diagnosis accuracy: the loop, measured against the floor.

Round 1 measured the loop against authored expectations (the golden
evals). This module measures it against what supervisors recorded
actually happening — the outcome loop:

- **top-1 / top-2 accuracy** — did the diagnosis' first (or first
  two) hypotheses name the fault the floor confirmed?
- **per-fault-class precision / recall** — on top-1 predictions:
  of the times the agent named class X, how often was X confirmed
  (precision); of the times X was confirmed, how often did the
  agent name it first (recall). Counts are printed next to the
  rates so a 100% on one case reads as what it is.
- **rounds-to-converge** — the mean loop rounds over confirmed
  cases that converged (what the loop costs when it is right).
- **plan acceptance** — how supervisors disposed of the drafted
  plans (accepted / accepted_with_changes / rejected).
- **downtime error** — actual minus estimated downtime over the
  outcomes that recorded an actual: signed mean (bias) and mean
  absolute error (accuracy).

Everything here is a pure function over case records — the same
code path serves the in-memory double and PostgreSQL, and a record
only enters a metric when it carries the fields that metric needs.
Partial outcomes count for what they recorded and are excluded,
visibly, from what they did not.
"""

from __future__ import annotations

from .schemas import EXIT_CONVERGED, OUTCOME_DECISIONS, CaseResult

EMPTY_NOTE = (
    "no supervisor outcomes recorded yet — record one via "
    "POST /api/cases/{case_id}/outcome to start measuring diagnosis quality"
)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def diagnosis_stats(records: list[CaseResult]) -> dict:
    """Aggregate the outcome-carrying records into accuracy stats."""
    with_outcome = [r for r in records if r.outcome is not None]
    # A record scores on diagnosis only when a fault was confirmed
    # AND the loop produced hypotheses to score (a safety stop has
    # none — it is counted separately, never silently dropped).
    scored = [
        r
        for r in with_outcome
        if r.outcome.confirmed_fault and r.diagnosis.hypotheses
    ]
    confirmed_no_diagnosis = sum(
        1 for r in with_outcome if r.outcome.confirmed_fault and not r.diagnosis.hypotheses
    )

    top1_correct = sum(
        1 for r in scored if r.diagnosis.hypotheses[0].fault_class == r.outcome.confirmed_fault
    )
    top2_correct = sum(
        1
        for r in scored
        if r.outcome.confirmed_fault
        in [h.fault_class for h in r.diagnosis.hypotheses[:2]]
    )

    per_class: dict[str, dict] = {}
    for r in scored:
        predicted = r.diagnosis.hypotheses[0].fault_class
        actual = r.outcome.confirmed_fault
        for cls in {predicted, actual}:
            per_class.setdefault(
                cls, {"fault_class": cls, "tp": 0, "predicted": 0, "actual": 0}
            )
        per_class[predicted]["predicted"] += 1
        per_class[actual]["actual"] += 1
        if predicted == actual:
            per_class[actual]["tp"] += 1
    per_fault_class = []
    for cls in sorted(per_class):
        row = per_class[cls]
        row["precision"] = (
            round(row["tp"] / row["predicted"], 4) if row["predicted"] else None
        )
        row["recall"] = round(row["tp"] / row["actual"], 4) if row["actual"] else None
        per_fault_class.append(row)

    converged_rounds = [
        len(r.rounds)
        for r in scored
        if r.exit_reason == EXIT_CONVERGED
    ]

    decisions = {d: 0 for d in OUTCOME_DECISIONS}
    for r in with_outcome:
        if r.outcome.decision in decisions:
            decisions[r.outcome.decision] += 1
    decided = sum(decisions.values())
    acceptance = {
        "counts": decisions,
        "shares": {
            d: (round(decisions[d] / decided, 4) if decided else None)
            for d in OUTCOME_DECISIONS
        },
        # The drafted plan was used, as-is or with changes.
        "used_rate": (
            round((decisions["accepted"] + decisions["accepted_with_changes"]) / decided, 4)
            if decided
            else None
        ),
    }

    downtime_errors = [
        r.outcome.downtime_minutes_actual - r.plan.estimated_downtime_min
        for r in with_outcome
        if r.outcome.downtime_minutes_actual is not None and r.plan.steps
    ]

    return {
        "available": bool(with_outcome),
        "note": "" if with_outcome else EMPTY_NOTE,
        "cases_total": len(records),
        "outcomes_recorded": len(with_outcome),
        "fault_scored": len(scored),
        "confirmed_without_diagnosis": confirmed_no_diagnosis,
        "top1_correct": top1_correct,
        "top1_accuracy": (
            round(top1_correct / len(scored), 4) if scored else None
        ),
        "top2_correct": top2_correct,
        "top2_accuracy": (
            round(top2_correct / len(scored), 4) if scored else None
        ),
        "per_fault_class": per_fault_class,
        "converged_confirmed": len(converged_rounds),
        "mean_rounds_to_converge": _mean([float(n) for n in converged_rounds]),
        "plan_acceptance": acceptance,
        "downtime": {
            "n": len(downtime_errors),
            "mean_error_min": _mean([float(e) for e in downtime_errors]),
            "mean_absolute_error_min": _mean(
                [float(abs(e)) for e in downtime_errors]
            ),
        },
        "resolved": {
            "recorded": sum(1 for r in with_outcome if r.outcome.resolved is not None),
            "resolved_true": sum(1 for r in with_outcome if r.outcome.resolved),
        },
    }
