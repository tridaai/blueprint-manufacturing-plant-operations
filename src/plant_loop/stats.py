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
- **downtime economics** — the same comparison priced (economics.py):
  estimated vs actual minutes *and* dollars, plant-wide and per
  line, over the same population. A case on a line the plant model
  does not rate counts in the minutes and never in the dollars —
  the section says how many of its cases are minutes-only, and
  every figure carries its case count beside it.
- **the learning loop** — what the plant has written back to its
  memory (confirmed faults, by machine) and how the loop scores on
  *repeat* faults: scored cases whose machine + fault a supervisor
  had already confirmed on an earlier case, with their top-1 rate
  set beside the overall top-1 rate. Both numbers are reported
  plainly, as counts over small samples — the section claims no
  uplift; it shows the two rates and lets the reader judge.

Everything here is a pure function over case records (the learning
section also reads the store's history entries when the caller
passes them) — the same code path serves the in-memory double and
PostgreSQL, and a record only enters a metric when it carries the
fields that metric needs. Partial outcomes count for what they
recorded and are excluded, visibly, from what they did not.
"""

from __future__ import annotations

from . import economics as economics_module
from .schemas import EXIT_CONVERGED, OUTCOME_DECISIONS, CaseResult, HistoryEntry

EMPTY_NOTE = (
    "no supervisor outcomes recorded yet — record one via "
    "POST /api/cases/{case_id}/outcome to start measuring diagnosis quality"
)

NO_REPEATS_NOTE = (
    "no repeat-fault cases yet — no scored case so far matches a machine + "
    "fault a supervisor had already confirmed on an earlier case"
)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _cost_block(pairs: list[tuple]) -> dict:
    """Estimated-vs-actual over (estimated_min, actual_min,
    estimated_usd, actual_usd) tuples — one line's worth, or the
    plant's. Dollar means run over the rated pairs only; a pair
    whose line carries no rate has both dollar fields None and
    counts in the minutes alone."""
    rated = [p for p in pairs if p[2] is not None]
    return {
        "n": len(pairs),
        "rated_n": len(rated),
        "minutes_only_n": len(pairs) - len(rated),
        "estimated_minutes": sum(p[0] for p in pairs),
        "actual_minutes": sum(p[1] for p in pairs),
        "mean_error_min": _mean([float(p[1] - p[0]) for p in pairs]),
        "mean_absolute_error_min": _mean([float(abs(p[1] - p[0])) for p in pairs]),
        "estimated_usd": (
            round(sum(p[2] for p in rated), 2) if rated else None
        ),
        "actual_usd": (
            round(sum(p[3] for p in rated), 2) if rated else None
        ),
        "mean_error_usd": _mean([p[3] - p[2] for p in rated]),
        "mean_absolute_error_usd": _mean([abs(p[3] - p[2]) for p in rated]),
    }


def _economics_section(records: list[CaseResult]) -> dict:
    """Cost accuracy over the downtime population: the outcomes
    that recorded an actual downtime on a case that carried a plan
    (the same population the minutes-only ``downtime`` block uses,
    so the two blocks always describe the same cases)."""
    population = [
        r
        for r in records
        if r.outcome is not None
        and r.outcome.downtime_minutes_actual is not None
        and r.plan.steps
    ]
    pairs_by_line: dict[str, list[tuple]] = {}
    rates: dict[str, float | None] = {}
    for r in population:
        econ = economics_module.for_record(r)
        if econ is None:  # a safety stop has no plan; unreachable guard
            continue
        pair = (
            econ.estimated_minutes,
            econ.actual_minutes,
            econ.estimated_usd,
            econ.actual_usd,
        )
        pairs_by_line.setdefault(r.line_id, []).append(pair)
        rates[r.line_id] = econ.rate_usd_per_hour
    by_line = []
    for line_id in sorted(pairs_by_line):
        block = _cost_block(pairs_by_line[line_id])
        block = {"line_id": line_id, "rate_usd_per_hour": rates[line_id], **block}
        by_line.append(block)
    plant = _cost_block([p for pairs in pairs_by_line.values() for p in pairs])
    note = ""
    if not population:
        note = (
            "no recorded actuals yet — cost accuracy starts with the "
            "first outcome that records an actual downtime"
        )
    elif plant["minutes_only_n"]:
        note = (
            f"{plant['minutes_only_n']} of {plant['n']} case(s) sit on lines "
            "the plant model does not rate — they count in the minutes "
            "and never in the dollars"
        )
    return {"note": note, "by_line": by_line, **plant}


def _learning_loop(
    records: list[CaseResult],
    scored: list[CaseResult],
    top1_accuracy: float | None,
    history_entries: list[HistoryEntry] | None,
) -> dict:
    """The learning-loop section: write-backs + repeat-fault scoring.

    A *repeat* case is a scored case whose machine + confirmed
    fault a supervisor had already confirmed on an earlier case
    (records are read in record order — created, then case id — so
    "earlier" is the plant's own chronology, and a case never
    counts as its own precedent). The write-back counts come from
    the store's history entries when the caller passes them; the
    pure-records call derives the same counts from the confirmed
    outcomes themselves, since every recorded confirmation writes
    exactly one entry.
    """
    if history_entries is not None:
        written_back = len(history_entries)
        by_machine: dict[str, int] = {}
        for entry in history_entries:
            by_machine[entry.machine_id] = by_machine.get(entry.machine_id, 0) + 1
    else:
        confirmed_records = [
            r for r in records if r.outcome is not None and r.outcome.confirmed_fault
        ]
        written_back = len(confirmed_records)
        by_machine = {}
        for r in confirmed_records:
            by_machine[r.machine_id] = by_machine.get(r.machine_id, 0) + 1

    seen: set[tuple[str, str]] = set()
    repeat_ids: set[str] = set()
    timeline = sorted(
        (r for r in records if r.outcome is not None and r.outcome.confirmed_fault),
        key=lambda r: (r.created_at or "", r.case_id),
    )
    for r in timeline:
        key = (r.machine_id, r.outcome.confirmed_fault)
        if key in seen:
            repeat_ids.add(r.case_id)
        seen.add(key)
    repeats = [r for r in scored if r.case_id in repeat_ids]
    repeat_correct = sum(
        1
        for r in repeats
        if r.diagnosis.hypotheses[0].fault_class == r.outcome.confirmed_fault
    )
    return {
        "confirmed_written_back": written_back,
        "written_back_by_machine": dict(sorted(by_machine.items())),
        "repeat_cases": len(repeats),
        "repeat_top1_correct": repeat_correct,
        "repeat_top1_accuracy": (
            round(repeat_correct / len(repeats), 4) if repeats else None
        ),
        "overall_top1_accuracy": top1_accuracy,
        "note": "" if repeats else NO_REPEATS_NOTE,
    }


def diagnosis_stats(
    records: list[CaseResult], history_entries: list[HistoryEntry] | None = None
) -> dict:
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

    top1_accuracy = round(top1_correct / len(scored), 4) if scored else None

    return {
        "available": bool(with_outcome),
        "note": "" if with_outcome else EMPTY_NOTE,
        "cases_total": len(records),
        "outcomes_recorded": len(with_outcome),
        "fault_scored": len(scored),
        "confirmed_without_diagnosis": confirmed_no_diagnosis,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_accuracy,
        "learning_loop": _learning_loop(records, scored, top1_accuracy, history_entries),
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
        "economics": _economics_section(records),
        "resolved": {
            "recorded": sum(1 for r in with_outcome if r.outcome.resolved is not None),
            "resolved_true": sum(1 for r in with_outcome if r.outcome.resolved),
        },
    }
