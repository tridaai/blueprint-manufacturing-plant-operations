"""Downtime economics: the board, priced in the plant's own rates.

The board loop orders work by severity (priority.py) and stops
there: a supervisor reading the board sees points, not money. But
a plant runs on money per hour of downtime, and the supervisor —
not the formula — is where a dollar judgement belongs. This module
prices the board for that judgement, in code, from two facts the
loop already carries:

- the plan's **estimated downtime** (the SOP's estimate the offline
  planner copies, the model backends' own figure), and
- the line's **downtime rate** — authored data in the plant model
  (``data/kb.json``), like the SOPs and the shelf: what an hour of
  that line standing still costs, in the plant's own accounting.

Three disciplines keep the pricing honest:

- **No rate, no dollars.** A line the plant model does not rate
  (Utilities, whose downtime is not charged to a product line here)
  contributes minutes only. Its minutes are summed separately in
  every total and never folded into a dollar figure — an absent
  rate is a fact about the plant model, not a licence to invent
  one. Every dollar figure carries the rate it used and where the
  rate lives, so a number can always be traced to its basis.
- **Economics inform; they do not reorder.** Priority stays the
  severity formula's output, byte for byte. Exposure is printed
  beside the order, not fed into it: a dollar-heavy maintenance
  alert still waits behind a cheap line-down case, because the
  plant's triage doctrine is a safety-and-severity judgement, not
  a financial one. The supervisor sees both numbers and decides —
  that overlay is a human's job, and round 6 deliberately keeps it
  one.
- **Safety is never priced.** A safety stop carries no economics
  at all — not a zero, an absence. Nothing in this module reads,
  annotates or totals a safety record's money, because there is
  none: the escalation path stays exactly as bare as it was.

Actuals join when the supervisor records the outcome: the same
rate prices the actual minutes, so the stats can put estimated and
actual dollars side by side (stats.py) with the same counts-next-
to-rates honesty the diagnosis numbers use.

Everything here is a pure function over case records plus the KB —
the same arithmetic serves the record stamp, the board read model,
the run total, the stats and the handoff, over both stores.
"""

from __future__ import annotations

from . import kb
from .schemas import (
    EXIT_SAFETY_STOP,
    BoardEconomics,
    CaseEconomics,
    CaseResult,
)


def price(minutes: int, rate: float | None) -> float | None:
    """Minutes at a rate, in dollars (None when there is no rate)."""
    if rate is None:
        return None
    return round(minutes * rate / 60.0, 2)


def for_record(record: CaseResult) -> CaseEconomics | None:
    """One case's economics snapshot, or None for a safety stop.

    The estimate is the plan's own: a record with no plan carries
    no estimate (0 minutes — the stats only ever average records
    that carry a plan, the downtime section's population). The
    actuals come from the recorded outcome when one exists.
    """
    if record.exit_reason == EXIT_SAFETY_STOP:
        return None
    rate = kb.line_downtime_rate(record.line_id)
    estimated = record.plan.estimated_downtime_min if record.plan.steps else 0
    actual = record.outcome.downtime_minutes_actual if record.outcome else None
    return CaseEconomics(
        line_id=record.line_id,
        rate_usd_per_hour=rate,
        rate_basis=(
            f"plant model downtime rate for {record.line_id} (data/kb.json)"
            if rate is not None
            else ""
        ),
        estimated_minutes=estimated,
        estimated_usd=price(estimated, rate),
        actual_minutes=actual,
        actual_usd=price(actual, rate) if actual is not None else None,
    )


def board_summary(records: list[CaseResult]) -> BoardEconomics:
    """A set of records priced as one board.

    Safety stops are skipped whole (they carry no economics);
    every other record contributes its estimated minutes, split by
    whether its line is rated. The largest single exposure names
    the case a supervisor glancing at the total should look at
    first — the earliest such case in the list wins ties, so the
    caller passes records in board order.
    """
    summary = BoardEconomics()
    exposure = 0.0
    for record in records:
        econ = for_record(record)
        if econ is None:
            continue
        summary.estimated_minutes += econ.estimated_minutes
        if econ.rate_usd_per_hour is None:
            summary.cases_minutes_only += 1
            summary.unrated_minutes += econ.estimated_minutes
            continue
        summary.cases_costed += 1
        exposure += econ.estimated_usd or 0.0
        if (
            summary.largest_exposure_usd is None
            or (econ.estimated_usd or 0.0) > summary.largest_exposure_usd
        ):
            summary.largest_exposure_case_id = record.case_id
            summary.largest_exposure_usd = econ.estimated_usd
    if summary.cases_costed:
        summary.estimated_exposure_usd = round(exposure, 2)
    return summary
