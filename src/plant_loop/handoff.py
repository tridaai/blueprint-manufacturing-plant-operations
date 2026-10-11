"""The shift handoff report: the board, written down for the next shift.

A board run already writes a handoff summary for *its* remainder.
This report is the standing version — generated on demand from the
current records, for the supervisor handing the plant over:

- cases at the gate awaiting a decision (board order);
- escalations (rounds exhausted, budget spent) with their open gaps;
- blocked-external cases, with the part the plant is waiting on —
  and, when the wait is the board's own allocation (an earlier
  case's claim holds the shelf's last unit), the claiming case,
  named from the blocked record's own gap text;
- safety stops (always listed separately — they outrank everything);
- open reliability-review items the watch has raised;
- the faults supervisors have confirmed and written back to the
  machines' history — the memory the next shift's diagnoses start
  from, named so the incoming supervisor knows what the plant has
  newly learned;
- the board priced in downtime dollars (economics.py): the
  estimated exposure still open at the gate, the largest single
  open exposure, and the closed cases' recorded actuals — minutes
  only on lines the plant model does not rate, and safety stops
  never priced at all;
- the latest board run, and — when it was cut by the shift budget —
  what it did not reach.

Every fact is composed here, in code, from the stored records. No
model writes any part of it: a handoff that invents a gap or drops
a safety stop is worse than no handoff. The JSON payload and the
plain-text rendering are built from the same dict, so the text a
supervisor pastes into the shift log says exactly what the API
returned.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from . import economics as economics_module
from .events import format_timestamp
from .ports import Store
from .schemas import (
    EXIT_BLOCKED_EXTERNAL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_EXHAUSTED,
    EXIT_SAFETY_STOP,
    CaseResult,
)

_ESCALATION_EXITS = (EXIT_EXHAUSTED, EXIT_BUDGET_EXCEEDED)

# The claim note the verify stage writes into a part gap when an
# earlier case's allocation caused the shortfall (loops.py via
# allocation.py): "claimed by <case>, higher priority on this
# board" / "claimed by <case> (board run <id>)". The handoff reads
# the claimer back out of that text — the gap was composed when
# the case blocked, so it stays the truth of that run even if the
# claim has since been released at a gate.
_CLAIM_NOTE_RE = re.compile(r"claimed by ([^,(]+)")


def _allocation_blocked(record: CaseResult) -> list[dict]:
    """Parts this blocked case lost to another case's claim."""
    found: list[dict] = []
    for gap in record.open_gaps:
        if gap.kind != "part_unavailable":
            continue
        match = _CLAIM_NOTE_RE.search(gap.detail)
        if match:
            found.append(
                {"part_id": gap.ref, "claimed_by": match.group(1).strip()}
            )
    return found


def _gate_order(records: list[CaseResult]) -> list[CaseResult]:
    """Awaiting cases in board order (the store.board() ordering)."""
    waiting = [r for r in records if r.status == "awaiting_supervisor"]
    waiting.sort(key=lambda r: (-r.priority, r.reported_at or "", r.case_id))
    return waiting


def _brief(record: CaseResult) -> dict:
    return {
        "case_id": record.case_id,
        "type": record.type,
        "title": record.title,
        "machine_id": record.machine_id,
        "line_id": record.line_id,
        "exit_reason": record.exit_reason,
        "priority": record.priority,
        "rounds": len(record.rounds),
    }


def build_handoff(store: Store, *, now: datetime | None = None) -> dict:
    """The handoff payload for the board as it stands right now."""
    moment = now or datetime.now(timezone.utc)
    records = store.all_cases()
    waiting = _gate_order(records)

    awaiting = [_brief(r) for r in waiting]
    escalations = [
        {
            **_brief(r),
            "open_gaps": [
                {"kind": g.kind, "detail": g.detail} for g in r.open_gaps
            ],
        }
        for r in waiting
        if r.exit_reason in _ESCALATION_EXITS
    ]
    blocked = [
        {
            **_brief(r),
            "missing_parts": [
                p.part_id for p in r.plan.required_parts if not p.satisfied
            ],
            "allocation_blocked": _allocation_blocked(r),
        }
        for r in waiting
        if r.exit_reason == EXIT_BLOCKED_EXTERNAL
    ]
    safety_stops = [
        _brief(r) for r in waiting if r.exit_reason == EXIT_SAFETY_STOP
    ]
    reliability_items = [
        item.model_dump()
        for item in store.all_board_items()
        if item.status == "open"
    ]
    confirmed_faults = [
        {
            "entry_id": e.entry_id,
            "machine_id": e.machine_id,
            "fault_class": e.fault_class,
            "case_id": e.case_id,
            "date": e.date,
            "recorded_at": e.recorded_at,
            "summary": e.summary,
        }
        for e in sorted(
            store.all_history_entries(), key=lambda e: (e.date, e.entry_id)
        )
    ]

    # Downtime economics: the exposure still open at the gate
    # (the awaiting records, priced as one board in gate order, so
    # the largest exposure is the first such case the incoming
    # supervisor meets) and what the closed cases actually cost —
    # actual minutes from recorded outcomes, priced at the same
    # plant-model rates, split by rated and unrated lines. Safety
    # stops contribute nothing: they are never priced (economics.py).
    open_summary = economics_module.board_summary(waiting)
    # (Safety stops are excluded before counting, not skipped
    # mid-sum: for_record returns None for them, and a case the
    # economics never touch must not leak into the totals either.)
    closed_pairs = [
        (r, economics_module.for_record(r))
        for r in records
        if r.outcome is not None and r.outcome.downtime_minutes_actual is not None
    ]
    closed_pairs = [(r, e) for r, e in closed_pairs if e is not None]
    closed_records = [r for r, _e in closed_pairs]
    closed_rated_usd = 0.0
    closed_rated_n = 0
    closed_minutes_only = 0
    closed_unrated_minutes = 0
    for _r, econ in closed_pairs:
        if econ.actual_usd is None:
            closed_minutes_only += 1
            closed_unrated_minutes += econ.actual_minutes or 0
        else:
            closed_rated_n += 1
            closed_rated_usd += econ.actual_usd
    economics = {
        "open": open_summary.model_dump(),
        "closed": {
            "cases_with_actuals": len(closed_records),
            "actual_minutes": sum(
                r.outcome.downtime_minutes_actual for r in closed_records
            ),
            "rated_cases": closed_rated_n,
            "actual_usd": round(closed_rated_usd, 2) if closed_rated_n else None,
            "minutes_only_cases": closed_minutes_only,
            "unrated_minutes": closed_unrated_minutes,
        },
    }

    runs = store.all_board_runs()
    last_run = None
    if runs:
        latest = max(runs, key=lambda run: (run.get("created_at", ""), run.get("run_id", "")))
        telemetry = latest.get("telemetry") or {}
        tokens_spent = int(telemetry.get("input_tokens") or 0) + int(
            telemetry.get("output_tokens") or 0
        )
        cut = latest.get("status") == "shift_budget_exceeded"
        last_run = {
            "run_id": latest.get("run_id", ""),
            "status": latest.get("status", ""),
            "created_at": latest.get("created_at", ""),
            "processed": len(latest.get("processed") or []),
            "tokens_spent": tokens_spent,
            "cut_by_budget": cut,
            "remaining": latest.get("remaining") or [],
            "budget_note": (
                "the run was cut by the shift budget — the cases under "
                "'remaining' were not worked and carry to the next shift"
                if cut
                else ""
            ),
        }

    return {
        "generated_at": format_timestamp(moment),
        "counts": {
            "awaiting_decision": len(awaiting),
            "escalations": len(escalations),
            "blocked_external": len(blocked),
            "safety_stops": len(safety_stops),
            "reliability_items": len(reliability_items),
        },
        "awaiting_decision": awaiting,
        "escalations": escalations,
        "blocked_external": blocked,
        "safety_stops": safety_stops,
        "reliability_items": reliability_items,
        "confirmed_faults": confirmed_faults,
        "economics": economics,
        "last_board_run": last_run,
    }


def _case_line(entry: dict) -> str:
    machine = entry["machine_id"] or "—"
    return (
        f"  - {entry['case_id']} · {entry['type']} · {machine} · "
        f"{entry['exit_reason']} · priority {entry['priority']} · {entry['title']}"
    )


def render_text(handoff: dict) -> str:
    """The handoff as plain text — what goes in the shift log."""
    counts = handoff["counts"]
    lines = [
        f"SHIFT HANDOFF — generated {handoff['generated_at']}",
        (
            f"At the gate: {counts['awaiting_decision']} awaiting decision · "
            f"{counts['escalations']} escalation(s) · "
            f"{counts['blocked_external']} blocked external · "
            f"{counts['safety_stops']} safety stop(s) · "
            f"{counts['reliability_items']} reliability item(s)"
        ),
        "",
        "AWAITING A SUPERVISOR DECISION:",
    ]
    lines += [_case_line(e) for e in handoff["awaiting_decision"]] or ["  (none)"]
    lines += ["", "ESCALATIONS (open gaps):"]
    if handoff["escalations"]:
        for entry in handoff["escalations"]:
            lines.append(_case_line(entry))
            for gap in entry["open_gaps"]:
                lines.append(f"      gap: {gap['kind']} — {gap['detail']}")
    else:
        lines.append("  (none)")
    lines += ["", "BLOCKED ON THE OUTSIDE WORLD:"]
    if handoff["blocked_external"]:
        for entry in handoff["blocked_external"]:
            parts = ", ".join(entry["missing_parts"]) or "see record"
            line = _case_line(entry) + f" · waiting on: {parts}"
            for alloc in entry.get("allocation_blocked", []):
                line += (
                    f" · blocked by allocation: {alloc['part_id']} "
                    f"claimed by {alloc['claimed_by']}"
                )
            lines.append(line)
    else:
        lines.append("  (none)")
    lines += ["", "SAFETY STOPS:"]
    lines += [_case_line(e) for e in handoff["safety_stops"]] or ["  (none)"]
    lines += ["", "RELIABILITY REVIEW ITEMS:"]
    if handoff["reliability_items"]:
        for item in handoff["reliability_items"]:
            lines.append(
                f"  - {item['item_id']} · {item['machine_id']} · {item['fault_class']} · "
                f"{item['count']} cases in {item['window_days']} days · "
                f"evidence: {', '.join(item['case_ids'])}"
            )
    else:
        lines.append("  (none)")
    lines += ["", "FAULTS CONFIRMED BY SUPERVISORS (WRITTEN BACK TO MACHINE HISTORY):"]
    if handoff["confirmed_faults"]:
        for entry in handoff["confirmed_faults"]:
            lines.append(
                f"  - {entry['machine_id']} · {entry['fault_class']} · "
                f"confirmed in case {entry['case_id']} · {entry['date']}"
            )
    else:
        lines.append("  (none yet — no supervisor outcome has confirmed a fault)")
    econ = handoff["economics"]
    open_e, closed = econ["open"], econ["closed"]
    lines += ["", "DOWNTIME ECONOMICS (estimates priced at the plant model's line rates):"]
    if open_e["estimated_exposure_usd"] is not None:
        text = (
            f"  Open at the gate: ~${open_e['estimated_exposure_usd']:,.2f} of "
            f"estimated exposure over {open_e['estimated_minutes']} min"
        )
        if open_e["unrated_minutes"]:
            text += (
                f" ({open_e['unrated_minutes']} of those minutes on lines "
                "the plant model does not rate — minutes only)"
            )
        lines.append(text)
        lines.append(
            f"  Largest open exposure: {open_e['largest_exposure_case_id']} — "
            f"${open_e['largest_exposure_usd']:,.2f}"
        )
    elif open_e["estimated_minutes"]:
        lines.append(
            f"  Open at the gate: {open_e['estimated_minutes']} min estimated — "
            "all on lines the plant model does not rate (minutes only, no dollars)"
        )
    else:
        lines.append("  Open at the gate: no estimated exposure")
    if closed["cases_with_actuals"]:
        text = (
            f"  Closed cases: {closed['cases_with_actuals']} with recorded "
            f"actuals — {closed['actual_minutes']} min actual"
        )
        if closed["actual_usd"] is not None:
            text += f", ${closed['actual_usd']:,.2f} on rated lines"
        if closed["unrated_minutes"]:
            text += (
                f" ({closed['unrated_minutes']} min on unrated lines, "
                "minutes only)"
            )
        lines.append(text)
    else:
        lines.append("  Closed cases: no recorded actuals yet")
    run = handoff["last_board_run"]
    lines += ["", "LAST BOARD RUN:"]
    if run is None:
        lines.append("  (no board run recorded)")
    else:
        lines.append(
            f"  {run['run_id']} · {run['status']} · {run['processed']} case(s) worked · "
            f"{run['tokens_spent']} tokens"
        )
        if run["cut_by_budget"]:
            lines.append(f"  CUT BY SHIFT BUDGET — {run['budget_note']}")
            for item in run["remaining"]:
                lines.append(f"    not worked: {item['case_id']} ({item['reason']})")
    return "\n".join(lines) + "\n"
