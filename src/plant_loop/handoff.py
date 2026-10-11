"""The shift handoff report: the board, written down for the next shift.

A board run already writes a handoff summary for *its* remainder.
This report is the standing version — generated on demand from the
current records, for the supervisor handing the plant over:

- cases at the gate awaiting a decision (board order);
- escalations (rounds exhausted, budget spent) with their open gaps;
- blocked-external cases, with the part the plant is waiting on;
- safety stops (always listed separately — they outrank everything);
- open reliability-review items the watch has raised;
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

from datetime import datetime, timezone

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
            lines.append(_case_line(entry) + f" · waiting on: {parts}")
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
