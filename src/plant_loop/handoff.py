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
- preventive maintenance (pm.py): the standing PM obligations the
  incoming shift inherits — interval, next due point, whether it is
  already overdue, the open cycle case — the proposals still
  awaiting a supervisor's decision, and every deferral on record,
  so an inspection that keeps being put off is visible in the log,
  not just true;
- the coming parts shortages (outlook.py): the parts whose
  projected demand — standing claims, open cases' unsatisfied
  plan requirements, and the PM cycles falling due inside the
  outlook's horizon — passes the shelf, each with the shortfall
  and the moment it bites, so the next shift sees a shortage
  while it is still a date on a calendar rather than a blocked
  case at the workbench;
- the knowledge gaps (gaps.py): the blocks the plant's own
  missing knowledge caused — case records whose diagnosed fault
  has no SOP on record for their machine — grouped by the
  missing SOP, with the rounds the absence burned and whether a
  case is blocked on it right now, so the incoming supervisor
  (and the KB maintainer behind them) sees the pattern, not just
  the latest escalation;
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
from . import gaps as gaps_module
from . import outlook as outlook_module
from .events import format_timestamp
from .ports import Store
from .schemas import (
    EXIT_BLOCKED_EXTERNAL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_EXHAUSTED,
    EXIT_SAFETY_STOP,
    CaseResult,
    PmObligation,
    PmProposal,
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

    # Carry-over: the latest board run's outgoing manifest — the
    # unfinished business the next shift is offered, with what
    # each case is waiting on — and, when the latest run itself
    # accepted a predecessor's board, how that inheritance
    # settled (the claims renewed and released at the turnover).
    runs = store.all_board_runs()
    carry_over = None
    if runs:
        # The most recently saved run (the store's own ordering) —
        # its manifest is the offer currently on the table.
        latest_any = runs[-1]
        manifest = latest_any.get("carry_over") or []
        if manifest:
            standing = {
                c.case_id
                for c in store.all_part_claims()
                if c.status == "claimed"
            }
            carry_over = {
                "run_id": latest_any.get("run_id", ""),
                "entries": manifest,
                "claims_standing_on_carried": sum(
                    1 for e in manifest if e["case_id"] in standing
                ),
                "settled": latest_any.get("carry_over_settled") or {},
            }

    # Preventive maintenance (pm.py): the standing obligations and
    # what is owed on them, composed from the PM records. Overdue
    # is judged against the handoff's own moment — the same clock
    # the text render and the API caller see. Retired obligations
    # ride separately, with their reasons: a handoff that quietly
    # dropped them would read as if the duty had never existed,
    # and the next shift deserves to know why it stopped.
    stamp = format_timestamp(moment)
    pm_records = store.all_pm_records()
    pm_obligations = [r for r in pm_records if isinstance(r, PmObligation)]
    pm = {
        "obligations": [
            {
                "obligation_id": o.obligation_id,
                "machine_id": o.machine_id,
                "fault_class": o.fault_class,
                "interval_days": o.interval_days,
                "next_due_at": o.next_due_at,
                "overdue": bool(o.next_due_at) and o.next_due_at <= stamp,
                "last_performed_at": o.last_performed_at,
                "cycle_case_id": o.cycle_case_id,
                "cycles_completed": o.cycles_completed,
                "deferrals": len(o.deferrals),
            }
            for o in pm_obligations
            if o.status == "standing"
        ],
        "retired": [
            {
                "obligation_id": o.obligation_id,
                "machine_id": o.machine_id,
                "fault_class": o.fault_class,
                "retired_at": o.retired_at,
                "retired_by": o.retired_by,
                "retirement_reason": o.retirement_reason,
                "retirement_kind": o.retirement_kind,
                "decommission_id": o.decommission_id,
            }
            for o in pm_obligations
            if o.status == "retired"
        ],
        "proposals_pending": [
            {
                "proposal_id": r.proposal_id,
                "machine_id": r.machine_id,
                "fault_class": r.fault_class,
                "fault_count": r.fault_count,
                "proposed_interval_days": r.proposed_interval_days,
            }
            for r in pm_records
            if isinstance(r, PmProposal) and r.status == "proposed"
        ],
        "deferrals": [
            {
                "obligation_id": o.obligation_id,
                "machine_id": o.machine_id,
                "fault_class": o.fault_class,
                **d,
            }
            for o in pm_obligations
            for d in o.deferrals
        ],
    }

    # The parts outlook (outlook.py): the same projection the API
    # and CLI serve, computed at the handoff's own moment over the
    # authored horizon — the shortages section the render prints
    # for the incoming shift.
    parts_outlook = outlook_module.build_outlook(store, now=moment).model_dump()

    # The knowledge gaps (gaps.py): the same aggregation the API
    # and CLI serve, computed at the handoff's own moment — the
    # missing SOPs and what their absence has cost, so the pattern
    # hands over with the board. A read over the same records:
    # it drafts nothing and unblocks nothing.
    knowledge_gaps = gaps_module.build_gap_report(store, now=moment).model_dump()

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
        "carry_over": carry_over,
        "pm": pm,
        "parts_outlook": parts_outlook,
        "knowledge_gaps": knowledge_gaps,
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
    carry = handoff.get("carry_over")
    lines += ["", "CARRY-OVER — THE NEXT SHIFT INHERITS THIS BOARD:"]
    if not carry:
        lines.append("  (no board run has offered carry-over yet)")
    else:
        lines.append(
            f"  From board run {carry['run_id']} — "
            f"{len(carry['entries'])} case(s) carry over:"
        )
        for entry in carry["entries"]:
            lines.append(
                f"  - {entry['case_id']} · {entry['kind'].replace('_', ' ')} · "
                f"waiting on: {entry['waiting_on']}"
            )
        settled = carry.get("settled") or {}
        if settled:
            lines.append(
                f"  At the last turnover: {len(settled.get('renewed_claims', []))} "
                f"claim(s) renewed for carry-over, "
                f"{len(settled.get('released_claims', []))} claim(s) released "
                "— run ended."
            )
            if settled.get("dropped"):
                lines.append(
                    f"  Dropped at acceptance: {', '.join(settled['dropped'])}"
                )
        elif carry["claims_standing_on_carried"]:
            lines.append(
                f"  {carry['claims_standing_on_carried']} claim(s) stand on "
                "the carried cases until a successor run accepts the board."
            )
    pm = handoff.get("pm") or {
        "obligations": [],
        "proposals_pending": [],
        "deferrals": [],
        "retired": [],
    }
    lines += ["", "PREVENTIVE MAINTENANCE:"]
    if not pm["obligations"] and not pm["proposals_pending"] and not pm.get("retired"):
        lines.append("  (no standing obligations; no proposals awaiting a decision)")
    for entry in pm["obligations"]:
        lines.append(
            f"  - {entry['obligation_id']} · {entry['machine_id']} · "
            f"{entry['fault_class']} · every {entry['interval_days']} days · "
            f"next due {entry['next_due_at']}"
            + (" — OVERDUE" if entry["overdue"] else "")
        )
        lines.append(
            f"      last performed {entry['last_performed_at'] or 'never'} · "
            f"cycles completed {entry['cycles_completed']} · "
            f"open cycle case {entry['cycle_case_id'] or 'none'} · "
            f"deferrals {entry['deferrals']}"
        )
    for entry in pm["proposals_pending"]:
        lines.append(
            f"  - proposal {entry['proposal_id']} awaiting a decision · "
            f"{entry['machine_id']} · {entry['fault_class']} · "
            f"{entry['fault_count']} faults · proposed interval "
            f"{entry['proposed_interval_days']} days"
        )
    for entry in pm.get("retired", []):
        lines.append(
            f"  - retired: {entry['obligation_id']} · {entry['machine_id']} · "
            f"{entry['fault_class']} · retired {entry['retired_at']} by "
            f"{entry['retired_by'] or '-'} — {entry['retirement_reason']}"
            + (
                f" · via decommission {entry['decommission_id']}"
                if entry["retirement_kind"] == "machine_decommission"
                else ""
            )
        )
    for entry in pm["deferrals"]:
        lines.append(
            f"  - deferred: {entry['obligation_id']} · {entry['machine_id']} · "
            f"cycle case {entry['case_id']} · {entry['at']} by "
            f"{entry['by'] or '-'} — {entry['reason']} · due rolled "
            f"{entry['previous_due_at']} -> {entry['rolled_to']}"
        )
    outlook = handoff.get("parts_outlook") or {"shortages": [], "horizon_days": 0}
    lines += ["", f"COMING PARTS SHORTAGES ({outlook['horizon_days']}-DAY OUTLOOK):"]
    if outlook["shortages"]:
        for part in outlook["shortages"]:
            lines.append(
                f"  - {part['part_id']} · short {part['shortfall']} · projected "
                f"{part['projected_demand']} vs "
                f"{part['on_hand'] if part['stocked'] else 0} on hand · bites "
                f"{part['bites_at']} · {part['bites_note']}"
            )
    else:
        lines.append(
            "  (none — projected demand inside the horizon is covered "
            "by the shelf)"
        )
    gaps = (handoff.get("knowledge_gaps") or {}).get("gaps") or []
    lines += ["", "KNOWLEDGE GAPS — WHAT THE PLANT DOESN'T KNOW:"]
    if gaps:
        for gap in gaps:
            lines.append(
                f"  - {gap['machine_id']} · {gap['fault_class']} — no SOP on "
                f"record · {gap['blocked_cases']} case(s) blocked · "
                f"{gap['rounds_burned']} round(s) burned · "
                + (
                    f"blocked now: {', '.join(gap['open_case_ids'])}"
                    if gap["open_now"]
                    else "not blocking now — history"
                )
            )
            if gap.get("standing_obligation_ids"):
                lines.append(
                    "      standing PM obligation(s) on the same hole: "
                    + ", ".join(gap["standing_obligation_ids"])
                )
        lines.append(
            "  (the report advises the knowledge base's maintainer — "
            "it drafts no SOP; authoring one is a human's act)"
        )
    else:
        lines.append(
            "  (none — every non-converged case's diagnosed fault has "
            "an SOP on record)"
        )
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
