"""The work-order packet: a gated case, printed for the floor.

The draft work order on a record is written for the supervisor who
approves it. The packet is written for the technician who carries
it: the same facts, laid out for paper — the plan steps in order,
the mandatory safety steps again as a check-off list (a step you
must tick is harder to skip than a step you must remember), the
parts with any substitution flagged in the open, and blank sign-off
lines for the supervisor, the technician, and the completion facts
the floor writes back.

Only a converged case has a verified work order to packet. Any
other exit produced an escalation draft instead, and issuing a
floor packet from an unverified plan would put unchecked steps on
the floor — so :func:`build_packet` refuses (the API maps that to
409). A case the supervisor *rejected* is refused too: its draft
must not reach the floor dressed as a work order. An approved case
packets with the approval on its banner; an awaiting one packets
as what it is — a draft that authorises nothing until the gate.

Composed entirely in code from the stored record, JSON and text
from the same dict — the handoff idiom (handoff.py). No model
writes any part of it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .events import format_timestamp
from .schemas import EXIT_CONVERGED, GATE_NOTICE, CaseResult


def build_packet(record: CaseResult, *, now: datetime | None = None) -> dict:
    """The packet payload for one gated case.

    Raises :class:`ValueError` when the case has no verified work
    order to packet (a non-converged exit, or a rejected draft).
    """
    if record.exit_reason != EXIT_CONVERGED:
        raise ValueError(
            f"case {record.case_id} exited {record.exit_reason} — there is no verified "
            "work order to packet; the escalation draft on the record is this case's document"
        )
    if record.status == "rejected":
        raise ValueError(
            f"case {record.case_id} was rejected at the supervisor gate — its draft "
            "must not reach the floor as a work-order packet"
        )
    moment = now or datetime.now(timezone.utc)
    authorised = record.status == "approved"
    banner = (
        f"APPROVED AT THE GATE by {record.decision_by or 'a supervisor'} — "
        "this packet carries the approved plan"
        if authorised
        else "DRAFT — NOT AUTHORISED UNTIL A SUPERVISOR APPROVES AT THE GATE"
    )
    top = record.diagnosis.top()
    steps = [
        {
            "n": index,
            "step_id": step.step_id,
            "kind": step.kind,
            "mandatory": step.mandatory,
            "text": step.text,
            "part_id": step.part_id,
            "qty": step.qty,
        }
        for index, step in enumerate(record.plan.steps, start=1)
    ]
    safety_checklist = [
        {"step_id": step.step_id, "text": step.text}
        for step in record.plan.steps
        if step.kind == "safety" or step.mandatory
    ]
    parts = [
        {
            "part_id": req.part_id,
            "qty": req.qty,
            "on_hand": req.available_qty,
            "substituted": bool(req.substitute_part_id),
            "substitute_part_id": req.substitute_part_id,
        }
        for req in record.plan.required_parts
    ]
    return {
        "case_id": record.case_id,
        "generated_at": format_timestamp(moment),
        "authorised": authorised,
        "banner": banner,
        "gate_notice": GATE_NOTICE,
        "case": {
            "case_id": record.case_id,
            "type": record.type,
            "title": record.title,
            "machine_id": record.machine_id,
            "line_id": record.line_id,
            "priority": record.priority,
            "priority_reason": record.priority_reason,
            "reported_at": record.reported_at,
            "created_at": record.created_at,
            "status": record.status,
        },
        "diagnosis": (
            {
                "fault_class": top.fault_class,
                "confidence": top.confidence,
                "sop_id": top.sop_id,
                "statement": top.statement,
                "evidence": list(top.evidence),
            }
            if top is not None
            else None
        ),
        "steps": steps,
        "safety_checklist": safety_checklist,
        "parts": parts,
        "substitutions": [
            s.model_dump() for s in record.plan.substitutions if s.status == "applied"
        ],
        "estimated_downtime_min": record.plan.estimated_downtime_min,
        "rounds": len(record.rounds),
        "signoff": {
            "supervisor": "",
            "supervisor_at": "",
            "technician": "",
            "technician_at": "",
            "actual_downtime_min": None,
            "parts_used": [],
            "resolved": None,
        },
    }


def render_text(packet: dict) -> str:
    """The packet as plain text — what gets printed for the floor."""
    case = packet["case"]
    lines = [
        f"WORK ORDER PACKET — {case['case_id']} · {case['title']}",
        packet["banner"],
        packet["gate_notice"],
        "",
        "CASE",
        f"  Type: {case['type']} · Machine: {case['machine_id'] or '—'} · Line: {case['line_id'] or '—'}",
        f"  Priority: {case['priority']} ({case['priority_reason']})",
        f"  Reported: {case['reported_at'] or '—'} · Record created: {case['created_at'] or '—'}",
        "",
        "DIAGNOSIS",
    ]
    diagnosis = packet["diagnosis"]
    if diagnosis is None:
        lines.append("  (no diagnosis on the record)")
    else:
        lines.append(
            f"  {diagnosis['fault_class']} (confidence {diagnosis['confidence']:.2f}, "
            f"SOP {diagnosis['sop_id'] or '—'})"
        )
        lines.append(f"  {diagnosis['statement']}")
        for item in diagnosis["evidence"]:
            lines.append(f"  evidence: {item}")
    lines += ["", "MANDATORY SAFETY CHECKLIST — check off each step before work starts:"]
    if packet["safety_checklist"]:
        for item in packet["safety_checklist"]:
            lines.append(f"  [ ] {item['text']}")
    else:
        lines.append("  (the SOP for this fault specifies no mandatory safety steps)")
    lines += ["", "PLAN STEPS, IN ORDER:"]
    for step in packet["steps"]:
        part = f" [part: {step['part_id']} x{step['qty']}]" if step["part_id"] else ""
        lines.append(f"  {step['n']}. ({step['kind']}) {step['text']}{part}")
    lines += ["", "PARTS:"]
    if packet["parts"]:
        by_part = {s["original_part_id"]: s for s in packet["substitutions"]}
        for part in packet["parts"]:
            if part["substituted"]:
                sub = by_part.get(part["part_id"])
                lines.append(
                    f"  - {part['part_id']} x{part['qty']} — SUBSTITUTION: "
                    f"{part['substitute_part_id']} is fitted in place of {part['part_id']} "
                    f"({part['on_hand']} of the original on hand)"
                )
                if sub is not None:
                    if sub["basis"]:
                        lines.append(f"    basis: {sub['basis']}")
                    for delta in sub["spec_deltas"]:
                        note = f" ({delta['note']})" if delta["note"] else ""
                        lines.append(
                            f"    delta: {delta['attribute']} {delta['original']} "
                            f"-> {delta['substitute']}{note}"
                        )
            else:
                lines.append(f"  - {part['part_id']} x{part['qty']} — {part['on_hand']} on hand")
    else:
        lines.append("  (no parts required)")
    lines += [
        "",
        f"ESTIMATED DOWNTIME: {packet['estimated_downtime_min']} min "
        f"(loop: {packet['rounds']} round(s))",
        "",
        "SIGN-OFF",
        "  Supervisor: ______________________________  Date/time: ______________",
        "  Technician: ______________________________  Date/time: ______________",
        "  Completion: actual downtime ______ min · parts used ______________________ · resolved (Y/N) ______",
    ]
    return "\n".join(lines) + "\n"
