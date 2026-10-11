"""Drafts: the work order and the escalation note.

Drafts are deterministic templates over the loop's outputs — the
model never writes them, so their language is controlled by
construction (and still screened afterwards by the guardrails'
draft-language check). Both carry the same spine: what the case is,
what the loop concluded and on what evidence, what is proposed or
needed, and the gate notice. A draft is a proposal to a supervisor;
it authorises nothing, and says so.
"""

from __future__ import annotations

from . import kb
from .schemas import GATE_NOTICE, CaseResult, ExceptionCase


def _header(case: ExceptionCase) -> list[str]:
    machine = kb.machines().get(case.machine_id, {})
    line = kb.lines().get(case.line_id, {})
    return [
        f"Case {case.case_id} · {case.type} · {case.title}",
        f"Machine: {case.machine_id or '—'} ({machine.get('name', '')}) · "
        f"Line: {line.get('name', case.line_id or '—')}",
        f"Product/batch: {case.product or '—'} / {case.batch or '—'} · "
        f"Reported by {case.reporter or 'unknown'} at {case.reported_at or '—'}",
    ]


def _substitution_lines(result: CaseResult, status: str) -> list[str]:
    """The plan's substitution records of one status, rendered."""
    lines = []
    for record in result.plan.substitutions:
        if record.status != status:
            continue
        lines.append(f"  - {record.original_part_id} -> {record.substitute_part_id}")
        if record.basis:
            lines.append(f"    basis: {record.basis}")
        for delta in record.spec_deltas:
            note = f" ({delta.note})" if delta.note else ""
            lines.append(
                f"    delta: {delta.attribute} {delta.original} -> {delta.substitute}{note}"
            )
        if record.reason:
            lines.append(f"    {record.reason}")
    return lines


def build_work_order(case: ExceptionCase, result: CaseResult) -> str:
    """The draft work order for a converged case."""
    lines = ["DRAFT WORK ORDER — NOT AUTHORISED UNTIL A SUPERVISOR APPROVES", ""]
    lines += _header(case)
    top = result.diagnosis.top()
    lines.append("")
    if top is not None:
        lines.append(f"Diagnosis: {top.fault_class} (confidence {top.confidence:.2f}, SOP {top.sop_id or '—'})")
        lines.append(f"  {top.statement}")
        for item in top.evidence:
            lines.append(f"  evidence: {item}")
    applied = [s for s in result.plan.substitutions if s.status == "applied"]
    if applied:
        lines.append("")
        lines.append(
            "SUBSTITUTION — this plan runs on substitute part(s). "
            "The supervisor approves the substitution knowingly, or not at all:"
        )
        lines += _substitution_lines(result, "applied")
    lines.append("")
    lines.append("MANDATORY SAFETY STEPS (non-removable):")
    safety_steps = [s for s in result.plan.steps if s.kind == "safety" or s.mandatory]
    if safety_steps:
        for step in safety_steps:
            lines.append(f"  - {step.text}")
    else:
        lines.append("  - (none specified by the SOP for this fault)")
    lines.append("")
    lines.append("Proposed steps, in order:")
    for index, step in enumerate(result.plan.steps, start=1):
        part = f" [part: {step.part_id} ×{step.qty}]" if step.part_id else ""
        lines.append(f"  {index}. ({step.kind}) {step.text}{part}")
    lines.append("")
    if result.plan.required_parts:
        lines.append("Parts:")
        for req in result.plan.required_parts:
            if req.substitute_part_id:
                lines.append(
                    f"  - {req.part_id} ×{req.qty} — out of stock; approved substitute "
                    f"{req.substitute_part_id} on hand ({req.available_qty} of the original)"
                )
            else:
                lines.append(f"  - {req.part_id} ×{req.qty} — {req.available_qty} on hand")
    lines.append(f"Estimated downtime: {result.plan.estimated_downtime_min} min")
    lines.append(f"Loop: {len(result.rounds)} round(s), exit {result.exit_reason}.")
    lines.append("")
    lines.append(GATE_NOTICE)
    return "\n".join(lines)


_EXIT_TITLES = {
    "safety_stop": "ESCALATION — SAFETY (immediate supervisor attention)",
    "blocked_external": "ESCALATION — BLOCKED ON PARTS / VENDOR",
    "exhausted": "ESCALATION — LOOP DID NOT CONVERGE",
    "budget_exceeded": "ESCALATION — CASE BUDGET EXCEEDED",
}


def build_escalation(case: ExceptionCase, result: CaseResult) -> str:
    """The draft escalation note for every non-converged exit."""
    lines = [f"DRAFT {_EXIT_TITLES.get(result.exit_reason, 'ESCALATION')} — FOR SUPERVISOR DECISION", ""]
    lines += _header(case)
    lines.append("")
    if result.exit_reason == "safety_stop":
        lines.append(
            "This case stopped before any diagnosis round: safety cases and safety "
            "bypasses are never worked by the agent loop. A supervisor owns this "
            "case from here."
        )
        if case.description:
            lines.append(f"Observation: {case.description}")
    top = result.diagnosis.top()
    if top is not None:
        lines.append(f"Best diagnosis reached: {top.fault_class} (confidence {top.confidence:.2f})")
        for item in top.evidence:
            lines.append(f"  evidence: {item}")
    if result.open_gaps:
        lines.append("")
        lines.append("Open gaps the loop could not close:")
        for gap in result.open_gaps:
            lines.append(f"  - [{gap.kind}] {gap.detail}")
    if result.exit_reason == "blocked_external":
        searched = {s.original_part_id for s in result.plan.substitutions}
        missing = [r.part_id for r in result.plan.required_parts if not r.satisfied]
        lines.append("")
        lines.append("Substitute search (run at the verify stage, in code):")
        if result.plan.substitutions:
            lines += _substitution_lines(result, "rejected")
        for part_id in missing:
            if part_id not in searched:
                lines.append(
                    f"  - {part_id}: no substitute is listed in the KB "
                    "compatibility relation for this part"
                )
        lines.append("")
        lines.append(
            "What is needed from outside the loop: the part(s) above ordered or the "
            "vendor engaged. The plan on the record is ready to execute the moment "
            "the part lands — it is not authorised until a supervisor approves it."
        )
    lines.append("")
    lines.append(f"Loop: {len(result.rounds)} round(s), exit {result.exit_reason}.")
    lines.append("")
    lines.append(GATE_NOTICE)
    return "\n".join(lines)
