"""Guardrails as code: the checks a plan and a draft must survive.

The verify round asks the reasoner to criticise the plan; these
checks do not ask — they measure, against ground truth the model
cannot argue with (the SOP's mandatory safety list, the shelf count
in inventory, the banned-action screen):

- **safety completeness** — every mandatory safety step of the
  diagnosed SOP must be present in the plan. A plan missing one can
  never converge, however confident the diagnosis.
- **parts reality** — every required part must be satisfied against
  inventory (directly or via the SOP-approved substitute); a plan
  step that names a part the plant does not stock is a defect.
- **banned actions** — no plan step may recommend defeating a safety
  device or skipping lockout/tagout (screening.py). A safety-class
  banned action stops the loop immediately.
- **SOP contradiction** — a step that cites another machine's SOP
  does not belong in this plan.
- **draft language** — a draft work order is a proposal, not a
  report: it may not state work as performed, completed, or the
  machine as restored. Screened on the final text.

Each check produces :class:`~plant_loop.schemas.Gap` defects for the
loop (so the loop can work them) and a pass/fail line for the
record's guardrail section.
"""

from __future__ import annotations

import re

from . import kb, screening
from .schemas import (
    ActionPlan,
    CaseResult,
    Diagnosis,
    ExceptionCase,
    Gap,
    GuardrailCheck,
    GuardrailInfo,
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _safety_step_present(mandatory: dict, plan: ActionPlan) -> bool:
    """Is one mandatory SOP safety step present in the plan?

    Matched by step id (code-built plans) or by token overlap with
    the step text (model-built plans) — the substance must be there,
    not merely a step with a safety-flavoured title.
    """
    target = _tokens(mandatory["text"])
    if not target:
        return True
    for step in plan.steps:
        if step.step_id.endswith(mandatory["step_id"]) and (step.kind == "safety" or step.mandatory):
            return True
        if step.kind == "safety" or step.mandatory:
            overlap = len(target & _tokens(step.text)) / len(target)
            if overlap >= 0.55:
                return True
    return False


def verify_plan(case: ExceptionCase, diagnosis: Diagnosis, plan: ActionPlan) -> list[Gap]:
    """The code-side verify: gaps the reasoner's critique cannot veto."""
    gaps: list[Gap] = []
    top = diagnosis.top()
    sop = kb.sop_for(case.machine_id, top.fault_class) if top else None

    # 1 · mandatory safety steps — the non-negotiable list.
    for mandatory in kb.mandatory_safety_steps(sop):
        if not _safety_step_present(mandatory, plan):
            gaps.append(
                Gap(
                    kind="missing_safety_step",
                    detail=(
                        f"plan omits the mandatory safety step '{mandatory['step_id']}' "
                        f"from {sop['sop_id']}: {mandatory['text']}"
                    ),
                    ref=mandatory["step_id"],
                    blocking=True,
                    safety=True,
                )
            )

    # 2 · parts reality — requirements the shelf cannot satisfy, and
    # steps naming parts the plant does not stock at all.
    for req in plan.required_parts:
        if not req.satisfied:
            gaps.append(
                Gap(
                    kind="part_unavailable",
                    detail=(
                        f"part {req.part_id} needs {req.qty}, shelf holds {req.available_qty}, "
                        "and no approved substitute is available"
                    ),
                    ref=req.part_id,
                    blocking=True,
                )
            )
    for step in plan.steps:
        if step.part_id and kb.part_qty(step.part_id) is None:
            gaps.append(
                Gap(
                    kind="part_unavailable",
                    detail=f"step '{step.step_id}' names part {step.part_id}, which the plant does not stock",
                    ref=step.part_id,
                    blocking=True,
                )
            )

    # 3 · banned actions — screened over every step, whoever wrote it.
    for step in plan.steps:
        banned, is_safety = screening.screen_plan_step(step.text)
        if banned:
            gaps.append(
                Gap(
                    kind="banned_action",
                    detail=f"step '{step.step_id}' recommends a banned action: {step.text}",
                    ref=step.step_id,
                    blocking=True,
                    safety=is_safety,
                )
            )

    # 4 · SOP contradiction — a step citing another machine's SOP.
    machine_sops = {s["sop_id"] for s in kb.sops_for_machine(case.machine_id)}
    for step in plan.steps:
        if step.sop_ref and step.sop_ref not in machine_sops:
            gaps.append(
                Gap(
                    kind="sop_contradiction",
                    detail=(
                        f"step '{step.step_id}' cites {step.sop_ref}, which is not an SOP "
                        f"for {case.machine_id or 'this case'}"
                    ),
                    ref=step.step_id,
                    blocking=True,
                )
            )
    return gaps


# Language a draft may never use: the past tense of work nobody has
# authorised, let alone done.
_PERFORMED_PHRASES = (
    "work has been performed",
    "has been repaired",
    "has been fixed",
    "repair is complete",
    "repair completed",
    "we replaced",
    "we fitted",
    "we have replaced",
    "machine restarted",
    "back in production",
    "returned to production",
)


def draft_language_ok(text: str) -> tuple[bool, str]:
    lowered = (text or "").lower()
    for phrase in _PERFORMED_PHRASES:
        if phrase in lowered:
            return False, f"draft states work as performed: '...{phrase}...'"
    return True, "draft proposes work; it states nothing as performed"


def run_guardrails(result: CaseResult) -> GuardrailInfo:
    """The final guardrail verdict printed on the record.

    Recomputed from the finished record (diagnosis, plan, draft) so
    the section always describes the record it sits on — including
    the honest failures on exhausted / blocked cases.
    """
    case = ExceptionCase(
        case_id=result.case_id, type=result.type, title=result.title,
        machine_id=result.machine_id, line_id=result.line_id,
    )
    gaps = verify_plan(case, result.diagnosis, result.plan)
    kinds = {g.kind for g in gaps}
    top = result.diagnosis.top()
    language_ok, language_detail = draft_language_ok(result.draft.text)
    checks = [
        GuardrailCheck(
            name="evidence_present",
            passed=bool(top and top.evidence),
            detail=(
                f"top hypothesis {top.fault_class} cites {len(top.evidence)} evidence item(s)"
                if top else "no diagnosis on the record"
            ),
        ),
        GuardrailCheck(
            name="safety_steps_complete",
            passed="missing_safety_step" not in kinds,
            detail="every mandatory SOP safety step is in the plan"
            if "missing_safety_step" not in kinds
            else "the plan omits a mandatory SOP safety step — it cannot be authorised as-is",
        ),
        GuardrailCheck(
            name="parts_available",
            passed="part_unavailable" not in kinds,
            detail="every required part is on the shelf or substituted per the SOP"
            if "part_unavailable" not in kinds
            else "a required part is unavailable — outside help or stock is needed",
        ),
        GuardrailCheck(
            name="no_banned_actions",
            passed="banned_action" not in kinds,
            detail="no plan step recommends defeating a safety device or skipping lockout/tagout"
            if "banned_action" not in kinds
            else "a plan step recommends a banned action",
        ),
        GuardrailCheck(
            name="draft_language",
            passed=language_ok,
            detail=language_detail,
        ),
    ]
    return GuardrailInfo(passed=all(c.passed for c in checks), checks=checks)
