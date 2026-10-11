"""Guardrails as code: the checks a plan cannot talk its way past."""

from __future__ import annotations

from plant_loop import guardrails, kb
from plant_loop.backends import OfflineReasoner
from plant_loop.ports import CaseContext
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import ActionPlan, PartRequirement, PlanStep

CASES = {c.case_id: c for c in load_sample_cases()}
REASONER = OfflineReasoner()


def _diagnose_and_plan(case_id: str):
    case = CASES[case_id]
    ctx = CaseContext(case=case, case_text=f"{case.title}\n{case.description}")
    diagnosis = REASONER.diagnose(ctx, focus=[]).payload
    plan = REASONER.draft_plan(ctx, diagnosis, focus=[]).payload
    return case, diagnosis, plan


def test_clean_plan_has_no_code_gaps():
    case, diagnosis, plan = _diagnose_and_plan("EXC-105")
    assert guardrails.verify_plan(case, diagnosis, plan) == []


def test_missing_safety_step_is_a_blocking_safety_gap():
    case, diagnosis, plan = _diagnose_and_plan("EXC-105")
    plan.steps = [s for s in plan.steps if s.kind != "safety"]
    gaps = guardrails.verify_plan(case, diagnosis, plan)
    missing = [g for g in gaps if g.kind == "missing_safety_step"]
    assert len(missing) == 2  # both PRESS-1 mandatory steps
    assert all(g.blocking and g.safety for g in missing)


def test_unavailable_part_is_a_gap_with_the_part_named():
    case, diagnosis, plan = _diagnose_and_plan("EXC-107")
    gaps = guardrails.verify_plan(case, diagnosis, plan)
    part_gaps = [g for g in gaps if g.kind == "part_unavailable"]
    assert part_gaps and part_gaps[0].ref == "SPN-CART"


def test_step_naming_an_unstocked_part_is_a_gap():
    case, diagnosis, plan = _diagnose_and_plan("EXC-105")
    plan.steps.append(PlanStep(step_id="x", text="Fit the generic spare.", part_id="NOPE-1", qty=1))
    gaps = guardrails.verify_plan(case, diagnosis, plan)
    assert any(g.kind == "part_unavailable" and g.ref == "NOPE-1" for g in gaps)


def test_banned_action_step_is_flagged_safety():
    case, diagnosis, plan = _diagnose_and_plan("EXC-105")
    plan.steps.append(PlanStep(step_id="bad", text="Jumper the interlock switch and cycle the press."))
    gaps = guardrails.verify_plan(case, diagnosis, plan)
    banned = [g for g in gaps if g.kind == "banned_action"]
    assert banned and banned[0].safety is True


def test_foreign_sop_reference_is_a_contradiction():
    case, diagnosis, plan = _diagnose_and_plan("EXC-105")
    plan.steps.append(PlanStep(step_id="odd", text="Follow the welder procedure.", sop_ref="SOP-WELD4-DRIFT"))
    gaps = guardrails.verify_plan(case, diagnosis, plan)
    assert any(g.kind == "sop_contradiction" for g in gaps)


def test_substitute_part_satisfies_the_requirement():
    # The coolant case: CLNT-FLT is out of stock. The drafted plan
    # now carries the requirement honestly unsatisfied — substitution
    # moved out of the planner in round 3 — and the loop's verify
    # stage searches the KB compatibility relation, applies
    # CLNT-FLT-E, and only then does the plan verify clean, with the
    # substitution recorded on the plan.
    from plant_loop.schemas import ExceptionCase

    from helpers import make_controller

    case = ExceptionCase.model_validate({
        "case_id": "T-COOL", "type": "machine_down", "title": "Coolant flow low",
        "description": "", "machine_id": "CNC-3", "line_id": "line-b",
        "readings": {"coolant_flow_l_min": 3.0},
    })
    ctx = CaseContext(case=case, case_text="coolant flow low")
    diagnosis = REASONER.diagnose(ctx, focus=[]).payload
    plan = REASONER.draft_plan(ctx, diagnosis, focus=[]).payload
    req = next(r for r in plan.required_parts if r.part_id == "CLNT-FLT")
    assert not req.satisfied and not req.substitute_part_id  # the planner no longer swaps

    outcome = make_controller().run_case(case, "coolant flow low")
    assert outcome.exit_reason == "converged"
    req = next(r for r in outcome.plan.required_parts if r.part_id == "CLNT-FLT")
    assert req.satisfied and req.substitute_part_id == "CLNT-FLT-E"
    applied = [s for s in outcome.plan.substitutions if s.status == "applied"]
    assert [s.substitute_part_id for s in applied] == ["CLNT-FLT-E"]
    assert guardrails.verify_plan(case, outcome.diagnosis, outcome.plan) == []


def test_draft_language_screen():
    ok, _ = guardrails.draft_language_ok("Fit the replacement bearing and run the run-in cycle.")
    assert ok
    bad, detail = guardrails.draft_language_ok("We replaced the bearing and the machine is back in production.")
    assert not bad and "performed" in detail


def test_every_sop_required_part_is_on_the_register():
    """KB integrity: the planner resolves parts against the register,
    so every part an SOP names must be ON it — stocked, stocked-out,
    or stocked-out-with-a-substitute are the three honest states (the
    stocked-out-no-substitute ones are what blocked_external is for)."""
    for sop in kb.sops():
        for req in sop.get("required_parts", []):
            assert kb.part_qty(req["part_id"]) is not None, (
                f"{sop['sop_id']} requires {req['part_id']}, which is not on the register"
            )
