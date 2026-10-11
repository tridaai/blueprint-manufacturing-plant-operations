"""Round 3: the verify stage's substitute search.

The search used to be a habit of the offline planner, which swapped
a stocked-out part while drafting and left no paperwork. These
tests pin the new ownership: the loop's verify stage re-measures
requirements against the shelf, walks the KB compatibility relation,
checks SOP constraints, applies a winner to the plan as a recorded
substitution — and documents every rejection on the record of the
case it could not save.
"""

from __future__ import annotations

from plant_loop import kb, substitution
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import ExceptionCase, Gap

from helpers import ScriptedCritiqueReasoner, make_controller

CASES = {c.case_id: c for c in load_sample_cases()}

COOLANT_CASE = {
    "case_id": "T-COOL",
    "type": "machine_down",
    "title": "CNC-3 coolant flow low, spindle hot",
    "description": "Coolant flow alarm; spindle temperature climbing.",
    "machine_id": "CNC-3",
    "line_id": "line-b",
    "readings": {"coolant_flow_l_min": 3.0, "spindle_temp_c": 84},
    "reporter": "J. Okafor (operator)",
    "reported_at": "2026-10-11T06:40:00Z",
}


def _text(case) -> str:
    return f"{case.title}\n{case.description}\n{case.notes}"


# ---------------------------------------------------------------------------
# The search itself, against the KB
# ---------------------------------------------------------------------------


def test_evaluate_finds_the_extended_life_filter():
    sop = kb.sop_for("CNC-3", "coolant_system_fault")
    winner, records = substitution.evaluate(sop, "CLNT-FLT", 1)
    assert winner is not None and winner.substitute_part_id == "CLNT-FLT-E"
    assert winner.status == "applied"
    assert "spin-on" in winner.basis
    deltas = {d.attribute: d for d in winner.spec_deltas}
    assert deltas["micron_rating_um"].original == 25
    assert deltas["micron_rating_um"].substitute == 10
    assert records == [winner]  # the first candidate cleared; search stopped


def test_evaluate_rejects_the_generic_fan_on_its_rating():
    sop = kb.sop_for("COMP-7", "cooling_fan_failure")
    winner, records = substitution.evaluate(sop, "FAN-C7", 1)
    assert winner is None
    assert len(records) == 1
    record = records[0]
    assert record.substitute_part_id == "FAN-C7-G" and record.status == "rejected"
    # The rejection names the constraint it failed — the SOP's
    # airflow floor — not a shrug.
    assert "airflow_m3h" in record.reason and "420" in record.reason


def test_evaluate_with_no_relation_considers_nothing():
    sop = kb.sop_for("CNC-2", "spindle_cartridge_failure")
    winner, records = substitution.evaluate(sop, "SPN-CART", 1)
    assert winner is None and records == []


def test_evaluate_rejects_a_candidate_the_shelf_cannot_cover():
    sop = kb.sop_for("CNC-3", "coolant_system_fault")
    winner, records = substitution.evaluate(sop, "CLNT-FLT", 99)
    assert winner is None
    assert "shelf holds 4" in records[0].reason and "needs 99" in records[0].reason


def test_constraint_check_fails_closed_on_an_unspecced_candidate():
    sop = kb.sop_for("COMP-7", "cooling_fan_failure")
    # BRG-6208 carries no airflow spec at all — unverifiable is a fail.
    violations = substitution.constraint_violations(sop, "FAN-C7", "BRG-6208")
    assert violations and "no airflow_m3h spec" in violations[0]


# ---------------------------------------------------------------------------
# The loop, with scripted critics isolating the code pass
# ---------------------------------------------------------------------------


def test_verify_stage_applies_the_substitute_and_converges():
    case = ExceptionCase.model_validate(COOLANT_CASE)
    controller = make_controller(ScriptedCritiqueReasoner([[]]))
    outcome = controller.run_case(case, _text(case))
    assert outcome.exit_reason == "converged"
    assert len(outcome.rounds) == 1
    applied = [s for s in outcome.plan.substitutions if s.status == "applied"]
    assert len(applied) == 1
    record = applied[0]
    assert record.original_part_id == "CLNT-FLT"
    assert record.substitute_part_id == "CLNT-FLT-E"
    assert record.basis and record.spec_deltas  # the record carries the why
    # The plan itself was revised onto the substitute.
    req = next(r for r in outcome.plan.required_parts if r.part_id == "CLNT-FLT")
    assert req.satisfied and req.substitute_part_id == "CLNT-FLT-E"
    step = next(s for s in outcome.plan.steps if s.step_id == "replace-filter")
    assert step.part_id == "CLNT-FLT-E" and "SUBSTITUTION" in step.text
    # And the timeline shows the search happened.
    assert outcome.rounds[0].substitutions == ["CLNT-FLT->CLNT-FLT-E applied"]


def test_rejected_substitute_keeps_the_case_blocked_with_the_reason_on_record():
    controller = make_controller(ScriptedCritiqueReasoner([[]]))
    outcome = controller.run_case(CASES["EXC-106"], _text(CASES["EXC-106"]))
    assert outcome.exit_reason == "blocked_external"
    rejected = [s for s in outcome.plan.substitutions if s.status == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].substitute_part_id == "FAN-C7-G"
    assert "airflow_m3h" in rejected[0].reason
    gap = next(g for g in outcome.open_gaps if g.kind == "part_unavailable")
    assert gap.ref == "FAN-C7"
    # The surviving gap documents the search — the escalation a
    # supervisor reads says what was tried and why it failed.
    assert "substitute search" in gap.detail and "FAN-C7-G rejected" in gap.detail
    assert outcome.rounds[0].substitutions == ["FAN-C7-G rejected for FAN-C7"]


def test_no_substitute_listed_blocks_with_the_search_documented_as_empty():
    controller = make_controller(ScriptedCritiqueReasoner([[]]))
    outcome = controller.run_case(CASES["EXC-107"], _text(CASES["EXC-107"]))
    assert outcome.exit_reason == "blocked_external"
    assert outcome.plan.substitutions == []
    gap = next(g for g in outcome.open_gaps if g.kind == "part_unavailable")
    assert "no substitute is listed" in gap.detail


def test_a_critics_own_part_gap_is_retired_when_the_pass_resolves_the_part():
    # The model critic flags CLNT-FLT itself; the pass applies the
    # substitute, so the critic's (accurate, now stale) gap must not
    # survive to block a plan the shelf can satisfy.
    case = ExceptionCase.model_validate(COOLANT_CASE)
    model_gap = Gap(kind="part_unavailable", detail="the filter is out of stock", ref="CLNT-FLT")
    controller = make_controller(ScriptedCritiqueReasoner([[model_gap]]))
    outcome = controller.run_case(case, _text(case))
    assert outcome.exit_reason == "converged"
    assert outcome.open_gaps == []


def test_a_planner_cannot_smuggle_an_unverified_substitute_past_the_pass():
    # A plan that arrives claiming a substitute the search would
    # reject (the generic fan, marked satisfied by the planner) is
    # re-measured: the claim is stripped and the case stays blocked.
    from plant_loop.backends import OfflineReasoner
    from plant_loop.ports import RoundOutput
    from plant_loop.schemas import PartRequirement

    class SmugglingPlanner(OfflineReasoner):
        def draft_plan(self, ctx, diagnosis, focus):
            out = super().draft_plan(ctx, diagnosis, focus)
            out.payload.required_parts = [
                PartRequirement(
                    part_id="FAN-C7", qty=1, available_qty=0,
                    satisfied=True, substitute_part_id="FAN-C7-G",
                )
            ]
            return RoundOutput(
                payload=out.payload, input_tokens=out.input_tokens,
                output_tokens=out.output_tokens, latency_ms=out.latency_ms,
                backend=out.backend, model=out.model,
            )

    controller = make_controller(SmugglingPlanner())
    outcome = controller.run_case(CASES["EXC-106"], _text(CASES["EXC-106"]))
    assert outcome.exit_reason == "blocked_external"
    req = next(r for r in outcome.plan.required_parts if r.part_id == "FAN-C7")
    assert not req.satisfied and not req.substitute_part_id
