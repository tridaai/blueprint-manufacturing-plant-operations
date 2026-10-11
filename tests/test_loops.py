"""Loop-controller behaviour, driven directly (no graph, no HTTP).

The scripted-critique cases isolate the controller: the offline
diagnose/plan are known-good, so the exit under test is decided by
the scripted gap sequence and the policy alone.
"""

from __future__ import annotations

from plant_loop.loops import BoardLoopController, BoardPolicy, LoopPolicy
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import Gap

from helpers import CountingReasoner, ScriptedCritiqueReasoner, make_controller

CASES = {c.case_id: c for c in load_sample_cases()}


def _text(case) -> str:
    return f"{case.title}\n{case.description}\n{case.notes}"


def test_converges_round_one_when_verify_is_clean():
    controller = make_controller(ScriptedCritiqueReasoner([[]]))
    outcome = controller.run_case(CASES["EXC-105"], _text(CASES["EXC-105"]))
    assert outcome.exit_reason == "converged"
    assert len(outcome.rounds) == 1
    assert outcome.open_gaps == []


def test_converges_round_two_after_gap_closure():
    gap = Gap(kind="competing_hypothesis", detail="rival not separated", ref="tool_imbalance")
    controller = make_controller(ScriptedCritiqueReasoner([[gap], []]))
    outcome = controller.run_case(CASES["EXC-105"], _text(CASES["EXC-105"]))
    assert outcome.exit_reason == "converged"
    assert len(outcome.rounds) == 2
    assert outcome.rounds[0].gaps_opened  # the gap was on the timeline
    assert "competing_hypothesis:tool_imbalance" in outcome.rounds[1].gaps_closed


def test_exhausts_when_gaps_never_close():
    gap = Gap(kind="competing_hypothesis", detail="still ambiguous", ref="motor_overheat")
    controller = make_controller(ScriptedCritiqueReasoner([[gap]]))
    outcome = controller.run_case(CASES["EXC-109"], _text(CASES["EXC-109"]))
    assert outcome.exit_reason == "exhausted"
    assert len(outcome.rounds) == 3
    assert outcome.open_gaps and outcome.open_gaps[0].kind == "competing_hypothesis"


def test_budget_exceeded_beats_exhaustion():
    gap = Gap(kind="competing_hypothesis", detail="still ambiguous", ref="motor_overheat")
    controller = make_controller(
        ScriptedCritiqueReasoner([[gap]]), policy=LoopPolicy(max_rounds=3, case_token_budget=1)
    )
    outcome = controller.run_case(CASES["EXC-109"], _text(CASES["EXC-109"]))
    assert outcome.exit_reason == "budget_exceeded"
    assert len(outcome.rounds) == 1


def test_safety_case_stops_at_round_zero_without_reasoning():
    reasoner = CountingReasoner()
    controller = make_controller(reasoner)
    outcome = controller.run_case(CASES["EXC-102"], _text(CASES["EXC-102"]))
    assert outcome.exit_reason == "safety_stop"
    assert outcome.rounds == []
    assert reasoner.calls == {"diagnose": 0, "plan": 0, "critique": 0}
    assert outcome.input_tokens == 0 and outcome.output_tokens == 0


def test_blocked_external_when_part_unavailable():
    controller = make_controller()
    outcome = controller.run_case(CASES["EXC-107"], _text(CASES["EXC-107"]))
    assert outcome.exit_reason == "blocked_external"
    assert len(outcome.rounds) == 1
    assert any(g.kind == "part_unavailable" and g.ref == "SPN-CART" for g in outcome.open_gaps)


def test_offline_two_round_convergence_on_the_bearing_case():
    controller = make_controller()
    outcome = controller.run_case(CASES["EXC-101"], _text(CASES["EXC-101"]))
    assert outcome.exit_reason == "converged"
    assert len(outcome.rounds) == 2
    top = outcome.diagnosis.top()
    assert top.fault_class == "spindle_bearing_wear"
    # Round 2's discriminating check ruled the rival out.
    rival = next(h for h in outcome.diagnosis.hypotheses if h.fault_class == "tool_imbalance")
    assert rival.confidence <= 0.2


def test_board_controller_order_and_handoff():
    controller = BoardLoopController(policy=BoardPolicy(max_cases=3))
    result, outcomes = controller.run(
        load_sample_cases(),
        process=lambda case: make_controller().run_case(case, _text(case)),
    )
    assert result.processed == ["EXC-102", "EXC-107", "EXC-105"]
    assert result.status == "shift_budget_exceeded"
    assert [item["case_id"] for item in result.remaining] == [
        "EXC-101", "EXC-109", "EXC-103", "EXC-108", "EXC-104", "EXC-106",
    ]
    assert "handed to the next shift" in result.handoff_summary
    assert len(outcomes) == 3
