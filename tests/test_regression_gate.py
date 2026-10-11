"""The reasoner regression gate (regression.py): the harness, the
comparison, and every FAIL path of the verdict.

The pin-breaking candidates live HERE, in the tests, by design —
never in a shipped profile. Two constructions:

- *record-level*: a real candidate record with its pins tampered
  (a safety case that received model calls; a converged case whose
  plan dropped a mandatory step). compare_runs operates on records,
  so this is the honest unit at which a pin violation exists: the
  shipped loop's code guardrails make both unreachable in a live
  run (proven below), and the gate's pin criteria stand guard over
  records from any harness, present or future.
- *through the loop*: a tests-only planner that strips safety
  steps, driven through run_suite. Its plans never converge — the
  verify round holds them, the records say so, and the gate fails
  the candidate on exit regressions instead. That is the guardrail
  doing its job, and this test pins the division of labour.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from plant_loop import regression
from plant_loop.backends import OfflineReasoner
from plant_loop.stubs import StubCandidateReasoner

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_BASELINE = REPO_ROOT / "evals" / "gate" / "baseline-offline.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return regression.run_suite(OfflineReasoner(), label="offline-baseline")


@pytest.fixture(scope="module")
def candidate() -> dict:
    return regression.run_suite(StubCandidateReasoner(), label="stub-candidate")


def _case(record: dict, case_id: str) -> dict:
    return next(c for c in record["cases"] if c["case_id"] == case_id)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def test_suite_is_the_golden_case_entries():
    cases = regression.load_suite()
    assert [c.case_id for c in cases] == [f"G{i:02d}" for i in range(1, 21)]
    safety = [c.case_id for c in cases if c.type == "safety_observation"]
    assert safety == ["G02", "G18"]


def test_baseline_record_is_deterministic_and_matches_the_committed_one(baseline):
    again = regression.run_suite(OfflineReasoner(), label="offline-baseline")
    assert again == baseline
    committed = json.loads(COMMITTED_BASELINE.read_text(encoding="utf-8"))
    assert committed == baseline


def test_baseline_record_shape_and_pins(baseline):
    assert baseline["config"] == {
        "label": "offline-baseline",
        "backend": "offline",
        "model": "rules-v1",
        "surface": "",
    }
    assert baseline["cases_total"] == 20
    assert baseline["totals"]["model_calls"] == 72
    for case_id in ("G02", "G18"):
        record = _case(baseline, case_id)
        assert record["exit"] == "safety_stop"
        assert record["model_calls"] == 0
        assert record["pins"]["safety_zero_calls"] is True
    assert regression.pin_violations(baseline) == []


def test_config_stamp_names_the_surface():
    class Fake:
        name = "muse-responses"
        model = "muse-spark-1.3"

    stamp = regression.config_stamp(Fake(), label="spark-responses")
    assert stamp == {
        "label": "spark-responses",
        "backend": "muse-responses",
        "model": "muse-spark-1.3",
        "surface": "responses",
    }


# ---------------------------------------------------------------------------
# The stub candidate: a good swap PASSES, differences reported
# ---------------------------------------------------------------------------


def test_stub_candidate_passes_with_its_two_differences(baseline, candidate):
    report = regression.compare_runs(baseline, candidate)
    assert report["verdict"] == "PASS"
    assert all(c["passed"] for c in report["criteria"])
    assert report["diagnosis"]["agreement"] == 0.95
    assert report["diagnosis"]["differences"] == [
        {"case_id": "G14", "baseline": "wire_feed_fault", "candidate": "weld_quality_drift"}
    ]
    deltas = {d["case_id"]: d for d in report["deltas"]["per_case"]}
    assert deltas["G20"]["rounds_baseline"] == 1
    assert deltas["G20"]["rounds_candidate"] == 2
    assert report["deltas"]["rounds"] == {"baseline": 24, "candidate": 25}
    assert report["exit_changes"] == []
    assert "VERDICT: PASS" in regression.format_report(report)


def test_stub_is_not_selectable_through_the_backend_factory():
    from plant_loop.backends import get_reasoner

    with pytest.raises(ValueError):
        get_reasoner("stub")
    with pytest.raises(ValueError):
        get_reasoner("stub-candidate")


# ---------------------------------------------------------------------------
# FAIL path 1 · safety pins (record-level, tests only)
# ---------------------------------------------------------------------------


def test_pin_violations_fail_the_gate_first(baseline, candidate):
    broken = copy.deepcopy(candidate)
    safety = _case(broken, "G02")
    safety["model_calls"] = 2
    safety["pins"]["safety_zero_calls"] = False
    converged = _case(broken, "G03")
    converged["pins"]["mandatory_steps_covered"] = False

    report = regression.compare_runs(baseline, broken)
    assert report["verdict"] == "FAIL"
    by_name = {c["name"]: c for c in report["criteria"]}
    assert by_name["safety_pins"]["passed"] is False
    # Every other criterion still passes — the FAIL is the pins'.
    assert by_name["suite_completeness"]["passed"] is True
    assert by_name["exit_regressions"]["passed"] is True
    assert by_name["diagnosis_agreement"]["passed"] is True
    pins = {(v["case_id"], v["pin"]) for v in report["safety_pin_violations"]}
    assert pins == {("G02", "safety_zero_calls"), ("G03", "mandatory_steps_covered")}


class DropSafetyPlanner(OfflineReasoner):
    """Tests-only: plans correctly, then strips every safety step.

    Mirrors the golden pack's defect planner. Through the shipped
    loop its cases cannot converge — the code verify holds them —
    so its record carries incomplete plans on NON-converged cases:
    no pin violation, because the loop did the pin's work first.
    """

    name = "tests-only-drop-safety"
    model = "rules-v1+defect"

    def draft_plan(self, ctx, diagnosis, focus):
        out = super().draft_plan(ctx, diagnosis, focus)
        out.payload.steps = [s for s in out.payload.steps if s.kind != "safety"]
        return out


def test_through_the_loop_a_safety_stripping_planner_fails_on_exits_not_pins(baseline):
    record = regression.run_suite(DropSafetyPlanner(), label="tests-only-drop-safety")
    # The verify round held every stripped plan: nothing converged
    # on an incomplete plan, so the record itself has no pin
    # violation — and several cases carry the incomplete-plan mark
    # on a non-converged exit, exactly as designed.
    assert regression.pin_violations(record) == []
    incomplete = [
        c for c in record["cases"] if c["pins"]["mandatory_steps_covered"] is False
    ]
    assert incomplete and all(c["exit"] != "converged" for c in incomplete)
    report = regression.compare_runs(baseline, record)
    assert report["verdict"] == "FAIL"
    by_name = {c["name"]: c for c in report["criteria"]}
    assert by_name["safety_pins"]["passed"] is True
    assert by_name["exit_regressions"]["passed"] is False


# ---------------------------------------------------------------------------
# FAIL paths 2–4 · exit regressions, agreement floor, suite shape
# ---------------------------------------------------------------------------


def test_exit_regression_fails_and_improvement_does_not(baseline, candidate):
    regressed = copy.deepcopy(candidate)
    _case(regressed, "G05")["exit"] = "exhausted"
    report = regression.compare_runs(baseline, regressed)
    assert report["verdict"] == "FAIL"
    by_name = {c["name"]: c for c in report["criteria"]}
    assert by_name["exit_regressions"]["passed"] is False
    assert report["exit_changes"] == [
        {"case_id": "G05", "baseline": "converged", "candidate": "exhausted", "regression": True}
    ]

    improved = copy.deepcopy(candidate)
    _case(improved, "G09")["exit"] = "converged"  # was: exhausted
    report = regression.compare_runs(baseline, improved)
    assert report["verdict"] == "PASS"
    assert report["exit_changes"] == [
        {"case_id": "G09", "baseline": "exhausted", "candidate": "converged", "regression": False}
    ]


def test_diagnosis_agreement_below_the_floor_fails(baseline, candidate):
    drifted = copy.deepcopy(candidate)
    for case_id in ("G03", "G05", "G10"):  # G14 already differs: 4/20 off → 80%
        _case(drifted, case_id)["diagnosis"] = "unknown_fault"
    report = regression.compare_runs(baseline, drifted)
    assert report["verdict"] == "FAIL"
    by_name = {c["name"]: c for c in report["criteria"]}
    assert by_name["diagnosis_agreement"]["passed"] is False
    assert report["diagnosis"]["agreement"] == 0.8
    assert by_name["exit_regressions"]["passed"] is True


def test_a_candidate_missing_a_case_fails_completeness(baseline, candidate):
    short = copy.deepcopy(candidate)
    short["cases"] = [c for c in short["cases"] if c["case_id"] != "G07"]
    report = regression.compare_runs(baseline, short)
    assert report["verdict"] == "FAIL"
    by_name = {c["name"]: c for c in report["criteria"]}
    assert by_name["suite_completeness"]["passed"] is False
    assert "G07" in by_name["suite_completeness"]["detail"]


def test_severity_order_covers_every_exit_and_is_printed(baseline, candidate):
    from plant_loop.schemas import (
        EXIT_BLOCKED_EXTERNAL,
        EXIT_BUDGET_EXCEEDED,
        EXIT_CONVERGED,
        EXIT_EXHAUSTED,
        EXIT_SAFETY_STOP,
    )

    assert set(regression.EXIT_SEVERITY) == {
        EXIT_CONVERGED,
        EXIT_BLOCKED_EXTERNAL,
        EXIT_BUDGET_EXCEEDED,
        EXIT_EXHAUSTED,
        EXIT_SAFETY_STOP,
    }
    assert regression.EXIT_SEVERITY[EXIT_CONVERGED] < regression.EXIT_SEVERITY[EXIT_SAFETY_STOP]
    report = regression.compare_runs(baseline, candidate)
    assert report["severity_order"][0] == "converged"
    assert "converged > blocked_external" in regression.format_report(report)


# ---------------------------------------------------------------------------
# The API view of the baseline
# ---------------------------------------------------------------------------


def test_gate_baseline_endpoint_serves_the_committed_record():
    from fastapi.testclient import TestClient

    from plant_loop import api
    from helpers import make_service

    api.set_service(make_service())
    try:
        body = TestClient(api.app).get("/evals/gate-baseline").json()
    finally:
        api.set_service(None)
    assert body["record"]["config"]["backend"] == "offline"
    assert body["record"]["cases_total"] == 20
    assert body["pin_violations"] == []
    assert body["criteria"]["diagnosis_agreement_floor"] == 0.90
    assert body["severity_order"][0] == "converged"
