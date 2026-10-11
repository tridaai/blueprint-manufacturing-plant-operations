"""Round 2: outcomes, live diagnosis stats, the handoff, the watch.

The stats fixtures are hand-computed: records are built directly
(no pipeline) so every expected number below is arithmetic the
reader can check, not a snapshot of whatever the code produced.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from plant_loop import api
from plant_loop.events import format_timestamp
from plant_loop.handoff import build_handoff, render_text
from plant_loop.loops import ReliabilityPolicy
from plant_loop.reliability import ReliabilityWatch, effective_fault_class
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import (
    ActionPlan,
    BoardRunResult,
    CaseResult,
    Diagnosis,
    Hypothesis,
    PlanStep,
    RoundRecord,
    SupervisorOutcome,
)
from plant_loop.stats import diagnosis_stats
from plant_loop.store import InMemoryStore

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}


def record(
    case_id: str,
    *,
    machine_id: str = "CNC-3",
    exit_reason: str = "converged",
    rounds: int = 2,
    hypotheses: list[str] | None = None,
    estimated_downtime: int = 60,
    created_at: str = "2026-10-11T06:00:00Z",
    status: str = "awaiting_supervisor",
    outcome: SupervisorOutcome | None = None,
) -> CaseResult:
    plan = ActionPlan(
        steps=[PlanStep(step_id="s1", text="Do the work.")] if estimated_downtime else [],
        estimated_downtime_min=estimated_downtime,
    )
    return CaseResult(
        case_id=case_id,
        status=status,
        exit_reason=exit_reason,
        type="machine_down",
        title=f"Case {case_id}",
        machine_id=machine_id,
        line_id="line-b",
        priority=700.0,
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(fault_class=f, statement=f"maybe {f}", confidence=0.8)
                for f in (hypotheses or [])
            ]
        ),
        plan=plan,
        rounds=[RoundRecord(round=i + 1) for i in range(rounds)],
        reported_at=created_at,
        created_at=created_at,
        outcome=outcome,
    )


def outcome(**kwargs) -> SupervisorOutcome:
    base = {"decision": "accepted"}
    base.update(kwargs)
    return SupervisorOutcome(**base)


# ---------------------------------------------------------------------------
# Live diagnosis stats — hand-computed fixtures
# ---------------------------------------------------------------------------


def test_stats_zero_outcome_state_is_honest():
    stats = diagnosis_stats([record("A"), record("B")])
    assert stats["available"] is False
    assert "no supervisor outcomes recorded yet" in stats["note"]
    assert stats["outcomes_recorded"] == 0
    assert stats["top1_accuracy"] is None
    assert stats["plan_acceptance"]["used_rate"] is None
    assert stats["downtime"]["n"] == 0


def test_stats_hand_computed():
    records = [
        # top-1 right, converged in 2 rounds, downtime 50 actual vs 60 est.
        record(
            "A",
            hypotheses=["spindle_bearing_wear", "coolant_system_fault"],
            outcome=outcome(confirmed_fault="spindle_bearing_wear",
                            downtime_minutes_actual=50, resolved=True),
        ),
        # top-1 wrong, top-2 right (confirmed is the second hypothesis).
        record(
            "B",
            hypotheses=["coolant_system_fault", "spindle_bearing_wear"],
            rounds=3,
            outcome=outcome(decision="accepted_with_changes",
                            confirmed_fault="spindle_bearing_wear",
                            downtime_minutes_actual=90),
        ),
        # top-1 wrong, top-2 wrong; the loop never converged (the
        # supervisor worked it by hand); plan rejected; no downtime.
        record(
            "C",
            exit_reason="exhausted",
            hypotheses=["tool_imbalance"],
            outcome=outcome(decision="rejected", confirmed_fault="motor_overheat"),
        ),
        # confirmed, but no hypotheses (safety-stop shape): counted apart.
        record(
            "D",
            exit_reason="safety_stop",
            rounds=0,
            hypotheses=[],
            outcome=outcome(confirmed_fault="motor_overheat"),
        ),
        # partial outcome: decision only, no confirmed fault — counts for
        # acceptance, excluded from fault scoring.
        record("E", hypotheses=["tool_imbalance"], outcome=outcome(decision="accepted")),
        # no outcome at all: invisible to every metric.
        record("F", hypotheses=["tool_imbalance"]),
    ]
    stats = diagnosis_stats(records)
    assert stats["available"] is True
    assert stats["cases_total"] == 6
    assert stats["outcomes_recorded"] == 5
    assert stats["fault_scored"] == 3
    assert stats["confirmed_without_diagnosis"] == 1
    assert stats["top1_correct"] == 1
    assert stats["top1_accuracy"] == round(1 / 3, 4)
    assert stats["top2_correct"] == 2
    assert stats["top2_accuracy"] == round(2 / 3, 4)
    # converged confirmed cases: A (2 rounds) and B (3 rounds).
    assert stats["converged_confirmed"] == 2
    assert stats["mean_rounds_to_converge"] == 2.5
    # acceptance: accepted A, D, E; accepted_with_changes B; rejected C.
    counts = stats["plan_acceptance"]["counts"]
    assert counts == {"accepted": 3, "accepted_with_changes": 1, "rejected": 1}
    assert stats["plan_acceptance"]["used_rate"] == round(4 / 5, 4)
    # downtime: A error -10, B error +30 → mean +10, MAE 20.
    assert stats["downtime"]["n"] == 2
    assert stats["downtime"]["mean_error_min"] == 10.0
    assert stats["downtime"]["mean_absolute_error_min"] == 20.0
    assert stats["resolved"] == {"recorded": 1, "resolved_true": 1}


def test_stats_per_fault_class_precision_recall():
    records = [
        record("A", hypotheses=["spindle_bearing_wear"],
               outcome=outcome(confirmed_fault="spindle_bearing_wear")),
        record("B", hypotheses=["spindle_bearing_wear"],
               outcome=outcome(confirmed_fault="spindle_bearing_wear")),
        record("C", hypotheses=["spindle_bearing_wear"],
               outcome=outcome(confirmed_fault="coolant_system_fault")),
        record("D", hypotheses=["tool_imbalance"],
               outcome=outcome(confirmed_fault="coolant_system_fault")),
    ]
    rows = {r["fault_class"]: r for r in diagnosis_stats(records)["per_fault_class"]}
    bearing = rows["spindle_bearing_wear"]  # predicted 3, actual 2, tp 2
    assert (bearing["tp"], bearing["predicted"], bearing["actual"]) == (2, 3, 2)
    assert bearing["precision"] == round(2 / 3, 4)
    assert bearing["recall"] == 1.0
    coolant = rows["coolant_system_fault"]  # predicted 0, actual 2, tp 0
    assert coolant["precision"] is None
    assert coolant["recall"] == 0.0
    tool = rows["tool_imbalance"]  # predicted 1, actual 0
    assert tool["precision"] == 0.0
    assert tool["recall"] is None


# ---------------------------------------------------------------------------
# Outcome capture over HTTP
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def _run_sample(client, case_id="EXC-101"):
    response = client.post("/cases/analyze", json=CASES[case_id].model_dump())
    assert response.status_code == 201
    return response.json()


def test_outcome_endpoint_round_trip(client):
    http, _service = client
    _run_sample(http)
    response = http.post(
        "/api/cases/EXC-101/outcome",
        json={
            "decision": "accepted_with_changes",
            "confirmed_fault": "spindle_bearing_wear",
            "work_performed": "Replaced the spindle bearing set.",
            "downtime_minutes_actual": 95,
            "parts_used": ["BRG-6208"],
            "resolved": True,
            "by": "A. Supervisor",
        },
    )
    assert response.status_code == 200
    record = response.json()
    assert record["outcome"]["confirmed_fault"] == "spindle_bearing_wear"
    assert record["outcome"]["recorded_by"] == "A. Supervisor"
    assert record["outcome"]["recorded_at"]
    stats = http.get("/api/stats/diagnosis").json()
    assert stats["available"] is True
    assert stats["outcomes_recorded"] == 1
    assert stats["top1_accuracy"] == 1.0


def test_outcome_validation_is_strict(client):
    http, _service = client
    _run_sample(http)
    bad_fault = http.post(
        "/api/cases/EXC-101/outcome",
        json={"decision": "accepted", "confirmed_fault": "gremlins"},
    )
    assert bad_fault.status_code == 422
    bad_decision = http.post(
        "/api/cases/EXC-101/outcome", json={"decision": "shrugged"}
    )
    assert bad_decision.status_code == 422
    bad_downtime = http.post(
        "/api/cases/EXC-101/outcome",
        json={"decision": "accepted", "downtime_minutes_actual": -5},
    )
    assert bad_downtime.status_code == 422
    unknown = http.post("/api/cases/NOPE/outcome", json={"decision": "accepted"})
    assert unknown.status_code == 404
    ok = http.post("/api/cases/EXC-101/outcome", json={"decision": "accepted"})
    assert ok.status_code == 200
    again = http.post("/api/cases/EXC-101/outcome", json={"decision": "accepted"})
    assert again.status_code == 409


def test_fault_classes_endpoint(client):
    http, _service = client
    classes = http.get("/api/fault-classes").json()["fault_classes"]
    assert "spindle_bearing_wear" in classes
    assert classes == sorted(classes)


def test_stats_endpoint_empty_state(client):
    http, _service = client
    body = http.get("/api/stats/diagnosis").json()
    assert body["available"] is False
    assert "no supervisor outcomes recorded yet" in body["note"]


# ---------------------------------------------------------------------------
# Shift handoff
# ---------------------------------------------------------------------------


def _seeded_store() -> InMemoryStore:
    store = InMemoryStore()
    store.save_case(record("W1", exit_reason="converged"))
    store.save_case(
        record(
            "W2",
            exit_reason="exhausted",
            hypotheses=["tool_imbalance"],
        )
    )
    blocked = record("W3", exit_reason="blocked_external", hypotheses=["motor_overheat"])
    from plant_loop.schemas import PartRequirement

    blocked.plan.required_parts = [
        PartRequirement(part_id="MTR-9", qty=1, available_qty=0, satisfied=False)
    ]
    store.save_case(blocked)
    store.save_case(record("W4", exit_reason="safety_stop", hypotheses=[], rounds=0))
    decided = record("W5")
    decided.status = "approved"
    store.save_case(decided)
    run = BoardRunResult(
        run_id="board-test1",
        status="shift_budget_exceeded",
        processed=["W1", "W2"],
        exits={"W1": "converged", "W2": "exhausted"},
        remaining=[{"case_id": "W9", "reason": "shift case budget reached (2 cases)"}],
        created_at="2026-10-11T07:00:00Z",
    )
    payload = run.model_dump()
    payload["telemetry"]["input_tokens"] = 1000
    payload["telemetry"]["output_tokens"] = 500
    store.save_board_run(payload)
    return store


def test_handoff_sections_and_counts():
    payload = build_handoff(_seeded_store(), now=datetime(2026, 10, 11, 8, 0, tzinfo=timezone.utc))
    assert payload["counts"] == {
        "awaiting_decision": 4,
        "escalations": 1,
        "blocked_external": 1,
        "safety_stops": 1,
        "reliability_items": 0,
    }
    assert [e["case_id"] for e in payload["escalations"]] == ["W2"]
    assert payload["blocked_external"][0]["missing_parts"] == ["MTR-9"]
    assert [e["case_id"] for e in payload["safety_stops"]] == ["W4"]
    run = payload["last_board_run"]
    assert run["cut_by_budget"] is True
    assert run["tokens_spent"] == 1500
    assert run["remaining"][0]["case_id"] == "W9"


def test_handoff_text_matches_payload():
    payload = build_handoff(_seeded_store(), now=datetime(2026, 10, 11, 8, 0, tzinfo=timezone.utc))
    text = render_text(payload)
    assert "SHIFT HANDOFF" in text
    assert "At the gate: 4 awaiting decision" in text
    assert "W2" in text and "W3" in text and "W4" in text
    assert "waiting on: MTR-9" in text
    assert "CUT BY SHIFT BUDGET" in text
    assert "not worked: W9" in text
    # the decided case is not handed off as awaiting
    assert "W5" not in text


def test_handoff_over_http(client):
    http, service = client
    _run_sample(http)
    body = http.get("/api/board/handoff").json()
    assert body["counts"]["awaiting_decision"] == 1
    assert "SHIFT HANDOFF" in body["text"]
    text_response = http.get("/api/board/handoff", params={"format": "text"})
    assert text_response.headers["content-type"].startswith("text/plain")
    assert text_response.text == body["text"]


# ---------------------------------------------------------------------------
# The recurring-fault watch
# ---------------------------------------------------------------------------

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _at(days_ago: float) -> str:
    return format_timestamp(NOW - timedelta(days=days_ago))


def _fault_case(case_id: str, days_ago: float, fault: str = "spindle_bearing_wear",
                machine: str = "CNC-3", confirmed: bool = False) -> CaseResult:
    rec = record(
        case_id,
        machine_id=machine,
        hypotheses=[fault],
        created_at=_at(days_ago),
    )
    if confirmed:
        rec.outcome = outcome(confirmed_fault=fault)
    return rec


def test_watch_raises_item_at_threshold_with_evidence():
    store = InMemoryStore()
    watch = ReliabilityWatch(store, ReliabilityPolicy(threshold=3, window_days=14))
    assert watch.check(_fault_case("R1", 10), now=NOW) is None
    store.save_case(_fault_case("R1", 10))
    store.save_case(_fault_case("R2", 5))
    trigger = _fault_case("R3", 0)
    store.save_case(trigger)
    item = watch.check(trigger, now=NOW)
    assert item is not None
    assert item.kind == "reliability_review"
    assert item.status == "open"
    assert item.case_ids == ["R1", "R2", "R3"]
    assert item.count == 3
    assert "spindle_bearing_wear" in item.title and "CNC-3" in item.title
    assert store.all_board_items() == [item]


def test_watch_window_edges():
    store = InMemoryStore()
    watch = ReliabilityWatch(store, ReliabilityPolicy(threshold=3, window_days=14))
    store.save_case(_fault_case("OLD", 14.5))  # outside the window: never counts
    store.save_case(_fault_case("EDGE", 14))   # exactly at the edge: inside
    store.save_case(_fault_case("MID", 7))
    trigger = _fault_case("NEW", 0)
    store.save_case(trigger)
    item = watch.check(trigger, now=NOW)
    assert item is not None
    # oldest first; OLD fell outside the 14-day window.
    assert item.case_ids == ["EDGE", "MID", "NEW"]
    assert item.count == 3


def test_watch_below_threshold_and_wrong_machine_or_fault():
    store = InMemoryStore()
    watch = ReliabilityWatch(store, ReliabilityPolicy(threshold=3, window_days=14))
    store.save_case(_fault_case("A", 1))
    store.save_case(_fault_case("B", 1, machine="PRESS-1"))
    store.save_case(_fault_case("C", 1, fault="tool_imbalance"))
    trigger = _fault_case("D", 0)
    store.save_case(trigger)
    # only A and D share machine+fault -> 2 < 3
    assert watch.check(trigger, now=NOW) is None
    assert store.all_board_items() == []


def test_watch_dedup_one_open_item():
    store = InMemoryStore()
    watch = ReliabilityWatch(store, ReliabilityPolicy(threshold=3, window_days=14))
    for i, days in enumerate((3, 2, 1)):
        store.save_case(_fault_case(f"K{i}", days))
    first = watch.check(_fault_case("K2", 1), now=NOW)
    assert first is not None
    store.save_case(_fault_case("K3", 0))
    assert watch.check(_fault_case("K3", 0), now=NOW) is None
    assert len(store.all_board_items()) == 1


def test_watch_uses_confirmed_fault_over_diagnosis():
    rec = record("X", hypotheses=["tool_imbalance"],
                 outcome=outcome(confirmed_fault="motor_overheat"))
    assert effective_fault_class(rec) == "motor_overheat"
    rec2 = record("Y", hypotheses=["tool_imbalance"])
    assert effective_fault_class(rec2) == "tool_imbalance"


def test_watch_fires_from_service_outcomes():
    service = make_service()
    # three runs of the same sample case shape on CNC-3 (offline backend
    # diagnoses spindle_bearing_wear), then outcomes confirm the fault.
    for i in range(3):
        case = CASES["EXC-101"].model_copy(update={"case_id": f"EXC-9{i}"})
        service.run_case(case)
    items = service.store.all_board_items()
    assert len(items) == 1
    assert items[0].machine_id == "CNC-3"
    assert items[0].fault_class == "spindle_bearing_wear"
    assert items[0].count >= 3
    # and it is listed over HTTP-shaped reads
    assert service.board_items()[0].item_id == items[0].item_id
