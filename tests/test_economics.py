"""Round 6: downtime economics — the board in dollars.

Fixtures are hand-computed like the stats fixtures in
test_outcomes.py: records are built directly, and every expected
number below is arithmetic the reader can check against the plant
model's rates (line-a $2,400/hr, line-b $1,800/hr, line-c $950/hr,
line-d $720/hr, line-e — Utilities — deliberately unrated).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from plant_loop import api, kb
from plant_loop.economics import board_summary, for_record, price
from plant_loop.handoff import build_handoff, render_text
from plant_loop.priority import board_order
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import (
    ActionPlan,
    CaseResult,
    ExceptionCase,
    PlanStep,
    SupervisorOutcome,
)
from plant_loop.stats import diagnosis_stats
from plant_loop.store import InMemoryStore

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}


def record(
    case_id: str,
    *,
    line_id: str = "line-b",
    exit_reason: str = "converged",
    estimated_downtime: int = 60,
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
        machine_id="CNC-3",
        line_id=line_id,
        priority=700.0,
        plan=plan,
        reported_at="2026-10-11T06:00:00Z",
        created_at="2026-10-11T06:00:00Z",
        outcome=outcome,
    )


def outcome(**kwargs) -> SupervisorOutcome:
    base = {"decision": "accepted"}
    base.update(kwargs)
    return SupervisorOutcome(**base)


# ---------------------------------------------------------------------------
# Rates: authored plant-model data, absence is a fact
# ---------------------------------------------------------------------------


def test_rates_live_in_the_plant_model():
    assert kb.line_downtime_rate("line-a") == 2400.0
    assert kb.line_downtime_rate("line-b") == 1800.0
    assert kb.line_downtime_rate("line-c") == 950.0
    assert kb.line_downtime_rate("line-d") == 720.0
    # Utilities carries no rate: minutes only, never an invented one.
    assert kb.line_downtime_rate("line-e") is None
    assert kb.line_downtime_rate("line-unknown") is None
    assert kb.line_downtime_rate("") is None


def test_price_math():
    assert price(120, 1800.0) == 3600.0
    assert price(40, 950.0) == 633.33
    assert price(90, None) is None
    assert price(0, 1800.0) == 0.0


# ---------------------------------------------------------------------------
# Per-case economics
# ---------------------------------------------------------------------------


def test_for_record_rated_line():
    econ = for_record(record("A", line_id="line-b", estimated_downtime=120))
    assert econ is not None
    assert econ.rate_usd_per_hour == 1800.0
    assert "data/kb.json" in econ.rate_basis  # the rate is cited, not bare
    assert econ.estimated_minutes == 120
    assert econ.estimated_usd == 3600.0
    assert econ.actual_minutes is None and econ.actual_usd is None


def test_for_record_unrated_line_is_minutes_only():
    econ = for_record(record("A", line_id="line-e", estimated_downtime=90))
    assert econ is not None
    assert econ.rate_usd_per_hour is None
    assert econ.rate_basis == ""
    assert econ.estimated_minutes == 90
    assert econ.estimated_usd is None  # no rate → no dollars, ever


def test_for_record_safety_stop_is_never_priced():
    econ = for_record(record("A", exit_reason="safety_stop", estimated_downtime=0))
    assert econ is None


def test_for_record_actuals_join_at_the_same_rate():
    rec = record(
        "A",
        line_id="line-b",
        estimated_downtime=120,
        outcome=outcome(downtime_minutes_actual=95),
    )
    econ = for_record(rec)
    assert econ.actual_minutes == 95
    assert econ.actual_usd == 2850.0  # 95 min at the estimate's own rate


def test_board_summary_splits_rated_unrated_and_skips_safety():
    summary = board_summary(
        [
            record("A", line_id="line-b", estimated_downtime=120),   # $3,600
            record("B", line_id="line-a", estimated_downtime=150),   # $6,000
            record("C", line_id="line-e", estimated_downtime=90),    # minutes only
            record("D", exit_reason="safety_stop", estimated_downtime=0),
        ]
    )
    assert summary.cases_costed == 2
    assert summary.cases_minutes_only == 1
    assert summary.estimated_minutes == 360
    assert summary.unrated_minutes == 90
    assert summary.estimated_exposure_usd == 9600.0
    assert summary.largest_exposure_case_id == "B"
    assert summary.largest_exposure_usd == 6000.0


def test_board_summary_no_rated_cases_has_no_dollar_total():
    summary = board_summary([record("C", line_id="line-e", estimated_downtime=90)])
    assert summary.estimated_exposure_usd is None
    assert summary.largest_exposure_usd is None
    assert summary.unrated_minutes == 90


# ---------------------------------------------------------------------------
# Economics inform; they do not reorder
# ---------------------------------------------------------------------------


def test_priority_order_ignores_dollars():
    # The expensive case: a maintenance alert on CNC-2 (line-b) whose
    # plan estimates 240 min — $7,200 of exposure once priced.
    expensive = CASES["EXC-107"].model_copy(
        update={"case_id": "EXP", "type": "maintenance_alert", "downtime_minutes": 0}
    )
    # The cheap case: a line-down on WELD-4 (line-c), 40 min — $633.33.
    cheap = CASES["EXC-103"].model_copy(
        update={"case_id": "CHP", "type": "machine_down"}
    )
    ordered = board_order([expensive, cheap])
    assert [c.case_id for c in ordered] == ["CHP", "EXP"]


# ---------------------------------------------------------------------------
# Cost accuracy in the stats
# ---------------------------------------------------------------------------


def test_stats_economics_hand_computed():
    records = [
        # line-b: est 120 / act 95 → -25 min, -$750.
        record("A", line_id="line-b", estimated_downtime=120,
               outcome=outcome(downtime_minutes_actual=95)),
        # line-b: est 60 / act 60 → exact.
        record("B", line_id="line-b", estimated_downtime=60,
               outcome=outcome(downtime_minutes_actual=60)),
        # line-e (unrated): est 90 / act 120 → +30 min, no dollars.
        record("C", line_id="line-e", estimated_downtime=90,
               outcome=outcome(downtime_minutes_actual=120)),
        # no actual recorded: outside the population entirely.
        record("D", line_id="line-a", estimated_downtime=150,
               outcome=outcome()),
    ]
    econ = diagnosis_stats(records)["economics"]
    assert econ["n"] == 3
    assert econ["rated_n"] == 2
    assert econ["minutes_only_n"] == 1
    assert econ["estimated_minutes"] == 270
    assert econ["actual_minutes"] == 275
    assert econ["mean_error_min"] == round(5 / 3, 4)
    assert econ["mean_absolute_error_min"] == round(55 / 3, 4)
    assert econ["estimated_usd"] == 5400.0
    assert econ["actual_usd"] == 4650.0
    assert econ["mean_error_usd"] == -375.0
    assert econ["mean_absolute_error_usd"] == 375.0
    assert "minutes-only" in econ["note"] or "does not rate" in econ["note"]
    by_line = {row["line_id"]: row for row in econ["by_line"]}
    line_b = by_line["line-b"]
    assert line_b["rate_usd_per_hour"] == 1800.0
    assert line_b["n"] == 2
    assert line_b["mean_error_min"] == -12.5
    assert line_b["estimated_usd"] == 5400.0
    line_e = by_line["line-e"]
    assert line_e["rate_usd_per_hour"] is None
    assert line_e["estimated_usd"] is None and line_e["actual_usd"] is None
    assert line_e["mean_error_usd"] is None
    assert line_e["mean_error_min"] == 30.0


def test_stats_economics_empty_population_is_honest():
    econ = diagnosis_stats([record("A")])["economics"]
    assert econ["n"] == 0
    assert econ["by_line"] == []
    assert econ["estimated_usd"] is None
    assert "no recorded actuals yet" in econ["note"]


# ---------------------------------------------------------------------------
# The handoff's economics section
# ---------------------------------------------------------------------------


def _handoff_store() -> InMemoryStore:
    store = InMemoryStore()
    store.save_case(record("W1", line_id="line-b", estimated_downtime=60))   # $1,800 open
    store.save_case(record("W6", line_id="line-e", estimated_downtime=90))   # 90 min, unrated
    store.save_case(record("W4", exit_reason="safety_stop", estimated_downtime=0))
    closed = record(
        "W5", line_id="line-b", estimated_downtime=60, status="approved",
        outcome=outcome(downtime_minutes_actual=30),                          # $900 actual
    )
    store.save_case(closed)
    return store


def test_handoff_economics_payload():
    payload = build_handoff(
        _handoff_store(), now=datetime(2026, 10, 11, 8, 0, tzinfo=timezone.utc)
    )
    econ = payload["economics"]
    assert econ["open"]["estimated_exposure_usd"] == 1800.0
    assert econ["open"]["estimated_minutes"] == 150  # safety stop contributes nothing
    assert econ["open"]["unrated_minutes"] == 90
    assert econ["open"]["largest_exposure_case_id"] == "W1"
    assert econ["closed"]["cases_with_actuals"] == 1
    assert econ["closed"]["actual_minutes"] == 30
    assert econ["closed"]["actual_usd"] == 900.0


def test_handoff_economics_text():
    payload = build_handoff(
        _handoff_store(), now=datetime(2026, 10, 11, 8, 0, tzinfo=timezone.utc)
    )
    text = render_text(payload)
    assert "DOWNTIME ECONOMICS" in text
    assert "Open at the gate: ~$1,800.00 of estimated exposure over 150 min" in text
    assert "90 of those minutes on lines the plant model does not rate" in text
    assert "Largest open exposure: W1 — $1,800.00" in text
    assert "Closed cases: 1 with recorded actuals — 30 min actual, $900.00 on rated lines" in text
    # the safety stop's own section stays bare of money
    safety_block = text.split("SAFETY STOPS:")[1].split("RELIABILITY")[0]
    assert "$" not in safety_block


# ---------------------------------------------------------------------------
# Through the service: stamps, run totals, read models, HTTP
# ---------------------------------------------------------------------------


def test_service_stamps_records_and_totals_the_run():
    service = make_service()
    result = service.run_board([CASES["EXC-101"], CASES["EXC-106"]])
    rec = service.get_record("EXC-101")
    assert rec.economics is not None
    assert rec.economics.estimated_usd == 3600.0
    assert rec.economics.rate_basis.startswith("plant model downtime rate")
    fan = service.get_record("EXC-106")
    assert fan.economics is not None and fan.economics.estimated_usd is None
    assert fan.economics.estimated_minutes == 90
    safety = service.run_case(CASES["EXC-102"]).result
    assert safety.economics is None
    run = result.economics
    assert run is not None
    assert run.estimated_exposure_usd == 3600.0
    assert run.unrated_minutes == 90
    assert run.largest_exposure_case_id == "EXC-101"
    # the stored run payload carries the same totals
    stored = service.get_board_run(result.run_id)
    assert stored["economics"]["estimated_exposure_usd"] == 3600.0


def test_service_outcome_refreshes_the_economics_stamp():
    service = make_service()
    service.run_case(CASES["EXC-101"])
    service.record_outcome(
        "EXC-101",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="spindle_bearing_wear",
            downtime_minutes_actual=95,
        ),
    )
    econ = service.get_record("EXC-101").economics
    assert econ.actual_minutes == 95
    assert econ.actual_usd == 2850.0
    assert econ.estimated_usd == 3600.0


def test_board_read_model_carries_exposure():
    service = make_service()
    service.run_case(CASES["EXC-101"])
    service.run_case(CASES["EXC-102"])
    rows = {row["case_id"]: row for row in service.board()}
    assert rows["EXC-101"]["exposure_usd"] == 3600.0
    assert rows["EXC-101"]["estimated_minutes"] == 120
    assert rows["EXC-102"]["exposure_usd"] is None
    assert rows["EXC-102"]["estimated_minutes"] is None


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def test_economics_over_http(client):
    http, _service = client
    response = http.post("/cases/analyze", json=CASES["EXC-101"].model_dump())
    assert response.status_code == 201
    assert response.json()["economics"]["estimated_usd"] == 3600.0
    board = http.get("/board").json()
    assert board[0]["exposure_usd"] == 3600.0
    run = http.post("/board/run", json={}).json()
    assert run["economics"]["estimated_exposure_usd"] == 19053.33
    assert run["economics"]["largest_exposure_case_id"] == "EXC-107"
    stats = http.get("/api/stats/diagnosis").json()
    assert stats["economics"]["n"] == 0  # no actuals yet — counted, not hidden
