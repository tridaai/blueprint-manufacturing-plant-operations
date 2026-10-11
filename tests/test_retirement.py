"""Round 10: machines and obligations end — retirement and decommission.

Round 8 left PM obligations immortal. Now a supervisor can
retire one (actor + reason, one-way, the reason on the record),
and can decommission a machine: its standing obligations retire
automatically with the fixed reason "machine decommissioned"
and the decommission record linked, its open cases are listed
on the record for the supervisor's per-case disposition (keep,
or close as moot), and new cases for it are refused at intake.
Safety records are never touched by any of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from helpers import make_service

from plant_loop import api
from plant_loop.events import format_timestamp
from plant_loop.handoff import build_handoff, render_text
from plant_loop.lifecycle import MachineLifecycle
from plant_loop.loops import ReliabilityPolicy
from plant_loop.pm import PmManager
from plant_loop.reliability import ReliabilityWatch
from plant_loop.schemas import (
    CaseResult,
    Diagnosis,
    ExceptionCase,
    Hypothesis,
    SupervisorOutcome,
)
from plant_loop.store import InMemoryStore

T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
SUP = "Sam Supervisor"


def _manager(store: InMemoryStore) -> PmManager:
    return PmManager(store, ReliabilityWatch(store, ReliabilityPolicy()))


def _fault(case_id: str, moment: datetime, machine_id: str = "CNC-3") -> CaseResult:
    # Fabricated the way test_pm.py fabricates: the watch counts a
    # case by its created_at and its top hypothesis's fault class.
    return CaseResult(
        case_id=case_id,
        type="machine_down",
        machine_id=machine_id,
        line_id="line-b",
        created_at=format_timestamp(moment),
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(
                    fault_class="spindle_bearing_wear",
                    statement="fabricated for the retirement tests",
                    confidence=0.9,
                    sop_id="SOP-X",
                )
            ]
        ),
    )


def _standing_obligation(store: InMemoryStore, manager: PmManager) -> object:
    records = []
    for i in range(3):
        record = _fault(f"F-{i + 1}", T0 + timedelta(days=4 * i + (i > 0)))
        store.save_case(record)
        records.append(record)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    _decided, obligation = manager.decide(
        proposal.proposal_id,
        "approved",
        by="Sup",
        interval_days=14,
        last_performed_at="2026-09-01T00:00:00Z",
        now=T0 + timedelta(days=10),
    )
    assert obligation is not None
    return obligation  # due 2026-09-15


def _quality(case_id: str, machine_id: str = "WELD-4") -> ExceptionCase:
    return ExceptionCase(
        case_id=case_id,
        type="quality_nonconformance",
        title="Porosity above limit on bracket welds",
        description="QC inspection found porosity in 6.2 percent of welds.",
        machine_id=machine_id,
        line_id="line-c",
        product="Bracket BR-118",
        batch="B-4490",
        readings={"qc_porosity_pct": 6.2, "gas_flow_l_min": 9.5},
        reporter="S. Patel (quality)",
        reported_at="2026-10-11T05:40:00Z",
        downtime_minutes=0,
    )


def _coolant(case_id: str) -> ExceptionCase:
    return ExceptionCase(
        case_id=case_id,
        type="machine_down",
        title=f"{case_id} coolant flow low",
        description="Coolant flow alarm; spindle temperature climbing.",
        machine_id="CNC-3",
        line_id="line-b",
        product="Gear blank GB-220",
        batch="B-4482",
        readings={"coolant_flow_l_min": 3.0, "spindle_temp_c": 84},
        reporter="operator",
        reported_at="2026-10-11T06:40:00Z",
        downtime_minutes=30,
    )


def _safety(case_id: str) -> ExceptionCase:
    return ExceptionCase(
        case_id=case_id,
        type="safety_observation",
        title="Guard interlock switch intermittent",
        description="Operator found the guard interlock switch intermittent.",
        machine_id="CNC-3",
        line_id="line-b",
        reporter="M. Reyes (operator)",
        reported_at="2026-10-11T05:58:00Z",
        downtime_minutes=0,
    )


# ---------------------------------------------------------------------------
# Manual retirement
# ---------------------------------------------------------------------------


def test_retire_stamps_the_record_and_is_one_way():
    store = InMemoryStore()
    manager = _manager(store)
    obligation = _standing_obligation(store, manager)
    retired = manager.retire_obligation(
        obligation.obligation_id,
        by=SUP,
        reason="Spindle rebuilt — pattern closed.",
        now=T0 + timedelta(days=20),
    )
    assert retired.status == "retired"
    assert retired.retirement_kind == "manual"
    assert retired.retired_by == SUP
    assert retired.retirement_reason == "Spindle rebuilt — pattern closed."
    assert retired.retired_at == format_timestamp(T0 + timedelta(days=20))
    assert retired.decommission_id == ""
    # One-way: a second retirement is refused, and the record stands.
    with pytest.raises(ValueError, match="already retired"):
        manager.retire_obligation(obligation.obligation_id, by=SUP, reason="again")
    assert manager.obligations()[0].status == "retired"


def test_retire_requires_actor_and_reason_and_a_real_obligation():
    store = InMemoryStore()
    manager = _manager(store)
    obligation = _standing_obligation(store, manager)
    with pytest.raises(ValueError, match="actor"):
        manager.retire_obligation(obligation.obligation_id, reason="why")
    with pytest.raises(ValueError, match="reason"):
        manager.retire_obligation(obligation.obligation_id, by=SUP)
    with pytest.raises(KeyError):
        manager.retire_obligation("PMO-nope", by=SUP, reason="why")
    # A proposal id is not an obligation.
    with pytest.raises(KeyError):
        manager.retire_obligation(obligation.proposal_id, by=SUP, reason="why")
    assert manager.obligations()[0].status == "standing"


def test_a_retired_obligation_surfaces_no_due_cases():
    due_point = datetime(2026, 9, 15, tzinfo=timezone.utc)
    # Control: standing, the obligation fires at its due point.
    store = InMemoryStore()
    manager = _manager(store)
    _standing_obligation(store, manager)
    assert len(manager.due_cases(now=due_point)) == 1
    # Retired before its due point: silence — at the point, and
    # arbitrarily far past it.
    store2 = InMemoryStore()
    manager2 = _manager(store2)
    obligation2 = _standing_obligation(store2, manager2)
    manager2.retire_obligation(obligation2.obligation_id, by=SUP, reason="done")
    assert manager2.due_cases(now=due_point) == []
    assert manager2.due_cases(now=due_point + timedelta(days=90)) == []


def test_retirement_spends_the_evidence_and_a_fresh_pattern_starts_over():
    store = InMemoryStore()
    manager = _manager(store)
    obligation = _standing_obligation(store, manager)
    manager.retire_obligation(obligation.obligation_id, by=SUP, reason="pattern closed")
    # The cited faults are spent: replaying them drafts nothing.
    assert manager.evaluate(store.get_case("F-3"), now=T0 + timedelta(days=30)) is None
    # A fresh threshold drafts a fresh proposal citing fresh faults.
    fresh_records = []
    for i, cid in enumerate(("F-4", "F-5", "F-6")):
        record = _fault(cid, T0 + timedelta(days=30 + 4 * i))
        store.save_case(record)
        fresh_records.append(record)
    proposal = manager.evaluate(fresh_records[-1], now=T0 + timedelta(days=40))
    assert proposal is not None
    assert [e["case_id"] for e in proposal.evidence] == ["F-4", "F-5", "F-6"]
    _decided, new_obligation = manager.decide(proposal.proposal_id, "approved", by="Sup")
    assert new_obligation is not None
    assert new_obligation.obligation_id != obligation.obligation_id
    assert new_obligation.status == "standing"
    # The old obligation is untouched by the new cycle.
    old = next(o for o in manager.obligations() if o.obligation_id == obligation.obligation_id)
    assert old.status == "retired"


# ---------------------------------------------------------------------------
# Decommission
# ---------------------------------------------------------------------------


def test_decommission_validates_the_act():
    store = InMemoryStore()
    lifecycle = MachineLifecycle(store, _manager(store))
    with pytest.raises(KeyError):
        lifecycle.decommission("NOPE-9", by=SUP, reason="gone")
    with pytest.raises(ValueError, match="actor"):
        lifecycle.decommission("CNC-3", reason="gone")
    with pytest.raises(ValueError, match="reason"):
        lifecycle.decommission("CNC-3", by=SUP)
    with pytest.raises(ValueError, match="not a record timestamp"):
        lifecycle.decommission("CNC-3", by=SUP, reason="gone", effective_at="last Friday")
    record = lifecycle.decommission(
        "CNC-3", by=SUP, reason="Cell consolidated onto CNC-2.",
        effective_at="2026-10-12T00:00:00Z", now=T0,
    )
    assert record.record_id.startswith("DEC-")
    assert record.effective_at == "2026-10-12T00:00:00Z"
    assert record.decided_at == format_timestamp(T0)
    with pytest.raises(ValueError, match="already decommissioned"):
        lifecycle.decommission("CNC-3", by=SUP, reason="again")


def test_decommission_retires_standing_obligations_with_the_link():
    store = InMemoryStore()
    manager = _manager(store)
    obligation = _standing_obligation(store, manager)
    lifecycle = MachineLifecycle(store, manager)
    record = lifecycle.decommission("CNC-3", by=SUP, reason="Scrapped.", now=T0)
    assert record.retired_obligation_ids == [obligation.obligation_id]
    retired = manager.obligations()[0]
    assert retired.status == "retired"
    assert retired.retirement_kind == "machine_decommission"
    assert retired.retirement_reason == "machine decommissioned"
    assert retired.decommission_id == record.record_id
    assert retired.retired_by == SUP
    # And it drafts nothing ever again, even on fresh-looking faults.
    fresh = _fault("F-9", T0 + timedelta(days=60))
    store.save_case(fresh)
    assert manager.evaluate(fresh, now=T0 + timedelta(days=60)) is None


def test_decommission_leaves_a_manual_retirement_as_it_stands():
    store = InMemoryStore()
    manager = _manager(store)
    obligation = _standing_obligation(store, manager)
    manager.retire_obligation(obligation.obligation_id, by=SUP, reason="Pattern closed.")
    lifecycle = MachineLifecycle(store, manager)
    record = lifecycle.decommission("CNC-3", by=SUP, reason="Scrapped.", now=T0)
    assert record.retired_obligation_ids == []
    retired = manager.obligations()[0]
    assert retired.retirement_kind == "manual"
    assert retired.retirement_reason == "Pattern closed."
    assert retired.decommission_id == ""


def test_open_cases_are_the_unfinished_business_only():
    store = InMemoryStore()
    awaiting = CaseResult(case_id="X-1", machine_id="CNC-3")
    decided = CaseResult(case_id="X-2", machine_id="CNC-3", status="approved")
    settled = CaseResult(
        case_id="X-3", machine_id="CNC-3", status="approved",
        outcome=SupervisorOutcome(decision="accepted", recorded_at="2026-10-01T00:00:00Z"),
    )
    dropped = CaseResult(case_id="X-4", machine_id="CNC-3", status="dropped")
    other_machine = CaseResult(case_id="X-5", machine_id="WELD-4")
    for record_case in (awaiting, decided, settled, dropped, other_machine):
        store.save_case(record_case)
    lifecycle = MachineLifecycle(store, _manager(store))
    record = lifecycle.decommission("CNC-3", by=SUP, reason="Scrapped.", now=T0)
    assert record.open_case_ids == ["X-1", "X-2"]


# ---------------------------------------------------------------------------
# Dispositions
# ---------------------------------------------------------------------------


def _decommissioned_store_with_cases() -> tuple[InMemoryStore, MachineLifecycle]:
    store = InMemoryStore()
    for cid in ("X-1", "X-2"):
        store.save_case(CaseResult(case_id=cid, machine_id="CNC-3"))
    lifecycle = MachineLifecycle(store, _manager(store))
    lifecycle.decommission("CNC-3", by=SUP, reason="Scrapped.", now=T0)
    return store, lifecycle


def test_disposition_validates_and_records_once():
    store, lifecycle = _decommissioned_store_with_cases()
    with pytest.raises(KeyError):
        lifecycle.dispose("WELD-4", "X-1", "keep", by=SUP)
    with pytest.raises(ValueError, match="unknown disposition"):
        lifecycle.dispose("CNC-3", "X-1", "shred", by=SUP)
    with pytest.raises(ValueError, match="actor"):
        lifecycle.dispose("CNC-3", "X-1", "keep")
    with pytest.raises(ValueError, match="not on"):
        lifecycle.dispose("CNC-3", "X-9", "keep", by=SUP)
    record = lifecycle.dispose("CNC-3", "X-1", "keep", by=SUP, now=T0)
    assert record.dispositions[0]["case_id"] == "X-1"
    assert record.dispositions[0]["disposition"] == "keep"
    with pytest.raises(ValueError, match="already has a recorded disposition"):
        lifecycle.dispose("CNC-3", "X-1", "close_moot", by=SUP)


def test_keep_leaves_the_case_record_untouched_and_moot_closes_it():
    store, lifecycle = _decommissioned_store_with_cases()
    before = store.get_case("X-1").model_dump()
    lifecycle.dispose("CNC-3", "X-1", "keep", by=SUP, now=T0)
    assert store.get_case("X-1").model_dump() == before

    lifecycle.dispose("CNC-3", "X-2", "close_moot", by=SUP, now=T0)
    closed = store.get_case("X-2")
    assert closed.status == "closed_moot"
    assert closed.closed_as_moot["machine_id"] == "CNC-3"
    assert closed.closed_as_moot["by"] == SUP
    assert closed.closed_as_moot["decommission_id"].startswith("DEC-")


def test_a_case_that_settled_since_the_act_needs_no_disposition():
    store, lifecycle = _decommissioned_store_with_cases()
    case = store.get_case("X-1")
    case.status = "approved"
    case.outcome = SupervisorOutcome(
        decision="accepted", recorded_at="2026-10-02T00:00:00Z"
    )
    store.save_case(case)
    with pytest.raises(ValueError, match="no longer open"):
        lifecycle.dispose("CNC-3", "X-1", "keep", by=SUP)


# ---------------------------------------------------------------------------
# Service level: intake, kept work, claims, the safety discipline
# ---------------------------------------------------------------------------


def _service_with_weld_obligation(service) -> object:
    for cid in ("RT-A", "RT-B", "RT-C"):
        service.run_case(_quality(cid))
    proposals = [p for p in service.pm_proposals() if p.status == "proposed"]
    assert len(proposals) == 1
    _proposal, obligation = service.decide_pm_proposal(
        proposals[0].proposal_id,
        "approved",
        by=SUP,
        interval_days=14,
        last_performed_at="2026-09-01T00:00:00Z",
    )
    assert obligation is not None
    return obligation


def test_service_manual_retirement_silences_the_board():
    service = make_service()
    obligation = _service_with_weld_obligation(service)
    retired = service.retire_pm_obligation(
        obligation.obligation_id, by=SUP, reason="Torch head replaced — pattern closed."
    )
    assert retired.status == "retired"
    # A board run well past the due point (2026-09-15) stays silent.
    run = service.run_board([])
    assert run.pm_cases == []
    assert [o for o in service.pm_obligations() if o.status == "standing"] == []


def test_intake_refuses_the_dead_machine_and_names_the_state():
    service = make_service()
    record = service.decommission_machine(
        "CNC-3", by=SUP, reason="Cell consolidated onto CNC-2.",
        effective_at="2026-10-12T00:00:00Z",
    )
    with pytest.raises(ValueError, match="DECOMMISSIONED") as excinfo:
        service.run_case(_coolant("RT-Z"))
    assert record.record_id in str(excinfo.value)
    with pytest.raises(ValueError, match="DECOMMISSIONED"):
        service.run_board([_coolant("RT-Y")])
    # An empty board is not intake: it runs, and stays silent.
    run = service.run_board([])
    assert run.pm_cases == []
    # Other machines are unaffected.
    assert service.run_case(_quality("RT-W")).result.exit_reason == "converged"


def test_decommission_lists_cases_and_moot_close_releases_claims():
    service = make_service()
    service.run_case(_coolant("RT-1"))  # converges, claims CLNT-FLT-E
    service.run_case(_safety("RT-S"))  # safety stop at the gate
    claims = service.allocation.claims_for_case("RT-1")
    assert claims and claims[0].status == "claimed"

    record = service.decommission_machine("CNC-3", by=SUP, reason="Scrapped.")
    assert sorted(record.open_case_ids) == ["RT-1", "RT-S"]
    # Nothing was disposed by the act itself — both still wait.
    assert service.get_record("RT-1").status == "awaiting_supervisor"
    safety = service.get_record("RT-S")
    assert safety.status == "awaiting_supervisor"
    assert safety.exit_reason == "safety_stop"

    service.dispose_machine_case("CNC-3", "RT-1", "close_moot", by=SUP)
    closed = service.get_record("RT-1")
    assert closed.status == "closed_moot"
    assert closed.closed_as_moot["decommission_id"] == record.record_id
    released = service.allocation.claims_for_case("RT-1")
    assert released[0].status == "released"
    assert released[0].release_reason == "released_by_gate_rejection"

    # The safety case: kept, still at the gate, still a safety stop.
    service.dispose_machine_case("CNC-3", "RT-S", "keep", by=SUP)
    safety = service.get_record("RT-S")
    assert safety.status == "awaiting_supervisor"
    assert safety.exit_reason == "safety_stop"


def test_a_kept_case_is_still_worked_by_the_loop():
    service = make_service()
    board1 = service.run_board([_coolant("RT-K")])
    assert board1.exits["RT-K"] == "converged"
    service.decommission_machine("CNC-3", by=SUP, reason="Scrapped.")
    service.dispose_machine_case("CNC-3", "RT-K", "keep", by=SUP)
    # Carried into the next shift's board as a gate case, standing.
    board2 = service.run_board([], carry_over_from=board1.run_id)
    assert board2.carry_over_settled["inherited_at_gate"] == ["RT-K"]
    # And the gate still works on it: decision and outcome land.
    service.record_decision("RT-K", "approved", by=SUP)
    record = service.record_outcome(
        "RT-K",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="coolant_system_fault",
            work_performed="Filter replaced.",
            recorded_at="2026-10-12T09:00:00Z",
        ),
    )
    assert record.outcome is not None
    assert record.status == "approved"


def test_a_kept_pm_cycle_outcome_never_resurrects_the_obligation():
    service = make_service()
    obligation = _service_with_weld_obligation(service)
    board = service.run_board([])  # due 2026-09-15: the cycle case fires
    assert len(board.pm_cases) == 1
    cycle_case_id = board.pm_cases[0]
    record = service.decommission_machine("WELD-4", by=SUP, reason="Cell retired.")
    assert record.retired_obligation_ids == [obligation.obligation_id]
    assert cycle_case_id in record.open_case_ids
    service.dispose_machine_case("WELD-4", cycle_case_id, "keep", by=SUP)
    service.record_decision(cycle_case_id, "approved", by=SUP)
    service.record_outcome(
        cycle_case_id,
        SupervisorOutcome(
            decision="accepted",
            work_performed="Final inspection done before the cell came down.",
            recorded_at="2026-10-12T09:00:00Z",
        ),
    )
    after = next(o for o in service.pm_obligations() if o.obligation_id == obligation.obligation_id)
    assert after.status == "retired"
    assert after.cycles_completed == 0  # the outcome did not stamp the cycle
    assert after.last_performed_at == "2026-09-01T00:00:00Z"
    # And no new cycle ever surfaces for it.
    assert service.run_board([]).pm_cases == []


# ---------------------------------------------------------------------------
# The handoff tells the truth about the endings
# ---------------------------------------------------------------------------


def test_handoff_shows_both_retirements_with_their_reasons():
    service = make_service()
    weld = _service_with_weld_obligation(service)
    service.retire_pm_obligation(
        weld.obligation_id, by=SUP, reason="Torch head replaced — pattern closed."
    )
    handoff = build_handoff(service.store, now=datetime(2026, 10, 12, tzinfo=timezone.utc))
    retired = {r["obligation_id"]: r for r in handoff["pm"]["retired"]}
    assert weld.obligation_id in retired
    assert retired[weld.obligation_id]["retirement_reason"] == (
        "Torch head replaced — pattern closed."
    )
    assert retired[weld.obligation_id]["retirement_kind"] == "manual"
    assert all(o["obligation_id"] != weld.obligation_id for o in handoff["pm"]["obligations"])
    text = render_text(handoff)
    assert "Torch head replaced — pattern closed." in text


def test_handoff_names_the_decommission_behind_an_auto_retirement():
    store = InMemoryStore()
    manager = _manager(store)
    obligation = _standing_obligation(store, manager)
    lifecycle = MachineLifecycle(store, manager)
    record = lifecycle.decommission("CNC-3", by=SUP, reason="Scrapped.", now=T0)
    handoff = build_handoff(store, now=T0 + timedelta(days=30))
    (entry,) = handoff["pm"]["retired"]
    assert entry["obligation_id"] == obligation.obligation_id
    assert entry["retirement_reason"] == "machine decommissioned"
    assert entry["retirement_kind"] == "machine_decommission"
    assert entry["decommission_id"] == record.record_id
    assert handoff["pm"]["obligations"] == []
    text = render_text(handoff)
    assert "machine decommissioned" in text
    assert record.record_id in text


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------


def _client(service) -> TestClient:
    api.set_service(service)
    return TestClient(api.app)


def test_api_retire_and_decommission_flow():
    service = make_service()
    client = _client(service)
    obligation = _service_with_weld_obligation(service)

    # Retire: the actor/reason discipline surfaces as 409s.
    r = client.post(f"/api/pm/obligations/{obligation.obligation_id}/retire", json={"by": SUP})
    assert r.status_code == 409
    r = client.post("/api/pm/obligations/PMO-nope/retire", json={"by": SUP, "reason": "x"})
    assert r.status_code == 404
    r = client.post(
        f"/api/pm/obligations/{obligation.obligation_id}/retire",
        json={"by": SUP, "reason": "Pattern closed."},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "retired"
    assert r.json()["retirement_reason"] == "Pattern closed."
    r = client.post(
        f"/api/pm/obligations/{obligation.obligation_id}/retire",
        json={"by": SUP, "reason": "again"},
    )
    assert r.status_code == 409

    # Machines: state, decommission, dispositions, intake refusal.
    r = client.get("/api/machines")
    assert r.status_code == 200
    assert all(m["state"] == "in_service" for m in r.json())
    service.run_case(_coolant("RT-API"))
    r = client.post("/api/machines/CNC-3/decommission", json={"reason": "Scrapped."})
    assert r.status_code == 409  # no actor
    r = client.post("/api/machines/NOPE-9/decommission", json={"by": SUP, "reason": "x"})
    assert r.status_code == 404
    r = client.post(
        "/api/machines/CNC-3/decommission",
        json={"by": SUP, "reason": "Scrapped.", "effective_at": "2026-10-12T00:00:00Z"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["open_case_ids"] == ["RT-API"]
    r = client.post(
        "/api/machines/CNC-3/dispositions",
        json={"case_id": "RT-API", "disposition": "shred", "by": SUP},
    )
    assert r.status_code == 422
    r = client.post(
        "/api/machines/CNC-3/dispositions",
        json={"case_id": "RT-API", "disposition": "close_moot", "by": SUP},
    )
    assert r.status_code == 200
    assert service.get_record("RT-API").status == "closed_moot"
    # Intake for the dead machine is refused, state named.
    r = client.post("/cases/analyze", json=_coolant("RT-API-2").model_dump())
    assert r.status_code == 409
    assert "DECOMMISSIONED" in r.json()["detail"]
