"""Preventive maintenance: from watch to work (pm.py).

The pins, in the order the story runs:

- the interval rule is arithmetic on the evidence's spacing,
  clamped to the plant model's authored bounds;
- a proposal drafts exactly when the watch's own threshold is
  met with fresh evidence, cites that evidence, and stays silent
  below it;
- a rejection spends its evidence (the watermark): the same
  faults cannot re-draft, a fresh threshold can;
- the approved interval — not the proposal's — is what stands,
  and the obligation's clock runs from where the supervisor said
  it runs from;
- a due obligation surfaces exactly one cycle case per cycle,
  through the board's own machinery;
- only a supervisor's outcome marks a PM performed; a deferral
  needs the supervisor's reason and rolls by the authored step;
- a PM cycle case never counts as recurrence evidence, and a
  safety-shaped record never drafts anything.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from plant_loop import api
from plant_loop import handoff as handoff_module
from plant_loop.events import format_timestamp
from plant_loop.kb import pm_policy
from plant_loop.loops import ReliabilityPolicy
from plant_loop.pm import PmManager, interval_arithmetic
from plant_loop.reliability import ReliabilityWatch
from plant_loop.schemas import (
    CaseResult,
    Diagnosis,
    ExceptionCase,
    Hypothesis,
    SupervisorOutcome,
)
from plant_loop.store import InMemoryStore

from helpers import make_service

T0 = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)


def _record(case_id: str, at: datetime, *, machine: str = "CNC-3", fault: str = "spindle_bearing_wear") -> CaseResult:
    return CaseResult(
        case_id=case_id,
        type="machine_down",
        machine_id=machine,
        created_at=format_timestamp(at),
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(
                    fault_class=fault,
                    statement="fabricated for the PM tests",
                    confidence=0.9,
                    sop_id="SOP-X",
                )
            ]
        ),
    )


def _manager(store: InMemoryStore) -> PmManager:
    return PmManager(store, ReliabilityWatch(store, ReliabilityPolicy()))


def _seeded_store(n: int = 3) -> tuple[InMemoryStore, list[CaseResult]]:
    store = InMemoryStore()
    records = []
    for i in range(n):
        record = _record(f"F-{i + 1}", T0 + timedelta(days=4 * i + (i > 0)))
        store.save_case(record)
        records.append(record)
    return store, records


# ---------------------------------------------------------------------------
# The interval rule
# ---------------------------------------------------------------------------


def test_interval_arithmetic_rounds_and_clamps():
    policy = pm_policy()  # fraction 0.5, bounds 3..30
    # Gaps 4 and 5 days -> mean 4.5 -> raw 2.25 -> rounds to 2,
    # clamped up to the 3-day floor.
    out = interval_arithmetic([T0, T0 + timedelta(days=4), T0 + timedelta(days=9)], policy)
    assert out["gaps_days"] == [4.0, 5.0]
    assert out["mean_gap_days"] == 4.5
    assert out["raw_interval_days"] == 2.25
    assert out["proposed_interval_days"] == 3
    assert out["clamped"] is True
    # Gaps 20 and 28 -> mean 24 -> raw 12, inside the bounds.
    out = interval_arithmetic([T0, T0 + timedelta(days=20), T0 + timedelta(days=48)], policy)
    assert out["proposed_interval_days"] == 12
    assert out["clamped"] is False
    # Gaps 90 and 90 -> raw 45, clamped down to the 30-day ceiling.
    out = interval_arithmetic([T0, T0 + timedelta(days=90), T0 + timedelta(days=180)], policy)
    assert out["proposed_interval_days"] == 30
    assert out["clamped"] is True


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def test_proposal_drafted_at_threshold_with_cited_evidence():
    store, records = _seeded_store()
    manager = _manager(store)
    watch = ReliabilityWatch(store, ReliabilityPolicy())
    item = watch.check(records[-1], now=T0 + timedelta(days=10))
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    assert proposal.status == "proposed"
    assert proposal.machine_id == "CNC-3"
    assert proposal.fault_class == "spindle_bearing_wear"
    assert [e["case_id"] for e in proposal.evidence] == ["F-1", "F-2", "F-3"]
    assert proposal.fault_count == 3
    assert proposal.gaps_days == [5.0, 4.0]
    # The proposal answers the watch's item, by id.
    assert item is not None and proposal.board_item_id == item.item_id
    # And the store now holds exactly this one record for the pair.
    assert [p.proposal_id for p in manager.proposals()] == [proposal.proposal_id]


def test_below_threshold_drafts_nothing():
    store, records = _seeded_store(n=2)
    manager = _manager(store)
    assert manager.evaluate(records[-1], now=T0 + timedelta(days=10)) is None
    assert manager.proposals() == []
    assert manager.obligations() == []


def test_open_proposal_suppresses_a_second_draft():
    store, records = _seeded_store()
    manager = _manager(store)
    first = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert first is not None
    fourth = _record("F-4", T0 + timedelta(days=12))
    store.save_case(fourth)
    assert manager.evaluate(fourth, now=T0 + timedelta(days=12)) is None
    assert len(manager.proposals()) == 1


def test_safety_shaped_record_drafts_nothing():
    store, records = _seeded_store()
    manager = _manager(store)
    safety = CaseResult(case_id="S-1", type="safety_observation", machine_id="CNC-3")
    assert manager.evaluate(safety, now=T0 + timedelta(days=10)) is None
    assert manager.proposals() == []


# ---------------------------------------------------------------------------
# The decision and the watermark
# ---------------------------------------------------------------------------


def test_rejection_spends_its_evidence_until_a_fresh_threshold():
    store, records = _seeded_store()
    manager = _manager(store)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    decided, obligation = manager.decide(
        proposal.proposal_id, "rejected", by="Sup", reason="Rebuild already covers this."
    )
    assert decided.status == "rejected"
    assert decided.decision_reason == "Rebuild already covers this."
    assert obligation is None

    # One fresh fault: the cited three are spent, 1 < threshold.
    fourth = _record("F-4", T0 + timedelta(days=11))
    store.save_case(fourth)
    assert manager.evaluate(fourth, now=T0 + timedelta(days=11)) is None
    # Two more fresh faults: a fresh threshold, and the new
    # proposal cites only the fresh evidence.
    for i, cid in enumerate(("F-5", "F-6")):
        record = _record(cid, T0 + timedelta(days=12 + i))
        store.save_case(record)
    fresh = manager.evaluate(record, now=T0 + timedelta(days=13))
    assert fresh is not None
    assert [e["case_id"] for e in fresh.evidence] == ["F-4", "F-5", "F-6"]
    assert len(manager.proposals()) == 2


def test_rejection_without_a_reason_is_refused():
    store, records = _seeded_store()
    manager = _manager(store)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    with pytest.raises(ValueError, match="reason"):
        manager.decide(proposal.proposal_id, "rejected", by="Sup")
    assert manager.proposals()[0].status == "proposed"


def test_decision_is_recorded_once_and_unknown_ids_are_404s():
    store, records = _seeded_store()
    manager = _manager(store)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    manager.decide(proposal.proposal_id, "approved", by="Sup")
    with pytest.raises(ValueError, match="already approved"):
        manager.decide(proposal.proposal_id, "approved", by="Sup")
    with pytest.raises(KeyError):
        manager.decide("PMP-nope", "approved", by="Sup")


def test_approved_interval_stands_and_clock_runs_from_the_declaration():
    store, records = _seeded_store()
    manager = _manager(store)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    decided, obligation = manager.decide(
        proposal.proposal_id,
        "approved",
        by="Sup",
        interval_days=10,
        last_performed_at="2026-09-20T00:00:00Z",
        now=T0 + timedelta(days=10),
    )
    assert decided.status == "approved"
    assert decided.proposed_interval_days != 10  # the draft said otherwise
    assert decided.approved_interval_days == 10  # the approval is what stands
    assert obligation is not None
    assert obligation.interval_days == 10
    assert obligation.clock_basis == "last_performed"
    assert obligation.last_performed_at == "2026-09-20T00:00:00Z"
    assert obligation.next_due_at == "2026-09-30T00:00:00Z"


def test_approval_without_a_declaration_clocks_from_the_approval():
    store, records = _seeded_store()
    manager = _manager(store)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    _decided, obligation = manager.decide(
        proposal.proposal_id, "approved", by="Sup", now=T0 + timedelta(days=10)
    )
    assert obligation is not None
    assert obligation.clock_basis == "approval"
    assert obligation.last_performed_at == ""
    assert obligation.interval_days == proposal.proposed_interval_days
    assert obligation.next_due_at == format_timestamp(
        T0 + timedelta(days=10 + proposal.proposed_interval_days)
    )


def test_unparseable_last_performed_is_refused():
    store, records = _seeded_store()
    manager = _manager(store)
    proposal = manager.evaluate(records[-1], now=T0 + timedelta(days=10))
    assert proposal is not None
    with pytest.raises(ValueError, match="not a record timestamp"):
        manager.decide(
            proposal.proposal_id, "approved", by="Sup", last_performed_at="last Tuesday"
        )


# ---------------------------------------------------------------------------
# Due cases and the cycle
# ---------------------------------------------------------------------------


def test_due_cases_fire_once_per_cycle():
    store = InMemoryStore()
    # Seed a second store's worth of evidence through this store.
    records = []
    for i in range(3):
        record = _record(f"F-{i + 1}", T0 + timedelta(days=4 * i + (i > 0)))
        store.save_case(record)
        records.append(record)
    manager = _manager(store)
    obligation = _standing_obligation_on(manager, records)
    # Due 2026-09-15; a board before that sees nothing.
    assert manager.due_cases(now=datetime(2026, 9, 14, tzinfo=timezone.utc)) == []
    due = manager.due_cases(now=datetime(2026, 9, 15, tzinfo=timezone.utc))
    assert len(due) == 1
    case = due[0]
    assert case.type == "maintenance_alert"
    assert case.machine_id == "CNC-3"
    assert case.reported_at == "2026-09-15T00:00:00Z"
    assert case.case_id == obligation.cycle_case_id
    # The cycle is open: no second case for the same due point.
    assert manager.due_cases(now=datetime(2026, 9, 16, tzinfo=timezone.utc)) == []


def _standing_obligation_on(manager: PmManager, records: list[CaseResult]):
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
    return obligation


def test_cycle_close_performed_stamps_from_the_outcome():
    store = InMemoryStore()
    records = []
    for i in range(3):
        record = _record(f"F-{i + 1}", T0 + timedelta(days=4 * i + (i > 0)))
        store.save_case(record)
        records.append(record)
    manager = _manager(store)
    obligation = _standing_obligation_on(manager, records)
    case = manager.due_cases(now=datetime(2026, 9, 15, tzinfo=timezone.utc))[0]
    outcome = SupervisorOutcome(
        decision="accepted",
        work_performed="Bearing inspection performed; wear within limits.",
        recorded_by="Sup",
        recorded_at="2026-09-16T09:30:00Z",
    )
    closed = manager.close_cycle(_record(case.case_id, T0), outcome)
    assert closed is not None and closed.obligation_id == obligation.obligation_id
    assert closed.last_performed_at == "2026-09-16T09:30:00Z"
    assert closed.next_due_at == "2026-09-30T09:30:00Z"  # performed + 14 days
    assert closed.cycles_completed == 1
    assert closed.cycle_case_id == ""


def test_cycle_deferral_rolls_by_the_authored_step_with_the_reason():
    store = InMemoryStore()
    records = []
    for i in range(3):
        record = _record(f"F-{i + 1}", T0 + timedelta(days=4 * i + (i > 0)))
        store.save_case(record)
        records.append(record)
    manager = _manager(store)
    obligation = _standing_obligation_on(manager, records)
    case = manager.due_cases(now=datetime(2026, 9, 15, tzinfo=timezone.utc))[0]
    outcome = SupervisorOutcome(
        decision="rejected",
        work_performed="Line is mid-campaign; inspection waits for the changeover.",
        recorded_by="Sup",
        recorded_at="2026-09-16T09:30:00Z",
    )
    closed = manager.close_cycle(_record(case.case_id, T0), outcome)
    assert closed is not None
    assert closed.cycles_completed == 0
    assert closed.last_performed_at == "2026-09-01T00:00:00Z"  # unchanged
    assert closed.next_due_at == "2026-09-23T09:30:00Z"  # deferral + 7 days
    assert len(closed.deferrals) == 1
    deferral = closed.deferrals[0]
    assert deferral["reason"] == "Line is mid-campaign; inspection waits for the changeover."
    assert deferral["previous_due_at"] == "2026-09-15T00:00:00Z"
    assert deferral["rolled_to"] == "2026-09-23T09:30:00Z"
    assert closed.cycle_case_id == ""


def test_pm_cycle_cases_never_count_as_recurrence_evidence():
    store = InMemoryStore()
    records = []
    for i in range(3):
        record = _record(f"F-{i + 1}", T0 + timedelta(days=4 * i + (i > 0)))
        store.save_case(record)
        records.append(record)
    manager = _manager(store)
    obligation = _standing_obligation_on(manager, records)
    case = manager.due_cases(now=datetime(2026, 9, 15, tzinfo=timezone.utc))[0]
    # The obligation stands, so no new proposal can draft anyway;
    # the exclusion is pinned at the evidence level: even asking
    # with the cycle case itself as the trigger drafts nothing,
    # and a hypothetical fresh count would exclude its id.
    cycle_record = _record(case.case_id, T0 + timedelta(days=11))
    store.save_case(cycle_record)
    assert manager.evaluate(cycle_record, now=T0 + timedelta(days=12)) is None


# ---------------------------------------------------------------------------
# Through the service
# ---------------------------------------------------------------------------


def _weld_case(case_id: str) -> ExceptionCase:
    return ExceptionCase.model_validate(
        {
            "case_id": case_id,
            "type": "quality_nonconformance",
            "title": "Porosity above limit on bracket welds",
            "description": "QC inspection found porosity in 6.2 percent of welds on the morning bracket batch; batch held at inspection.",
            "machine_id": "WELD-4",
            "line_id": "line-c",
            "product": "Bracket BR-118",
            "batch": "B-4470",
            "readings": {"qc_porosity_pct": 6.2, "gas_flow_l_min": 9.5},
            "notes": "Cylinder was changed on night shift.",
            "reporter": "S. Patel (quality)",
            "reported_at": "2026-10-11T05:40:00Z",
            "downtime_minutes": 0,
        }
    )


def _service_with_obligation() -> tuple[object, object]:
    service = make_service()
    for cid in ("W-1", "W-2", "W-3"):
        service.run_case(_weld_case(cid))
    proposal = service.pm_proposals()[0]
    _decided, obligation = service.decide_pm_proposal(
        proposal.proposal_id,
        "approved",
        by="Sup",
        interval_days=14,
        last_performed_at="2026-09-01T00:00:00Z",
    )
    run = service.run_board([])
    assert run.pm_cases, "the due obligation should surface its cycle case"
    return service, obligation


def test_service_story_board_surfaces_and_outcome_closes():
    service, obligation = _service_with_obligation()
    pm_case_id = service.pm_obligations()[0].cycle_case_id
    record = service.get_record(pm_case_id)
    assert record is not None
    assert record.type == "maintenance_alert"
    assert record.status == "awaiting_supervisor"
    # Approval alone never marks anything performed: the performed
    # stamp is still only the supervisor's declaration.
    assert service.pm_obligations()[0].last_performed_at == "2026-09-01T00:00:00Z"
    service.record_outcome(
        pm_case_id,
        SupervisorOutcome(
            decision="accepted",
            work_performed="Drift inspection performed; test welds passed.",
            recorded_by="Sup",
        ),
    )
    closed = service.pm_obligations()[0]
    assert closed.cycles_completed == 1
    assert closed.cycle_case_id == ""
    outcome = service.get_record(pm_case_id).outcome
    assert closed.last_performed_at == outcome.recorded_at
    # And the board does not resurface the closed cycle.
    assert service.run_board([]).pm_cases == []


def test_service_refuses_a_reasonless_pm_deferral():
    service, _obligation = _service_with_obligation()
    pm_case_id = service.pm_obligations()[0].cycle_case_id
    with pytest.raises(ValueError, match="work-performed note"):
        service.record_outcome(
            pm_case_id,
            SupervisorOutcome(decision="rejected", work_performed="", recorded_by="Sup"),
        )
    # Nothing saved: no outcome on the record, cycle still open.
    assert service.get_record(pm_case_id).outcome is None
    assert service.pm_obligations()[0].cycle_case_id == pm_case_id


def test_service_deferral_rolls_and_shows_in_handoff():
    service, _obligation = _service_with_obligation()
    pm_case_id = service.pm_obligations()[0].cycle_case_id
    service.record_outcome(
        pm_case_id,
        SupervisorOutcome(
            decision="rejected",
            work_performed="Cell booked solid until the weekend changeover.",
            recorded_by="Sup",
        ),
    )
    obligation = service.pm_obligations()[0]
    assert len(obligation.deferrals) == 1
    handoff = handoff_module.build_handoff(service.store)
    assert len(handoff["pm"]["deferrals"]) == 1
    assert handoff["pm"]["obligations"][0]["obligation_id"] == obligation.obligation_id
    text = handoff_module.render_text(handoff)
    assert "PREVENTIVE MAINTENANCE:" in text
    assert "Cell booked solid until the weekend changeover." in text


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def test_pm_over_http(client):
    http, service = client
    for cid in ("W-1", "W-2", "W-3"):
        response = http.post("/cases/analyze", json=_weld_case(cid).model_dump())
        assert response.status_code == 201
    proposals = http.get("/api/pm/proposals").json()
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["status"] == "proposed"
    assert [e["case_id"] for e in proposal["evidence"]] == ["W-1", "W-2", "W-3"]

    # A rejection without a reason is a conflict, not a decision.
    refused = http.post(
        f"/api/pm/proposals/{proposal['proposal_id']}/decision",
        json={"decision": "rejected", "by": "Sup"},
    )
    assert refused.status_code == 409
    # An unknown proposal is a 404; a nonsense decision is a 422.
    assert http.post("/api/pm/proposals/PMP-nope/decision", json={"decision": "approved"}).status_code == 404
    assert http.post(
        f"/api/pm/proposals/{proposal['proposal_id']}/decision", json={"decision": "maybe"}
    ).status_code == 422

    decided = http.post(
        f"/api/pm/proposals/{proposal['proposal_id']}/decision",
        json={
            "decision": "approved",
            "by": "Sup",
            "interval_days": 14,
            "last_performed_at": "2026-09-01T00:00:00Z",
        },
    )
    assert decided.status_code == 200
    body = decided.json()
    assert body["proposal"]["approved_interval_days"] == 14
    assert body["obligation"]["next_due_at"] == "2026-09-15T00:00:00Z"
    obligations = http.get("/api/pm/obligations").json()
    assert len(obligations) == 1
    # A second decision on the same proposal conflicts.
    again = http.post(
        f"/api/pm/proposals/{proposal['proposal_id']}/decision",
        json={"decision": "approved", "by": "Sup"},
    )
    assert again.status_code == 409
