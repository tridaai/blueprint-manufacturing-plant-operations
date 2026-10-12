"""Downtime window planning (windows.py): the window, the plan,
the confirmation.

The staging mirrors the golden entry (evals, kind "windows"):
real converged cases run through the offline loop and approved
at the gate, PM obligations saved as records — then the window
over them. What is pinned:

- work-readiness, clause by clause: awaiting a decision is not
  ready, a rejected case is not ready, a safety stop is never
  ready (escalated, not scheduled), an outcome recorded ends
  readiness, and readiness reads the allocation ledger — a case
  whose claims no longer cover its plan is not ready, however
  approved it is;
- packing in board priority order, greedy, never reordered to
  fill a gap, with the minutes arithmetic on the plan; an item
  longer than the window is OVERSIZED, an item with no estimate
  on record is UNESTIMATED and never given a number;
- obligations interleaved on the board's own footing — the
  priority formula read as the maintenance alert their cycle
  would be — and due exactly when the window's end covers their
  next-due moment;
- confirmation as a supervisor's act over a fingerprint:
  drift (a member outcomed after computation) refuses naming
  the case, re-planning is the remedy; a confirmed window books
  its packed items against overlapping windows, and never
  re-plans or re-confirms;
- the overlay invariant: planning and confirming change no case
  record, byte for byte;
- declaration refusals by name; the API surface and its status
  codes; the handoff naming confirmed windows still ahead.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from plant_loop import windows as windows_module
from plant_loop.api import app
from plant_loop.handoff import build_handoff
from plant_loop.schemas import (
    CaseResult,
    ExceptionCase,
    PmObligation,
    SupervisorOutcome,
)

from helpers import make_service

START = "2026-10-17T08:00:00Z"
SCOPE_LINES = ["line-b"]
SCOPE_MACHINES = ["CNC-3", "WELD-4", "PACK-6", "PRESS-1"]

# The staging payloads: the golden board's own shapes (the same
# faults the loop is known to converge on, offline).
CASE_A = {  # CNC-3 spindle bearing wear — est 120, priority 900
    "case_id": "WIN-A", "type": "machine_down",
    "title": "CNC-3 spindle vibration high, line stopped",
    "description": "Spindle vibration alarm tripped at 06:10 and the machine was stopped by the operator. Audible rumble from the spindle head at speed.",
    "machine_id": "CNC-3", "line_id": "line-b",
    "product": "Gear blank GB-220", "batch": "B-4471",
    "readings": {"spindle_vibration_mm_s": 8.4, "spindle_temp_c": 86, "tool_runout_mm": 0.008},
    "notes": "Operator: rumble gets worse above 6000 rpm. No recent tool change.",
    "reporter": "J. Okafor (operator)", "reported_at": "2026-10-15T06:12:00Z",
    "downtime_minutes": 0,
}
CASE_B = {  # WELD-4 weld quality drift — est 40, priority 500
    "case_id": "WIN-B", "type": "quality_nonconformance",
    "title": "Porosity above limit on bracket welds",
    "description": "QC inspection found porosity in 6.2 percent of welds on the morning bracket batch; batch held at inspection.",
    "machine_id": "WELD-4", "line_id": "line-c",
    "product": "Bracket BR-118", "batch": "B-4470",
    "readings": {"qc_porosity_pct": 6.2, "gas_flow_l_min": 9.5},
    "notes": "Cylinder was changed on night shift.",
    "reporter": "S. Patel (quality)", "reported_at": "2026-10-15T05:40:00Z",
    "downtime_minutes": 0,
}
CASE_C = {  # PACK-6 film shortage — est 20, priority 312.5 (substitute)
    "case_id": "WIN-C", "type": "material_shortage",
    "title": "Packing film out of stock at PACK-6",
    "description": "Last roll of 450 mm packing film used at 06:30; stores show zero rolls. Packing can continue on the approved grade B film if confirmed.",
    "machine_id": "PACK-6", "line_id": "line-d",
    "product": "Retail carton RC-9", "batch": "B-4472",
    "readings": {"material": "FILM-PKG", "rolls_remaining": 0},
    "notes": "", "reporter": "T. Nguyen (stores)",
    "reported_at": "2026-10-15T06:31:00Z", "downtime_minutes": 0,
}
CASE_E = {  # CONV-5 belt tracking — converged, approved, OUT of scope
    "case_id": "WIN-E", "type": "machine_down",
    "title": "CONV-5 belt tracking fault, line stopped",
    "description": "Belt drifting to one side, edge fraying past the wear mark.",
    "machine_id": "CONV-5", "line_id": "line-d",
    "product": "Assembly kit AK-12", "batch": "B-4621",
    "readings": {"belt_drift_mm": 9}, "notes": "",
    "reporter": "floor-lead", "reported_at": "2026-10-15T06:00:00Z",
    "downtime_minutes": 0,
}


def _obligation(oid: str, machine: str, fault: str, line: str, due: str) -> PmObligation:
    return PmObligation(
        obligation_id=oid, proposal_id="PMP-STAGED", machine_id=machine,
        fault_class=fault, line_id=line, interval_days=30,
        approved_by="Staging supervisor", approved_at="2026-10-01T00:00:00Z",
        clock_basis="approval", next_due_at=due,
        created_at="2026-10-01T00:00:00Z",
    )


def _run_approved(service, payload: dict) -> None:
    result = service.run_case(ExceptionCase.model_validate(payload)).result
    assert result.exit_reason == "converged", (payload["case_id"], result.exit_reason)
    service.record_decision(payload["case_id"], "approved", by="Staging supervisor")


def _staged(service):
    """The golden staging: A/B/C approved and in scope, E approved
    out of scope, and four obligations — deferred, oversized,
    unestimated, and not yet due."""
    for payload in (CASE_A, CASE_B, CASE_C, CASE_E):
        _run_approved(service, payload)
    service.store.save_pm_record(
        _obligation("PMO-W1", "PRESS-1", "hydraulic_pressure_loss", "line-a", "2026-10-17T06:00:00Z"))
    service.store.save_pm_record(
        _obligation("PMO-W2", "CNC-2", "spindle_cartridge_failure", "line-b", "2026-10-17T09:00:00Z"))
    service.store.save_pm_record(
        _obligation("PMO-W3", "CNC-3", "bearing_seat_fretting", "line-b", "2026-10-17T09:30:00Z"))
    service.store.save_pm_record(
        _obligation("PMO-W4", "WELD-4", "wire_feed_fault", "line-c", "2026-11-05T09:00:00Z"))
    return service


def _declare(service, **overrides):
    kwargs = dict(
        by="Dana (shift lead)", starts_at=START, length_min=180,
        line_ids=SCOPE_LINES, machine_ids=SCOPE_MACHINES,
    )
    kwargs.update(overrides)
    return service.declare_window(**kwargs)


# -- readiness -------------------------------------------------------------------


def test_readiness_definition_clause_by_clause():
    service = _staged(make_service())
    window = _declare(service)
    # Awaiting its decision: a fresh converged case, never decided.
    fresh = service.run_case(ExceptionCase.model_validate(
        {**CASE_B, "case_id": "WIN-WAIT", "batch": "B-9999"})).result
    assert fresh.status == "awaiting_supervisor"
    ready, why = windows_module.case_readiness(
        service.store.get_case("WIN-WAIT"), service.allocation)
    assert not ready and "not approved" in why
    # The approved staged cases are ready; the decision is what
    # made them ready.
    ready, _ = windows_module.case_readiness(
        service.store.get_case("WIN-A"), service.allocation)
    assert ready
    # An outcome ends readiness: the work is no longer ahead.
    service.record_outcome("WIN-A", SupervisorOutcome(
        decision="accepted", confirmed_fault="spindle_bearing_wear",
        work_performed="bearing replaced", recorded_by="Dana"))
    ready, why = windows_module.case_readiness(
        service.store.get_case("WIN-A"), service.allocation)
    assert not ready and "outcome" in why
    assert window.packed  # the earlier plan stands as computed


def test_readiness_reads_the_ledger_not_the_approval():
    """A claim released after approval ends readiness — readiness
    is read from the ledger as it stands, never assumed."""
    service = _staged(make_service())
    record = service.store.get_case("WIN-C")
    ready, _ = windows_module.case_readiness(record, service.allocation)
    assert ready  # the substitute FILM-PKG-B is claimed for it
    released = service.allocation.release_for_case("WIN-C", by="turnover")
    assert released
    ready, why = windows_module.case_readiness(
        service.store.get_case("WIN-C"), service.allocation)
    assert not ready and "ledger" in why


def test_safety_stop_is_never_work_ready():
    """A safety case escalated at round 0 cannot be scheduled —
    and even a hand-built approved-looking record is refused by
    the safety clause first."""
    service = make_service()
    payload = {
        "case_id": "SAFE-1", "type": "safety_observation",
        "title": "Guard interlock bypassed on PRESS-1",
        "description": "Operator reports the guard interlock switch taped over.",
        "machine_id": "PRESS-1", "line_id": "line-a",
        "product": "", "batch": "", "readings": {}, "notes": "",
        "reporter": "floor", "reported_at": "2026-10-15T06:00:00Z",
    }
    record = service.run_case(ExceptionCase.model_validate(payload)).result
    assert record.exit_reason == "safety_stop"
    ready, why = windows_module.case_readiness(record, service.allocation)
    assert not ready and "safety stop" in why


# -- the plan ----------------------------------------------------------------------


def test_plan_packs_exactly_in_board_order_with_arithmetic():
    service = _staged(make_service())
    window = _declare(service)
    assert [i.item_id for i in window.packed] == ["WIN-A", "WIN-B", "WIN-C"]
    assert [(i.start_offset_min, i.end_offset_min) for i in window.packed] == [
        (0, 120), (120, 160), (160, 180)]
    assert window.minutes_used == 180 and window.minutes_left == 0
    # The order is the board's: scores descend, and each item
    # carries the reason the board printed for it.
    scores = [i.priority for i in window.packed]
    assert scores == sorted(scores, reverse=True) == [900.0, 500.0, 312.5]
    assert all(i.priority_reason for i in window.packed)
    # The obligation interleaved by the same formula — after the
    # cases here, on its own maintenance footing.
    assert [i.item_id for i in window.deferred] == ["PMO-W1"]
    deferred = window.deferred[0]
    assert deferred.priority == 225.0 and "maintenance_alert" in deferred.priority_reason
    assert "short by 150 min" in deferred.note
    # Oversized and unestimated are listed, never squeezed or
    # numbered.
    assert [i.item_id for i in window.oversized] == ["PMO-W2"]
    assert window.oversized[0].estimate_min == 240
    assert [i.item_id for i in window.unestimated] == ["PMO-W3"]
    assert window.unestimated[0].estimate_min is None
    # Not due within the span: in no plan section (it stands in
    # the fingerprint snapshot, marked not-a-candidate — the
    # snapshot is the plan's basis, not the plan). Out of scope:
    # nowhere at all.
    section_ids = [i.item_id for s in (
        window.packed, window.deferred, window.oversized,
        window.unestimated, window.booked_elsewhere) for i in s]
    assert "PMO-W4" not in section_ids and "WIN-E" not in section_ids
    assert "WIN-E" not in window.model_dump_json()
    assert window.status == "proposed" and window.plan_basis


def test_packing_never_reorders_to_fill_a_gap():
    """A smaller low-priority item does not jump ahead of a
    bigger higher-priority one that no longer fits: the bigger
    is deferred and the smaller is considered against what is
    left — the order is the plant's priority, shown honestly."""
    service = _staged(make_service())
    # 150 minutes: A (120) packs, B (40) no longer fits and is
    # deferred, C (20) still fits the remaining 30 — considered in
    # order, not promoted ahead of anything.
    window = _declare(service, length_min=150)
    assert [i.item_id for i in window.packed] == ["WIN-A", "WIN-C"]
    assert [i.item_id for i in window.deferred] == ["WIN-B", "PMO-W1"]
    assert "short by 10 min" in window.deferred[0].note
    assert window.minutes_used == 140 and window.minutes_left == 10


def test_unestimated_case_is_listed_never_numbered():
    """A converged approved case whose plan states no estimate
    (hand-built record: the offline planner always copies the
    SOP's figure for a converged plan, so this shape is staged
    directly) is UNESTIMATED — listed, and packed never."""
    service = make_service()
    record = CaseResult(
        case_id="NOEST-1", status="approved", decision="approved",
        decision_by="Dana", exit_reason="converged",
        machine_id="CNC-3", line_id="line-b", priority=900.0,
        priority_reason="staged", exception=ExceptionCase.model_validate(
            {**CASE_A, "case_id": "NOEST-1"}),
    )
    record.plan.estimated_downtime_min = 0
    service.store.save_case(record)
    window = _declare(service)
    assert [i.item_id for i in window.unestimated] == ["NOEST-1"]
    assert window.unestimated[0].estimate_min is None
    assert window.packed == [] and window.minutes_used == 0


def test_obligation_with_open_cycle_is_planned_through_its_case():
    service = _staged(make_service())
    obligation = _obligation(
        "PMO-W5", "PRESS-1", "hydraulic_pressure_loss", "line-a",
        "2026-10-17T06:30:00Z")
    obligation.cycle_case_id = "CYCLE-OPEN-1"
    service.store.save_pm_record(obligation)
    window = _declare(service)
    blob = window.model_dump_json()
    assert "PMO-W5" not in [i.item_id for s in (
        window.packed, window.deferred, window.oversized, window.unestimated)
        for i in s]
    assert any("PMO-W5" in note for note in window.notes)


# -- confirmation ------------------------------------------------------------------


def _snapshots(service):
    return {c.case_id: c.model_dump_json() for c in service.store.all_cases()}


def test_confirmation_books_members_and_cases_do_not_move():
    service = _staged(make_service())
    window = _declare(service)
    before = _snapshots(service)
    window = service.confirm_window(window.window_id, by="Ruth (plant manager)")
    assert window.status == "confirmed"
    assert window.confirmed_by == "Ruth (plant manager)" and window.confirmed_at
    # The overlay invariant: confirmation changed no case record.
    assert _snapshots(service) == before
    # A second window over the same span plans around the booked.
    second = _declare(service, starts_at="2026-10-17T10:30:00Z", length_min=120)
    assert [i.item_id for i in second.booked_elsewhere] == ["WIN-A", "WIN-B", "WIN-C"]
    assert all(
        i.note.startswith("booked into confirmed window") for i in second.booked_elsewhere)
    assert not second.packed
    # And a non-overlapping window books nothing.
    third = _declare(service, starts_at="2026-10-17T11:00:00Z", length_min=60)
    assert third.booked_elsewhere == []


def test_confirmation_refuses_on_drift_and_names_it():
    service = _staged(make_service())
    window = _declare(service)
    service.record_outcome("WIN-B", SupervisorOutcome(
        decision="accepted", confirmed_fault="weld_quality_drift",
        work_performed="nozzle changed, welds clean", recorded_by="Ruth"))
    try:
        service.confirm_window(window.window_id, by="Ruth")
        raise AssertionError("confirmation should have refused")
    except ValueError as exc:
        message = str(exc)
    assert "WIN-B" in message and "Re-plan" in message
    # The refusal changed nothing: the window is still proposed.
    assert service.get_window(window.window_id).status == "proposed"
    # Re-planning is the remedy: the fresh plan drops the
    # outcomed case, and confirms clean.
    window = service.plan_window(window.window_id)
    assert [i.item_id for i in window.packed] == ["WIN-A", "WIN-C"]
    window = service.confirm_window(window.window_id, by="Ruth")
    assert window.status == "confirmed"


def test_confirmation_is_once_and_a_confirmed_window_never_replans():
    service = _staged(make_service())
    window = _declare(service)
    service.confirm_window(window.window_id, by="Ruth")
    for act in (
        lambda: service.confirm_window(window.window_id, by="Ruth"),
        lambda: service.plan_window(window.window_id),
    ):
        try:
            act()
            raise AssertionError("should have refused")
        except ValueError:
            pass


def test_declaration_refusals_are_named():
    service = _staged(make_service())
    for kwargs, needle in (
        (dict(by=""), "supervisor's name"),
        (dict(starts_at="not a time"), "not a record timestamp"),
        (dict(length_min=0), "at least 1 minute"),
        (dict(line_ids=[], machine_ids=[]), "at least one line or machine"),
        (dict(machine_ids=["NOPE-9"]), "unknown machine"),
        (dict(line_ids=["line-z"]), "unknown line"),
    ):
        try:
            _declare(service, **kwargs)
            raise AssertionError(f"should have refused: {kwargs}")
        except ValueError as exc:
            assert needle in str(exc), (kwargs, exc)


# -- surfaces ------------------------------------------------------------------------


def test_window_api_surface_and_status_codes():
    service = _staged(make_service())
    import plant_loop.api as api_module

    api_module.set_service(service)
    try:
        _window_api_flow(TestClient(app), service)
    finally:
        api_module.set_service(None)


def _window_api_flow(client, service):
    declared = client.post("/api/windows", json={
        "by": "Dana", "starts_at": START, "length_min": 180,
        "line_ids": SCOPE_LINES, "machine_ids": SCOPE_MACHINES})
    assert declared.status_code == 200
    window = declared.json()
    assert [i["item_id"] for i in window["packed"]] == ["WIN-A", "WIN-B", "WIN-C"]

    listing = client.get("/api/windows").json()
    assert listing["counts"] == {"planned": 1, "proposed": 1, "confirmed": 0}
    text = client.get(f"/api/windows/{window['window_id']}?format=text")
    assert text.status_code == 200 and "180 used" in text.text

    confirmed = client.post(
        f"/api/windows/{window['window_id']}/confirm", json={"by": "Ruth"})
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "confirmed"
    again = client.post(
        f"/api/windows/{window['window_id']}/confirm", json={"by": "Ruth"})
    assert again.status_code == 409

    assert client.get("/api/windows/WIN-nope").status_code == 404
    bad = client.post("/api/windows", json={
        "by": "", "starts_at": START, "length_min": 60,
        "line_ids": ["line-b"], "machine_ids": []})
    assert bad.status_code == 409
    counts = client.get("/api/windows").json()["counts"]
    assert counts == {"planned": 1, "proposed": 0, "confirmed": 1}


def test_handoff_names_confirmed_windows_still_ahead():
    from datetime import datetime, timezone

    service = _staged(make_service())
    window = _declare(service)
    moment = datetime(2026, 10, 17, 7, 0, tzinfo=timezone.utc)
    # Proposed: not a schedule yet — the handoff does not name it.
    assert build_handoff(service.store, now=moment)["windows"] == []
    service.confirm_window(window.window_id, by="Ruth")
    handoff = build_handoff(service.store, now=moment)
    assert [w["window_id"] for w in handoff["windows"]] == [window.window_id]
    assert handoff["windows"][0]["packed_cases"] == ["WIN-A", "WIN-B", "WIN-C"]
    from plant_loop.handoff import render_text as handoff_text

    assert "CONFIRMED DOWNTIME WINDOWS AHEAD" in handoff_text(handoff)
    assert window.window_id in handoff_text(handoff)
    # Once the window is over, it is history, not a handover.
    later = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    assert build_handoff(service.store, now=later)["windows"] == []
