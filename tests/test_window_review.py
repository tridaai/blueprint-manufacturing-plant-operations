"""Window review (windows.py): what the window actually got done.

The pins, in the order the story runs:

- a proposed window has no review (a proposal is not a promise),
  and a window whose span is still open refuses by name — the
  review opens when the span closes, not before;
- a packed case is judged by its outcome: accepted is performed,
  accepted WITH CHANGES is performed (the inclusion is pinned),
  rejected is deferred with its reason, and no outcome is not
  worked — still open, or closed without the work (closed as
  moot names the decommission; a drop names the acceptance);
- a packed PM obligation is judged by the cycle case that
  answered the due point the plan packed: performed, rejected
  (rolled), dropped at a board acceptance (also rolled — a
  deferral), still open, or never surfaced;
- the minutes are the plan's own estimates restated — planned,
  performed, deferred, not worked — sums only;
- the ledger lists confirmed windows in span order with each
  window's own arithmetic and never a cross-window score;
- the handoff names an ended window whose work is still open —
  and does not name one whose unfinished item is settled history
  (closed as moot);
- the review writes nothing: cases, obligations and the window
  itself stand byte-identical across it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from plant_loop import api
from plant_loop import handoff as handoff_module
from plant_loop.schemas import (
    CaseResult,
    ExceptionCase,
    PmObligation,
    SupervisorOutcome,
)
from plant_loop.windows import window_ledger

from helpers import make_service

SUP = "Dana (shift supervisor)"
START = "2026-10-17T08:00:00Z"  # Saturday; ends 11:00 at 180 min
AFTER = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)

CASE_A = {  # CNC-3 spindle bearing — est 120, priority 900
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
CASE_B = {  # WELD-4 weld drift — est 40, priority 500
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
CASE_C = {  # PACK-6 film shortage, substitute — est 20, priority 312.5
    "case_id": "WIN-C", "type": "material_shortage",
    "title": "Packing film out of stock at PACK-6",
    "description": "Last roll of 450 mm packing film used at 06:30; stores show zero rolls. Packing can continue on the approved grade B film if confirmed.",
    "machine_id": "PACK-6", "line_id": "line-d",
    "product": "Retail carton RC-9", "batch": "B-4472",
    "readings": {"material": "FILM-PKG", "rolls_remaining": 0},
    "notes": "", "reporter": "T. Nguyen (stores)",
    "reported_at": "2026-10-15T06:31:00Z", "downtime_minutes": 0,
}


def _run_approved(service, payload: dict) -> None:
    result = service.run_case(ExceptionCase.model_validate(payload)).result
    assert result.exit_reason == "converged"
    service.record_decision(payload["case_id"], "approved", by=SUP)


def _staged_board(service) -> None:
    for payload in (CASE_A, CASE_B, CASE_C):
        _run_approved(service, payload)


def _confirmed_case_window(service):
    """The round-15 window: A + B + C pack 180 minutes exactly."""
    _staged_board(service)
    window = service.declare_window(
        by=SUP, starts_at=START, length_min=180,
        line_ids=["line-b"], machine_ids=["CNC-3", "WELD-4", "PACK-6", "PRESS-1"],
    )
    assert [i.item_id for i in window.packed] == ["WIN-A", "WIN-B", "WIN-C"]
    return service.confirm_window(window.window_id, by=SUP)


def _outcome(service, case_id: str, decision: str, note: str) -> None:
    service.record_outcome(
        case_id,
        SupervisorOutcome(
            decision=decision, work_performed=note, recorded_by=SUP
        ),
    )


def _item(review: dict, item_id: str) -> dict:
    return next(i for i in review["items"] if i["item_id"] == item_id)


# -- the refusals ------------------------------------------------------------------


def test_review_refuses_a_proposed_window_by_name():
    service = make_service()
    _staged_board(service)
    window = service.declare_window(
        by=SUP, starts_at=START, length_min=180,
        line_ids=["line-b"], machine_ids=["CNC-3", "WELD-4", "PACK-6"],
    )
    with pytest.raises(ValueError, match="only a confirmed window"):
        service.window_review(window.window_id, now=AFTER)


def test_review_refuses_while_the_span_is_open_and_opens_at_its_end():
    service = make_service()
    window = _confirmed_case_window(service)
    with pytest.raises(ValueError) as refused:
        service.window_review(
            window.window_id,
            now=datetime(2026, 10, 17, 10, 59, tzinfo=timezone.utc),
        )
    assert "cannot be reviewed yet" in str(refused.value)
    assert "2026-10-17T11:00:00Z" in str(refused.value)  # says when it opens
    # The span is half-open: at its end the review opens.
    review = service.window_review(
        window.window_id,
        now=datetime(2026, 10, 17, 11, 0, tzinfo=timezone.utc),
    )
    assert review["counts"]["packed"] == 3


def test_review_of_an_unknown_window_is_a_key_error():
    service = make_service()
    with pytest.raises(KeyError):
        service.window_review("WIN-nope", now=AFTER)


# -- cases, judged by their outcomes -----------------------------------------------


def test_review_judges_cases_by_their_outcomes():
    service = make_service()
    window = _confirmed_case_window(service)
    _outcome(service, "WIN-A", "accepted", "Bearing set replaced and run-in.")
    _outcome(service, "WIN-B", "accepted_with_changes",
             "Weld schedule adjusted at the gate; rework done on the "
             "window's own settings.")
    _outcome(service, "WIN-C", "rejected",
             "Film delivery slipped to Monday; the window's hours went "
             "to the bearing job instead.")

    review = service.window_review(window.window_id, now=AFTER)
    assert review["counts"] == {
        "packed": 3, "performed": 2, "deferred": 1, "not_worked": 0,
    }
    # The plan's own estimates, restated by state — sums only.
    assert review["minutes"] == {
        "planned": 180, "performed": 160, "deferred": 20, "not_worked": 0,
    }
    assert review["performed_share"] == round(160 / 180, 4)

    a = _item(review, "WIN-A")
    assert a["state"] == "performed"
    assert "outcome accepted" in a["detail"]
    b = _item(review, "WIN-B")
    # accepted_with_changes is in the performed set — the work
    # happened, on the gate's terms. Pinned here by name.
    assert b["state"] == "performed"
    assert "accepted_with_changes" in b["detail"]
    c = _item(review, "WIN-C")
    assert c["state"] == "deferred"
    assert "outcome rejected" in c["detail"]
    assert "Film delivery slipped" in c["detail"]  # the reason travels
    # The wording discipline: estimates restated, never measured time.
    assert "planned minutes performed" in review["basis"]
    assert "never time worked" in review["basis"]


def test_review_late_outcome_counts_when_it_lands():
    service = make_service()
    window = _confirmed_case_window(service)
    before = service.window_review(window.window_id, now=AFTER)
    assert _item(before, "WIN-A")["state"] == "not_worked"
    _outcome(service, "WIN-A", "accepted", "Done on the next shift's hours.")
    after = service.window_review(
        window.window_id,
        now=datetime(2026, 10, 18, 8, 0, tzinfo=timezone.utc),
    )
    assert _item(after, "WIN-A")["state"] == "performed"
    assert after["reviewed_at"].startswith("2026-10-18")


def test_review_untouched_case_is_not_worked_and_still_open():
    service = make_service()
    window = _confirmed_case_window(service)
    _outcome(service, "WIN-A", "accepted", "Bearing set replaced.")
    review = service.window_review(window.window_id, now=AFTER)
    for case_id in ("WIN-B", "WIN-C"):
        row = _item(review, case_id)
        assert row["state"] == "not_worked"
        assert row["still_open"] is True
        assert "still open" in row["detail"]
    assert review["counts"]["not_worked"] == 2
    assert review["minutes"]["not_worked"] == 60  # 40 + 20, the plan's figures


def test_review_closed_moot_is_not_worked_but_settled():
    service = make_service()
    window = _confirmed_case_window(service)
    # The state lifecycle.py leaves when a supervisor disposes a
    # case as moot after its machine's decommission — staged as
    # the record stands, the way test_windows stages its records.
    record = service.store.get_case("WIN-C")
    record.status = "closed_moot"
    record.closed_as_moot = {
        "decommission_id": "DEC-1", "machine_id": "PACK-6",
        "by": SUP, "at": "2026-10-17T13:05:00Z",
    }
    service.store.save_case(record)
    review = service.window_review(window.window_id, now=AFTER)
    row = _item(review, "WIN-C")
    assert row["state"] == "not_worked"
    assert row["still_open"] is False  # settled history, not open work
    assert "closed as moot" in row["detail"]
    assert "decommissioned" in row["detail"]


# -- PM obligations, judged by their cycle cases ------------------------------------


def _pm_window(service):
    """A window over PRESS-1 alone: PMO-W1 (est 150) packs."""
    service.store.save_pm_record(PmObligation(
        obligation_id="PMO-W1", proposal_id="PMP-TEST", machine_id="PRESS-1",
        fault_class="hydraulic_pressure_loss", line_id="line-a",
        interval_days=30, approved_by=SUP, approved_at="2026-10-01T00:00:00Z",
        clock_basis="approval", next_due_at="2026-10-17T06:00:00Z",
        created_at="2026-10-01T00:00:00Z",
    ))
    window = service.declare_window(
        by=SUP, starts_at=START, length_min=180,
        line_ids=["line-a"], machine_ids=["PRESS-1"],
    )
    assert [i.item_id for i in window.packed] == ["PMO-W1"]
    return service.confirm_window(window.window_id, by=SUP)


def _surface_cycle(service, *, status: str = "approved") -> CaseResult:
    """The cycle case a board run surfaces for the due point, as
    the store record it becomes."""
    due = service.pm_manager.due_cases(
        now=datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc))
    assert len(due) == 1
    case = due[0]
    record = CaseResult(
        case_id=case.case_id, type=case.type, machine_id=case.machine_id,
        line_id=case.line_id, reported_at=case.reported_at,
        title=case.title, status=status,
    )
    service.store.save_case(record)
    return record


def _close_cycle(service, record: CaseResult, decision: str, note: str) -> None:
    """An outcome on the cycle case + pm.py's own cycle close —
    the two halves of what service.record_outcome does with one."""
    outcome = SupervisorOutcome(
        decision=decision, work_performed=note, recorded_by=SUP,
        recorded_at="2026-10-17T10:15:00Z",
    )
    record.outcome = outcome
    service.store.save_case(record)
    service.pm_manager.close_cycle(record, outcome)


def test_review_pm_cycle_performed_counts_performed():
    service = make_service()
    window = _pm_window(service)
    record = _surface_cycle(service)
    _close_cycle(service, record, "accepted",
                 "Hydraulics inspection performed; seals sound.")
    review = service.window_review(window.window_id, now=AFTER)
    row = _item(review, "PMO-W1")
    assert row["state"] == "performed"
    assert record.case_id in row["detail"]
    assert review["minutes"] == {
        "planned": 150, "performed": 150, "deferred": 0, "not_worked": 0,
    }
    assert review["performed_share"] == 1.0


def test_review_pm_cycle_rejected_is_deferred_and_rolled():
    service = make_service()
    window = _pm_window(service)
    record = _surface_cycle(service)
    _close_cycle(service, record, "rejected",
                 "Press is mid-campaign; inspection waits for the "
                 "changeover.")
    review = service.window_review(window.window_id, now=AFTER)
    row = _item(review, "PMO-W1")
    assert row["state"] == "deferred"
    assert "rolled its due point to" in row["detail"]


def test_review_pm_cycle_dropped_at_acceptance_is_a_deferral():
    service = make_service()
    window = _pm_window(service)
    record = _surface_cycle(service, status="awaiting_supervisor")
    # The drop as carryover.py stamps it, and pm.py's own roll.
    record.status = "dropped"
    record.dropped_from_board = {
        "from_run": "RUN-1", "into_run": "RUN-2", "by": SUP,
        "reason": "Next shift takes the press work as one block.",
        "at": "2026-10-18T06:00:00Z",
    }
    service.store.save_case(record)
    service.pm_manager.note_board_drop(
        record.case_id, by=SUP,
        reason="Next shift takes the press work as one block.",
    )
    review = service.window_review(window.window_id, now=AFTER)
    row = _item(review, "PMO-W1")
    assert row["state"] == "deferred"
    assert "dropped at a board acceptance" in row["detail"]


def test_review_pm_no_cycle_surfaced_is_not_worked():
    service = make_service()
    window = _pm_window(service)
    review = service.window_review(window.window_id, now=AFTER)
    row = _item(review, "PMO-W1")
    assert row["state"] == "not_worked"
    assert row["still_open"] is True
    assert "no cycle case surfaced" in row["detail"]
    assert "2026-10-17T06:00:00Z" in row["detail"]  # the packed due point


def test_review_pm_cycle_still_open_is_not_worked():
    service = make_service()
    window = _pm_window(service)
    record = _surface_cycle(service)
    review = service.window_review(window.window_id, now=AFTER)
    row = _item(review, "PMO-W1")
    assert row["state"] == "not_worked"
    assert row["still_open"] is True
    assert record.case_id in row["detail"]
    assert "still open" in row["detail"]


# -- the ledger, the handoff, and purity ---------------------------------------------


def test_ledger_rows_are_span_ordered_and_never_scored():
    service = make_service()
    later = _pm_window(service)  # Oct 17, declared first
    _staged_board(service)
    earlier = service.declare_window(
        by=SUP, starts_at="2026-10-10T08:00:00Z", length_min=180,
        line_ids=["line-b"], machine_ids=["CNC-3", "WELD-4", "PACK-6"],
    )
    earlier = service.confirm_window(earlier.window_id, by=SUP)
    rows = window_ledger(service.store, now=AFTER)
    assert [r["window_id"] for r in rows] == [earlier.window_id, later.window_id]
    first, second = rows
    assert first["review_open"] is True  # its span closed Oct 10
    assert first["counts"]["packed"] == 3
    assert second["review_open"] is True
    assert second["counts"]["packed"] == 1
    for row in rows:  # no cross-window score exists to leak rankings
        assert "score" not in row and "rank" not in row


def test_ledger_row_for_an_open_window_says_when_review_opens():
    service = make_service()
    window = _confirmed_case_window(service)
    rows = window_ledger(
        service.store, now=datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc))
    assert len(rows) == 1
    row = rows[0]
    assert row["review_open"] is False
    assert row["counts"] is None
    assert "2026-10-17T11:00:00Z" in row["review_note"]


def test_handoff_names_the_window_whose_work_is_still_open():
    service = make_service()
    window = _confirmed_case_window(service)
    _outcome(service, "WIN-A", "accepted", "Bearing set replaced.")
    handoff = handoff_module.build_handoff(service.store, now=AFTER)
    named = handoff["windows_unfinished"]
    assert [w["window_id"] for w in named] == [window.window_id]
    assert {i["item_id"] for i in named[0]["unfinished"]} == {"WIN-B", "WIN-C"}
    text = handoff_module.render_text(handoff)
    assert "DOWNTIME WINDOWS OVER — WORK STILL OPEN" in text
    assert "WIN-B" in text


def test_handoff_stays_silent_when_nothing_open_remains():
    service = make_service()
    window = _pm_window(service)
    record = _surface_cycle(service)
    _close_cycle(service, record, "accepted", "Inspection performed.")
    handoff = handoff_module.build_handoff(service.store, now=AFTER)
    assert handoff["windows_unfinished"] == []


def test_handoff_does_not_name_a_window_settled_as_moot():
    service = make_service()
    window = _confirmed_case_window(service)
    record = service.store.get_case("WIN-C")
    record.status = "closed_moot"
    record.closed_as_moot = {
        "decommission_id": "DEC-1", "machine_id": "PACK-6",
        "by": SUP, "at": "2026-10-17T13:05:00Z",
    }
    service.store.save_case(record)
    _outcome(service, "WIN-A", "accepted", "Done.")
    _outcome(service, "WIN-B", "accepted", "Done.")
    handoff = handoff_module.build_handoff(service.store, now=AFTER)
    assert handoff["windows_unfinished"] == []


def test_review_writes_nothing():
    service = make_service()
    window = _confirmed_case_window(service)
    _pm_window(service)  # a second confirmed window in the same store
    _outcome(service, "WIN-A", "accepted", "Bearing set replaced.")

    def snapshot() -> str:
        cases = sorted(
            c.model_dump_json() for c in service.store.all_cases())
        records = sorted(
            r.model_dump_json() for r in service.store.all_pm_records())
        windows = sorted(
            w.model_dump_json() for w in service.store.all_window_records())
        return "\n".join(cases + records + windows)

    before = snapshot()
    service.window_review(window.window_id, now=AFTER)
    service.window_ledger(now=AFTER)
    handoff_module.build_handoff(service.store, now=AFTER)
    assert snapshot() == before


# -- the HTTP surface ------------------------------------------------------------------


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def test_review_over_http_once_the_span_has_closed(client):
    test_client, service = client
    # A window in the recent past: its span closed before the
    # wall clock's now, so the live endpoint can answer for it.
    _staged_board(service)
    window = service.declare_window(
        by=SUP, starts_at="2026-10-10T08:00:00Z", length_min=180,
        line_ids=["line-b"], machine_ids=["CNC-3", "WELD-4", "PACK-6"],
    )
    window = service.confirm_window(window.window_id, by=SUP)
    _outcome(service, "WIN-A", "accepted", "Bearing set replaced.")
    response = test_client.get(f"/api/windows/{window.window_id}/review")
    assert response.status_code == 200
    body = response.json()
    assert body["counts"] == {
        "packed": 3, "performed": 1, "deferred": 0, "not_worked": 2,
    }
    text = test_client.get(
        f"/api/windows/{window.window_id}/review", params={"format": "text"})
    assert text.status_code == 200
    assert "WINDOW REVIEW" in text.text
    listing = test_client.get("/api/windows").json()
    assert [r["window_id"] for r in listing["ledger"]] == [window.window_id]
    assert listing["ledger"][0]["counts"]["performed"] == 1


def test_review_over_http_refusals(client):
    test_client, service = client
    _staged_board(service)
    proposed = service.declare_window(
        by=SUP, starts_at="2026-10-17T08:00:00Z", length_min=180,
        line_ids=["line-b"], machine_ids=["CNC-3", "WELD-4", "PACK-6"],
    )
    assert test_client.get(
        f"/api/windows/{proposed.window_id}/review").status_code == 409
    running = service.confirm_window(proposed.window_id, by=SUP)
    # The synthetic Saturday is ahead of the wall clock: still open.
    response = test_client.get(f"/api/windows/{running.window_id}/review")
    assert response.status_code == 409
    assert "cannot be reviewed yet" in response.json()["detail"]
    assert test_client.get("/api/windows/WIN-nope/review").status_code == 404
