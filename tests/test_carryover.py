"""Round 7: the next shift inherits the board — shift carry-over.

Before this round a board run ended and its unfinished business
existed only as records plus handoff prose; the next board was
assembled fresh, by hand. Now the run record carries a computed
carry-over manifest, a successor run settles it explicitly
(blocked cases re-worked with their loop state intact, gate
cases inherited standing, unworked cases joined fresh), the
claims turn over with the board (renewed for what carries,
released_run_ended for what does not), and a supervisor can drop
a carried case at acceptance — audited, with its claim freed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from helpers import CountingReasoner, make_service

from plant_loop import api
from plant_loop.loops import BoardPolicy
from plant_loop.schemas import ExceptionCase, SupervisorOutcome


def _belt(case_id: str, line_id: str, downtime: int) -> ExceptionCase:
    return ExceptionCase(
        case_id=case_id,
        type="machine_down",
        title=f"{case_id} belt tracking fault",
        description="Belt drifting to one side, edge fraying past the wear mark.",
        machine_id="CONV-5",
        line_id=line_id,
        product="Assembly kit AK-12",
        batch="B-4480",
        readings={"belt_drift_mm": 9},
        reporter="floor-lead",
        reported_at="2026-10-11T06:00:00Z",
        downtime_minutes=downtime,
    )


def _coolant(case_id: str, downtime: int = 30) -> ExceptionCase:
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
        downtime_minutes=downtime,
    )


def _quality(case_id: str) -> ExceptionCase:
    return ExceptionCase(
        case_id=case_id,
        type="quality_nonconformance",
        title="Porosity above limit on bracket welds",
        description="QC inspection found porosity in 6.2 percent of welds.",
        machine_id="WELD-4",
        line_id="line-c",
        product="Bracket BR-118",
        batch="B-4490",
        readings={"qc_porosity_pct": 6.2, "gas_flow_l_min": 9.5},
        reporter="S. Patel (quality)",
        reported_at="2026-10-11T05:40:00Z",
        downtime_minutes=0,
    )


def _safety(case_id: str) -> ExceptionCase:
    return ExceptionCase(
        case_id=case_id,
        type="safety_observation",
        title="Weld cell light curtain lens cracked",
        description="Operator found the light curtain lens cracked at start of shift.",
        machine_id="WELD-4",
        line_id="line-c",
        reporter="M. Reyes (operator)",
        reported_at="2026-10-11T05:58:00Z",
        downtime_minutes=0,
    )


def _contention_board(service, *, max_cases: int | None = None):
    """D wins the belt and converges · C converges on a substitute ·
    B blocks on D's claim · A is cut by the case budget."""
    policy = BoardPolicy(max_cases=max_cases) if max_cases else None
    return service.run_board(
        [
            _quality("CO-A"),
            _belt("CO-B", "line-d", 10),
            _coolant("CO-C"),
            _belt("CO-D", "line-a", 40),
        ],
        board_policy=policy,
    )


# ---------------------------------------------------------------------------
# The manifest: computed in code when the run ends
# ---------------------------------------------------------------------------


def test_the_manifest_names_every_kind_of_unfinished_business():
    service = make_service()
    result = _contention_board(service, max_cases=3)
    assert result.processed == ["CO-D", "CO-C", "CO-B"]
    assert [item["case_id"] for item in result.remaining] == ["CO-A"]

    manifest = {e["case_id"]: e for e in result.carry_over}
    assert manifest["CO-D"]["kind"] == "at_gate"
    assert manifest["CO-D"]["claim_ids"] == ["claim-CO-D-BELT-5M"]
    assert "work order" in manifest["CO-D"]["waiting_on"]
    assert manifest["CO-C"]["kind"] == "at_gate"
    assert manifest["CO-B"]["kind"] == "blocked"
    assert manifest["CO-B"]["rounds_used"] == 1
    assert "BELT-5M" in manifest["CO-B"]["waiting_on"]
    assert manifest["CO-A"]["kind"] == "not_worked"
    assert manifest["CO-A"]["has_payload"] is True

    # The stored run record is the contract: same manifest, and
    # every queued case's intake payload rides along.
    stored = service.get_board_run(result.run_id)
    assert stored["carry_over"] == result.carry_over
    assert set(stored["case_payloads"]) == {"CO-A", "CO-B", "CO-C", "CO-D"}
    assert "Carry-over to the next shift: 4 case(s)" in result.handoff_summary


# ---------------------------------------------------------------------------
# Acceptance: re-work with history, inherit at gate, join fresh
# ---------------------------------------------------------------------------


def test_acceptance_reworks_the_blocked_case_with_its_history_intact():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    # Free the belt the honest way: D is approved and its work is
    # recorded done before the turnover.
    service.record_decision("CO-D", "approved", by="Sam Supervisor")
    service.record_outcome(
        "CO-D",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="belt_tracking_fault",
            downtime_minutes_actual=55,
            parts_used=["BELT-5M"],
            resolved=True,
        ),
    )

    run2 = service.run_board([], carry_over_from=run1.run_id)
    settled = run2.carry_over_settled
    assert run2.carried_from == run1.run_id
    assert settled["reworked"] == ["CO-B"]
    assert settled["inherited_at_gate"] == ["CO-C"]
    assert settled["fresh"] == ["CO-A"]
    assert run2.processed == ["CO-B", "CO-A"]
    assert run2.exits == {"CO-B": "converged", "CO-A": "converged"}

    # History intact: the re-worked record's rounds are the case's
    # own — round 1 from the first board, round 2 from this one —
    # and the trace says the loop continued rather than restarted.
    record = service.store.get_case("CO-B")
    assert [r.round for r in record.rounds] == [1, 2]
    prepare = next(s for s in record.trace if s.name == "prepare")
    assert any("carry-over resume" in d for d in prepare.details)

    # The inherited gate case was not re-worked: its record is the
    # run-1 record, untouched, still waiting.
    inherited = service.store.get_case("CO-C")
    assert inherited.status == "awaiting_supervisor"
    assert len(inherited.rounds) == 1


def test_a_blocked_case_that_stays_blocked_keeps_its_round_count():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    # Nobody frees the belt: D still waits, claim standing.
    run2 = service.run_board([], carry_over_from=run1.run_id)
    record = service.store.get_case("CO-B")
    assert record.exit_reason == "blocked_external"
    assert [r.round for r in record.rounds] == [1, 2]  # resumed, then blocked again
    # And it is offered again on run 2's own manifest.
    kinds = {e["case_id"]: e["kind"] for e in run2.carry_over}
    assert kinds["CO-B"] == "blocked"


def test_no_carry_over_reference_starts_clean():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    claims_before = [
        (c.claim_id, c.status, c.run_id) for c in service.allocation.all_claims()
    ]
    run2 = service.run_board([_coolant("CO-NEW")])
    assert run2.carried_from == ""
    assert run2.carry_over_settled == {}
    assert run2.processed == ["CO-NEW"]
    # The earlier board's claims are exactly as it left them.
    claims_after = [
        (c.claim_id, c.status, c.run_id) for c in service.allocation.all_claims()
    ]
    assert claims_after[: len(claims_before)] == claims_before


def test_unknown_predecessor_and_stray_drops_are_refused():
    service = make_service()
    with pytest.raises(KeyError):
        service.run_board([], carry_over_from="board-nope")
    run1 = _contention_board(service, max_cases=3)
    with pytest.raises(ValueError):
        service.run_board(
            [],
            carry_over_from=run1.run_id,
            drop=[{"case_id": "CO-GHOST", "by": "Sam", "reason": "not on the board"}],
        )


# ---------------------------------------------------------------------------
# Claims at the turnover
# ---------------------------------------------------------------------------


def test_claims_turn_over_renewed_for_carried_released_for_done():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    service.record_decision("CO-D", "approved", by="Sam Supervisor")
    service.record_outcome(
        "CO-D",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="belt_tracking_fault",
            downtime_minutes_actual=55,
            parts_used=["BELT-5M"],
            resolved=True,
        ),
    )
    run2 = service.run_board([], carry_over_from=run1.run_id)

    claims = {c.claim_id: c for c in service.allocation.all_claims()}
    renewed = claims["claim-CO-C-CLNT-FLT-E"]
    assert renewed.status == "claimed"
    assert renewed.run_id == run2.run_id
    assert renewed.carried_from_run == run1.run_id
    assert run2.carry_over_settled["renewed_claims"] == [renewed.claim_id]

    done = claims["claim-CO-D-BELT-5M"]
    assert done.status == "released"
    assert done.release_reason == "released_run_ended"
    assert run2.carry_over_settled["released_claims"] == [done.claim_id]

    # The freed belt is what let the blocked case converge — and
    # its new claim is a fresh claim of the accepting run.
    assert claims["claim-CO-B-BELT-5M"].run_id == run2.run_id
    assert claims["claim-CO-B-BELT-5M"].carried_from_run == ""


def test_an_approved_cases_claim_survives_until_its_outcome_lands():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    service.record_decision("CO-D", "approved", by="Sam Supervisor")
    # Approved, work not yet recorded: the work order will draw
    # the belt, so the turnover must not free it.
    run2 = service.run_board([], carry_over_from=run1.run_id)
    claim = service.allocation.claims_for_case("CO-D")[0]
    assert claim.status == "claimed"
    assert service.allocation.effective_qty("BELT-5M") == 0
    # The blocked case therefore blocks again on the renewed board.
    assert run2.exits["CO-B"] == "blocked_external"

    # Once the outcome lands, the case is done — the NEXT turnover
    # releases the claim and the blocked case finally converges.
    service.record_outcome(
        "CO-D",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="belt_tracking_fault",
            downtime_minutes_actual=55,
            parts_used=["BELT-5M"],
            resolved=True,
        ),
    )
    run3 = service.run_board([], carry_over_from=run2.run_id)
    claim = service.allocation.claims_for_case("CO-D")[0]
    assert claim.status == "released"
    assert claim.release_reason == "released_run_ended"
    assert run3.exits["CO-B"] == "converged"


def test_a_renewed_claim_still_releases_on_gate_rejection():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    service.run_board([], carry_over_from=run1.run_id)
    service.record_decision("CO-C", "rejected", by="Sam Supervisor", reason="no crew")
    claim = service.allocation.claims_for_case("CO-C")[0]
    assert claim.status == "released"
    assert claim.release_reason == "released_by_gate_rejection"
    assert claim.released_by == "Sam Supervisor"


def test_the_claim_note_names_a_carried_claim_as_carried():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    service.run_board([], carry_over_from=run1.run_id)
    # CO-C's substitute claim was renewed into run 2. A new coolant
    # case in run 3 measures the shelf net of it; force the issue
    # by claiming the rest of the shelf first.
    record = service.run_case(_coolant("CO-LATE")).result
    assert record.exit_reason == "converged"  # 4 on the shelf, 1 claimed — room yet
    from plant_loop.allocation import RELEASE_RUN_ENDED  # noqa: F401  (vocabulary pin)

    assert service.allocation.note_for("CLNT-FLT-E", 4) == (
        "claimed by CO-C (carried over from board run " + run1.run_id + ")"
    )


# ---------------------------------------------------------------------------
# Drops at acceptance
# ---------------------------------------------------------------------------


def test_dropping_a_carried_case_frees_its_part_and_stamps_the_record():
    service = make_service()
    run1 = service.run_board(
        [_belt("CO-L", "line-d", 10), _belt("CO-W", "line-a", 40)]
    )
    assert run1.exits == {"CO-W": "converged", "CO-L": "blocked_external"}

    run2 = service.run_board(
        [],
        carry_over_from=run1.run_id,
        drop=[
            {
                "case_id": "CO-W",
                "by": "Sam Supervisor",
                "reason": "handled on paper by the day crew",
            }
        ],
    )
    assert run2.carry_over_settled["dropped"] == ["CO-W"]

    record = service.store.get_case("CO-W")
    assert record.status == "dropped"
    assert record.dropped_from_board["by"] == "Sam Supervisor"
    assert record.dropped_from_board["from_run"] == run1.run_id
    # The case has left the board: it no longer waits at the gate.
    assert "CO-W" not in {row["case_id"] for row in service.board()}

    claim = service.allocation.claims_for_case("CO-W")[0]
    assert claim.status == "released"
    assert claim.release_reason == "released_run_ended"
    assert claim.released_by == "Sam Supervisor"

    # And the case it was blocking takes the freed part.
    assert run2.exits == {"CO-L": "converged"}
    assert service.allocation.claims_for_case("CO-L")[0].status == "claimed"


def test_a_fresh_submission_supersedes_the_carried_entry():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    run2 = service.run_board([_coolant("CO-C")], carry_over_from=run1.run_id)
    assert run2.carry_over_settled["superseded"] == ["CO-C"]
    assert set(run2.processed) == {"CO-A", "CO-B", "CO-C"}
    # The old claim was swept at the turnover; the fresh run of
    # the case earned the part again, and the ledger row was
    # revived as a fresh claim of the new run (one row per
    # case + part — a released claim never blocks a re-claim).
    claims = service.allocation.claims_for_case("CO-C")
    assert len(claims) == 1
    assert claims[0].status == "claimed"
    assert claims[0].run_id == run2.run_id
    assert claims[0].carried_from_run == ""
    assert claims[0].release_reason == ""


# ---------------------------------------------------------------------------
# Safety: carried as a record, never re-worked
# ---------------------------------------------------------------------------


def test_a_safety_case_carries_as_a_gate_record_and_is_never_reworked():
    reasoner = CountingReasoner()
    service = make_service(reasoner=reasoner)
    run1 = service.run_board([_safety("CO-SAFE")])
    assert run1.exits["CO-SAFE"] == "safety_stop"
    kinds = {e["case_id"]: e["kind"] for e in run1.carry_over}
    assert kinds["CO-SAFE"] == "at_gate"
    waiting = {e["case_id"]: e["waiting_on"] for e in run1.carry_over}
    assert "SAFETY" in waiting["CO-SAFE"]

    run2 = service.run_board([], carry_over_from=run1.run_id)
    assert run2.carry_over_settled["inherited_at_gate"] == ["CO-SAFE"]
    assert "CO-SAFE" not in run2.processed  # inherited standing, not re-worked
    # No model call was spent on it on either board: it escalated
    # at round 0 on the first, and carry-over never re-enters it.
    assert reasoner.calls == {"diagnose": 0, "plan": 0, "critique": 0}
    assert service.store.get_case("CO-SAFE").status == "awaiting_supervisor"


def test_an_unworked_safety_case_carries_as_a_fresh_intake_and_stops_at_round_0():
    reasoner = CountingReasoner()
    service = make_service(reasoner=reasoner)
    run1 = service.run_board([_safety("CO-SAFE")], board_policy=BoardPolicy(max_cases=0))
    assert run1.processed == []
    assert {e["case_id"]: e["kind"] for e in run1.carry_over} == {
        "CO-SAFE": "not_worked"
    }
    run2 = service.run_board([], carry_over_from=run1.run_id)
    assert run2.carry_over_settled["fresh"] == ["CO-SAFE"]
    assert run2.exits == {"CO-SAFE": "safety_stop"}
    assert reasoner.calls == {"diagnose": 0, "plan": 0, "critique": 0}


# ---------------------------------------------------------------------------
# Records, not recounts; the handoff; inheritance chains
# ---------------------------------------------------------------------------


def test_a_reworked_case_counts_once_in_the_stats():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    service.record_decision("CO-D", "approved", by="Sam Supervisor")
    service.record_outcome(
        "CO-D",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="belt_tracking_fault",
            downtime_minutes_actual=55,
            parts_used=["BELT-5M"],
            resolved=True,
        ),
    )
    service.run_board([], carry_over_from=run1.run_id)
    service.record_decision("CO-B", "approved", by="Sam Supervisor")
    service.record_outcome(
        "CO-B",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="belt_tracking_fault",
            downtime_minutes_actual=40,
            parts_used=["BELT-5M"],
            resolved=True,
        ),
    )
    stats = service.diagnosis_stats()
    # Four cases exist; CO-B was worked on two boards and is
    # still ONE case in the plant's numbers.
    assert stats["cases_total"] == 4
    assert stats["outcomes_recorded"] == 2
    assert stats["economics"]["n"] == 2


def test_inheritance_chains_across_runs():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    run2 = service.run_board([], carry_over_from=run1.run_id)
    # CO-C was inherited standing by run 2 — and run 2's own
    # manifest re-offers it, so a third run still inherits it.
    assert "CO-C" in {e["case_id"] for e in run2.carry_over}
    run3 = service.run_board([], carry_over_from=run2.run_id)
    assert "CO-C" in run3.carry_over_settled["inherited_at_gate"]
    claim = service.allocation.claims_for_case("CO-C")[0]
    assert claim.status == "claimed"
    assert claim.run_id == run3.run_id
    assert claim.carried_from_run == run2.run_id


def test_the_handoff_carries_the_manifest_and_the_turnover():
    service = make_service()
    run1 = _contention_board(service, max_cases=3)
    handoff = service.handoff()
    carry = handoff["carry_over"]
    assert carry["run_id"] == run1.run_id
    assert {e["case_id"] for e in carry["entries"]} == {
        "CO-A",
        "CO-B",
        "CO-C",
        "CO-D",
    }
    assert "CARRY-OVER — THE NEXT SHIFT INHERITS THIS BOARD:" in handoff["text"]
    assert "CO-B · blocked · waiting on:" in handoff["text"]

    service.run_board([], carry_over_from=run1.run_id)
    handoff = service.handoff()
    settled = handoff["carry_over"]["settled"]
    # Both gate cases were still waiting, so both claims renewed.
    assert sorted(settled["renewed_claims"]) == [
        "claim-CO-C-CLNT-FLT-E",
        "claim-CO-D-BELT-5M",
    ]
    assert "2 claim(s) renewed for carry-over" in handoff["text"]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def test_carry_over_over_http(client):
    http, service = client
    body = {
        "cases": [
            _belt("CO-L", "line-d", 10).model_dump(),
            _belt("CO-W", "line-a", 40).model_dump(),
        ]
    }
    run1 = http.post("/board/run", json=body).json()
    assert run1["carry_over"]  # the offer rides the run response
    manifest = http.get(f"/board/runs/{run1['run_id']}/carry-over")
    assert manifest.status_code == 200
    kinds = {e["case_id"]: e["kind"] for e in manifest.json()["carry_over"]}
    assert kinds == {"CO-W": "at_gate", "CO-L": "blocked"}

    run2 = http.post(
        "/board/run",
        json={
            "cases": [],
            "carry_over_from": run1["run_id"],
            "drop": [
                {"case_id": "CO-W", "by": "Sam Supervisor", "reason": "paper"}
            ],
        },
    )
    assert run2.status_code == 201
    payload = run2.json()
    assert payload["carried_from"] == run1["run_id"]
    assert payload["carry_over_settled"]["dropped"] == ["CO-W"]
    assert payload["exits"] == {"CO-L": "converged"}

    missing = http.post(
        "/board/run", json={"cases": [], "carry_over_from": "board-nope"}
    )
    assert missing.status_code == 404
    assert http.get("/board/runs/board-nope/carry-over").status_code == 404
