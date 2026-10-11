"""Round 5: the board shares one shelf — board-level parts allocation.

Before this round each case's verify stage measured the shelf as
if it were alone on it: two cases could plan around the same last
unit, both converge, and the floor found out at the workbench.
Now a converged case claims the parts its plan will draw, later
cases measure the shelf net of the claims standing, a case that
loses a part to a claim blocks with the claiming case named, and a
gate rejection releases the claim for future runs — never
rewriting the finished run it came from.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from helpers import CountingReasoner, make_service

from plant_loop import api, substitution
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import ExceptionCase, PartClaim


def _belt_case(case_id: str, line_id: str, downtime: int) -> ExceptionCase:
    """A CONV-5 belt case: diagnoses belt_tracking_fault, whose SOP
    requires BELT-5M — of which the shelf holds exactly one."""
    return ExceptionCase(
        case_id=case_id,
        type="machine_down",
        line_id=line_id,
        machine_id="CONV-5",
        title=f"{case_id} belt tracking fault",
        reported_by="floor-lead",
        reported_at="2026-10-11T06:00:00Z",
        downtime_min=downtime,
        description="Belt drifting to one side, edge fraying past the wear mark.",
        readings={"belt_drift_mm": 9.0},
    )


def _sample(case_id: str) -> ExceptionCase:
    return next(c for c in load_sample_cases() if c.case_id == case_id)


def _coolant_case(case_id: str) -> ExceptionCase:
    """A CNC-3 coolant case: the shelf holds no CLNT-FLT, so its
    plan converges on the CLNT-FLT-E substitute (the G10 shape)."""
    return ExceptionCase(
        case_id=case_id,
        type="machine_down",
        line_id="line-b",
        machine_id="CNC-3",
        title=f"{case_id} coolant flow low",
        reported_by="operator",
        reported_at="2026-10-11T06:40:00Z",
        downtime_min=30,
        description="Coolant flow alarm; spindle temperature climbing.",
        readings={"coolant_flow_l_min": 3.0, "spindle_temp_c": 84},
    )


# ---------------------------------------------------------------------------
# Contention: one belt, two cases, one board run
# ---------------------------------------------------------------------------


def test_two_cases_one_last_unit_the_loser_names_the_claimer():
    service = make_service()
    hi = _belt_case("ALC-HI", "line-a", 40)  # line criticality 3 -> works first
    lo = _belt_case("ALC-LO", "line-d", 10)  # line criticality 2 -> works second
    result = service.run_board([lo, hi])  # reversed input; board order rules

    assert result.processed == ["ALC-HI", "ALC-LO"]
    assert result.exits["ALC-HI"] == "converged"
    assert result.exits["ALC-LO"] == "blocked_external"

    # The run record carries the ledger: one claim, made by the
    # higher-priority case, still standing.
    assert [(c.part_id, c.case_id, c.status) for c in result.allocations] == [
        ("BELT-5M", "ALC-HI", "claimed")
    ]
    stored = service.get_board_run(result.run_id)
    assert stored["allocations"][0]["claim_id"] == "claim-ALC-HI-BELT-5M"

    # The loser's gap names the claim — never a bare stockout.
    record = service.store.get_case("ALC-LO")
    gap = next(g for g in record.open_gaps if g.kind == "part_unavailable")
    assert "claimed by ALC-HI, higher priority on this board" in gap.detail


def test_gate_rejection_releases_the_claim_for_a_fresh_run_only():
    service = make_service()
    service.run_board(
        [_belt_case("ALC-HI", "line-a", 40), _belt_case("ALC-LO", "line-d", 10)]
    )
    assert service.store.get_case("ALC-LO").exit_reason == "blocked_external"

    service.record_decision(
        "ALC-HI", "rejected", by="Sam Supervisor", reason="crew reassigned"
    )
    claim = service.allocation.claims_for_case("ALC-HI")[0]
    assert claim.status == "released"
    assert claim.release_reason == "released_by_gate_rejection"
    assert claim.released_by == "Sam Supervisor"
    assert claim.released_at  # stamped with when

    # The finished run is a record: the blocked case stays blocked
    # even though its blocker is gone. A fresh run re-reads the
    # shelf — and the lower case now converges on the freed part.
    assert service.store.get_case("ALC-LO").exit_reason == "blocked_external"
    fresh = service.run_board([_belt_case("ALC-LO2", "line-d", 10)])
    assert fresh.exits["ALC-LO2"] == "converged"
    assert [(c.part_id, c.case_id) for c in fresh.allocations] == [
        ("BELT-5M", "ALC-LO2")
    ]


def test_gate_approval_keeps_the_claim_standing():
    service = make_service()
    service.run_board([_belt_case("ALC-HI", "line-a", 40)])
    service.record_decision("ALC-HI", "approved", by="Sam Supervisor")
    claim = service.allocation.claims_for_case("ALC-HI")[0]
    assert claim.status == "claimed"  # the work order will draw the part
    assert service.allocation.effective_qty("BELT-5M") == 0


def test_a_claim_from_an_earlier_run_is_named_as_such():
    service = make_service()
    service.run_board([_belt_case("ALC-HI", "line-a", 40)])
    service.record_decision("ALC-HI", "approved", by="Sam Supervisor")
    # A later case outside that run still cannot draw the belt —
    # and its gap names the earlier run's claim as an earlier claim,
    # not as this board's priority decision.
    record = service.run_case(_belt_case("ALC-LATE", "line-d", 10)).result
    assert record.exit_reason == "blocked_external"
    gap = next(g for g in record.open_gaps if g.kind == "part_unavailable")
    assert "claimed by ALC-HI (board run board-" in gap.detail
    assert "higher priority on this board" not in gap.detail


# ---------------------------------------------------------------------------
# No contention: claims are recorded, nothing else changes
# ---------------------------------------------------------------------------


def test_single_case_run_claims_and_behaves_as_before():
    service = make_service()
    record = service.run_case(_sample("EXC-101")).result
    assert record.exit_reason == "converged"
    assert len(record.rounds) == 2  # exactly as the golden pack pins this case
    claims = service.allocation.all_claims()
    assert [(c.part_id, c.case_id, c.qty, c.run_id, c.status) for c in claims] == [
        ("BRG-6208", "EXC-101", 1, "single", "claimed")
    ]


def test_a_substituted_plan_claims_the_substitute_it_draws():
    service = make_service()
    record = service.run_case(_coolant_case("ALC-COOL")).result
    assert record.exit_reason == "converged"
    claims = service.allocation.all_claims()
    assert len(claims) == 1
    claim = claims[0]
    assert claim.part_id == "CLNT-FLT-E"  # the part actually drawn
    assert claim.original_part_id == "CLNT-FLT"  # the requirement it satisfies


# ---------------------------------------------------------------------------
# The substitution search reads effective stock too
# ---------------------------------------------------------------------------


def test_substitute_search_sees_claims_and_names_the_claimer():
    service = make_service()
    # Every CLNT-FLT-E on the shelf is already claimed elsewhere.
    service.store.save_part_claim(
        PartClaim(
            claim_id="claim-OTHER-CLNT-FLT-E",
            run_id="board-earlier",
            case_id="OTHER",
            part_id="CLNT-FLT-E",
            qty=4,
            status="claimed",
            claimed_at="2026-10-11T05:00:00Z",
        )
    )
    from plant_loop import kb

    sop = kb.sop_for("CNC-3", "coolant_system_fault")
    winner, records = substitution.evaluate(
        sop,
        "CLNT-FLT",
        1,
        qty_of=service.allocation.effective_qty,
        note_of=lambda p, n: service.allocation.note_for(p, n),
    )
    assert winner is None  # the shelf-alone search would have found it
    assert "claimed by OTHER (board run board-earlier)" in records[0].reason

    # And the case that needs it blocks, with the claim named in
    # its substitution paperwork as well as its gap.
    record = service.run_case(_coolant_case("ALC-COOL")).result
    assert record.exit_reason == "blocked_external"
    assert any(
        "claimed by OTHER" in s.reason
        for s in record.plan.substitutions
        if s.status == "rejected"
    )


# ---------------------------------------------------------------------------
# Safety: claims never touch the safety path
# ---------------------------------------------------------------------------


def test_safety_cases_never_claim_and_are_never_delayed():
    reasoner = CountingReasoner()
    service = make_service(reasoner=reasoner)
    record = service.run_case(_sample("EXC-102")).result  # safety_observation
    assert record.exit_reason == "safety_stop"
    assert reasoner.calls == {"diagnose": 0, "plan": 0, "critique": 0}
    assert service.allocation.all_claims() == []


def test_a_contended_board_still_escalates_safety_first():
    service = make_service()
    result = service.run_board(
        [
            _belt_case("ALC-LO", "line-d", 10),
            _sample("EXC-102"),
            _belt_case("ALC-HI", "line-a", 40),
        ]
    )
    assert result.exits["EXC-102"] == "safety_stop"
    assert service.allocation.claims_for_case("EXC-102") == []
    # The belt contention resolves exactly as it does without the
    # safety case on the board.
    assert result.exits["ALC-HI"] == "converged"
    assert result.exits["ALC-LO"] == "blocked_external"


# ---------------------------------------------------------------------------
# Surfaces: packet + handoff carry the allocation
# ---------------------------------------------------------------------------


def test_packet_marks_its_parts_claimed_for_this_case():
    service = make_service()
    result = service.run_board([_belt_case("ALC-HI", "line-a", 40)])
    packet = service.case_packet("ALC-HI")
    part = packet["parts"][0]
    assert part["claimed"] is True
    assert part["claimed_on_run"] == result.run_id
    assert f"claimed for this case on board run {result.run_id}" in packet["text"]


def test_handoff_names_the_case_blocked_by_an_allocation():
    service = make_service()
    service.run_board(
        [_belt_case("ALC-HI", "line-a", 40), _belt_case("ALC-LO", "line-d", 10)]
    )
    handoff = service.handoff()
    blocked = {e["case_id"]: e for e in handoff["blocked_external"]}
    assert blocked["ALC-LO"]["allocation_blocked"] == [
        {"part_id": "BELT-5M", "claimed_by": "ALC-HI"}
    ]
    assert "blocked by allocation: BELT-5M claimed by ALC-HI" in handoff["text"]


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app)
    finally:
        api.set_service(None)


def test_allocations_endpoint_and_board_run_payload(client):
    body = {
        "cases": [
            _belt_case("ALC-LO", "line-d", 10).model_dump(),
            _belt_case("ALC-HI", "line-a", 40).model_dump(),
        ]
    }
    run = client.post("/board/run", json=body)
    assert run.status_code == 201
    payload = run.json()
    assert [(c["part_id"], c["case_id"], c["status"]) for c in payload["allocations"]] == [
        ("BELT-5M", "ALC-HI", "claimed")
    ]
    # The stored run record carries the same ledger.
    stored = client.get(f"/board/runs/{payload['run_id']}").json()
    assert stored["allocations"][0]["claim_id"] == "claim-ALC-HI-BELT-5M"
    # And the standing ledger endpoint shows the claim — then the
    # release, once the gate rejects the case.
    ledger = client.get("/api/allocations").json()
    assert [(c["part_id"], c["status"]) for c in ledger] == [("BELT-5M", "claimed")]
    client.post(
        "/cases/ALC-HI/decision", json={"decision": "rejected", "by": "Sam Supervisor"}
    )
    ledger = client.get("/api/allocations").json()
    assert ledger[0]["status"] == "released"
    assert ledger[0]["release_reason"] == "released_by_gate_rejection"
