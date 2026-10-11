"""Round 11: the board looks ahead — projected parts demand.

Until now every parts measurement was present-tense: the verify
stage blocks a case when its part is not there *now*, and the
allocation ledger nets the claims standing *now*. The outlook
projects what the record already knows about the future —
standing claims, open cases' unsatisfied plan requirements, and
PM cycles falling due inside a stated horizon — against the
plant model's authored shelf, and names the shortage rows with
the shortfall and the moment it bites.

The fixtures here are fabricated records with hand-computed
expectations against the seeded shelf (data/kb.json: BRG-6208 2,
HYD-SEAL 3, LINER-045 2, BELT-5M 1, CLNT-FLT 0, CLNT-FLT-E 4,
AIR-FLT 5): the tests check the projection's arithmetic and its
discipline (one count per unit, retired obligations silent,
unquantified stated never invented, a pure read), not a snapshot
of its own output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from helpers import make_service

from plant_loop import api
from plant_loop.handoff import build_handoff, render_text
from plant_loop.outlook import build_outlook
from plant_loop.schemas import (
    ActionPlan,
    CaseResult,
    Diagnosis,
    Hypothesis,
    PartClaim,
    PartRequirement,
    PlanStep,
    PmObligation,
)
from plant_loop.store import InMemoryStore

AS_OF = datetime(2026, 10, 11, 0, 0, tzinfo=timezone.utc)
AS_OF_STAMP = "2026-10-11T00:00:00Z"


# ---------------------------------------------------------------------------
# Fixtures (fabricated records, the test_pm / test_retirement way)
# ---------------------------------------------------------------------------


def _record(
    case_id: str,
    *,
    status: str = "awaiting_supervisor",
    exit_reason: str = "blocked_external",
    machine_id: str = "CNC-3",
    fault: str = "coolant_system_fault",
    requirements: tuple = (),
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=status,
        exit_reason=exit_reason,
        type="machine_down",
        title=f"fabricated case {case_id}",
        machine_id=machine_id,
        line_id="line-b",
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(fault_class=fault, statement="fabricated", confidence=0.9)
            ]
        ),
        plan=ActionPlan(
            steps=[PlanStep(step_id="s1", text="Do the work.")],
            required_parts=[PartRequirement(**r) for r in requirements],
        ),
        created_at="2026-10-10T08:00:00Z",
        reported_at="2026-10-10T07:55:00Z",
    )


def _claim(
    case_id: str,
    part_id: str,
    qty: int,
    *,
    status: str = "claimed",
    original: str = "",
) -> PartClaim:
    return PartClaim(
        claim_id=f"claim-{case_id}-{part_id}",
        run_id="run-1",
        case_id=case_id,
        part_id=part_id,
        original_part_id=original,
        qty=qty,
        status=status,
        claimed_at="2026-10-10T09:00:00Z",
        released_at="2026-10-10T10:00:00Z" if status == "released" else "",
        release_reason="released_by_gate_rejection" if status == "released" else "",
    )


def _obligation(
    obligation_id: str,
    machine_id: str,
    fault_class: str,
    next_due_at: str,
    *,
    status: str = "standing",
    cycle_case_id: str = "",
) -> PmObligation:
    return PmObligation(
        obligation_id=obligation_id,
        machine_id=machine_id,
        fault_class=fault_class,
        line_id="line-b",
        interval_days=14,
        next_due_at=next_due_at,
        status=status,
        cycle_case_id=cycle_case_id,
    )


def _part(outlook, part_id: str):
    return next((p for p in outlook.parts if p.part_id == part_id), None)


# ---------------------------------------------------------------------------
# The arithmetic: three sources, traced
# ---------------------------------------------------------------------------


def test_all_three_sources_counted_and_traced():
    store = InMemoryStore()
    # HYD-SEAL, shelf 3: a standing claim of 2, an open blocked
    # case still needing 1, and a PM cycle due inside the horizon
    # whose SOP (SOP-PRESS1-HYD) names 1 more.
    store.save_part_claim(_claim("C-CLM", "HYD-SEAL", 2))
    store.save_case(
        _record(
            "C-BLK",
            machine_id="PRESS-1",
            fault="hydraulic_pressure_loss",
            requirements=({"part_id": "HYD-SEAL", "qty": 1, "satisfied": False},),
        )
    )
    store.save_pm_record(
        _obligation("PMO-HYD", "PRESS-1", "hydraulic_pressure_loss", "2026-10-20T00:00:00Z")
    )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "HYD-SEAL")
    assert part is not None
    assert (part.on_hand, part.claimed_qty, part.case_demand_qty, part.pm_demand_qty) == (3, 2, 1, 1)
    assert part.projected_demand == 4
    assert part.shortfall == 1 and part.shortage_risk
    # On the books: 2 claimed + 1 planned = 3 = the shelf — not
    # short yet. The PM cycle due 2026-10-20 is what tips it.
    assert part.bites_at == "2026-10-20T00:00:00Z"
    assert "PMO-HYD" in part.bites_note
    assert [p.part_id for p in outlook.shortages] == ["HYD-SEAL"]
    lines = {(l.source_kind, l.source_id) for l in part.lines}
    assert lines == {
        ("claim", "C-CLM"),
        ("case_plan", "C-BLK"),
        ("pm_cycle", "PMO-HYD"),
    }
    assert outlook.as_of == AS_OF_STAMP and outlook.horizon_days == 14
    assert outlook.horizon_end == "2026-10-25T00:00:00Z"


def test_exactly_zero_shortfall_is_covered_not_risk():
    store = InMemoryStore()
    # LINER-045, shelf 2: claim 1 + open plan 1 = exactly the shelf.
    store.save_part_claim(_claim("C-1", "LINER-045", 1))
    store.save_case(
        _record(
            "C-2",
            machine_id="WELD-4",
            fault="wire_feed_fault",
            requirements=({"part_id": "LINER-045", "qty": 1, "satisfied": False},),
        )
    )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "LINER-045")
    assert part.projected_demand == 2 and part.on_hand == 2
    assert part.shortfall == 0 and not part.shortage_risk
    assert part.bites_at == "" and outlook.shortages == []


def test_already_short_bites_at_the_projection_moment():
    store = InMemoryStore()
    # BELT-5M, shelf 1: two open cases each waiting on one.
    for case_id in ("C-1", "C-2"):
        store.save_case(
            _record(
                case_id,
                machine_id="CONV-5",
                fault="belt_tracking_fault",
                requirements=({"part_id": "BELT-5M", "qty": 1, "satisfied": False},),
            )
        )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "BELT-5M")
    assert part.projected_demand == 2 and part.shortfall == 1
    assert part.bites_at == AS_OF_STAMP
    assert "already short" in part.bites_note


def test_overdue_cycle_counts_and_bites_now():
    store = InMemoryStore()
    # LINER-045, shelf 2, fully claimed; a wire-feed PM cycle
    # already overdue (due 2026-10-05) adds its SOP's 1 — the
    # shortfall exists today, so it bites at the projection
    # moment, not at the date in the past.
    store.save_part_claim(_claim("C-1", "LINER-045", 2))
    store.save_pm_record(
        _obligation("PMO-W", "WELD-4", "wire_feed_fault", "2026-10-05T00:00:00Z")
    )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "LINER-045")
    assert part.pm_demand_qty == 1 and part.projected_demand == 3
    assert part.shortfall == 1 and part.bites_at == AS_OF_STAMP


# ---------------------------------------------------------------------------
# One count per unit
# ---------------------------------------------------------------------------


def test_claim_covered_requirement_is_not_counted_twice():
    store = InMemoryStore()
    # A converged case: its satisfied requirement IS its claim.
    store.save_case(
        _record(
            "C-CONV",
            exit_reason="converged",
            machine_id="PRESS-1",
            fault="hydraulic_pressure_loss",
            requirements=({"part_id": "HYD-SEAL", "qty": 1, "satisfied": True},),
        )
    )
    store.save_part_claim(_claim("C-CONV", "HYD-SEAL", 1))
    # A plan that verified its parts but never converged (rounds
    # exhausted on another gap): satisfied, unclaimed — its draw
    # lands on the part the plan runs on, the substitute.
    store.save_case(
        _record(
            "C-EXH",
            exit_reason="exhausted",
            requirements=(
                {
                    "part_id": "CLNT-FLT",
                    "qty": 1,
                    "satisfied": True,
                    "substitute_part_id": "CLNT-FLT-E",
                },
            ),
        )
    )
    outlook = build_outlook(store, now=AS_OF)
    hyd = _part(outlook, "HYD-SEAL")
    assert hyd.claimed_qty == 1 and hyd.case_demand_qty == 0
    sub = _part(outlook, "CLNT-FLT-E")
    assert sub.case_demand_qty == 1 and sub.claimed_qty == 0
    # The original part carries no line of its own: the plan runs
    # on the substitute, and the projection follows the plan.
    assert _part(outlook, "CLNT-FLT") is None


def test_open_cycle_obligation_counts_through_its_case_only():
    store = InMemoryStore()
    # The obligation's cycle is already on a board; counting the
    # obligation as well would spend the same unit twice.
    store.save_pm_record(
        _obligation(
            "PMO-C",
            "CNC-3",
            "coolant_system_fault",
            "2026-10-12T00:00:00Z",
            cycle_case_id="PMO-C-1",
        )
    )
    store.save_case(
        _record(
            "PMO-C-1",
            requirements=({"part_id": "CLNT-FLT", "qty": 1, "satisfied": False},),
        )
    )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "CLNT-FLT")
    assert part.case_demand_qty == 1 and part.pm_demand_qty == 0
    assert part.projected_demand == 1


def test_retired_obligation_contributes_nothing():
    store = InMemoryStore()
    # CLNT-FLT's shelf is empty: if the retired obligation's due
    # cycle counted, this would be a shortage row. Its duty
    # ended on the record, and the projection believes the record.
    store.save_pm_record(
        _obligation(
            "PMO-R",
            "CNC-3",
            "coolant_system_fault",
            "2026-10-13T00:00:00Z",
            status="retired",
        )
    )
    outlook = build_outlook(store, now=AS_OF)
    assert _part(outlook, "CLNT-FLT") is None
    assert outlook.shortages == [] and outlook.unquantified_pm == []


def test_released_claims_and_finished_cases_contribute_nothing():
    store = InMemoryStore()
    store.save_part_claim(_claim("C-REL", "HYD-SEAL", 2, status="released"))
    for case_id, status in (
        ("C-REJ", "rejected"),
        ("C-DRP", "dropped"),
        ("C-MOOT", "closed_moot"),
    ):
        store.save_case(
            _record(
                case_id,
                status=status,
                machine_id="PRESS-1",
                fault="hydraulic_pressure_loss",
                requirements=({"part_id": "HYD-SEAL", "qty": 1, "satisfied": False},),
            )
        )
    # An approved case still counts: its plan will draw.
    store.save_case(
        _record(
            "C-APR",
            status="approved",
            exit_reason="blocked_external",
            machine_id="PRESS-1",
            fault="hydraulic_pressure_loss",
            requirements=({"part_id": "HYD-SEAL", "qty": 1, "satisfied": False},),
        )
    )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "HYD-SEAL")
    assert part.claimed_qty == 0 and part.case_demand_qty == 1
    assert [l.source_id for l in part.lines] == ["C-APR"]


# ---------------------------------------------------------------------------
# The horizon, and what cannot be quantified
# ---------------------------------------------------------------------------


def test_horizon_window_includes_its_last_day_only():
    store = InMemoryStore()
    store.save_pm_record(
        _obligation("PMO-LATE", "PRESS-1", "hydraulic_pressure_loss", "2026-10-31T00:00:00Z")
    )
    store.save_pm_record(
        _obligation("PMO-EDGE", "WELD-4", "wire_feed_fault", "2026-10-25T00:00:00Z")
    )
    outlook = build_outlook(store, now=AS_OF)  # 14 days, to 2026-10-25
    assert _part(outlook, "HYD-SEAL") is None  # due day 20: outside
    edge = _part(outlook, "LINER-045")
    assert edge is not None and edge.pm_demand_qty == 1  # due on the last day: in
    wider = build_outlook(store, now=AS_OF, horizon_days=30)
    assert _part(wider, "HYD-SEAL").pm_demand_qty == 1
    with pytest.raises(ValueError):
        build_outlook(store, now=AS_OF, horizon_days=0)


def test_obligation_without_an_sop_is_unquantified_never_invented():
    store = InMemoryStore()
    # No SOP covers PACK-6 + coolant_system_fault: the cycle's
    # parts cannot be projected, and the outlook says so.
    store.save_pm_record(
        _obligation("PMO-U", "PACK-6", "coolant_system_fault", "2026-10-12T00:00:00Z")
    )
    outlook = build_outlook(store, now=AS_OF)
    assert outlook.parts == []
    assert len(outlook.unquantified_pm) == 1
    entry = outlook.unquantified_pm[0]
    assert entry["obligation_id"] == "PMO-U" and "no SOP" in entry["reason"]


def test_unstocked_part_with_demand_is_a_shortage():
    store = InMemoryStore()
    store.save_case(
        _record(
            "C-X",
            requirements=({"part_id": "GEN-BEARING", "qty": 1, "satisfied": False},),
        )
    )
    outlook = build_outlook(store, now=AS_OF)
    part = _part(outlook, "GEN-BEARING")
    assert part.stocked is False and part.on_hand is None
    assert part.shortfall == 1 and part.shortage_risk
    assert part.bites_at == AS_OF_STAMP


# ---------------------------------------------------------------------------
# The read model's purity, the handoff, the API
# ---------------------------------------------------------------------------


def _snapshot(store: InMemoryStore) -> str:
    return json.dumps(
        {
            "cases": [r.model_dump() for r in store.all_cases()],
            "claims": [c.model_dump() for c in store.all_part_claims()],
            "pm": [
                r.model_dump()
                for r in store.all_pm_records()
            ],
            "runs": store.all_board_runs(),
        },
        sort_keys=True,
        default=str,
    )


def test_projection_is_a_pure_read():
    store = InMemoryStore()
    store.save_part_claim(_claim("C-CLM", "HYD-SEAL", 2))
    store.save_case(
        _record(
            "C-BLK",
            machine_id="PRESS-1",
            fault="hydraulic_pressure_loss",
            requirements=({"part_id": "HYD-SEAL", "qty": 1, "satisfied": False},),
        )
    )
    store.save_pm_record(
        _obligation("PMO-HYD", "PRESS-1", "hydraulic_pressure_loss", "2026-10-20T00:00:00Z")
    )
    before = _snapshot(store)
    first = build_outlook(store, now=AS_OF)
    second = build_outlook(store, now=AS_OF)
    assert _snapshot(store) == before  # board + claims byte-identical
    assert first.model_dump() == second.model_dump()  # deterministic


def test_handoff_carries_the_coming_shortages():
    store = InMemoryStore()
    # BRG-6208, shelf 2, fully claimed; the bearing PM cycle due
    # 2026-10-15 tips it — the handoff must carry it.
    store.save_part_claim(_claim("C-1", "BRG-6208", 1))
    store.save_part_claim(_claim("C-2", "BRG-6208", 1))
    store.save_pm_record(
        _obligation("PMO-BRG", "CNC-3", "spindle_bearing_wear", "2026-10-15T00:00:00Z")
    )
    payload = build_handoff(store, now=AS_OF)
    shortages = payload["parts_outlook"]["shortages"]
    assert [p["part_id"] for p in shortages] == ["BRG-6208"]
    assert shortages[0]["bites_at"] == "2026-10-15T00:00:00Z"
    text = render_text(payload)
    assert "COMING PARTS SHORTAGES (14-DAY OUTLOOK):" in text
    assert "BRG-6208 · short 1" in text and "PMO-BRG" in text
    quiet = render_text(build_handoff(InMemoryStore(), now=AS_OF))
    assert "COMING PARTS SHORTAGES" in quiet and "(none" in quiet


def test_service_and_api_shapes():
    service = make_service()
    outlook = service.parts_outlook()
    assert outlook.horizon_days == 14 and outlook.parts == []
    api.set_service(service)
    try:
        client = TestClient(api.app)
        body = client.get("/api/parts/outlook").json()
        assert body["horizon_days"] == 14 and body["shortages"] == []
        assert client.get("/api/parts/outlook?horizon_days=30").json()["horizon_days"] == 30
        assert client.get("/api/parts/outlook?horizon_days=0").status_code == 422
    finally:
        api.set_service(None)
