"""Round 12: what the plant doesn't know — the knowledge-gap report.

Until now a case whose diagnosed fault had no SOP on record for
its machine blocked in a way nothing named: offline the
diagnosis falls back to ``unknown_fault`` with no evidence, the
critique's ``unevidenced_hypothesis`` gap can never close, and
the case exhausts after burning every round — filed in the
handoff as a generic escalation, invisible as a *pattern*. The
gap report (gaps.py) aggregates those blocks by the missing
asset, prices them in rounds, and says whether the hole is
biting right now.

The fixtures here are fabricated records with hand-derived
expectations against the seeded plant model (data/kb.json:
FEED-9 is not a machine in it; CONV-5 has an SOP for
belt_tracking_fault; COMP-7 for cooling_fan_failure): the tests
check the membership rule, the ranking, the obligation
annotation (the outlook's own no-SOP test, referenced), the
advisory boundary (the schema can carry no SOP content), and the
discipline (a pure read — no writes, no reasoner calls), not a
snapshot of the implementation's output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from helpers import CountingReasoner, make_service

from plant_loop import api
from plant_loop.gaps import build_gap_report, render_text
from plant_loop.handoff import build_handoff, render_text as render_handoff
from plant_loop.schemas import (
    CaseResult,
    Diagnosis,
    Hypothesis,
    KnowledgeGap,
    PmObligation,
    RoundRecord,
)
from plant_loop.store import InMemoryStore

AS_OF = datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures (fabricated records, the test_outlook way)
# ---------------------------------------------------------------------------


def _record(
    case_id: str,
    *,
    machine_id: str = "FEED-9",
    fault: str = "unknown_fault",
    exit_reason: str = "exhausted",
    status: str = "awaiting_supervisor",
    rounds: int = 3,
    created: str = "2026-10-01T00:00:00Z",
    diagnosed: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=status,
        exit_reason=exit_reason,
        type="machine_down",
        title=f"fabricated case {case_id}",
        machine_id=machine_id,
        line_id="line-d",
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(fault_class=fault, statement="fabricated", confidence=0.3)
            ]
        )
        if diagnosed
        else Diagnosis(),
        rounds=[RoundRecord(round=i + 1) for i in range(rounds)],
        created_at=created,
        reported_at=created,
    )


def _obligation(
    obligation_id: str,
    machine_id: str,
    fault_class: str,
    *,
    status: str = "standing",
) -> PmObligation:
    return PmObligation(
        obligation_id=obligation_id,
        machine_id=machine_id,
        fault_class=fault_class,
        line_id="line-d",
        interval_days=14,
        next_due_at="2026-10-20T00:00:00Z",
        status=status,
    )


def _snapshot(store: InMemoryStore) -> str:
    return json.dumps(
        {
            "cases": [r.model_dump() for r in store.all_cases()],
            "claims": [c.model_dump() for c in store.all_part_claims()],
            "pm": [r.model_dump() for r in store.all_pm_records()],
        },
        sort_keys=True,
        default=str,
    )


# ---------------------------------------------------------------------------
# Membership: what counts as a knowledge block
# ---------------------------------------------------------------------------


def test_repeat_gap_groups_counts_and_prices():
    store = InMemoryStore()
    # The same missing SOP blocks twice, a month apart: the first
    # case was decided at the gate (history), the second still
    # waits (biting now). 3 rounds burned each.
    store.save_case(_record("F-OLD", created="2026-09-10T00:00:00Z", status="rejected"))
    store.save_case(_record("F-NEW", created="2026-10-10T00:00:00Z"))
    report = build_gap_report(store, now=AS_OF)
    assert report.cases_examined == 2 and report.blocked_cases == 2
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert (gap.machine_id, gap.fault_class) == ("FEED-9", "unknown_fault")
    assert gap.machine_on_register is False  # FEED-9 is not in the plant model
    assert gap.blocked_cases == 2 and gap.case_ids == ["F-OLD", "F-NEW"]
    assert gap.first_case_id == "F-OLD" and gap.latest_case_id == "F-NEW"
    assert gap.first_seen_at == "2026-09-10T00:00:00Z"
    assert gap.latest_seen_at == "2026-10-10T00:00:00Z"
    assert gap.rounds_burned == 6
    assert gap.exits == {"exhausted": 2}
    assert gap.open_now is True and gap.open_case_ids == ["F-NEW"]


def test_named_fault_without_an_sop_counts():
    # A diagnosis that names a real fault class the machine has
    # no SOP for (spindle_bearing_wear is CNC-3's, not CONV-5's):
    # the same hole, seen from the named side.
    store = InMemoryStore()
    store.save_case(_record("N-1", machine_id="CONV-5", fault="spindle_bearing_wear"))
    report = build_gap_report(store, now=AS_OF)
    assert [(g.machine_id, g.fault_class) for g in report.gaps] == [
        ("CONV-5", "spindle_bearing_wear")
    ]
    assert report.gaps[0].machine_on_register is True


def test_fault_with_an_sop_never_appears():
    # Exhausted on an unseparated pair (the G09 shape): the fault
    # HAS an SOP, so the block is evidence, not knowledge — no row.
    store = InMemoryStore()
    store.save_case(
        _record("B-1", machine_id="CONV-5", fault="belt_tracking_fault")
    )
    assert build_gap_report(store, now=AS_OF).gaps == []


def test_parts_block_is_not_a_knowledge_gap():
    # Blocked on the outside world (the G06 shape): the fault has
    # an SOP; the plant is waiting on a part, not on knowledge.
    store = InMemoryStore()
    store.save_case(
        _record(
            "P-1",
            machine_id="COMP-7",
            fault="cooling_fan_failure",
            exit_reason="blocked_external",
            rounds=1,
        )
    )
    assert build_gap_report(store, now=AS_OF).gaps == []


def test_converged_and_safety_records_never_count():
    store = InMemoryStore()
    # A converged case whose fault has no SOP reached a verified
    # plan (a live backend can do this) — it did not block.
    store.save_case(
        _record("C-1", fault="spindle_bearing_wear", exit_reason="converged", rounds=1)
    )
    # A safety stop has no diagnosis at all — it ends by rule.
    store.save_case(
        _record("S-1", exit_reason="safety_stop", rounds=0, diagnosed=False)
    )
    # A safety stop WITH a diagnosis (a banned-action plan): it
    # stopped on safety, which an SOP would not have changed.
    store.save_case(_record("S-2", exit_reason="safety_stop", rounds=1))
    report = build_gap_report(store, now=AS_OF)
    assert report.gaps == [] and report.blocked_cases == 0
    assert report.cases_examined == 3


def test_all_closed_gap_still_lists_as_history():
    store = InMemoryStore()
    store.save_case(_record("H-1", status="rejected"))
    store.save_case(_record("H-2", status="approved", created="2026-10-02T00:00:00Z"))
    report = build_gap_report(store, now=AS_OF)
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.blocked_cases == 2 and gap.open_now is False
    assert gap.open_case_ids == []


# ---------------------------------------------------------------------------
# Ranking: open-now first, then count, then rounds burned
# ---------------------------------------------------------------------------


def test_ranking_open_now_first_then_count():
    store = InMemoryStore()
    # GAP-CLOSED: 3 blocks, all decided — history, however long.
    for i in range(3):
        store.save_case(
            _record(f"C-{i}", fault="hydraulic_pressure_loss", status="rejected",
                    created=f"2026-09-0{i + 1}T00:00:00Z")
        )
    # GAP-OPEN-SMALL: 1 block, still at the gate — bites now.
    store.save_case(_record("O-1", machine_id="PRESS-9", fault="unknown_fault"))
    # GAP-OPEN-BIG: 2 blocks, one at the gate.
    store.save_case(
        _record("O-2", machine_id="WELD-9", fault="unknown_fault", status="rejected",
                created="2026-09-05T00:00:00Z")
    )
    store.save_case(
        _record("O-3", machine_id="WELD-9", fault="unknown_fault",
                created="2026-10-05T00:00:00Z")
    )
    report = build_gap_report(store, now=AS_OF)
    order = [(g.machine_id, g.blocked_cases, g.open_now) for g in report.gaps]
    assert order == [
        ("WELD-9", 2, True),   # open now, most blocks
        ("PRESS-9", 1, True),  # open now
        ("FEED-9", 3, False),  # history ranks below any live gap
    ]


# ---------------------------------------------------------------------------
# The obligation annotation: round 11's test, referenced not forked
# ---------------------------------------------------------------------------


def test_standing_obligation_on_the_same_hole_annotates_the_row():
    store = InMemoryStore()
    store.save_case(_record("F-1", fault="spindle_bearing_wear"))
    # FEED-9 has no SOP for spindle_bearing_wear (it is CNC-3's),
    # so the outlook's unquantified test names this obligation;
    # the gap row carries it — one authored SOP heals both.
    store.save_pm_record(_obligation("PMO-F", "FEED-9", "spindle_bearing_wear"))
    # A retired obligation annotates nothing: its duty ended.
    store.save_pm_record(
        _obligation("PMO-F-OLD", "FEED-9", "spindle_bearing_wear", status="retired")
    )
    # An obligation whose pair HAS an SOP is not on a hole.
    store.save_pm_record(
        _obligation("PMO-H", "PRESS-1", "hydraulic_pressure_loss")
    )
    report = build_gap_report(store, now=AS_OF)
    assert len(report.gaps) == 1
    assert report.gaps[0].standing_obligation_ids == ["PMO-F"]


# ---------------------------------------------------------------------------
# The advisory boundary, pinned by construction
# ---------------------------------------------------------------------------


def test_schema_has_no_field_that_could_carry_sop_content():
    fields = set(KnowledgeGap.model_fields)
    assert fields == {
        "machine_id",
        "fault_class",
        "machine_on_register",
        "blocked_cases",
        "case_ids",
        "first_case_id",
        "latest_case_id",
        "first_seen_at",
        "latest_seen_at",
        "rounds_burned",
        "exits",
        "open_now",
        "open_case_ids",
        "standing_obligation_ids",
    }
    # No prose-bearing field exists to smuggle a drafted procedure
    # through: identifiers, counts and timestamps only.
    for name in fields:
        for banned in ("text", "step", "content", "draft", "instruction"):
            assert banned not in name


def test_report_generates_nothing_and_calls_no_reasoner():
    reasoner = CountingReasoner()
    service = make_service(reasoner=reasoner)
    # A real case through the real loop: the feeder shape (G16).
    from plant_loop.schemas import ExceptionCase

    result = service.run_case(
        ExceptionCase(
            case_id="KG-LIVE",
            type="machine_down",
            title="Feeder unit jammed, cause unclear",
            description="The standalone feeder unit (not on the machine register) has jammed solid twice; no readings available from it.",
            machine_id="FEED-9",
            line_id="line-d",
            reported_at="2026-10-11T06:55:00Z",
            downtime_minutes=40,
        )
    ).result
    assert result.exit_reason == "exhausted"
    calls_before = dict(reasoner.calls)
    before = _snapshot(service.store)
    first = service.knowledge_gaps(now=AS_OF)
    second = service.knowledge_gaps(now=AS_OF)
    # The report is a read: the store stands byte-identical, the
    # reasoner was never called, and two builds agree exactly.
    assert _snapshot(service.store) == before
    assert dict(reasoner.calls) == calls_before
    assert first.model_dump() == second.model_dump()
    assert [(g.machine_id, g.fault_class, g.blocked_cases) for g in first.gaps] == [
        ("FEED-9", "unknown_fault", 1)
    ]
    assert first.gaps[0].rounds_burned == 3 and first.gaps[0].open_now


def test_empty_board_reports_no_gaps():
    report = build_gap_report(InMemoryStore(), now=AS_OF)
    assert report.gaps == [] and report.blocked_cases == 0
    assert report.cases_examined == 0 and report.as_of == "2026-10-12T00:00:00Z"
    assert "drafts nothing" in report.basis


# ---------------------------------------------------------------------------
# Surfaces: text render, handoff, API
# ---------------------------------------------------------------------------


def test_render_text_names_the_asset_and_the_cost():
    store = InMemoryStore()
    store.save_case(_record("F-OLD", created="2026-09-10T00:00:00Z", status="rejected"))
    store.save_case(_record("F-NEW", created="2026-10-10T00:00:00Z"))
    text = render_text(build_gap_report(store, now=AS_OF))
    assert "KNOWLEDGE GAPS" in text
    assert "FEED-9 · unknown_fault (NOT on the machine register)" in text
    assert "2 case(s) blocked, 6 round(s) burned" in text
    assert "BLOCKED NOW: F-NEW" in text
    quiet = render_text(build_gap_report(InMemoryStore(), now=AS_OF))
    assert "(none" in quiet


def test_handoff_carries_the_knowledge_gaps():
    store = InMemoryStore()
    store.save_case(_record("F-NEW", created="2026-10-10T00:00:00Z"))
    payload = build_handoff(store, now=AS_OF)
    gaps = payload["knowledge_gaps"]["gaps"]
    assert [(g["machine_id"], g["fault_class"]) for g in gaps] == [
        ("FEED-9", "unknown_fault")
    ]
    # The counts dict is untouched by the new section (the
    # round-11 discipline): the same five keys as ever.
    assert set(payload["counts"]) == {
        "awaiting_decision",
        "escalations",
        "blocked_external",
        "safety_stops",
        "reliability_items",
    }
    text = render_handoff(payload)
    assert "KNOWLEDGE GAPS — WHAT THE PLANT DOESN'T KNOW:" in text
    assert "FEED-9 · unknown_fault — no SOP on record" in text
    assert "blocked now: F-NEW" in text
    quiet = render_handoff(build_handoff(InMemoryStore(), now=AS_OF))
    assert "KNOWLEDGE GAPS" in quiet and "(none" in quiet


def test_service_and_api_shapes():
    service = make_service()
    report = service.knowledge_gaps(now=AS_OF)
    assert report.gaps == [] and report.cases_examined == 0
    api.set_service(service)
    try:
        client = TestClient(api.app)
        body = client.get("/api/knowledge/gaps").json()
        assert body["gaps"] == [] and body["blocked_cases"] == 0
        assert "drafts nothing" in body["basis"]
    finally:
        api.set_service(None)
