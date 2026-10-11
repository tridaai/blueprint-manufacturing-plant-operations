"""Round 4: the plant remembers — confirmed-fault write-back.

A supervisor's outcome with a confirmed fault is written back to
the machine's history (history.py), the diagnose context reads the
merged history (confirmed entries marked, ahead of the recorded
ones), and the offline diagnoser can separate a rival the readings
could not when the floor has already confirmed the leader on that
machine. The stats fixtures are hand-computed, like round 2's.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from plant_loop import api, history, kb
from plant_loop.backends import OfflineReasoner
from plant_loop.handoff import build_handoff, render_text
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import (
    CaseResult,
    Diagnosis,
    ExceptionCase,
    HistoryEntry,
    Hypothesis,
    RoundRecord,
    SupervisorOutcome,
)
from plant_loop.stats import NO_REPEATS_NOTE, diagnosis_stats
from plant_loop.store import InMemoryStore

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}


def _run(service, case_id: str, new_id: str | None = None):
    case = CASES[case_id]
    if new_id:
        case = case.model_copy(update={"case_id": new_id})
    return service.run_case(case).result


def _outcome(**kwargs) -> SupervisorOutcome:
    base = {"decision": "accepted"}
    base.update(kwargs)
    return SupervisorOutcome(**base)


# ---------------------------------------------------------------------------
# The write-back itself
# ---------------------------------------------------------------------------


def test_confirmed_fault_writes_back():
    service = make_service()
    record = _run(service, "EXC-101")
    service.record_outcome(
        "EXC-101",
        _outcome(
            decision="accepted_with_changes",
            confirmed_fault="spindle_bearing_wear",
            work_performed="Bearing set replaced under SOP-CNC3-BRG.",
        ),
    )
    entries = service.store.all_history_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.entry_id == "hist-EXC-101"
    assert entry.machine_id == "CNC-3"
    assert entry.fault_class == "spindle_bearing_wear"
    assert entry.case_id == "EXC-101"
    assert entry.source == "supervisor_confirmed"
    assert entry.date == record.reported_at[:10]
    assert "spindle_bearing_wear" in entry.summary
    assert entry.resolution == "Bearing set replaced under SOP-CNC3-BRG."


@pytest.mark.parametrize("decision", ["accepted", "accepted_with_changes", "rejected"])
def test_any_decision_with_a_confirmation_writes(decision):
    # The fault the floor found is ground truth even when the plan
    # was refused — the write-back keys on the confirmation, not
    # on the plan's disposition.
    service = make_service()
    _run(service, "EXC-101")
    service.record_outcome(
        "EXC-101", _outcome(decision=decision, confirmed_fault="spindle_bearing_wear")
    )
    assert len(service.store.all_history_entries()) == 1


@pytest.mark.parametrize("decision", ["accepted", "accepted_with_changes", "rejected"])
def test_outcome_without_a_confirmation_writes_nothing(decision):
    service = make_service()
    _run(service, "EXC-101")
    service.record_outcome("EXC-101", _outcome(decision=decision))
    assert service.store.all_history_entries() == []


def test_write_back_is_idempotent_with_the_outcome():
    service = make_service()
    _run(service, "EXC-101")
    service.record_outcome(
        "EXC-101", _outcome(confirmed_fault="spindle_bearing_wear")
    )
    # Re-applying the same record's confirmation is the same fact.
    again = service.store.get_case("EXC-101")
    assert history.record_confirmation(service.store, again) is not None
    assert len(service.store.all_history_entries()) == 1
    # And a second outcome (which would be the only way to confirm
    # "again") is refused by the outcome path itself.
    with pytest.raises(ValueError):
        service.record_outcome(
            "EXC-101", _outcome(confirmed_fault="spindle_bearing_wear")
        )
    assert len(service.store.all_history_entries()) == 1


def test_safety_stop_refuses_a_confirmed_fault():
    service = make_service()
    _run(service, "EXC-102")  # the seeded safety observation (WELD-4)
    with pytest.raises(ValueError, match="safety stop"):
        service.record_outcome(
            "EXC-102", _outcome(confirmed_fault="weld_quality_drift")
        )
    # The refusal happens before anything persists: no outcome on
    # the record, no entry in the plant's memory.
    assert service.store.get_case("EXC-102").outcome is None
    assert service.store.all_history_entries() == []
    # A safety outcome without a confirmed fault records fine.
    service.record_outcome("EXC-102", _outcome(decision="accepted"))
    assert service.store.get_case("EXC-102").outcome is not None
    assert service.store.all_history_entries() == []


def test_seeded_history_is_immutable_and_marked_recorded():
    service = make_service()
    _run(service, "EXC-101")
    service.record_outcome(
        "EXC-101", _outcome(confirmed_fault="spindle_bearing_wear")
    )
    merged = history.merged_history(service.store, "CNC-3", "spindle_bearing_wear")
    assert [e["source"] for e in merged] == ["supervisor_confirmed", "recorded"]
    assert merged[0]["case_id"] == "EXC-101"
    assert merged[1]["hist_id"] == "H-1042"
    # The KB file itself is untouched: the write-back lives in the
    # store, never in the packaged knowledge base.
    assert len(kb.load_kb()["history"]) == 8
    assert kb.history_for("CNC-3", "spindle_bearing_wear")[0]["hist_id"] == "H-1042"


# ---------------------------------------------------------------------------
# The diagnose context reads the merged history
# ---------------------------------------------------------------------------


class CapturingReasoner(OfflineReasoner):
    """Stashes every diagnose context it is handed."""

    def __init__(self) -> None:
        self.contexts: list = []

    def diagnose(self, ctx, focus):
        self.contexts.append(ctx)
        return super().diagnose(ctx, focus)


def test_diagnose_context_carries_the_marked_history():
    reasoner = CapturingReasoner()
    service = make_service(reasoner=reasoner)
    _run(service, "EXC-101")
    service.record_outcome(
        "EXC-101", _outcome(confirmed_fault="spindle_bearing_wear")
    )
    reasoner.contexts.clear()
    _run(service, "EXC-101", new_id="EXC-101-AGAIN")
    ctx = reasoner.contexts[0]
    assert ctx.history[0]["source"] == "supervisor_confirmed"
    assert ctx.history[0]["case_id"] == "EXC-101"
    assert any(e["source"] == "recorded" for e in ctx.history)
    confirmed_chunks = [
        c for c in ctx.kb_context if c.get("history_source") == "supervisor_confirmed"
    ]
    assert confirmed_chunks and confirmed_chunks[0]["text"].startswith(
        "Supervisor-confirmed history (case EXC-101"
    )
    assert any(c.get("history_source") == "recorded" for c in ctx.kb_context)


def _ambiguous(case_id: str, batch: str, vib: float, temp: float) -> ExceptionCase:
    """The CNC-3 signature the readings cannot separate: vibration
    high, temperature below the bearing corroboration, no runout."""
    return ExceptionCase(
        case_id=case_id,
        type="machine_down",
        title="CNC-3 spindle vibration high, no runout taken",
        description="Spindle vibration alarm; operator stopped the machine.",
        machine_id="CNC-3",
        line_id="line-b",
        product="Gear blank GB-220",
        batch=batch,
        readings={"spindle_vibration_mm_s": vib, "spindle_temp_c": temp},
        notes="No runout measurement was taken.",
        reporter="J. Okafor (operator)",
        reported_at="2026-10-11T06:12:00Z",
        downtime_minutes=40,
    )


def test_the_loop_learns_from_a_confirmation():
    service = make_service()
    baseline = service.run_case(_ambiguous("T-1", "B-1", 8.2, 74)).result
    assert baseline.exit_reason == "exhausted"
    assert len(baseline.rounds) == 3
    assert baseline.diagnosis.top().fault_class == "spindle_bearing_wear"
    assert baseline.diagnosis.hypotheses[1].fault_class == "tool_imbalance"

    service.record_outcome(
        "T-1",
        _outcome(
            confirmed_fault="spindle_bearing_wear",
            work_performed="Stripped the spindle: bearing set scored; replaced.",
        ),
    )

    follow = service.run_case(_ambiguous("T-2", "B-2", 8.6, 76)).result
    assert follow.exit_reason == "converged"
    assert len(follow.rounds) == 2
    top = follow.diagnosis.top()
    assert top.fault_class == "spindle_bearing_wear"
    assert any("supervisor-confirmed history" in e for e in top.evidence)
    rival = follow.diagnosis.hypotheses[1]
    assert rival.fault_class == "tool_imbalance"
    assert rival.confidence <= 0.2
    assert any("supervisor-confirmed precedent" in e for e in rival.evidence)


def test_recorded_history_alone_never_separates():
    # CNC-3 already has a *recorded* bearing precedent (H-1042) in
    # the KB — the baseline above exhausts anyway. Pin it directly:
    # with no confirmation anywhere, the ambiguous case exhausts.
    service = make_service()
    result = service.run_case(_ambiguous("T-9", "B-9", 8.2, 74)).result
    assert result.exit_reason == "exhausted"
    assert len(result.rounds) == 3


# ---------------------------------------------------------------------------
# Stats: the learning-loop section (hand-computed)
# ---------------------------------------------------------------------------


def _record(case_id, *, machine_id, hypotheses, confirmed, created_at) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status="awaiting_supervisor",
        exit_reason="converged",
        type="machine_down",
        title=f"Case {case_id}",
        machine_id=machine_id,
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(fault_class=f, statement="s", confidence=0.8)
                for f in hypotheses
            ]
        ),
        rounds=[RoundRecord(round=1)],
        reported_at=created_at,
        created_at=created_at,
        outcome=SupervisorOutcome(
            decision="accepted", confirmed_fault=confirmed, recorded_at=created_at
        ),
    )


def _stats_records() -> list[CaseResult]:
    return [
        # first confirmation of bearing on CNC-3 (not a repeat), top-1 right
        _record("R1", machine_id="CNC-3", hypotheses=["spindle_bearing_wear"],
                confirmed="spindle_bearing_wear", created_at="2026-10-01T06:00:00Z"),
        # repeat of CNC-3 bearing, top-1 right
        _record("R2", machine_id="CNC-3", hypotheses=["spindle_bearing_wear"],
                confirmed="spindle_bearing_wear", created_at="2026-10-05T06:00:00Z"),
        # repeat of CNC-3 bearing, top-1 wrong
        _record("R3", machine_id="CNC-3", hypotheses=["coolant_system_fault"],
                confirmed="spindle_bearing_wear", created_at="2026-10-06T06:00:00Z"),
        # first confirmation on PRESS-1 (not a repeat), top-1 right
        _record("R4", machine_id="PRESS-1", hypotheses=["hydraulic_pressure_loss"],
                confirmed="hydraulic_pressure_loss", created_at="2026-10-07T06:00:00Z"),
    ]


def test_stats_learning_section_hand_computed():
    section = diagnosis_stats(_stats_records())["learning_loop"]
    # overall: 3 of 4 scored are top-1 right.
    assert section["overall_top1_accuracy"] == 0.75
    # repeats: R2 (right) and R3 (wrong) -> 1/2.
    assert section["repeat_cases"] == 2
    assert section["repeat_top1_correct"] == 1
    assert section["repeat_top1_accuracy"] == 0.5
    # derived write-back counts (no entries passed): one per confirmation.
    assert section["confirmed_written_back"] == 4
    assert section["written_back_by_machine"] == {"CNC-3": 3, "PRESS-1": 1}
    assert section["note"] == ""


def test_stats_learning_section_prefers_store_entries_and_says_none_yet():
    records = _stats_records()[:1]  # one confirmation, no repeat possible
    entries = [
        HistoryEntry(entry_id="hist-X", machine_id="CNC-3",
                     fault_class="spindle_bearing_wear", case_id="X"),
        HistoryEntry(entry_id="hist-Y", machine_id="CNC-3",
                     fault_class="coolant_system_fault", case_id="Y"),
    ]
    section = diagnosis_stats(records, history_entries=entries)["learning_loop"]
    assert section["confirmed_written_back"] == 2
    assert section["written_back_by_machine"] == {"CNC-3": 2}
    assert section["repeat_cases"] == 0
    assert section["repeat_top1_accuracy"] is None
    assert section["note"] == NO_REPEATS_NOTE


# ---------------------------------------------------------------------------
# Handoff + HTTP surface
# ---------------------------------------------------------------------------


def test_handoff_names_confirmed_faults():
    store = InMemoryStore()
    store.save_history_entry(
        HistoryEntry(
            entry_id="hist-W1",
            machine_id="CNC-3",
            fault_class="spindle_bearing_wear",
            case_id="W1",
            date="2026-10-11",
            recorded_at="2026-10-11T09:00:00Z",
            summary="The floor confirmed spindle_bearing_wear.",
        )
    )
    payload = build_handoff(store)
    assert [e["case_id"] for e in payload["confirmed_faults"]] == ["W1"]
    assert payload["confirmed_faults"][0]["fault_class"] == "spindle_bearing_wear"
    # the counts block keeps exactly its round-2 shape
    assert set(payload["counts"]) == {
        "awaiting_decision", "escalations", "blocked_external",
        "safety_stops", "reliability_items",
    }
    text = render_text(payload)
    assert "FAULTS CONFIRMED BY SUPERVISORS" in text
    assert "confirmed in case W1" in text


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def test_learning_loop_over_http(client):
    http, _service = client
    http.post("/cases/analyze", json=CASES["EXC-101"].model_dump())
    refused = http.post(
        "/api/cases/EXC-102/outcome", json={"decision": "accepted"}
    )
    assert refused.status_code == 404  # EXC-102 was never run here
    ok = http.post(
        "/api/cases/EXC-101/outcome",
        json={"decision": "accepted", "confirmed_fault": "spindle_bearing_wear"},
    )
    assert ok.status_code == 200
    stats = http.get("/api/stats/diagnosis").json()
    assert stats["learning_loop"]["confirmed_written_back"] == 1
    assert stats["learning_loop"]["written_back_by_machine"] == {"CNC-3": 1}
    text = http.get("/api/board/handoff", params={"format": "text"}).text
    assert "FAULTS CONFIRMED BY SUPERVISORS" in text
    assert "spindle_bearing_wear" in text and "EXC-101" in text


def test_safety_outcome_refusal_over_http(client):
    http, _service = client
    http.post("/cases/analyze", json=CASES["EXC-102"].model_dump())
    response = http.post(
        "/api/cases/EXC-102/outcome",
        json={"decision": "accepted", "confirmed_fault": "weld_quality_drift"},
    )
    assert response.status_code == 409
    assert "safety stop" in response.json()["detail"]
