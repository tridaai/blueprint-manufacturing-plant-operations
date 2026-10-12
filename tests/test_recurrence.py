"""Round 14: recurrence — the fault that came back.

The link itself lives in recurrence.py and is stamped by the
loop controller the moment a case's first diagnosis lands:

- **detection, in code** — a prior case seeds a link only when
  it closed with an outcome that confirmed the very fault the
  new case diagnosed, on the same machine, inside the window
  (the watch's fortnight); the misdiagnosis close (an outcome
  confirming nothing), the still-open case, the dropped case
  and the safety stop all seed nothing;
- **the head start, bounded** — the prior episode enters the
  diagnosis context as one labelled recurrence chunk carrying
  the prior's outcome; the reasoner may agree or depart, and
  nothing forces it either way;
- **departure, marked not blocked** — a settled diagnosis that
  differs from the prior's confirmed fault stamps the record
  DIAGNOSIS_DEPARTED_FROM_PRIOR and the case is still gated on
  its own merits, through the unchanged verify checks;
- **counting** — the stats' recurrence section and the handoff's
  open-recurrence list read the links as recorded; a linked
  case that blocks on a missing SOP is one knowledge gap, not
  two.
"""

from __future__ import annotations

from plant_loop import kb, recurrence
from plant_loop.backends import OfflineReasoner
from plant_loop.gaps import build_gap_report
from plant_loop.handoff import build_handoff, render_text
from plant_loop.packet import build_packet, render_text as render_packet_text
from plant_loop.ports import RoundOutput
from plant_loop.recurrence import find_link
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import (
    CaseResult,
    Diagnosis,
    ExceptionCase,
    Hypothesis,
    RecurrenceLink,
    RoundRecord,
    SupervisorOutcome,
)
from plant_loop.stats import NO_LINKS_NOTE, diagnosis_stats
from plant_loop.store import InMemoryStore

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}
BEARING = "spindle_bearing_wear"
COOLANT = "coolant_system_fault"


def _prior(
    case_id: str,
    *,
    machine_id: str = "CNC-3",
    confirmed: str = BEARING,
    diagnosed: str | None = None,
    decision: str = "accepted",
    resolved: bool | None = True,
    reported_at: str = "2026-10-01T06:00:00Z",
    created_at: str = "",
    status: str = "approved",
    with_outcome: bool = True,
    case_type: str = "machine_down",
    exit_reason: str = "converged",
) -> CaseResult:
    """A fabricated closed record: the loop diagnosed
    ``diagnosed`` (default: the fault itself), the floor's
    outcome confirmed ``confirmed``."""
    return CaseResult(
        case_id=case_id,
        status=status,
        exit_reason=exit_reason,
        type=case_type,
        title=f"Case {case_id}",
        machine_id=machine_id,
        diagnosis=Diagnosis(
            hypotheses=[
                Hypothesis(
                    fault_class=diagnosed if diagnosed is not None else confirmed,
                    statement="s",
                    confidence=0.8,
                )
            ]
        ),
        rounds=[RoundRecord(round=1)],
        reported_at=reported_at,
        created_at=created_at or reported_at,
        outcome=(
            SupervisorOutcome(
                decision=decision,
                confirmed_fault=confirmed,
                work_performed="The work was done.",
                resolved=resolved,
                recorded_at="2026-10-02T09:00:00Z",
            )
            if with_outcome
            else None
        ),
    )


def _store_with(*records: CaseResult) -> InMemoryStore:
    store = InMemoryStore()
    for record in records:
        store.save_case(record)
    return store


def _link(store, **kwargs):
    base = {
        "case_id": "NEW-1",
        "machine_id": "CNC-3",
        "fault_class": BEARING,
        "reported_at": "2026-10-11T06:00:00Z",
    }
    base.update(kwargs)
    return find_link(store, **base)


# ---------------------------------------------------------------------------
# Detection: the qualifying set, stated exactly
# ---------------------------------------------------------------------------


def test_accepted_confirmation_links():
    link = _link(_store_with(_prior("P-1")))
    assert link is not None
    assert link.prior_case_id == "P-1"
    assert link.fault_class == BEARING
    assert link.outcome_decision == "accepted"
    assert link.outcome_resolved is True
    assert link.days_between == 10
    assert link.window_days == recurrence.WINDOW_DAYS == 14


def test_rejected_with_confirmation_still_links():
    # The write-back's doctrine (history.py): the fault the floor
    # found is ground truth even when the plan was refused.
    link = _link(_store_with(_prior("P-1", decision="rejected")))
    assert link is not None and link.outcome_decision == "rejected"


def test_outcome_confirming_nothing_seeds_nothing():
    # The misdiagnosis close: the floor found no fault to confirm.
    assert _link(_store_with(_prior("P-1", confirmed=""))) is None


def test_no_outcome_no_link():
    assert _link(_store_with(_prior("P-1", with_outcome=False))) is None
    # Dropped at a board turnover / closed as moot: no outcome,
    # so not closed-with-outcome, whatever else the record says.
    assert (
        _link(_store_with(_prior("P-1", with_outcome=False, status="dropped")))
        is None
    )
    assert (
        _link(
            _store_with(_prior("P-1", with_outcome=False, status="closed_moot"))
        )
        is None
    )


def test_safety_stop_never_seeds():
    # Fabricated directly (the service refuses this outcome on a
    # safety stop): the detector itself must still refuse it.
    stop = _prior(
        "P-1", case_type="safety_observation", exit_reason="safety_stop"
    )
    assert _link(_store_with(stop)) is None


def test_join_key_is_machine_and_confirmed_fault():
    assert _link(_store_with(_prior("P-1", machine_id="PRESS-1"))) is None
    assert _link(_store_with(_prior("P-1", confirmed=COOLANT))) is None
    # The prior's own diagnosis is not the key: the loop there
    # misread it, the floor confirmed this fault — it links, and
    # the link carries what the prior's loop had diagnosed.
    link = _link(_store_with(_prior("P-1", diagnosed="tool_imbalance")))
    assert link is not None
    assert link.prior_diagnosis == "tool_imbalance"


def test_a_case_never_recurs_itself():
    assert _link(_store_with(_prior("NEW-1"))) is None


def test_window_edges():
    store = _store_with(_prior("P-1", reported_at="2026-10-01T06:00:00Z"))
    at_edge = _link(store, reported_at="2026-10-15T06:00:00Z")
    assert at_edge is not None and at_edge.days_between == 14
    assert _link(store, reported_at="2026-10-15T07:00:00Z") is None  # 14d + 1h
    inside = _link(store, reported_at="2026-10-14T05:00:00Z")
    assert inside is not None and inside.days_between == 12
    # A prior dated after the new case is not a prior.
    assert _link(store, reported_at="2026-09-30T06:00:00Z") is None


def test_most_recent_qualifying_prior_wins():
    store = _store_with(
        _prior("P-OLD", reported_at="2026-10-02T06:00:00Z"),
        _prior("P-NEW", reported_at="2026-10-06T06:00:00Z"),
        _prior("P-OUT", reported_at="2026-09-01T06:00:00Z"),  # outside window
    )
    link = _link(store)
    assert link is not None
    assert link.prior_case_id == "P-NEW"
    assert link.days_between == 5


def test_undatable_prior_never_seeds():
    ghost = _prior("P-1", reported_at="", created_at="")
    assert _link(_store_with(ghost)) is None
    # A created stamp alone still dates the episode.
    stamped = _prior("P-1", reported_at="", created_at="2026-10-03T06:00:00Z")
    link = _link(_store_with(stamped))
    assert link is not None and link.days_between == 8


# ---------------------------------------------------------------------------
# The link's faces: chunk + departure
# ---------------------------------------------------------------------------


def test_context_chunk_carries_the_episode():
    link = _link(_store_with(_prior("P-1")))
    chunk = recurrence.context_chunk(link)
    assert chunk["chunk_id"] == "recurrence:P-1"
    assert chunk["source"] == "recurrence"
    assert "P-1" in chunk["text"]
    assert "accepted" in chunk["text"] and "resolved" in chunk["text"]
    assert "history, not an instruction" in chunk["text"]


def test_with_departure_is_code_against_the_final_diagnosis():
    link = _link(_store_with(_prior("P-1")))

    def diag(fault: str) -> Diagnosis:
        return Diagnosis(
            hypotheses=[Hypothesis(fault_class=fault, statement="s", confidence=0.7)]
        )

    assert recurrence.with_departure(link, diag(BEARING)).departed is False
    assert recurrence.with_departure(link, diag(COOLANT)).departed is True
    assert recurrence.with_departure(link, Diagnosis()).departed is False
    assert recurrence.with_departure(None, diag(BEARING)) is None


# ---------------------------------------------------------------------------
# End to end: the head start reaches the reasoner, the prior is untouched
# ---------------------------------------------------------------------------


class CapturingReasoner(OfflineReasoner):
    """Stashes a snapshot of every diagnose context it is handed
    (the chunk list is the loop's live one — snapshot the chunk
    dicts, or later rounds' appends would rewrite the capture)."""

    def __init__(self) -> None:
        self.contexts: list = []

    def diagnose(self, ctx, focus):
        import dataclasses

        self.contexts.append(
            dataclasses.replace(
                ctx, kb_context=[dict(c) for c in ctx.kb_context]
            )
        )
        return super().diagnose(ctx, focus)


def _closed_bearing_case(service, case_id: str = "EXC-101") -> None:
    service.run_case(CASES[case_id])
    service.record_outcome(
        case_id,
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault=BEARING,
            work_performed="Bearing set replaced under SOP-CNC3-BRG.",
            resolved=True,
        ),
    )


def test_followup_links_and_the_chunk_reaches_later_rounds():
    reasoner = CapturingReasoner()
    service = make_service(reasoner=reasoner)
    _closed_bearing_case(service)
    prior_before = service.store.get_case("EXC-101").model_dump_json()

    reasoner.contexts.clear()
    follow = service.run_case(
        CASES["EXC-101"].model_copy(
            update={"case_id": "REC-1", "reported_at": "2026-10-17T06:12:00Z"}
        )
    ).result

    link = follow.recurrence
    assert link is not None
    assert link.prior_case_id == "EXC-101"
    assert link.days_between == 6
    assert link.departed is False
    assert follow.diagnosis.top().fault_class == BEARING
    assert follow.exit_reason == "converged"
    # The chunk was not in the first diagnose context (the link
    # forms from that round's diagnosis) and is in every later
    # one — the head start informs the work after the first read.
    chunked = [
        any(c.get("source") == "recurrence" for c in ctx.kb_context)
        for ctx in reasoner.contexts
    ]
    assert chunked[0] is False
    assert len(chunked) > 1 and all(chunked[1:])
    chunk = next(
        c
        for c in reasoner.contexts[1].kb_context
        if c.get("source") == "recurrence"
    )
    assert "EXC-101" in chunk["text"] and "accepted" in chunk["text"]
    # The prior record was never touched by being linked from.
    assert service.store.get_case("EXC-101").model_dump_json() == prior_before


def test_no_prior_no_link_no_chunk():
    reasoner = CapturingReasoner()
    service = make_service(reasoner=reasoner)
    result = service.run_case(CASES["EXC-101"]).result
    assert result.recurrence is None
    assert not any(
        c.get("source") == "recurrence"
        for ctx in reasoner.contexts
        for c in ctx.kb_context
    )


def test_misdiagnosis_close_seeds_no_link_end_to_end():
    service = make_service()
    service.run_case(CASES["EXC-101"])
    service.record_outcome(
        "EXC-101", SupervisorOutcome(decision="rejected")  # confirms nothing
    )
    follow = service.run_case(
        CASES["EXC-101"].model_copy(
            update={"case_id": "REC-2", "reported_at": "2026-10-13T06:12:00Z"}
        )
    ).result
    assert follow.recurrence is None


# ---------------------------------------------------------------------------
# Departure: marked on the face, judged on its own merits
# ---------------------------------------------------------------------------


class DepartingReasoner(OfflineReasoner):
    """Round 1 reads the target case like the offline rules do;
    from round 2 it settles on another fault of the machine."""

    def __init__(self, target: str, as_fault: str) -> None:
        self.target = target
        self.as_fault = as_fault
        self.calls: dict[str, int] = {}

    def diagnose(self, ctx, focus):
        case_id = ctx.case.case_id
        self.calls[case_id] = self.calls.get(case_id, 0) + 1
        out = super().diagnose(ctx, focus)
        if case_id == self.target and self.calls[case_id] >= 2:
            sop = kb.sop_for(ctx.case.machine_id, self.as_fault) or {}
            diagnosis = Diagnosis(
                hypotheses=[
                    Hypothesis(
                        fault_class=self.as_fault,
                        statement=(
                            f"{sop.get('title', self.as_fault)} fits the "
                            "readings on the second look."
                        ),
                        confidence=0.7,
                        evidence=["departing read: the second look separates it"],
                        sop_id=sop.get("sop_id", ""),
                    )
                ],
                summary=f"Top hypothesis: {self.as_fault} at 0.70 confidence.",
            )
            return RoundOutput(
                payload=diagnosis,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                latency_ms=out.latency_ms,
                backend=out.backend,
                model=out.model,
            )
        return out


def test_departed_case_is_marked_and_still_gated_on_its_merits():
    service = make_service(reasoner=DepartingReasoner("DEP-1", COOLANT))
    _closed_bearing_case(service)
    record = service.run_case(
        CASES["EXC-101"].model_copy(
            update={"case_id": "DEP-1", "reported_at": "2026-10-15T06:12:00Z"}
        )
    ).result
    assert record.recurrence is not None
    assert record.recurrence.prior_case_id == "EXC-101"
    assert record.recurrence.departed is True
    assert record.diagnosis.top().fault_class == COOLANT
    # The verify stage invented nothing for the occasion: the
    # departed case converged through the ordinary checks, with
    # no gap kind anywhere naming the recurrence.
    assert record.exit_reason == "converged"
    assert record.open_gaps == []
    assert not any(
        "recurrence" in key
        for r in record.rounds
        for key in r.gaps_opened + r.gaps_closed
    )
    # And the gate treats it as any converged case.
    decided = service.record_decision("DEP-1", "approved", by="Test Supervisor")
    assert decided.status == "approved"
    # The packet prints the comeback and the departure.
    packet = build_packet(service.store.get_case("DEP-1"))
    assert packet["recurrence"]["departed"] is True
    text = render_packet_text(packet)
    assert "THIS FAULT CAME BACK" in text
    assert recurrence.DEPARTED_MARK in text


class ScriptedDiagnosisReasoner(OfflineReasoner):
    """The target case's diagnoses follow a script — one list of
    (fault, confidence) hypotheses per call; every other case
    reads by the offline rules."""

    def __init__(self, target: str, script: list[list[tuple[str, float]]]) -> None:
        self.target = target
        self.script = script
        self.calls = 0

    def diagnose(self, ctx, focus):
        out = super().diagnose(ctx, focus)
        if ctx.case.case_id != self.target:
            return out
        spec = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        diagnosis = Diagnosis(
            hypotheses=[
                Hypothesis(
                    fault_class=fault,
                    statement="Scripted read.",
                    confidence=confidence,
                    evidence=["scripted reasoner"],
                    sop_id=(kb.sop_for(ctx.case.machine_id, fault) or {}).get(
                        "sop_id", ""
                    ),
                )
                for fault, confidence in spec
            ],
            summary="Scripted diagnosis.",
        )
        return RoundOutput(
            payload=diagnosis,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            latency_ms=out.latency_ms,
            backend=out.backend,
            model=out.model,
        )


def test_the_link_keys_on_the_first_diagnosis_only():
    # Round 1 reads tool_imbalance leading a close bearing rival
    # (no closed prior for the leader — the check runs against
    # it, once); the open competing-hypothesis gap sends the case
    # back, and round 2 settles on the bearing fault, which HAS
    # a closed prior. No link forms: the check was sought at the
    # first diagnosis and a miss is final for the run — a
    # diagnosis that arrives at the fault only later gets the
    # merged-history context (round 4) but no recurrence link.
    service = make_service(
        reasoner=ScriptedDiagnosisReasoner(
            "SCR-1",
            [
                [("tool_imbalance", 0.55), (BEARING, 0.50)],
                [(BEARING, 0.70)],
            ],
        )
    )
    _closed_bearing_case(service)
    record = service.run_case(
        CASES["EXC-101"].model_copy(
            update={"case_id": "SCR-1", "reported_at": "2026-10-15T06:12:00Z"}
        )
    ).result
    assert record.diagnosis.top().fault_class == BEARING
    assert record.recurrence is None


# ---------------------------------------------------------------------------
# Counting: stats + handoff + the knowledge gaps' one-count rule
# ---------------------------------------------------------------------------


def _linked_record(
    case_id: str,
    *,
    machine_id: str,
    fault: str,
    settled: str,
    departed: bool,
    status: str = "awaiting_supervisor",
) -> CaseResult:
    record = _prior(
        case_id, machine_id=machine_id, confirmed=settled, with_outcome=False
    )
    record.status = status
    record.recurrence = RecurrenceLink(
        prior_case_id=f"P-{case_id}",
        machine_id=machine_id,
        fault_class=fault,
        outcome_decision="accepted",
        days_between=4,
        window_days=14,
        departed=departed,
    )
    return record


def test_stats_recurrence_section_hand_computed():
    records = [
        _linked_record("S-A", machine_id="CNC-3", fault=BEARING,
                       settled=BEARING, departed=False),
        _linked_record("S-B", machine_id="CNC-3", fault=COOLANT,
                       settled=BEARING, departed=True),
        _linked_record("S-C", machine_id="PRESS-1",
                       fault="hydraulic_pressure_loss",
                       settled="hydraulic_pressure_loss", departed=False),
        _prior("S-D", with_outcome=False),  # unlinked
    ]
    section = diagnosis_stats(records)["recurrence"]
    assert section["linked_cases"] == 3
    assert section["by_machine"] == {"CNC-3": 2, "PRESS-1": 1}
    assert section["window_days"] == 14
    assert section["settled_cases"] == 3
    assert section["agreed"] == 2
    assert section["departed"] == 1
    assert section["agreement_rate"] == 0.6667
    assert section["note"] == ""


def test_stats_recurrence_section_empty():
    section = diagnosis_stats([])["recurrence"]
    assert section["linked_cases"] == 0
    assert section["agreement_rate"] is None
    assert section["note"] == NO_LINKS_NOTE


def test_handoff_names_open_recurrences_until_they_are_decided():
    service = make_service()
    _closed_bearing_case(service)
    service.run_case(
        CASES["EXC-101"].model_copy(
            update={"case_id": "REC-1", "reported_at": "2026-10-17T06:12:00Z"}
        )
    )
    payload = build_handoff(service.store)
    assert [e["case_id"] for e in payload["recurrences"]] == ["REC-1"]
    entry = payload["recurrences"][0]
    assert entry["prior_case_id"] == "EXC-101"
    assert entry["days_between"] == 6
    assert entry["departed"] is False
    text = render_text(payload)
    assert "RECURRING FAULTS AT THE GATE" in text
    assert "confirmed in case EXC-101" in text
    # Decided, the case leaves the open list — the handoff is the
    # board as it stands, not a history of links.
    service.record_decision("REC-1", "approved", by="Test Supervisor")
    assert build_handoff(service.store)["recurrences"] == []


def test_linked_gap_case_counts_once_in_the_gap_report():
    # FEED-9 is not on the machine register: its cases exhaust at
    # unknown_fault (the golden G16 shape). The first closes with
    # the fault confirmed; the second links to it — and the gap
    # report counts each record once: two blocked cases, not
    # three, and no extra row for the link.
    feeder = ExceptionCase(
        case_id="GAP-P",
        type="machine_down",
        title="FEED-9 feeder drive fault, no response",
        description="Feeder unit unresponsive; no readings available.",
        machine_id="FEED-9",
        line_id="line-a",
        product="Assembly kit AK-12",
        batch="B-4601",
        notes="",
        reporter="floor-lead",
        reported_at="2026-10-05T06:00:00Z",
        downtime_minutes=30,
    )
    service = make_service()
    first = service.run_case(feeder).result
    assert first.exit_reason == "exhausted"
    assert first.diagnosis.top().fault_class == "unknown_fault"
    service.record_outcome(
        "GAP-P",
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault="unknown_fault",
            work_performed="Feeder drive replaced by the vendor's crew.",
            resolved=True,
        ),
    )
    second = service.run_case(
        feeder.model_copy(
            update={"case_id": "GAP-Q", "reported_at": "2026-10-09T06:00:00Z"}
        )
    ).result
    assert second.recurrence is not None
    assert second.recurrence.prior_case_id == "GAP-P"
    report = build_gap_report(service.store)
    assert len(report.gaps) == 1
    row = report.gaps[0]
    assert row.machine_id == "FEED-9" and row.fault_class == "unknown_fault"
    assert row.blocked_cases == 2
    assert row.case_ids == ["GAP-P", "GAP-Q"]
    assert report.blocked_cases == 2
