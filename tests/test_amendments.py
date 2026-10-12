"""Gate amendments: the supervisor edits, the machine re-checks.

Pins, by layer:

- **the passing swap** — verified in code, the claim moves with
  it (old part released ``released_by_amendment``, new part
  claimed), the effective plan draws the new part with the
  amendment marked, the packet prints the swap as the
  supervisor's, and approval then reads the amended plan;
- **the incompatible swap** — the seeded fan pairing (450 → 380
  against the SOP's 420 floor): AMENDMENT_FAILED with the spec
  delta and the constraint on the record, approval refused while
  it stands, rejection still open, no claim moved;
- **the claim conflict** — a swap onto a part another case holds
  fails naming the holder, and passes once the holder's claim is
  released at its own gate;
- **strikes** — a mandatory safety step is refused by name at
  recording (nothing is appended), a non-mandatory step strikes
  clean and is quoted on the amendment;
- **recording refusals** — free-text part, anonymous, reasonless,
  a decided case, a safety stop, a swap to itself, a part the
  plan does not draw, a step the plan does not have;
- **no model in re-verification** — a counting reasoner proves
  the amendment path makes zero reasoning calls;
- **the record's honesty** — the loop's original plan is
  byte-identical after every amendment; the stats count amended
  approvals as amended; the handoff names amended cases; a
  carried case keeps its amendment and its verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plant_loop import amendments as amendments_module
from plant_loop import api
from plant_loop.backends import OfflineReasoner
from plant_loop.schemas import ExceptionCase

from helpers import make_service

SAMPLES = {
    c["case_id"]: c
    for c in json.loads(
        (Path(__file__).parent.parent / "src/plant_loop/data/sample_cases.json").read_text()
    )
}

SUPERVISOR = "R. Alvarez"


def run(service, case_id):
    return service.run_case(ExceptionCase.model_validate(SAMPLES[case_id])).result


def swap(service, case_id, from_part, to_part, **kw):
    return service.amend_case(
        case_id,
        kind="part_swap",
        by=kw.get("by", SUPERVISOR),
        reason=kw.get("reason", "the supervisor's stated reason"),
        from_part_id=from_part,
        to_part_id=to_part,
    )


def strike(service, case_id, step_id, **kw):
    return service.amend_case(
        case_id,
        kind="step_strike",
        by=kw.get("by", SUPERVISOR),
        reason=kw.get("reason", "the supervisor's stated reason"),
        step_id=step_id,
    )


def claims_by_part(service, case_id):
    return {
        c.part_id: c for c in service.allocation.claims_for_case(case_id)
    }


# -- the passing swap ----------------------------------------------------------


def test_a_passing_swap_verifies_moves_the_claim_and_marks_the_plan():
    service = make_service()
    record = run(service, "EXC-103")  # weld drift, draws TIP-08
    assert record.exit_reason == "converged"
    original_plan = record.plan.model_dump_json()

    record = swap(
        service, "EXC-103", "TIP-08", "LINER-045",
        reason="tips inspected — the liner is the worn consumable",
    )
    amendment = record.amendments[-1]
    assert amendment.verification.status == "passed"
    assert amendment.slot_part_id == "TIP-08"
    assert amendment.from_part_id == "TIP-08"
    assert amendment.to_part_id == "LINER-045"
    # An unlisted pairing says so on its face.
    assert "not listed in the KB compatibility relation" in amendment.basis

    # The claim followed the verified plan.
    claims = claims_by_part(service, "EXC-103")
    assert claims["TIP-08"].status == "released"
    assert claims["TIP-08"].release_reason == "released_by_amendment"
    assert claims["TIP-08"].released_by == SUPERVISOR
    assert claims["LINER-045"].status == "claimed"
    assert claims["LINER-045"].original_part_id == "TIP-08"

    # The effective plan draws the new part, marked; the original
    # plan on the record is byte-identical.
    effective = amendments_module.apply_passing_amendments(record)
    req = effective.required_parts[0]
    assert req.substitute_part_id == "LINER-045" and req.satisfied
    step = next(s for s in effective.steps if s.step_id == "replace-tips")
    assert step.part_id == "LINER-045"
    assert "AMENDED AT THE GATE" in step.text
    assert record.plan.model_dump_json() == original_plan

    # Approval reads the amended plan's verification — and the
    # amendments survive the decision (the resume path included).
    record = service.record_decision("EXC-103", "approved", by=SUPERVISOR)
    assert record.status == "approved"
    assert len(record.amendments) == 1

    stats = service.diagnosis_stats()["amendments"]
    assert stats["amended_cases"] == 1
    assert stats["approved_after_amendment"] == 1
    assert stats["verification_passed"] == 1


def test_a_swap_back_to_the_slots_own_part_revives_its_claim():
    service = make_service()
    run(service, "EXC-103")
    swap(service, "EXC-103", "TIP-08", "LINER-045")
    record = swap(
        service, "EXC-103", "LINER-045", "TIP-08",
        reason="a fresh box of tips reached the cell — back to the plan's own part",
    )
    amendment = record.amendments[-1]
    assert amendment.verification.status == "passed"
    assert "the slot's own part" in amendment.basis
    claims = claims_by_part(service, "EXC-103")
    assert claims["TIP-08"].status == "claimed"
    assert claims["LINER-045"].status == "released"
    assert claims["LINER-045"].release_reason == "released_by_amendment"
    effective = amendments_module.apply_passing_amendments(record)
    assert effective.required_parts[0].substitute_part_id == ""


def test_a_kb_listed_swap_carries_the_authored_basis_and_deltas():
    service = make_service()
    run(service, "EXC-106")  # the fan case: blocked, FAN-C7 unsatisfied
    record = swap(service, "EXC-106", "FAN-C7", "FAN-C7-G")
    amendment = record.amendments[-1]
    # The pairing is the KB's own stopgap listing — its authored
    # basis and the airflow delta ride the amendment even though
    # the verification then refuses the pairing on the constraint.
    assert "stopgap" in amendment.basis
    deltas = {d.attribute: d for d in amendment.spec_deltas}
    assert deltas["airflow_m3h"].original == 450
    assert deltas["airflow_m3h"].substitute == 380


# -- the incompatible swap -------------------------------------------------------


def test_an_incompatible_swap_fails_with_deltas_named_and_blocks_approval():
    service = make_service()
    run(service, "EXC-106")
    claims_before = [
        c.model_dump_json() for c in service.allocation.all_claims()
    ]
    record = swap(
        service, "EXC-106", "FAN-C7", "FAN-C7-G",
        by="J. Okafor", reason="try the generic fan from stores",
    )
    amendment = record.amendments[-1]
    assert amendment.verification.status == "failed"
    assert amendments_module.amendment_state(record) == "amendment_failed"
    kinds = {g.kind for g in amendment.verification.gaps}
    assert "incompatible_part" in kinds
    detail = amendment.verification.detail
    assert "380" in detail and "420" in detail

    # A failed amendment moves no claim and changes no plan.
    assert [
        c.model_dump_json() for c in service.allocation.all_claims()
    ] == claims_before
    effective = amendments_module.apply_passing_amendments(record)
    assert effective.required_parts[0].substitute_part_id == ""

    # Approval is refused over the failure — naming it — while
    # rejection stays open.
    with pytest.raises(ValueError, match="FAILED re-verification"):
        service.record_decision("EXC-106", "approved", by="J. Okafor")
    record = service.record_decision(
        "EXC-106", "rejected", by="J. Okafor", reason="no compliant fan on the shelf"
    )
    assert record.status == "rejected"
    assert len(record.amendments) == 1  # the failed attempt stays on the record


def test_a_failed_amendment_can_be_followed_by_a_passing_one():
    service = make_service()
    run(service, "EXC-103")
    # Onto a part the shelf cannot cover: SEAL-BAR x1 is claimed by
    # no one here, but PRT-HEAD's single unit is free too — use the
    # fan's own failure shape instead: swap onto an unstocked slot
    # part (FAN-C7, qty 0) fails on availability.
    record = swap(service, "EXC-103", "TIP-08", "FAN-C7")
    assert record.amendments[-1].verification.status == "failed"
    with pytest.raises(ValueError, match="FAILED re-verification"):
        service.record_decision("EXC-103", "approved", by=SUPERVISOR)
    record = swap(service, "EXC-103", "TIP-08", "LINER-045")
    assert record.amendments[-1].verification.status == "passed"
    record = service.record_decision("EXC-103", "approved", by=SUPERVISOR)
    assert record.status == "approved"


# -- the claim conflict ------------------------------------------------------------


def _printer_case(case_id="AM-PRT"):
    return {
        "case_id": case_id,
        "type": "machine_down",
        "title": "PACK-6 label printer head failed",
        "description": "The label printer stopped mid-run; the head is dead.",
        "machine_id": "PACK-6",
        "line_id": "line-e",
        "product": "Retail cartons",
        "batch": "B-2210",
        "readings": {"printer_head_temp_c": 24.0},
        "notes": "Operator: no print at all since 05:40.",
        "reporter": "S. Whitfield (operator)",
        "reported_at": "2026-10-11T05:44:00Z",
        "downtime_minutes": 25,
    }


def test_a_swap_onto_a_claimed_part_fails_naming_the_holder_then_passes():
    service = make_service()
    printer = service.run_case(ExceptionCase.model_validate(_printer_case())).result
    assert printer.exit_reason == "converged"  # claims the only PRT-HEAD
    run(service, "EXC-108")  # seal case, draws SEAL-BAR

    record = swap(service, "EXC-108", "SEAL-BAR", "PRT-HEAD")
    amendment = record.amendments[-1]
    assert amendment.verification.status == "failed"
    assert "AM-PRT" in amendment.verification.detail

    # The holder is rejected at its own gate; its claim releases.
    service.record_decision("AM-PRT", "rejected", by=SUPERVISOR, reason="printer is being replaced whole")
    record = swap(service, "EXC-108", "SEAL-BAR", "PRT-HEAD")
    assert record.amendments[-1].verification.status == "passed"
    claims = claims_by_part(service, "EXC-108")
    assert claims["PRT-HEAD"].status == "claimed"
    assert claims["SEAL-BAR"].release_reason == "released_by_amendment"


# -- strikes -----------------------------------------------------------------------


def test_a_mandatory_safety_step_cannot_be_struck():
    service = make_service()
    run(service, "EXC-108")
    with pytest.raises(ValueError, match="mandatory safety step"):
        strike(service, "EXC-108", "safety-cool-pack6")
    # The refusal names the step and its SOP — and appends nothing.
    record = service.get_record("EXC-108")
    assert record.amendments == []


def test_a_non_mandatory_step_strikes_clean_and_is_quoted():
    service = make_service()
    record = run(service, "EXC-103")
    original_plan = record.plan.model_dump_json()
    record = strike(
        service, "EXC-103", "gas-check",
        reason="gas flow was verified at the last PM, two days ago",
    )
    amendment = record.amendments[-1]
    assert amendment.verification.status == "passed"
    assert "Measure shielding-gas flow" in amendment.step_text
    effective = amendments_module.apply_passing_amendments(record)
    assert "gas-check" not in [s.step_id for s in effective.steps]
    # Every mandatory safety step is still there.
    assert "safety-isolate-weld4" in [s.step_id for s in effective.steps]
    assert record.plan.model_dump_json() == original_plan


# -- recording refusals -------------------------------------------------------------


def test_recording_refusals_name_themselves_and_write_nothing():
    service = make_service()
    run(service, "EXC-102")  # safety stop — no plan at all
    run(service, "EXC-103")
    service.record_decision("EXC-103", "approved", by=SUPERVISOR)
    run(service, "EXC-108")

    with pytest.raises(KeyError):
        swap(service, "EXC-999", "TIP-08", "LINER-045")
    with pytest.raises(ValueError, match="safety stop"):
        strike(service, "EXC-102", "anything")
    with pytest.raises(ValueError, match="not awaiting|still awaiting"):
        strike(service, "EXC-103", "test-welds")  # already decided
    with pytest.raises(ValueError, match="must name its supervisor"):
        swap(service, "EXC-108", "SEAL-BAR", "PRT-HEAD", by=" ")
    with pytest.raises(ValueError, match="must carry the supervisor's reason"):
        swap(service, "EXC-108", "SEAL-BAR", "PRT-HEAD", reason="")
    with pytest.raises(ValueError, match="not on the plant's parts register"):
        swap(service, "EXC-108", "SEAL-BAR", "UNOBTAINIUM-9")
    with pytest.raises(ValueError, match="draws no part"):
        swap(service, "EXC-108", "TIP-08", "PRT-HEAD")
    with pytest.raises(ValueError, match="for itself changes nothing"):
        swap(service, "EXC-108", "SEAL-BAR", "SEAL-BAR")
    with pytest.raises(ValueError, match="has no step"):
        strike(service, "EXC-108", "no-such-step")
    assert service.get_record("EXC-108").amendments == []


def test_an_amendment_on_an_exhausted_case_amends_its_plan():
    # The gate holds more than converged cases: an exhausted case's
    # plan is on the record, and the supervisor may amend it under
    # the same re-verification — the scope line is the safety stop
    # (no plan), not the loop's exit.
    service = make_service()
    record = run(service, "EXC-109")
    assert record.exit_reason == "exhausted"
    record = strike(
        service, "EXC-109", "check-motor_overheat",
        reason="the motor was megger-tested at the last stop — the check is spent",
    )
    assert record.amendments[-1].verification.status == "passed"


# -- no model in re-verification ------------------------------------------------------


class CountingReasoner(OfflineReasoner):
    name = "offline"

    def __init__(self):
        self.calls = {"diagnose": 0, "plan": 0, "critique": 0}

    def diagnose(self, ctx, focus=None):
        self.calls["diagnose"] += 1
        return super().diagnose(ctx, focus=focus)

    def draft_plan(self, ctx, diagnosis, focus=None):
        self.calls["plan"] += 1
        return super().draft_plan(ctx, diagnosis, focus=focus)

    def critique(self, ctx, diagnosis, plan):
        self.calls["critique"] += 1
        return super().critique(ctx, diagnosis, plan)


def test_reverification_makes_no_model_calls():
    reasoner = CountingReasoner()
    service = make_service(reasoner=reasoner)
    run(service, "EXC-103")
    run(service, "EXC-106")
    before = dict(reasoner.calls)
    assert before["critique"] > 0  # the loop did consult the critic
    swap(service, "EXC-103", "TIP-08", "LINER-045")
    strike(service, "EXC-103", "gas-check")
    swap(service, "EXC-106", "FAN-C7", "FAN-C7-G")
    assert reasoner.calls == before


# -- surfaces ---------------------------------------------------------------------------


def test_the_packet_prints_the_amended_plan_with_the_swap_marked():
    service = make_service()
    run(service, "EXC-103")
    swap(
        service, "EXC-103", "TIP-08", "LINER-045",
        reason="tips inspected — the liner is the worn consumable",
    )
    packet = service.case_packet("EXC-103")
    text = packet["text"]
    assert "AMENDED AT THE GATE" in text
    assert "LINER-045 is fitted in place of TIP-08" in text
    assert "AMENDMENTS AT THE GATE" in text
    parts = {p["part_id"]: p for p in packet["parts"]}
    assert parts["TIP-08"]["amendment_id"] == "amd-EXC-103-1"
    assert parts["TIP-08"]["substitute_part_id"] == "LINER-045"


def test_the_packet_banner_names_a_failed_amendment():
    service = make_service()
    run(service, "EXC-103")
    swap(service, "EXC-103", "TIP-08", "FAN-C7")  # shelf 0 — fails
    packet = service.case_packet("EXC-103")
    assert "FAILED" in packet["banner"]
    assert "amd-EXC-103-1" in packet["banner"]


def test_the_handoff_names_amended_cases_and_failed_states():
    service = make_service()
    run(service, "EXC-103")
    swap(service, "EXC-103", "TIP-08", "LINER-045")
    swap(service, "EXC-103", "LINER-045", "FAN-C7")  # fails — shelf 0
    handoff = service.handoff()
    entries = {e["case_id"]: e for e in handoff["amended_cases"]}
    assert entries["EXC-103"]["amendment_state"] == "amendment_failed"
    assert len(entries["EXC-103"]["amendments"]) == 2
    assert "AMENDED AT THE GATE" in handoff["text"]
    assert "LATEST AMENDMENT FAILED" in handoff["text"]


def test_the_api_records_an_amendment_and_serves_the_effective_plan():
    service = make_service()
    api.set_service(service)
    client = TestClient(api.app)
    try:
        _api_flow(client)
    finally:
        api.set_service(None)


def _api_flow(client):
    resp = client.post("/cases/analyze", json=SAMPLES["EXC-103"])
    assert resp.status_code == 201
    resp = client.post(
        "/cases/EXC-103/amendment",
        json={
            "kind": "part_swap",
            "by": SUPERVISOR,
            "reason": "the liner is the worn consumable",
            "from_part_id": "TIP-08",
            "to_part_id": "LINER-045",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["amendments"][-1]["verification"]["status"] == "passed"
    resp = client.get("/cases/EXC-103")
    body = resp.json()
    assert body["amendment_state"] == "verified"
    assert body["effective_plan"]["required_parts"][0]["substitute_part_id"] == "LINER-045"
    assert body["plan"]["required_parts"][0]["substitute_part_id"] == ""
    # A mandatory strike is a 409 naming the refusal.
    resp = client.post(
        "/cases/EXC-103/amendment",
        json={
            "kind": "step_strike",
            "by": SUPERVISOR,
            "reason": "skip it",
            "step_id": "safety-isolate-weld4",
        },
    )
    assert resp.status_code == 409
    assert "mandatory safety step" in resp.json()["detail"]
    resp = client.post(
        "/cases/EXC-999/amendment",
        json={"kind": "step_strike", "by": SUPERVISOR, "reason": "x", "step_id": "y"},
    )
    assert resp.status_code == 404


def test_a_carried_amended_case_keeps_its_amendment_and_verification():
    service = make_service()
    case = ExceptionCase.model_validate(SAMPLES["EXC-103"])
    run1 = service.run_board([case])
    swap(service, "EXC-103", "TIP-08", "LINER-045")
    run2 = service.run_board([], carry_over_from=run1.run_id)
    assert "EXC-103" in run2.carry_over_settled.get("inherited_at_gate", [])
    record = service.get_record("EXC-103")
    assert record.status == "awaiting_supervisor"
    assert len(record.amendments) == 1
    assert record.amendments[0].verification.status == "passed"
    claims = claims_by_part(service, "EXC-103")
    assert claims["LINER-045"].status == "claimed"
    assert claims["LINER-045"].carried_from_run == run1.run_id
