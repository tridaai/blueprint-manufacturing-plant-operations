"""Round 3: the work-order packet.

The packet is the floor copy of a gated case — steps in order, the
mandatory safety steps as a check-off list, parts with substitutions
flagged, blank sign-off lines. These tests pin its contents against
the records (including the safety checklist against the SOP's own
list), and its one refusal: a case without a verified work order
gets no packet, because an unchecked plan must not reach the floor
dressed as one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from plant_loop import api, kb
from plant_loop.packet import build_packet, render_text
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import ExceptionCase

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}

COOLANT_CASE = {
    "case_id": "T-COOL",
    "type": "machine_down",
    "title": "CNC-3 coolant flow low, spindle hot",
    "description": "Coolant flow alarm; spindle temperature climbing.",
    "machine_id": "CNC-3",
    "line_id": "line-b",
    "readings": {"coolant_flow_l_min": 3.0, "spindle_temp_c": 84},
    "reporter": "J. Okafor (operator)",
    "reported_at": "2026-10-11T06:40:00Z",
}


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app), service
    finally:
        api.set_service(None)


def _run(service, case_id):
    return service.run_case(CASES[case_id]).result


# ---------------------------------------------------------------------------
# Contents
# ---------------------------------------------------------------------------


def test_packet_contents_for_a_converged_case():
    service = make_service()
    record = _run(service, "EXC-101")
    packet = service.case_packet("EXC-101")

    assert packet["authorised"] is False
    assert "NOT AUTHORISED" in packet["banner"]
    assert packet["case"]["machine_id"] == "CNC-3"
    assert packet["diagnosis"]["fault_class"] == record.diagnosis.top().fault_class
    assert packet["diagnosis"]["evidence"]  # evidence travels with the diagnosis
    # Steps are the plan's steps, in order, numbered from 1.
    assert [s["text"] for s in packet["steps"]] == [s.text for s in record.plan.steps]
    assert [s["n"] for s in packet["steps"]] == list(range(1, len(record.plan.steps) + 1))
    # The sign-off block is blank — the paper world fills it in.
    signoff = packet["signoff"]
    assert signoff["supervisor"] == "" and signoff["technician"] == ""
    assert signoff["actual_downtime_min"] is None and signoff["resolved"] is None


def test_packet_safety_checklist_is_exactly_the_sops_mandatory_list():
    service = make_service()
    record = _run(service, "EXC-101")
    packet = service.case_packet("EXC-101")
    sop = kb.sop_for("CNC-3", record.diagnosis.top().fault_class)
    mandatory = kb.mandatory_safety_steps(sop)
    assert [item["text"] for item in packet["safety_checklist"]] == [
        step["text"] for step in mandatory
    ]
    text = packet["text"]
    for step in mandatory:
        assert f"[ ] {step['text']}" in text  # each one is a check-off line


def test_packet_flags_the_substitution_in_json_and_text():
    service = make_service()
    service.run_case(ExceptionCase.model_validate(COOLANT_CASE))
    packet = service.case_packet("T-COOL")

    part = next(p for p in packet["parts"] if p["part_id"] == "CLNT-FLT")
    assert part["substituted"] is True and part["substitute_part_id"] == "CLNT-FLT-E"
    assert len(packet["substitutions"]) == 1
    substitution = packet["substitutions"][0]
    assert substitution["basis"] and substitution["spec_deltas"]

    text = packet["text"]
    assert "SUBSTITUTION" in text
    assert "CLNT-FLT-E is fitted in place of CLNT-FLT" in text
    assert "micron_rating_um 25 -> 10" in text  # the delta reaches the floor


def test_packet_text_is_rendered_from_the_same_dict():
    service = make_service()
    record = _run(service, "EXC-101")
    # The service packet carries the case's allocation claims
    # (round 5); build from the same claims and the text must match.
    payload = build_packet(record, claims=service.allocation.claims_for_case("EXC-101"))
    assert render_text(payload) == service.case_packet("EXC-101")["text"]


# ---------------------------------------------------------------------------
# The refusal, and the approved banner
# ---------------------------------------------------------------------------


def test_no_packet_for_a_case_without_a_verified_work_order():
    service = make_service()
    _run(service, "EXC-107")  # blocked_external — an escalation, not a work order
    with pytest.raises(ValueError) as excinfo:
        service.case_packet("EXC-107")
    assert "blocked_external" in str(excinfo.value)
    _run(service, "EXC-102")  # safety stop — round 0, no plan at all
    with pytest.raises(ValueError):
        service.case_packet("EXC-102")


def test_packet_after_approval_carries_the_approval():
    service = make_service()
    _run(service, "EXC-101")
    service.record_decision("EXC-101", "approved", by="A. Supervisor")
    packet = service.case_packet("EXC-101")
    assert packet["authorised"] is True
    assert "APPROVED AT THE GATE by A. Supervisor" in packet["banner"]
    assert "NOT AUTHORISED" not in packet["text"]


def test_no_packet_for_a_rejected_draft():
    service = make_service()
    _run(service, "EXC-101")
    service.record_decision("EXC-101", "rejected", by="A. Supervisor")
    with pytest.raises(ValueError) as excinfo:
        service.case_packet("EXC-101")
    assert "rejected" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------


def test_packet_over_http(client):
    http, service = client
    http.post("/cases/analyze", json=CASES["EXC-101"].model_dump())
    body = http.get("/api/cases/EXC-101/packet").json()
    assert body["case_id"] == "EXC-101"
    assert body["safety_checklist"] and "WORK ORDER PACKET" in body["text"]
    text_response = http.get("/api/cases/EXC-101/packet", params={"format": "text"})
    assert text_response.headers["content-type"].startswith("text/plain")
    assert text_response.text == body["text"]


def test_packet_http_status_codes(client):
    http, service = client
    assert http.get("/api/cases/NOPE/packet").status_code == 404
    http.post("/cases/analyze", json=CASES["EXC-107"].model_dump())
    response = http.get("/api/cases/EXC-107/packet")
    assert response.status_code == 409
    assert "blocked_external" in response.json()["detail"]
