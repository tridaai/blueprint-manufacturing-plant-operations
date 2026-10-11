"""HTTP surface: console, probes, case run, board run, the gate, metrics."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from plant_loop import api
from plant_loop.samples import load_sample_cases

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}


@pytest.fixture()
def client():
    service = make_service()
    api.set_service(service)
    try:
        yield TestClient(api.app)
    finally:
        api.set_service(None)


def test_health_and_console(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "Plant Operations Loop Agent" in page.text


def test_readiness_is_honest_about_test_doubles(client):
    body = client.get("/readiness").json()
    assert body["ready"] is True
    assert "TEST DOUBLE" in body["store"]


def test_analyze_then_decide_over_http(client):
    response = client.post("/cases/analyze", json=CASES["EXC-101"].model_dump())
    assert response.status_code == 201
    record = response.json()
    assert record["exit_reason"] == "converged"
    assert record["status"] == "awaiting_supervisor"
    assert len(record["rounds"]) == 2
    assert record["draft"]["kind"] == "work_order"
    fetched = client.get("/cases/EXC-101")
    assert fetched.status_code == 200
    decided = client.post("/cases/EXC-101/decision", json={"decision": "approved", "by": "A. Supervisor"})
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"
    again = client.post("/cases/EXC-101/decision", json={"decision": "rejected"})
    assert again.status_code == 409


def test_decision_validation_and_unknown_case(client):
    assert client.post("/cases/EXC-101/decision", json={"decision": "maybe"}).status_code == 422
    assert client.get("/cases/NOPE").status_code == 404
    assert client.post("/cases/NOPE/decision", json={"decision": "approved"}).status_code == 404


def test_board_run_over_http_with_budget(client):
    response = client.post("/board/run", json={"max_cases": 2})
    assert response.status_code == 201
    body = response.json()
    assert body["processed"] == ["EXC-102", "EXC-107"]
    assert body["status"] == "shift_budget_exceeded"
    assert len(body["remaining"]) == 7
    assert "handoff" not in body  # field is handoff_summary
    assert body["handoff_summary"]
    board = client.get("/board").json()
    assert [row["case_id"] for row in board] == ["EXC-102", "EXC-107"]
    run = client.get(f"/board/runs/{body['run_id']}")
    assert run.status_code == 200


def test_samples_metrics_and_evals_endpoints(client):
    assert len(client.get("/samples").json()) == 9
    client.post("/cases/analyze", json=CASES["EXC-102"].model_dump())
    metrics = client.get("/metrics").json()
    assert metrics["cases"] == 1 and metrics["safety_stops"] == 1
    evals = client.get("/evals/results")
    assert evals.status_code == 200


def test_api_key_gate_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.get("/board").status_code == 401
    assert client.get("/board", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # probes stay public
