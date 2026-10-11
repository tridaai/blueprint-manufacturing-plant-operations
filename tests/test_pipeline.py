"""Service-level pipeline: gate, decisions, board persistence, failures."""

from __future__ import annotations

import pytest

from plant_loop.backends import OfflineReasoner
from plant_loop.errors import ProviderError
from plant_loop.loops import BoardPolicy
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import CaseResult, DraftInfo, ExceptionCase

from helpers import make_service

CASES = {c.case_id: c for c in load_sample_cases()}


class BrokenReasoner(OfflineReasoner):
    def diagnose(self, ctx, focus):
        raise ProviderError("muse: provider exploded (HTTP 500)")


def test_run_case_reaches_the_gate_and_persists():
    service = make_service()
    run = service.run_case(CASES["EXC-105"])
    assert run.awaiting_supervisor is True
    record = run.result
    assert record.status == "awaiting_supervisor"
    assert record.exit_reason == "converged"
    assert record.draft.kind == "work_order"
    assert "never issues commands to machines" in record.gate_notice
    assert service.get_record("EXC-105") is not None
    assert service.objects.get("cases/EXC-105.json") is not None  # raw payload archived


def test_decision_resumes_the_gate_and_is_terminal():
    service = make_service()
    service.run_case(CASES["EXC-101"])
    record = service.record_decision("EXC-101", "approved", by="A. Supervisor", reason="sound plan")
    assert record.status == "approved"
    assert record.decision_by == "A. Supervisor"
    assert service.get_record("EXC-101").status == "approved"
    with pytest.raises(ValueError):
        service.record_decision("EXC-101", "rejected")


def test_decision_applies_directly_when_no_gate_is_paused():
    service = make_service()
    stored = CaseResult(
        case_id="EXT-1", type="machine_down", title="Stored elsewhere", machine_id="CNC-3",
        line_id="line-b", status="awaiting_supervisor", exit_reason="converged",
        draft=DraftInfo(kind="work_order", text="draft"),
        case=ExceptionCase(case_id="EXT-1", type="machine_down", title="Stored elsewhere"),
    )
    service.store.save_case(stored)
    record = service.record_decision("EXT-1", "rejected", by="B. Supervisor")
    assert record.status == "rejected"
    with pytest.raises(KeyError):
        service.record_decision("NOPE", "approved")


def test_board_run_persists_records_and_the_run():
    service = make_service()
    result = service.run_board(load_sample_cases(), board_policy=BoardPolicy(max_cases=4))
    assert result.processed == ["EXC-102", "EXC-107", "EXC-105", "EXC-101"]
    assert result.status == "shift_budget_exceeded"
    assert service.get_record("EXC-101") is not None
    saved = service.get_board_run(result.run_id)
    assert saved is not None and saved["status"] == "shift_budget_exceeded"
    assert saved["cases"][0]["case_id"] == "EXC-102"
    board = service.board()
    assert [row["case_id"] for row in board] == ["EXC-102", "EXC-107", "EXC-105", "EXC-101"]
    metrics = service.metrics()
    assert metrics["cases"] == 4
    assert metrics["by_exit"] == {"safety_stop": 1, "blocked_external": 1, "converged": 2}
    assert metrics["safety_stops"] == 1
    assert metrics["awaiting_supervisor"] == 4


def test_full_board_clears_and_every_case_is_gated():
    service = make_service()
    result = service.run_board(load_sample_cases())
    assert result.status == "board_clear"
    assert len(result.processed) == 9
    assert all(service.get_record(cid).status == "awaiting_supervisor" for cid in result.processed)


def test_provider_failure_is_logged_and_raised_not_recorded():
    service = make_service(reasoner=BrokenReasoner())
    with pytest.raises(ProviderError):
        service.run_case(CASES["EXC-105"])
    assert service.get_record("EXC-105") is None
    errors = service.store.all_errors()
    assert len(errors) == 1 and errors[0]["case_id"] == "EXC-105"
    assert service.metrics()["failed_runs"] == 1
