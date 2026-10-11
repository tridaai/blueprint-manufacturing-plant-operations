"""Board priority: doctrine in code, reproducible to the case."""

from __future__ import annotations

from plant_loop.priority import board_order, priority_for
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import ExceptionCase

CASES = {c.case_id: c for c in load_sample_cases()}


def test_safety_outranks_everything():
    safety, _ = priority_for(CASES["EXC-102"])
    for case_id, case in CASES.items():
        if case_id != "EXC-102":
            assert safety > priority_for(case)[0], case_id


def test_type_doctrine_order():
    def bare(type_: str) -> ExceptionCase:
        return ExceptionCase(case_id="X", type=type_, title="t", machine_id="CNC-3", line_id="line-e")

    scores = {t: priority_for(bare(t))[0] for t in (
        "safety_observation", "machine_down", "quality_nonconformance",
        "material_shortage", "maintenance_alert")}
    assert scores["safety_observation"] > scores["machine_down"]
    assert scores["machine_down"] > scores["quality_nonconformance"]
    assert scores["quality_nonconformance"] > scores["material_shortage"]
    assert scores["material_shortage"] > scores["maintenance_alert"]


def test_line_criticality_scales_and_downtime_adds():
    base = ExceptionCase(case_id="A", type="machine_down", title="t", machine_id="CNC-3",
                         line_id="line-e", downtime_minutes=0)
    crit = ExceptionCase(case_id="B", type="machine_down", title="t", machine_id="CNC-3",
                         line_id="line-b", downtime_minutes=0)
    down = ExceptionCase(case_id="C", type="machine_down", title="t", machine_id="CNC-3",
                         line_id="line-e", downtime_minutes=100)
    assert priority_for(crit)[0] > priority_for(base)[0]
    assert priority_for(down)[0] > priority_for(base)[0]
    # The reason string explains the arithmetic.
    assert "line criticality 3" in priority_for(crit)[1]
    assert "100 min already down" in priority_for(down)[1]


def test_sample_board_order_is_the_doctrine_order():
    ordered = [c.case_id for c in board_order(load_sample_cases())]
    assert ordered == [
        "EXC-102", "EXC-107", "EXC-105", "EXC-101", "EXC-109",
        "EXC-103", "EXC-108", "EXC-104", "EXC-106",
    ]


def test_ties_break_by_reported_time_then_id():
    a = ExceptionCase(case_id="B-2", type="maintenance_alert", title="t", reported_at="2026-10-11T07:00:00Z")
    b = ExceptionCase(case_id="A-1", type="maintenance_alert", title="t", reported_at="2026-10-11T06:00:00Z")
    c = ExceptionCase(case_id="A-0", type="maintenance_alert", title="t", reported_at="2026-10-11T06:00:00Z")
    assert [x.case_id for x in board_order([a, b, c])] == ["A-0", "A-1", "B-2"]
