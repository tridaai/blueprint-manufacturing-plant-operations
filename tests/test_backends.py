"""Reasoning backends: offline rules, the LLM contract, the factory."""

from __future__ import annotations

import json

import pytest

from plant_loop.backends import LLMReasoner, OfflineReasoner, get_reasoner, parse_json_object
from plant_loop.errors import ProviderError
from plant_loop.ports import CaseContext
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import Diagnosis

CASES = {c.case_id: c for c in load_sample_cases()}


def _ctx(case_id: str) -> CaseContext:
    case = CASES[case_id]
    return CaseContext(case=case, case_text=f"{case.title}\n{case.description}")


def test_offline_diagnosis_is_evidenced_and_ranked():
    out = OfflineReasoner().diagnose(_ctx("EXC-101"), focus=[])
    diagnosis = out.payload
    assert isinstance(diagnosis, Diagnosis)
    top = diagnosis.top()
    assert top.fault_class == "spindle_bearing_wear"
    assert top.evidence  # no hypothesis without evidence
    assert out.backend == "offline"
    assert out.input_tokens > 0


def test_offline_safety_case_still_diagnoses_if_asked():
    # The loop never asks (round-0 rule, tested in test_loops) — but the
    # backend itself is a pure function of its inputs.
    out = OfflineReasoner().diagnose(_ctx("EXC-103"), focus=[])
    assert out.payload.top().fault_class == "weld_quality_drift"


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('here you go: {"a": 1} done') == {"a": 1}
    assert parse_json_object("no json here") is None
    assert parse_json_object('[1, 2]') is None


_DIAGNOSIS_JSON = json.dumps({
    "hypotheses": [{
        "fault_class": "hydraulic_pressure_loss",
        "statement": "Seal bypass fits the readings.",
        "confidence": 0.8,
        "evidence": ["ram pressure drop 14%"],
        "sop_id": "SOP-PRESS1-HYD",
    }],
    "summary": "one hypothesis",
})


def test_llm_reasoner_validates_and_returns_payload():
    calls = []

    def complete(system, user, session):
        calls.append(user)
        return _DIAGNOSIS_JSON, {"input_tokens": 11, "output_tokens": 7}

    reasoner = LLMReasoner("openai", complete=complete, model="gpt-5-mini")
    out = reasoner.diagnose(_ctx("EXC-105"), focus=[])
    assert out.payload.top().fault_class == "hydraulic_pressure_loss"
    assert out.input_tokens == 11 and out.output_tokens == 7
    assert out.backend == "llm"


def test_llm_reasoner_retries_once_then_fails_loudly():
    attempts = {"n": 0}

    def garbage(system, user, session):
        attempts["n"] += 1
        return "not json at all", {"input_tokens": 1, "output_tokens": 1}

    reasoner = LLMReasoner("openai", complete=garbage, model="gpt-5-mini")
    with pytest.raises(ProviderError) as excinfo:
        reasoner.diagnose(_ctx("EXC-105"), focus=[])
    assert attempts["n"] == 2  # exactly one bounded retry
    assert "invalid answer twice" in str(excinfo.value)


def test_llm_reasoner_recovers_on_the_retry():
    replies = iter(["{broken", _DIAGNOSIS_JSON])

    def flaky(system, user, session):
        return next(replies), {"input_tokens": 3, "output_tokens": 3}

    reasoner = LLMReasoner("openai", complete=flaky, model="gpt-5-mini")
    out = reasoner.diagnose(_ctx("EXC-105"), focus=[])
    assert out.payload.top().fault_class == "hydraulic_pressure_loss"
    assert out.input_tokens == 6  # both attempts accounted


def test_factory_defaults_to_offline_and_rejects_unknown(monkeypatch):
    assert get_reasoner().name == "offline"
    assert get_reasoner("offline").name == "offline"
    with pytest.raises(ValueError):
        get_reasoner("carrier-pigeon")
