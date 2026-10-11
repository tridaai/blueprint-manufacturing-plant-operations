"""Muse Spark adapters, proven against stubbed transports.

No live Model API key exists in this environment, so the adapters
are verified by shaping: fake clients record exactly what the
adapter sends and return documented-shape responses. What is proven
here: the endpoint/model defaults, the Chat Completions request
shape, usage accounting, the Responses request shape, and — the
point of the Responses surface — reasoning continuity: the
encrypted reasoning item from round 1 is sent back on round 2, so
the loop's later rounds keep the earlier rounds' thread.

Live verification against api.meta.ai is pending a Model API key
(see the README's honest-status note); these tests are the
pre-live evidence.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from plant_loop.backends import MuseResponsesReasoner, MuseSparkReasoner, estimate_cost_usd
from plant_loop.errors import ProviderError
from plant_loop.ports import CaseContext
from plant_loop.samples import load_sample_cases
from plant_loop.schemas import Diagnosis

CASE = next(c for c in load_sample_cases() if c.case_id == "EXC-105")

DIAGNOSIS_JSON = json.dumps({
    "hypotheses": [{
        "fault_class": "hydraulic_pressure_loss",
        "statement": "Seal bypass fits the readings.",
        "confidence": 0.8,
        "evidence": ["ram pressure drop 14%"],
        "sop_id": "SOP-PRESS1-HYD",
    }],
    "summary": "one hypothesis",
})
PLAN_JSON = json.dumps({
    "steps": [
        {"step_id": "assess", "text": "Measure the pressure drop under load."},
        {"step_id": "loto-press1", "text": "Lock out the press at the main isolator and prove dead.",
         "kind": "safety", "mandatory": True},
    ],
    "required_parts": [],
    "estimated_downtime_min": 60,
    "summary": "plan",
})
CRITIQUE_JSON = json.dumps({"gaps": []})


def _content_for(prompt_text: str) -> str:
    # The phase prompts are told apart by their labelled sections:
    # the critique prompt carries the PLAN block, the plan prompt
    # carries the DIAGNOSIS block, the diagnose prompt neither.
    if "\nPLAN:\n" in prompt_text:
        return CRITIQUE_JSON
    if "\nDIAGNOSIS:\n" in prompt_text:
        return PLAN_JSON
    return DIAGNOSIS_JSON


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][-1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_content_for(user)))],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=9),
        )


class FakeChatClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["input"][-1]["content"]
        return SimpleNamespace(
            output=[SimpleNamespace(type="reasoning", id="rs_1", encrypted_content="enc-xyz-123")],
            output_text=_content_for(user),
            usage=SimpleNamespace(input_tokens=17, output_tokens=5),
        )


class FakeResponsesClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def _ctx() -> CaseContext:
    return CaseContext(case=CASE, case_text=f"{CASE.title}\n{CASE.description}")


# ---------------------------------------------------------------------------
# Chat Completions surface
# ---------------------------------------------------------------------------


def test_muse_chat_adapter_defaults_to_meta_endpoint_and_spark():
    adapter = MuseSparkReasoner(api_key="test-key", client=FakeChatClient())
    assert adapter.name == "muse"
    assert adapter.base_url == "https://api.meta.ai/v1"
    assert adapter.model == "muse-spark-1.3"


def test_muse_chat_request_shape_and_usage():
    client = FakeChatClient()
    adapter = MuseSparkReasoner(api_key="test-key", client=client)
    out = adapter.diagnose(_ctx(), focus=[])
    assert isinstance(out.payload, Diagnosis)
    assert out.input_tokens == 13 and out.output_tokens == 9
    call = client.chat.completions.calls[0]
    assert call["model"] == "muse-spark-1.3"
    assert call["temperature"] == 0
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert "PRESS-1 ram pressure sagging under load" in call["messages"][1]["content"]


def test_muse_chat_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ProviderError) as excinfo:
        MuseSparkReasoner()
    assert "MODEL_API_KEY" in str(excinfo.value)
    assert "dev.meta.ai" in str(excinfo.value)


def test_muse_env_overrides_model_and_base(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k")
    monkeypatch.setenv("MUSE_MODEL", "muse-spark-1.2")
    monkeypatch.setenv("MODEL_API_BASE_URL", "https://api.meta.ai/v1/")
    adapter = MuseSparkReasoner(client=FakeChatClient())
    assert adapter.model == "muse-spark-1.2"
    assert adapter.base_url == "https://api.meta.ai/v1"  # trailing slash trimmed


# ---------------------------------------------------------------------------
# Responses surface + reasoning continuity
# ---------------------------------------------------------------------------


def test_responses_adapter_name_and_include():
    client = FakeResponsesClient()
    adapter = MuseResponsesReasoner(api_key="test-key", client=client)
    assert adapter.name == "muse-responses"
    adapter.diagnose(_ctx(), focus=[])
    call = client.responses.calls[0]
    assert call["model"] == "muse-spark-1.3"
    assert call["include"] == ["reasoning.encrypted_content"]
    assert call["input"][-1]["role"] == "user"


def test_responses_reasoning_continuity_across_rounds():
    client = FakeResponsesClient()
    adapter = MuseResponsesReasoner(api_key="test-key", client=client)
    ctx = _ctx()
    diagnosis = adapter.diagnose(ctx, focus=[]).payload
    # Round 1 stored the encrypted reasoning item on the case session.
    assert ctx.session["reasoning_items"] == [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-xyz-123"}
    ]
    plan = adapter.draft_plan(ctx, diagnosis, focus=[]).payload
    assert plan.steps  # plan parsed on round 2
    second_call = client.responses.calls[1]
    # …and round 2 sent that item back FIRST, ahead of the new prompt:
    # the loop keeps its thread instead of restarting its reasoning.
    assert second_call["input"][0] == {
        "type": "reasoning", "id": "rs_1", "encrypted_content": "enc-xyz-123",
    }
    assert second_call["input"][-1]["role"] == "user"
    assert "\nDIAGNOSIS:\n" in second_call["input"][-1]["content"]


def test_cost_table_uses_published_spark_rates():
    # $1.25/M input + $4.25/M output (Meta Model API, standard tier).
    assert estimate_cost_usd("muse-spark-1.3", 1_000_000, 1_000_000) == pytest.approx(5.50)
    assert estimate_cost_usd("muse-spark-1.3", 0, 0) == 0.0
    assert estimate_cost_usd("some-unknown-model", 100, 100) is None


def test_factory_builds_muse_when_key_present(monkeypatch):
    pytest.importorskip("openai")
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    from plant_loop.backends import get_reasoner

    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    reasoner = get_reasoner("muse")
    assert reasoner.name == "muse" and reasoner.model == "muse-spark-1.3"
    monkeypatch.setenv("MODEL_SURFACE", "responses")
    reasoner = get_reasoner("muse")
    assert reasoner.name == "muse-responses"
