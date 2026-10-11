"""Shared builders for the suite: double-wired services and scripted reasoners."""

from __future__ import annotations

from plant_loop.backends import OfflineReasoner
from plant_loop.embeddings import HashingEmbedder, InMemoryVectorIndex
from plant_loop.kb import KBRetriever
from plant_loop.loops import CaseLoopController, LoopPolicy
from plant_loop.objectstore import InMemoryObjectStore
from plant_loop.ports import RoundOutput
from plant_loop.schemas import Critique, Gap
from plant_loop.service import PlantService
from plant_loop.store import InMemoryStore


def make_retriever() -> KBRetriever:
    return KBRetriever(HashingEmbedder(), InMemoryVectorIndex())


def make_service(reasoner=None, **service_kwargs) -> PlantService:
    return PlantService(
        reasoner=reasoner or OfflineReasoner(),
        retriever=make_retriever(),
        store=InMemoryStore(),
        objects=InMemoryObjectStore(),
        **service_kwargs,
    )


def make_controller(reasoner=None, policy: LoopPolicy | None = None) -> CaseLoopController:
    return CaseLoopController(
        reasoner=reasoner or OfflineReasoner(),
        retriever=make_retriever(),
        policy=policy or LoopPolicy(),
    )


class ScriptedCritiqueReasoner(OfflineReasoner):
    """Offline diagnose/plan, but the critique follows a script:
    a list of gap lists, one per verify round (the last repeats)."""

    name = "offline"

    def __init__(self, script: list[list[Gap]]) -> None:
        self._script = script
        self.critique_calls = 0

    def critique(self, ctx, diagnosis, plan) -> RoundOutput:
        index = min(self.critique_calls, len(self._script) - 1)
        self.critique_calls += 1
        gaps = self._script[index]
        return RoundOutput(
            payload=Critique(gaps=list(gaps)),
            input_tokens=10,
            output_tokens=5,
            latency_ms=0.1,
            backend=self.name,
            model=self.model,
        )


class CountingReasoner(OfflineReasoner):
    """Counts round calls (to prove safety cases make none)."""

    def __init__(self) -> None:
        self.calls = {"diagnose": 0, "plan": 0, "critique": 0}

    def diagnose(self, ctx, focus):
        self.calls["diagnose"] += 1
        return super().diagnose(ctx, focus)

    def draft_plan(self, ctx, diagnosis, focus):
        self.calls["plan"] += 1
        return super().draft_plan(ctx, diagnosis, focus)

    def critique(self, ctx, diagnosis, plan):
        self.calls["critique"] += 1
        return super().critique(ctx, diagnosis, plan)
