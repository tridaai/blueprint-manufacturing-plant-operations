"""Ports: the seams of the agent, declared in one place.

The loops depend on replaceable collaborators, each a
``typing.Protocol`` (structural — an implementation qualifies by
having the methods, not by inheriting):

- :class:`Reasoner` — the model seam. Three round-shaped calls
  (diagnose / draft_plan / critique), each returning a validated
  payload plus token usage and latency. Implementations live in
  ``backends.py``: the Muse Spark adapters (the lead profile), a
  general LLM adapter, and the deterministic offline rules.
- :class:`Embedder` / :class:`VectorIndex` / :class:`Retriever` —
  SOP + history retrieval for the diagnose round. Implementations
  live in ``embeddings.py`` and ``kb.py``.
- :class:`Store` — case records and board runs (PostgreSQL in
  production, in-memory as a test double). In ``store.py``.
- :class:`ObjectStore` — raw case payloads behind object keys
  (S3-compatible in production). In ``objectstore.py``.
- :class:`EventSink` — structured run events. In ``events.py``.
- :class:`Checkpointer` — LangGraph graph-state persistence behind
  the supervisor gate (PostgreSQL saver in production).

``wiring.py`` is the composition root; everything else programs
against these protocols. This module imports the schema models only,
so anything may import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Protocol, Sequence

if TYPE_CHECKING:  # annotation-only imports (no runtime cycles)
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
    from langchain_core.runnables import RunnableConfig

    from .events import RunEvent
    from .store import CaseRecord

from .schemas import ActionPlan, BoardItem, Critique, Diagnosis, ExceptionCase, Gap


@dataclass
class CaseContext:
    """Everything a reasoning round may look at for one case.

    ``case_text`` is the screened, fenced state text (floor notes with
    flagged sentences removed). ``session`` is an opaque per-case dict
    the loop owns and hands to every round: the Responses-API adapter
    uses it to carry Muse Spark's reasoning continuity across rounds;
    other backends ignore it.
    """

    case: ExceptionCase
    case_text: str
    kb_context: list[dict] = field(default_factory=list)
    session: dict = field(default_factory=dict)


@dataclass
class RoundOutput:
    """One reasoning round's validated payload + its cost."""

    payload: Any  # Diagnosis | ActionPlan | Critique, per the phase
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    backend: str = ""
    model: str = ""


class Reasoner(Protocol):
    """The reasoning seam: three round-shaped calls.

    Implementations must raise ``ProviderError`` on any failure —
    never return a partial or silently-degraded answer, and never
    substitute another backend. A case record always names the
    backend that produced it.
    """

    name: str
    model: str

    def diagnose(self, ctx: CaseContext, focus: list[Gap]) -> RoundOutput: ...

    def draft_plan(self, ctx: CaseContext, diagnosis: Diagnosis, focus: list[Gap]) -> RoundOutput: ...

    def critique(self, ctx: CaseContext, diagnosis: Diagnosis, plan: ActionPlan) -> RoundOutput: ...


class Embedder(Protocol):
    """The embedding seam (SOP / history retrieval)."""

    name: str
    model: str

    def embed(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    """Chunk vectors with nearest-neighbour search."""

    def add(self, chunk_id: str, vector: list[float], model: str) -> None: ...

    def nearest(
        self, vector: list[float], *, exclude_id: str | None = None, limit: int = 5
    ) -> list[tuple[str, float]]: ...


class Retriever(Protocol):
    """Query → the KB chunks a diagnose round should see."""

    def retrieve(self, query: str, *, limit: int = 5) -> list[dict]: ...


class Store(Protocol):
    """The record seam: case records and board runs, in and out."""

    def save_case(self, record: "CaseRecord") -> None: ...

    def get_case(self, case_id: str) -> "CaseRecord | None": ...

    def all_cases(self) -> list["CaseRecord"]: ...

    def save_board_run(self, run: dict) -> None: ...

    def get_board_run(self, run_id: str) -> dict | None: ...

    def all_board_runs(self) -> list[dict]: ...

    def save_board_item(self, item: "BoardItem") -> None: ...

    def get_board_item(self, item_id: str) -> "BoardItem | None": ...

    def all_board_items(self) -> list["BoardItem"]: ...

    def save_error(self, entry: dict) -> None: ...

    def all_errors(self) -> list[dict]: ...


class ObjectStore(Protocol):
    """The document seam: raw case payloads behind object keys,
    never local paths (S3-compatible in production; an in-memory
    double for tests). Implementations in ``objectstore.py``."""

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str: ...

    def get(self, key: str) -> bytes | None: ...

    def exists(self, key: str) -> bool: ...


class EventSink(Protocol):
    """The run-events seam: one ``emit`` per structured event."""

    def emit(self, event: "RunEvent") -> None: ...


class Checkpointer(Protocol):
    """The graph-state seam behind the supervisor gate.

    Mirrors the sync surface of LangGraph's ``BaseCheckpointSaver``;
    the shipped selections (Postgres saver / memory saver) satisfy it
    by construction.
    """

    def get_tuple(self, config: "RunnableConfig") -> "CheckpointTuple | None": ...

    def list(
        self,
        config: "RunnableConfig | None",
        *,
        filter: dict[str, Any] | None = None,
        before: "RunnableConfig | None" = None,
        limit: int | None = None,
    ) -> Iterator["CheckpointTuple"]: ...

    def put(
        self,
        config: "RunnableConfig",
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: dict[str, Any],
    ) -> "RunnableConfig": ...

    def put_writes(
        self,
        config: "RunnableConfig",
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None: ...

    def delete_thread(self, thread_id: str) -> None: ...
