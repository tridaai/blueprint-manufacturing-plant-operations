"""Service layer: runs the graphs, owns persistence and the gate.

The surfaces (API, CLI, demo, evals) talk to :class:`PlantService`,
never to the graphs directly. The service:

- runs one case (case graph → supervisor gate), saves the loop
  record, and archives the raw case payload to the object store;
- records the supervisor's decision — by resuming the graph when its
  checkpoint is live, or by applying the decision directly to the
  stored record when it is not (API restart, checkpoints off). The
  record is the source of truth either way, and the two paths
  produce the same record;
- runs the board (board graph over the case graph), persisting each
  case record as it reaches the gate, and the board run at the end;
- answers board/metrics/record reads from the store.

A failed reasoning call is never recorded as a guess: the error is
logged to the store's error log and surfaces as a ProviderError.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from langgraph.types import Command
from pydantic import ValidationError

from .events import CollectingSink, RunEvent, format_timestamp
from .graph import CaseRunOutcome, build_board_graph, build_case_graph
from .loops import BoardPolicy, CaseLoopController, LoopPolicy
from .objectstore import payload_bytes
from .ports import Checkpointer, EventSink, ObjectStore, Reasoner, Retriever, Store
from .schemas import BoardRunResult, CaseResult, ExceptionCase


class PlantService:
    def __init__(
        self,
        reasoner: Reasoner,
        retriever: Retriever,
        store: Store,
        objects: ObjectStore,
        checkpointer: Checkpointer | None = None,
        loop_policy: LoopPolicy | None = None,
        board_policy: BoardPolicy | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.retriever = retriever
        self.store = store
        self.objects = objects
        self._checkpointer = checkpointer
        self._checkpointer_ready = checkpointer is not None
        self.loop_policy = loop_policy or LoopPolicy.from_env()
        self.board_policy = board_policy or BoardPolicy.from_env()
        self._case_graphs: dict[int, object] = {}

    # -- collaborators ---------------------------------------------------

    def _get_checkpointer(self):
        """One checkpointer per service, created lazily and reused —
        the in-memory saver's state lives in the instance, so a fresh
        saver per call would lose every paused gate."""
        if not self._checkpointer_ready:
            from .checkpoints import default_checkpointer

            self._checkpointer = default_checkpointer()
            self._checkpointer_ready = True
        return self._checkpointer

    def _controller(self, sink: EventSink) -> CaseLoopController:
        return CaseLoopController(
            reasoner=self.reasoner,
            retriever=self.retriever,
            policy=self.loop_policy,
            sink=sink,
        )

    def _case_graph(self, sink: EventSink):
        key = id(sink)
        if key not in self._case_graphs:
            self._case_graphs[key] = build_case_graph(
                self._controller(sink), self._get_checkpointer(), sink
            )
        return self._case_graphs[key]

    # -- single case -------------------------------------------------------

    def run_case(self, case: ExceptionCase, sink: EventSink | None = None) -> CaseRunOutcome:
        """Run one case to the supervisor gate and persist the record."""
        sink = sink or CollectingSink()
        self._archive(case)
        graph = self._case_graph(sink)
        gate_mode = "pause" if self._get_checkpointer() is not None else "record"
        config = {"configurable": {"thread_id": case.case_id}}
        try:
            output = graph.invoke(
                {
                    "case": case.model_dump(),
                    "gate_mode": gate_mode,
                    "trace": [],
                    "started_at": time.time(),
                },
                config=config,
            )
        except Exception as exc:
            self.store.save_error(
                {"case_id": case.case_id, "error": str(exc),
                 "at": format_timestamp(datetime.now(timezone.utc))}
            )
            raise
        awaiting = "__interrupt__" in output
        if awaiting:
            state = graph.get_state(config)
            result = CaseResult.model_validate(state.values["result"])
        else:
            result = CaseResult.model_validate(output["result"])
        self.store.save_case(result)
        return CaseRunOutcome(result=result, awaiting_supervisor=awaiting)

    def record_decision(
        self, case_id: str, decision: str, by: str = "", reason: str = ""
    ) -> CaseResult:
        """Record the supervisor's approve/reject for a waiting case."""
        record = self.store.get_case(case_id)
        if record is None:
            raise KeyError(f"unknown case: {case_id}")
        if record.status != "awaiting_supervisor":
            raise ValueError(f"case {case_id} is not awaiting a supervisor (status: {record.status})")
        resumed = self._try_resume(case_id, {"decision": decision, "by": by, "reason": reason})
        if resumed is not None:
            self.store.save_case(resumed)
            return resumed
        record.decision = decision
        record.decision_by = by
        record.decision_reason = reason
        record.status = "approved" if decision == "approved" else "rejected"
        self.store.save_case(record)
        return record

    def _try_resume(self, case_id: str, payload: dict) -> CaseResult | None:
        """Resume a paused gate when its checkpoint is still live."""
        checkpointer = self._get_checkpointer()
        if checkpointer is None:
            return None
        sink = CollectingSink()
        graph = self._case_graph(sink)
        config = {"configurable": {"thread_id": case_id}}
        try:
            state = graph.get_state(config)
        except Exception:
            return None
        if not state or not state.next:  # nothing paused on this thread
            return None
        try:
            output = graph.invoke(Command(resume=payload), config=config)
        except Exception:
            return None
        if "__interrupt__" in output or "result" not in output:
            return None
        return CaseResult.model_validate(output["result"])

    # -- board ---------------------------------------------------------------

    def run_board(
        self,
        cases: list[ExceptionCase],
        sink: EventSink | None = None,
        board_policy: BoardPolicy | None = None,
    ) -> BoardRunResult:
        """Work a list of cases through the board graph, in priority
        order, to the shift budget. Each case record is persisted as
        it reaches the gate; the board run is saved at the end."""
        sink = sink or CollectingSink()
        for case in cases:
            self._archive(case)
        run_id = f"board-{uuid.uuid4().hex[:8]}"
        case_graph = self._case_graph(sink)
        board_graph = build_board_graph(
            case_graph,
            board_policy or self.board_policy,
            on_case_result=self.store.save_case,
            sink=sink,
        )
        output = board_graph.invoke(
            {"run_id": run_id, "queue": [c.model_dump() for c in cases]}
        )
        result = BoardRunResult.model_validate(output["board_result"])
        payload = result.model_dump()
        payload["cases"] = [self.store.get_case(cid).model_dump() if self.store.get_case(cid) else {}
                            for cid in result.processed]
        self.store.save_board_run(payload)
        return result

    # -- reads -----------------------------------------------------------------

    def get_record(self, case_id: str) -> CaseResult | None:
        return self.store.get_case(case_id)

    def board(self) -> list[dict]:
        return self.store.board()

    def metrics(self) -> dict:
        return self.store.metrics()

    def get_board_run(self, run_id: str) -> dict | None:
        return self.store.get_board_run(run_id)

    # -- internals ---------------------------------------------------------------

    def _archive(self, case: ExceptionCase) -> None:
        """Archive the raw case payload; a failed archive never blocks a run."""
        try:
            self.objects.put(
                f"cases/{case.case_id}.json",
                payload_bytes(case.model_dump()),
                content_type="application/json",
            )
        except Exception:
            pass


def validate_case(payload: dict) -> ExceptionCase:
    """Parse an inbound case payload into the schema (API/CLI helper)."""
    try:
        return ExceptionCase.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
