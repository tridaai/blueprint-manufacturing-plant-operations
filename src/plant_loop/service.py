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

from . import economics as economics_module
from . import handoff as handoff_module
from . import history as history_module
from . import packet as packet_module
from . import stats as stats_module
from .allocation import AllocationLedger
from .events import CollectingSink, RunEvent, format_timestamp
from .graph import CaseRunOutcome, build_board_graph, build_case_graph
from .loops import BoardPolicy, CaseLoopController, LoopPolicy, ReliabilityPolicy
from .objectstore import payload_bytes
from .ports import Checkpointer, EventSink, ObjectStore, Reasoner, Retriever, Store
from .reliability import ReliabilityWatch
from .schemas import (
    BoardItem,
    BoardRunResult,
    CaseResult,
    ExceptionCase,
    SupervisorOutcome,
)


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
        reliability_policy: ReliabilityPolicy | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.retriever = retriever
        self.store = store
        self.objects = objects
        self._checkpointer = checkpointer
        self._checkpointer_ready = checkpointer is not None
        self.loop_policy = loop_policy or LoopPolicy.from_env()
        self.board_policy = board_policy or BoardPolicy.from_env()
        self.reliability_policy = reliability_policy or ReliabilityPolicy.from_env()
        self._case_graphs: dict[int, object] = {}
        # The allocation ledger over this service's store (round
        # 5): converged cases claim the parts their plans will
        # draw, and every parts measurement reads the shelf net of
        # the claims standing. The run id of the board run in
        # flight ("" outside one — claims then record "single" and
        # notes name the claiming run as an earlier one).
        self.allocation = AllocationLedger(store)
        self._active_run_id = ""

    @property
    def reliability_watch(self) -> ReliabilityWatch:
        return ReliabilityWatch(self.store, self.reliability_policy)

    def _persist_case(self, record: CaseResult) -> None:
        """Save a case that reached the gate, claim its parts,
        then run the recurring-fault watch over it (reliability.py).

        The claim lands with the record: it is made here — the one
        place every gated record passes through, board graph and
        single runs alike — so the next case worked, in this run or
        any later one, measures the shelf net of it (allocation.py).
        Only converged records claim; the ledger refuses the rest.

        The economics stamp lands with the record too (economics.py):
        the case priced from its plan's estimated downtime at its
        line's plant-model rate, so every surface that reads the
        record — board, console, handoff — reads the same figure.
        A safety stop is stamped None: it is never priced.
        """
        record.economics = economics_module.for_record(record)
        self.store.save_case(record)
        self.allocation.claim_from_record(record, self._active_run_id or "single")
        self.reliability_watch.check(record)

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
            # The diagnoser reads the store's merged history view:
            # recorded KB history + the supervisor-confirmed
            # write-backs this store has collected (history.py).
            history_provider=lambda machine_id: history_module.merged_history(
                self.store, machine_id
            ),
            # The verify stage reads the shelf through the
            # allocation ledger: net of every claim standing, with
            # the claiming case named when a claim causes a
            # shortfall. The run id is read at call time, so a
            # controller cached for one board run names claims
            # correctly in the next.
            stock_provider=lambda part_id: self.allocation.effective_qty(part_id),
            claim_note_provider=lambda part_id, needed: self.allocation.note_for(
                part_id, needed, current_run_id=self._active_run_id
            ),
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
        self._persist_case(result)
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
            record = resumed
        else:
            record.decision = decision
            record.decision_by = by
            record.decision_reason = reason
            record.status = "approved" if decision == "approved" else "rejected"
            self.store.save_case(record)
        if record.status == "rejected":
            # A rejected plan will never draw its parts: its claims
            # are released back to the pool for future runs, stamped
            # with the rejection (allocation.py). An approval keeps
            # the claim standing — the work order will draw them.
            self.allocation.release_for_case(case_id, by=by)
        return record

    def record_outcome(self, case_id: str, outcome: SupervisorOutcome) -> CaseResult:
        """Record what actually happened after the gate.

        The outcome is the ground truth the diagnosis stats and the
        reliability watch read. It is recorded once — a second
        outcome for the same case is a conflict, not an edit (a
        corrected fact belongs in a corrected record, visibly).
        Recording one re-runs the watch: a confirmed fault can be
        the case that completes a pattern.

        A confirmed fault also closes the learning loop: it is
        written back to the machine's history (history.py), where
        the next diagnosis of that machine reads it. The write-back
        entry is built *before* anything is saved — it is the step
        that refuses a confirmed fault on a safety stop (a safety
        case has no diagnosis to confirm), so a refused outcome
        leaves no trace anywhere, rather than a record that says
        one thing and a memory that says another.
        """
        record = self.store.get_case(case_id)
        if record is None:
            raise KeyError(f"unknown case: {case_id}")
        if record.outcome is not None:
            raise ValueError(f"case {case_id} already has a recorded outcome")
        outcome.recorded_at = format_timestamp(datetime.now(timezone.utc))
        entry = history_module.entry_for_record(record, outcome)  # may refuse (safety stop)
        record.outcome = outcome
        # The actuals join the economics snapshot at the same rate
        # the estimate was priced at, so estimate and actual dollars
        # on one record are always the same arithmetic.
        record.economics = economics_module.for_record(record)
        self.store.save_case(record)
        if entry is not None:
            self.store.save_history_entry(entry)
        self.reliability_watch.check(record)
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
            on_case_result=self._persist_case,
            sink=sink,
        )
        # Claims made while this run works carry its id, and notes
        # inside the run name its claims as this board's own (the
        # providers read the id at call time). The graph works its
        # cases synchronously, so the id is set for exactly the
        # run's duration.
        self._active_run_id = run_id
        try:
            output = board_graph.invoke(
                {"run_id": run_id, "queue": [c.model_dump() for c in cases]}
            )
        finally:
            self._active_run_id = ""
        result = BoardRunResult.model_validate(output["board_result"])
        result.allocations = self.allocation.claims_for_run(run_id)
        worked = [self.store.get_case(cid) for cid in result.processed]
        # The run priced as one board: the estimated downtime
        # exposure its processed cases carry (economics.py), in
        # board order so the largest exposure names the first such
        # case the supervisor would meet.
        result.economics = economics_module.board_summary(
            [r for r in worked if r is not None]
        )
        payload = result.model_dump()
        payload["cases"] = [r.model_dump() if r else {} for r in worked]
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

    def diagnosis_stats(self) -> dict:
        return stats_module.diagnosis_stats(
            self.store.all_cases(), history_entries=self.store.all_history_entries()
        )

    def handoff(self) -> dict:
        """The shift handoff payload + its plain-text rendering."""
        payload = handoff_module.build_handoff(self.store)
        payload["text"] = handoff_module.render_text(payload)
        return payload

    def case_packet(self, case_id: str) -> dict:
        """The work-order packet for one gated case (JSON + text).

        Raises KeyError for an unknown case and ValueError when the
        case has no verified work order to packet — the transport
        maps those to 404 / 409, following the decision/outcome
        conventions.
        """
        record = self.store.get_case(case_id)
        if record is None:
            raise KeyError(f"unknown case: {case_id}")
        payload = packet_module.build_packet(
            record, claims=self.allocation.claims_for_case(case_id)
        )
        payload["text"] = packet_module.render_text(payload)
        return payload

    def allocations(self) -> list[dict]:
        """The allocation ledger as it stands: every claim ever
        recorded on this store, standing and released alike."""
        return [claim.model_dump() for claim in self.allocation.all_claims()]

    def board_items(self) -> list[BoardItem]:
        return self.store.all_board_items()

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
