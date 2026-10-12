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
- records the supervisor's amendments to a waiting case's plan
  (amendments.py) — built and re-verified in code at recording,
  appended to the record passed or failed, with a passing part
  swap moving the case's claims in the same act — and refuses an
  approval whose case's latest amendment failed re-verification;
- runs the board (board graph over the case graph), persisting each
  case record as it reaches the gate, and the board run at the end —
  settling a predecessor run's carry-over first when the run names
  one (carryover.py), stamping the finished run with its own
  outgoing carry-over manifest, and carrying the cycle cases of any
  standing PM obligation whose due point the run reaches (pm.py);
- drafts preventive-maintenance proposals where the recurring-fault
  watch's pattern earns them (pm.py — evaluated wherever the watch
  is checked), records the supervisor's decision on them, and lets
  a PM cycle case's outcome — and nothing else — close or roll its
  obligation's cycle;
- records the ends of things (lifecycle.py): a supervisor's
  retirement of a PM obligation, and a machine's decommission —
  which retires that machine's obligations automatically, lists
  its open cases for the supervisor's per-case disposition, and
  closes intake for the machine at the run paths above;
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

from . import amendments as amendments_module
from . import carryover as carryover_module
from . import recurrence as recurrence_module
from . import economics as economics_module
from . import gaps as gaps_module
from . import handoff as handoff_module
from . import history as history_module
from . import lifecycle as lifecycle_module
from . import outlook as outlook_module
from . import packet as packet_module
from . import stats as stats_module
from .allocation import RELEASE_BY_AMENDMENT, AllocationLedger
from .events import CollectingSink, RunEvent, format_timestamp
from .graph import CaseRunOutcome, build_board_graph, build_case_graph
from .lifecycle import MachineLifecycle
from .loops import BoardPolicy, CaseLoopController, LoopPolicy, ReliabilityPolicy
from .objectstore import payload_bytes
from .pm import PmManager
from .ports import Checkpointer, EventSink, ObjectStore, Reasoner, Retriever, Store
from .reliability import ReliabilityWatch
from .schemas import (
    BoardItem,
    BoardRunResult,
    CaseResult,
    ExceptionCase,
    KnowledgeGapReport,
    MachineRecord,
    PartsOutlook,
    PlanAmendment,
    PmObligation,
    PmProposal,
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

    @property
    def pm_manager(self) -> PmManager:
        return PmManager(
            self.store, ReliabilityWatch(self.store, self.reliability_policy)
        )

    @property
    def machine_lifecycle(self) -> MachineLifecycle:
        return MachineLifecycle(self.store, self.pm_manager)

    def _refuse_if_decommissioned(self, case: ExceptionCase) -> None:
        """The intake guard (lifecycle.py): no new work for a
        machine that no longer exists.

        Applied where cases ENTER — a single run, a board run's
        submitted cases. Cases already on the board never pass
        through here: carried cases settle from their stored
        records, and PM cycle cases are the obligations' own,
        which a decommission has already retired.
        """
        if not case.machine_id:
            return
        record = self.store.get_machine_record(case.machine_id)
        if record is not None:
            raise ValueError(
                f"machine {case.machine_id} is DECOMMISSIONED "
                f"({record.record_id}, effective {record.effective_at}, "
                f"by {record.decided_by}: {record.reason}) — new cases "
                "for it are refused at intake"
            )

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
        # The watch's item is the notice; the PM layer (pm.py) is
        # the follow-through: the same pattern, freshly counted by
        # the watch's own rule, drafts a preventive-maintenance
        # proposal for a supervisor to decide on.
        self.pm_manager.evaluate(record)

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
            # The recurrence check reads the same store: a fresh
            # diagnosis whose fault a supervisor already closed on
            # that machine links the two cases (recurrence.py).
            recurrence_provider=recurrence_module.provider_for(self.store),
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
        self._refuse_if_decommissioned(case)
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

    def amend_case(
        self,
        case_id: str,
        *,
        kind: str,
        by: str = "",
        reason: str = "",
        from_part_id: str = "",
        to_part_id: str = "",
        step_id: str = "",
    ) -> CaseResult:
        """Record a supervisor's amendment to a waiting case's plan.

        The amendment is built and re-verified in code
        (amendments.py) — no model is consulted — and appended to
        the record whether it passes or fails: the attempt and the
        verdict are both the record. A passing part swap moves the
        case's claims in the same act (the old part's claim
        released stamped ``released_by_amendment``, the new part
        claimed under the case's run), so the ledger always
        describes the plan currently at the gate; a failed
        amendment moves nothing. The original plan on the record
        is never edited.
        """
        record = self.store.get_case(case_id)
        if record is None:
            raise KeyError(f"unknown case: {case_id}")
        amendment = amendments_module.build_amendment(
            record,
            self.allocation,
            kind=kind,
            by=by,
            reason=reason,
            from_part_id=from_part_id,
            to_part_id=to_part_id,
            step_id=step_id,
        )
        record.amendments.append(amendment)
        self.store.save_case(record)
        if (
            amendment.kind == amendments_module.PART_SWAP
            and amendment.verification.status == amendments_module.VERIFIED
        ):
            self._move_claims_for_swap(record, amendment)
        return record

    def _move_claims_for_swap(self, record: CaseResult, amendment: PlanAmendment) -> None:
        """The allocation follows a verified swap: release the old
        drawn part's standing claim (stamped with the amendment),
        claim the new part under the run that held the old claim."""
        for claim in self.allocation.claims_for_case(record.case_id):
            if claim.status == "claimed" and claim.part_id == amendment.from_part_id:
                run_id = claim.run_id
                self.allocation.release_claim(
                    claim, by=amendment.by, reason=RELEASE_BY_AMENDMENT
                )
                self.allocation.claim_part(
                    record.case_id,
                    drawn=amendment.to_part_id,
                    original=(
                        amendment.slot_part_id
                        if amendment.to_part_id != amendment.slot_part_id
                        else ""
                    ),
                    qty=amendment.qty,
                    run_id=run_id,
                )
                return

    def record_decision(
        self, case_id: str, decision: str, by: str = "", reason: str = ""
    ) -> CaseResult:
        """Record the supervisor's approve/reject for a waiting case."""
        record = self.store.get_case(case_id)
        if record is None:
            raise KeyError(f"unknown case: {case_id}")
        if record.status != "awaiting_supervisor":
            raise ValueError(f"case {case_id} is not awaiting a supervisor (status: {record.status})")
        if decision == "approved":
            # The approval reads the CURRENT verification state:
            # a case whose latest amendment failed re-verification
            # cannot be approved over the failure — the supervisor
            # amends again, or rejects. Checked here, before either
            # decision path, so no stale approve can slip past it.
            failed = amendments_module.latest_failed_amendment(record)
            if failed is not None:
                raise ValueError(
                    f"case {case_id} cannot be approved: its latest amendment "
                    f"{failed.amendment_id} FAILED re-verification "
                    f"({failed.verification.detail}) — the supervisor may "
                    "amend again or reject; a plan whose current verification "
                    "fails is not approved over the failure"
                )
        resumed = self._try_resume(case_id, {"decision": decision, "by": by, "reason": reason})
        if resumed is not None:
            # Amendments are gate records the graph never saw (they
            # were recorded on the stored record while the thread
            # waited); the resumed record must not lose them.
            resumed.amendments = record.amendments
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
        pm_obligation = self.pm_manager.obligation_for_case(case_id)
        if (
            outcome.decision == "rejected"
            and not outcome.work_performed.strip()
            and pm_obligation is not None
            and pm_obligation.status == "standing"
        ):
            # A PM cycle case rejected without a note would defer
            # the obligation silently. The deferral's reason IS the
            # work-performed note — refuse before anything saves,
            # so a refused outcome leaves no trace (the same
            # discipline as the safety write-back refusal below).
            raise ValueError(
                f"case {case_id} is a preventive-maintenance cycle case: "
                "a rejected outcome defers the obligation, and a deferral "
                "needs the supervisor's reason — record it as the "
                "work-performed note"
            )
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
        # A PM cycle case's outcome closes or rolls its
        # obligation's cycle (pm.py) — performed stamps come from
        # this outcome alone. Any other case's outcome is then
        # evaluated like any gate arrival: a confirmed fault can be
        # the case that earns a proposal. (A cycle case's own
        # outcome never earns one — an inspection is not a
        # recurrence, and pm.evaluate knows its case ids.)
        self.pm_manager.close_cycle(record, outcome)
        self.pm_manager.evaluate(record)
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
        carry_over_from: str = "",
        drop: list[dict] | None = None,
    ) -> BoardRunResult:
        """Work a list of cases through the board graph, in priority
        order, to the shift budget. Each case record is persisted as
        it reaches the gate; the board run is saved at the end.

        With ``carry_over_from`` naming a finished board run, this
        run first settles that run's carry-over manifest
        (carryover.py): blocked cases are re-worked with their
        loop state intact, gate cases are inherited standing,
        unworked cases join as fresh intakes, ``drop`` entries
        leave the board audited, and the claims turn over with the
        board. Without it, the run starts clean — no inheritance,
        no settlement, the pre-round-7 behaviour exactly.
        """
        sink = sink or CollectingSink()
        for case in cases:
            self._refuse_if_decommissioned(case)
        for case in cases:
            self._archive(case)
        run_id = f"board-{uuid.uuid4().hex[:8]}"
        queue: list[dict] = [
            {"case": c.model_dump(), "resume": None} for c in cases
        ]
        settled: dict | None = None
        if carry_over_from:
            predecessor = self.store.get_board_run(carry_over_from)
            if predecessor is None:
                raise KeyError(f"unknown board run: {carry_over_from}")
            carried_queue, settled = carryover_module.settle(
                self.store,
                self.objects,
                self.allocation,
                predecessor,
                into_run_id=run_id,
                drops=drop,
                new_case_ids={c.case_id for c in cases},
            )
            queue = carried_queue + queue
        if settled is not None:
            # A PM cycle case dropped at this acceptance has been
            # deferred in every way that matters: its obligation
            # rolls like one (pm.py), on the drop's own by/reason.
            for dropped_id in settled.get("dropped", []):
                drop_info = next(
                    (d for d in (drop or []) if d.get("case_id") == dropped_id),
                    {},
                )
                self.pm_manager.note_board_drop(
                    dropped_id,
                    by=drop_info.get("by", ""),
                    reason=drop_info.get("reason", ""),
                )
        # Standing PM obligations whose due point this run reaches
        # send their cycle cases (pm.py): the board remembers the
        # date the way it remembers a carried board — in code, as
        # cases on the queue, ranked by the same priority formula.
        pm_cases = self.pm_manager.due_cases()
        for case in pm_cases:
            self._archive(case)
        queue = queue + [{"case": c.model_dump(), "resume": None} for c in pm_cases]
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
            output = board_graph.invoke({"run_id": run_id, "queue": queue})
        finally:
            self._active_run_id = ""
        result = BoardRunResult.model_validate(output["board_result"])
        result.pm_cases = [c.case_id for c in pm_cases]
        result.allocations = self.allocation.claims_for_run(run_id)
        worked = [self.store.get_case(cid) for cid in result.processed]
        # The run priced as one board: the estimated downtime
        # exposure its processed cases carry (economics.py), in
        # board order so the largest exposure names the first such
        # case the supervisor would meet.
        result.economics = economics_module.board_summary(
            [r for r in worked if r is not None]
        )
        if settled is not None:
            result.carried_from = carry_over_from
            result.carry_over_settled = settled
        # The run record is the contract the next shift settles
        # against: it carries every queued case's intake payload
        # (worked or not), and — computed now, while the records
        # are fresh — this run's own outgoing carry-over manifest.
        case_payloads = {
            entry["case"]["case_id"]: entry["case"] for entry in queue
        }
        draft_payload = result.model_dump()
        draft_payload["case_payloads"] = case_payloads
        manifest = carryover_module.compute_manifest(self.store, draft_payload)
        result.carry_over = manifest
        extra_lines: list[str] = []
        if settled is not None:
            extra_lines += carryover_module.settled_lines(settled)
        extra_lines += carryover_module.manifest_lines(manifest)
        if extra_lines:
            result.handoff_summary = (
                result.handoff_summary.rstrip("\n")
                + "\n"
                + "\n".join(extra_lines)
            )
        payload = result.model_dump()
        payload["cases"] = [r.model_dump() if r else {} for r in worked]
        payload["case_payloads"] = case_payloads
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

    def get_carry_over(self, run_id: str) -> dict:
        """A stored run's carry-over contract: the manifest it
        offers the next shift, and — for a run that itself accepted
        carry-over — how its inheritance settled. Runs saved
        before manifests existed answer with empty sections."""
        run = self.store.get_board_run(run_id)
        if run is None:
            raise KeyError(f"unknown board run: {run_id}")
        return {
            "run_id": run_id,
            "carried_from": run.get("carried_from", ""),
            "carry_over": run.get("carry_over", []),
            "carry_over_settled": run.get("carry_over_settled", {}),
        }

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

    def parts_outlook(
        self, horizon_days: int | None = None, *, now: datetime | None = None
    ) -> PartsOutlook:
        """The parts outlook (outlook.py): projected demand against
        the shelf over the horizon — standing claims, open cases'
        unsatisfied plan requirements, and PM cycles falling due.
        A pure projection: it reads the store and writes nothing."""
        return outlook_module.build_outlook(
            self.store,
            now=now,
            horizon_days=(
                horizon_days
                if horizon_days is not None
                else outlook_module.DEFAULT_HORIZON_DAYS
            ),
        )

    def knowledge_gaps(self, *, now: datetime | None = None) -> KnowledgeGapReport:
        """The knowledge-gap report (gaps.py): case blocks grouped
        by the missing SOP — the diagnosed fault with no SOP on
        record for its machine — with the rounds the absence has
        burned and whether it is biting right now. A pure
        projection that drafts nothing: advising the KB maintainer
        is its whole job; authoring the SOP stays a human's act."""
        return gaps_module.build_gap_report(self.store, now=now)

    def board_items(self) -> list[BoardItem]:
        return self.store.all_board_items()

    def pm_proposals(self) -> list[PmProposal]:
        return self.pm_manager.proposals()

    def pm_obligations(self) -> list[PmObligation]:
        return self.pm_manager.obligations()

    def decide_pm_proposal(
        self,
        proposal_id: str,
        decision: str,
        by: str = "",
        reason: str = "",
        interval_days: int | None = None,
        last_performed_at: str = "",
    ) -> tuple[PmProposal, PmObligation | None]:
        """Record the supervisor's decision on a PM proposal (pm.py)."""
        return self.pm_manager.decide(
            proposal_id,
            decision,
            by=by,
            reason=reason,
            interval_days=interval_days,
            last_performed_at=last_performed_at,
        )

    def retire_pm_obligation(
        self, obligation_id: str, by: str = "", reason: str = ""
    ) -> PmObligation:
        """Record a supervisor's manual retirement of a PM
        obligation (pm.py): actor and reason required, one-way,
        the reason on the record forever."""
        return self.pm_manager.retire_obligation(obligation_id, by=by, reason=reason)

    # -- machine lifecycle (lifecycle.py) -------------------------------------

    def machine_records(self) -> list[MachineRecord]:
        return self.store.all_machine_records()

    def machine_state(self, machine_id: str) -> str:
        return lifecycle_module.machine_state(self.store, machine_id)

    def machine_open_cases(self, machine_id: str) -> list[CaseResult]:
        return lifecycle_module.open_cases(self.store, machine_id)

    def decommission_machine(
        self,
        machine_id: str,
        by: str = "",
        reason: str = "",
        effective_at: str = "",
    ) -> MachineRecord:
        """Record a machine's decommission (lifecycle.py).

        The act retires the machine's standing PM obligations
        automatically (linked, with the fixed reason) and lists
        its open cases on the record for disposition. It closes,
        drops, and decides nothing about any case itself.
        """
        return self.machine_lifecycle.decommission(
            machine_id, by=by, reason=reason, effective_at=effective_at
        )

    def dispose_machine_case(
        self, machine_id: str, case_id: str, disposition: str, by: str = ""
    ) -> MachineRecord:
        """Record the supervisor's disposition of one open case
        of a decommissioned machine (lifecycle.py).

        Closing a case as moot releases its standing part claims,
        the way a gate rejection's claims release (allocation.py):
        the work will never draw them.
        """
        record = self.machine_lifecycle.dispose(machine_id, case_id, disposition, by=by)
        if disposition == lifecycle_module.DISPOSITION_CLOSE_MOOT:
            self.allocation.release_for_case(case_id, by=by)
        return record

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
