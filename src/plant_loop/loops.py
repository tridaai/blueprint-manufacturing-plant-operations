"""The loop controllers: where the agent's iteration is governed.

Two nested loops, both driven from this module — the LangGraph
graphs in ``graph.py`` wire these same step functions into nodes, so
the graph and the plain-code path can never drift apart.

**The case loop** (per exception)::

    diagnose → plan → verify ── gaps? ──→ back to diagnose (gaps only)
                        │
                        └──── exit ──→ draft → supervisor gate

The verify round merges two critics: the reasoner's critique and
the code checks in ``guardrails.verify_plan``. Before those checks
run, the verify stage does its parts work in code
(``substitution.py``): every requirement is re-measured against the
shelf, and for one the shelf cannot satisfy, the KB's compatibility
relation is searched — a candidate that is in stock *and* clears
the SOP's constraints for the slot is applied to the plan (recorded
as a first-class substitution), and the code verify then measures
the revised plan. Code wins ties — a model cannot talk a missing
safety step out of existence. Exits are decided in code
(:meth:`CaseLoopController.decide_exit`):

- ``SAFETY_STOP`` — a safety_observation case (round 0, before any
  reasoning), or a plan step that recommends a safety bypass.
- ``BLOCKED_EXTERNAL`` — a required part is unavailable and the
  substitute search found nothing usable: no candidate listed, none
  in stock, or every candidate rejected by the SOP's constraints.
  Looping cannot fix a stockout; the case escalates to whoever can
  order parts, with the search documented on the record.
- ``CONVERGED`` — verify found no gaps. The plan may be drafted as a
  work order (still only a draft; the supervisor decides).
- ``BUDGET_EXCEEDED`` — the case's token budget is spent.
- ``EXHAUSTED`` — ``max_rounds`` reached with gaps still open. The
  open gaps are listed on the escalation; the loop never papers over
  them by "converging" anyway.

**The board loop** (per shift run) works cases in the code-computed
priority order of ``priority.py`` and stops when the board is clear
or the shift budget (cases / tokens) is spent; whatever remains is
listed, with reasons, in the handoff summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from . import guardrails, history, kb, recurrence, substitution
from .config import env_int
from .events import NullSink, RunEvent
from .ports import CaseContext, EventSink, Reasoner, Retriever
from .priority import board_order, priority_for
from .schemas import (
    EXIT_BLOCKED_EXTERNAL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONVERGED,
    EXIT_EXHAUSTED,
    EXIT_SAFETY_STOP,
    ActionPlan,
    BoardRunResult,
    Critique,
    Diagnosis,
    ExceptionCase,
    Gap,
    RecurrenceLink,
    RoundRecord,
    Substitution,
)


@dataclass
class LoopPolicy:
    """The case loop's bounds. Every bound is a code constant or an
    env override — a prompt never talks the loop into another round."""

    max_rounds: int = 3
    case_token_budget: int = 200_000

    @classmethod
    def from_env(cls) -> "LoopPolicy":
        return cls(
            max_rounds=env_int("LOOP_MAX_ROUNDS", 3),
            case_token_budget=env_int("LOOP_CASE_TOKEN_BUDGET", 200_000),
        )


@dataclass
class BoardPolicy:
    """The board loop's shift budget."""

    max_cases: int = 20
    board_token_budget: int = 1_000_000

    @classmethod
    def from_env(cls) -> "BoardPolicy":
        return cls(
            max_cases=env_int("BOARD_MAX_CASES", 20),
            board_token_budget=env_int("BOARD_TOKEN_BUDGET", 1_000_000),
        )


@dataclass
class ReliabilityPolicy:
    """The recurring-fault watch's bounds (reliability.py).

    When the same machine produces ``threshold`` cases of the same
    fault class inside ``window_days``, the watch raises one
    ``reliability_review`` board item. Fixing the same bearing three
    times in a fortnight is a reliability problem wearing a
    maintenance costume — the watch exists to say so, in code.
    """

    threshold: int = 3
    window_days: int = 14

    @classmethod
    def from_env(cls) -> "ReliabilityPolicy":
        return cls(
            threshold=env_int("RELIABILITY_THRESHOLD", 3),
            window_days=env_int("RELIABILITY_WINDOW_DAYS", 14),
        )


@dataclass
class CaseOutcome:
    """What one case loop produced, before the record is assembled."""

    diagnosis: Diagnosis = field(default_factory=Diagnosis)
    plan: ActionPlan = field(default_factory=ActionPlan)
    open_gaps: list[Gap] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    exit_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    backend: str = ""
    model: str = ""
    recurrence: RecurrenceLink | None = None


@dataclass
class CaseRun:
    """Mutable state for one case loop (shared by controller + graph)."""

    case: ExceptionCase
    ctx: CaseContext
    diagnosis: Diagnosis = field(default_factory=Diagnosis)
    plan: ActionPlan = field(default_factory=ActionPlan)
    gaps: list[Gap] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    round_no: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    exit_reason: str = ""
    backend: str = ""
    model: str = ""
    # The recurrence link (recurrence.py): computed in code after
    # the first diagnose round, from the store the service bound;
    # None when the diagnosed fault has no qualifying closed prior.
    recurrence: RecurrenceLink | None = None
    # Whether the recurrence check has run for this case run. The
    # check is sought ONCE, against the first diagnosis; a miss is
    # final for the run (a later round settling on a fault with a
    # prior does not re-seek — see _note_recurrence).
    recurrence_checked: bool = False


def _gap_keys(gaps: list[Gap]) -> list[str]:
    return [f"{g.kind}:{g.ref}" for g in gaps]


def _substitution_line(record: Substitution) -> str:
    """One timeline line for a verify-stage search outcome."""
    if record.status == "applied":
        return f"{record.original_part_id}->{record.substitute_part_id} applied"
    return f"{record.substitute_part_id} rejected for {record.original_part_id}"


def _merge_gaps(model_gaps: list[Gap], code_gaps: list[Gap]) -> list[Gap]:
    """Union of both critics, deduped by (kind, ref); code wins."""
    merged: dict[tuple[str, str], Gap] = {}
    for gap in model_gaps:
        merged[(gap.kind, gap.ref)] = gap
    for gap in code_gaps:
        merged[(gap.kind, gap.ref)] = gap
    return list(merged.values())


class CaseLoopController:
    """Diagnose → plan → verify, looped on gaps, exited in code."""

    def __init__(
        self,
        reasoner: Reasoner,
        retriever: Retriever,
        policy: LoopPolicy | None = None,
        sink: EventSink | None = None,
        history_provider: Callable[[str], list[dict]] | None = None,
        stock_provider: Callable[[str], int | None] | None = None,
        claim_note_provider: Callable[[str, int], str] | None = None,
        recurrence_provider: Callable[[ExceptionCase, str], RecurrenceLink | None] | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.retriever = retriever
        self.policy = policy or LoopPolicy()
        self.sink = sink or NullSink()
        # Where the machine's merged history comes from (the service
        # binds it to the record store, so supervisor-confirmed
        # write-backs are in it). Unbound, the controller reads the
        # KB's recorded history alone — never an empty memory.
        self.history_provider = history_provider
        # Where shelf truth comes from (the service binds it to the
        # allocation ledger, so every measurement is net of the
        # claims earlier cases on the board have standing). Unbound,
        # the controller reads the KB shelf itself — a lone case on
        # an unshared shelf, as before round 5.
        self.stock_provider = stock_provider
        # Names the claiming case when a shortfall is a claim's
        # doing (allocation.py); unbound, shortfalls are unnamed.
        self.claim_note_provider = claim_note_provider
        # Where a fresh diagnosis checks for the fault's closed
        # prior (recurrence.py — the service binds it to the record
        # store). Unbound, no case links: a controller working
        # without the plant's records cannot claim a recurrence.
        self.recurrence_provider = recurrence_provider

    def _effective_qty(self, part_id: str) -> int | None:
        """Shelf qty as this case may see it: net of claims when a
        stock provider is bound, the bare shelf when it is not."""
        if self.stock_provider is not None:
            return self.stock_provider(part_id)
        return kb.part_qty(part_id)

    def _claim_note(self, part_id: str, needed: int) -> str:
        """The allocation note for a shortfall ('' when unbound or
        when no claim caused it)."""
        if self.claim_note_provider is not None:
            return self.claim_note_provider(part_id, needed)
        return ""

    # -- context ------------------------------------------------------------

    def machine_history(self, machine_id: str) -> list[dict]:
        """The merged history view for one machine (history.py)."""
        if self.history_provider is not None:
            return self.history_provider(machine_id)
        return history.merged_history(None, machine_id)

    def prepare(self, case: ExceptionCase, case_text: str) -> CaseRun:
        query = f"{case.machine_id} {case.type} {case.title} {case.description}"
        chunks = self.retriever.retrieve(query, limit=5)
        # The machine's own history is diagnose context by right, not
        # by retrieval luck: the merged view (confirmed write-backs +
        # recorded history, sources marked) rides on the context for
        # the offline signals, and any of its chunks the retriever
        # did not already surface are appended for the model
        # backends' prompts — marked, never silently blended.
        entries = self.machine_history(case.machine_id)
        seen = {chunk["chunk_id"] for chunk in chunks}
        for chunk in history.history_chunks(entries):
            if chunk["chunk_id"] not in seen:
                chunks.append(chunk)
                seen.add(chunk["chunk_id"])
        ctx = CaseContext(case=case, case_text=case_text, kb_context=chunks, history=entries)
        return CaseRun(case=case, ctx=ctx)

    # -- round steps (also the graph's nodes) -------------------------------

    def diagnose_round(self, run: CaseRun) -> None:
        self.sink.emit(RunEvent(type="round_started", case_id=run.case.case_id,
                                detail={"round": run.round_no, "phase": "diagnose"}))
        out = self.reasoner.diagnose(run.ctx, focus=run.gaps)
        run.diagnosis = out.payload
        self._account(run, out)
        self._note_recurrence(run)

    def _note_recurrence(self, run: CaseRun) -> None:
        """The recurrence check (recurrence.py), run once per case.

        The moment the loop's first diagnosis lands, its top fault
        is the join key: code asks the bound store whether the
        plant has already closed that fault on this machine inside
        the window. A hit is stamped on the run and enters the
        context as one labelled chunk, so every later round —
        diagnose, plan, critique — meets the prior episode with
        its outcome attached. The check runs once: the link keys
        on the first diagnosis the loop produced, and a diagnosis
        that settles elsewhere in later rounds is measured
        against it (the departure mark), never re-keyed.
        """
        if run.recurrence_checked or self.recurrence_provider is None:
            return
        top = run.diagnosis.top()
        if top is None:
            return
        run.recurrence_checked = True
        link = self.recurrence_provider(run.case, top.fault_class)
        if link is None:
            return
        run.recurrence = link
        chunk = recurrence.context_chunk(link)
        if chunk["chunk_id"] not in {c["chunk_id"] for c in run.ctx.kb_context}:
            run.ctx.kb_context.append(chunk)
        self.sink.emit(
            RunEvent(
                type="recurrence_linked",
                case_id=run.case.case_id,
                detail={
                    "prior_case_id": link.prior_case_id,
                    "fault_class": link.fault_class,
                    "days_between": link.days_between,
                },
            )
        )

    def plan_round(self, run: CaseRun) -> None:
        self.sink.emit(RunEvent(type="round_started", case_id=run.case.case_id,
                                detail={"round": run.round_no, "phase": "plan"}))
        out = self.reasoner.draft_plan(run.ctx, run.diagnosis, focus=run.gaps)
        run.plan = out.payload
        self._account(run, out)

    # -- the verify stage's parts work (code, every round) -------------------

    def _parts_pass(
        self, run: CaseRun
    ) -> tuple[list[Substitution], set[str], dict[str, str]]:
        """Shelf truth, then the substitute search (substitution.py).

        A plan's part requirements are claims — the planner may have
        guessed at stock, or resolved a part through a substitute
        nobody re-checked. The pass re-measures every requirement
        against inventory, then searches the KB compatibility
        relation for what the shelf cannot satisfy: a passing
        candidate is applied to the plan (steps revised, requirement
        satisfied by substitute, the substitution recorded), a
        failing one is recorded as rejected with its reason.

        Returns this round's search records, the part ids the pass
        resolved (directly or by substitution), and — for parts it
        could not resolve — the search documentation that the
        surviving gap carries onto the record.
        """
        plan = run.plan
        top = run.diagnosis.top()
        sop = kb.sop_for(run.case.machine_id, top.fault_class) if top else None
        records: list[Substitution] = []
        resolved: set[str] = set()
        notes: dict[str, str] = {}
        for req in plan.required_parts:
            # Shelf truth is the effective shelf: net of the claims
            # earlier cases on the board have standing (the stock
            # provider), so two cases can no longer both measure
            # the same last unit as available.
            on_hand = self._effective_qty(req.part_id)
            req.available_qty = on_hand or 0
            req.satisfied = on_hand is not None and on_hand >= req.qty
            # The pass owns the substitute marker too: a planner-set
            # substitute is a claim until the search below verifies
            # it; apply_to_plan sets the field only for a winner.
            req.substitute_part_id = ""
            if req.satisfied:
                resolved.add(req.part_id)
        for req in plan.required_parts:
            if req.satisfied:
                continue
            winner, considered = substitution.evaluate(
                sop,
                req.part_id,
                req.qty,
                qty_of=self._effective_qty,
                note_of=self._claim_note,
            )
            known = {
                (s.original_part_id, s.substitute_part_id, s.status)
                for s in plan.substitutions
            }
            for record in considered:
                records.append(record)
                key = (record.original_part_id, record.substitute_part_id, record.status)
                if key not in known:
                    plan.substitutions.append(record)
                    known.add(key)
            if winner is not None:
                substitution.apply_to_plan(plan, winner)
                resolved.add(req.part_id)
            else:
                summary = substitution.search_summary(req.part_id, considered)
                note = self._claim_note(req.part_id, req.qty)
                # A shortfall a claim caused is documented as the
                # board's own decision, never as a bare stockout.
                notes[req.part_id] = f"{note} — {summary}" if note else summary
        return records, resolved, notes

    def verify_round(self, run: CaseRun) -> None:
        self.sink.emit(RunEvent(type="round_started", case_id=run.case.case_id,
                                detail={"round": run.round_no, "phase": "verify"}))
        previous_keys = _gap_keys(run.gaps)
        out = self.reasoner.critique(run.ctx, run.diagnosis, run.plan)
        self._account(run, out)
        critique: Critique = out.payload
        # The parts pass runs before the code verify, so the code
        # checks measure the revised plan — and a part the pass
        # resolved is no longer a gap, whoever first reported it.
        search_records, resolved_parts, search_notes = self._parts_pass(run)
        code_gaps = guardrails.verify_plan(run.case, run.diagnosis, run.plan)
        for gap in code_gaps:
            if gap.kind == "part_unavailable" and gap.ref in search_notes:
                gap.detail = f"{gap.detail} — {search_notes[gap.ref]}"
        model_gaps = [
            gap
            for gap in critique.gaps
            if not (gap.kind == "part_unavailable" and gap.ref in resolved_parts)
        ]
        merged = _merge_gaps(model_gaps, code_gaps)
        current_keys = _gap_keys(merged)
        top = run.diagnosis.top()
        # Token fields hold cumulative totals at this point; the
        # caller converts the latest record to this round's slice via
        # finalize_round() so the timeline shows per-round cost.
        run.rounds.append(
            RoundRecord(
                round=run.round_no,
                top_fault_class=top.fault_class if top else "",
                hypotheses=len(run.diagnosis.hypotheses),
                steps=len(run.plan.steps),
                gaps_opened=[k for k in current_keys if k not in previous_keys],
                gaps_closed=[k for k in previous_keys if k not in current_keys],
                substitutions=[_substitution_line(r) for r in search_records],
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
            )
        )
        run.gaps = merged
        self.sink.emit(
            RunEvent(type="gaps_found", case_id=run.case.case_id,
                     detail={"count": len(merged), "kinds": [g.kind for g in merged]})
        )

    @staticmethod
    def finalize_round(run: CaseRun, round_input: int, round_output: int) -> None:
        """Convert the latest RoundRecord's cumulative token totals
        to this round's slice (the timeline reads per-round cost)."""
        if run.rounds:
            run.rounds[-1].input_tokens = run.input_tokens - round_input
            run.rounds[-1].output_tokens = run.output_tokens - round_output

    def _account(self, run: CaseRun, out) -> None:
        run.input_tokens += out.input_tokens
        run.output_tokens += out.output_tokens
        run.backend = out.backend or run.backend
        run.model = out.model or run.model

    # -- exit decision (pure code) ------------------------------------------

    def _blocked_part(self, run: CaseRun) -> str:
        """A part gap the loop cannot work: re-running the verify
        stage's own search finds no candidate that is in stock and
        clears the SOP's constraints for the slot."""
        top = run.diagnosis.top()
        sop = kb.sop_for(run.case.machine_id, top.fault_class) if top else None
        for gap in run.gaps:
            if gap.kind != "part_unavailable":
                continue
            req = next(
                (r for r in run.plan.required_parts if r.part_id == gap.ref), None
            )
            winner, _considered = substitution.evaluate(
                sop,
                gap.ref,
                req.qty if req else 1,
                qty_of=self._effective_qty,
                note_of=self._claim_note,
            )
            if winner is None:
                return gap.ref
        return ""

    def decide_exit(self, run: CaseRun) -> str:
        """The loop's exit for the current state ('' = keep looping)."""
        if any(g.kind == "banned_action" and g.safety for g in run.gaps):
            return EXIT_SAFETY_STOP
        if self._blocked_part(run):
            return EXIT_BLOCKED_EXTERNAL
        if not run.gaps:
            return EXIT_CONVERGED
        if run.input_tokens + run.output_tokens > self.policy.case_token_budget:
            return EXIT_BUDGET_EXCEEDED
        if run.round_no >= self.policy.max_rounds:
            return EXIT_EXHAUSTED
        return ""

    # -- the loop -------------------------------------------------------------

    def run_case(self, case: ExceptionCase, case_text: str) -> CaseOutcome:
        """Run one case to its exit. Safety cases stop at round 0."""
        started = time.perf_counter()
        run = self.prepare(case, case_text)
        if case.type == "safety_observation":
            # Round 0 by rule: no diagnosis, no plan, no model call.
            # A safety observation is a supervisor's case from the
            # moment it lands; the agent's whole job is to not bury it.
            run.exit_reason = EXIT_SAFETY_STOP
            self.sink.emit(RunEvent(type="loop_exit", case_id=case.case_id,
                                    detail={"exit_reason": EXIT_SAFETY_STOP, "rounds": 0}))
            return self._outcome(run, started)
        while True:
            run.round_no += 1
            round_in, round_out = run.input_tokens, run.output_tokens
            self.diagnose_round(run)
            self.plan_round(run)
            self.verify_round(run)
            self.finalize_round(run, round_in, round_out)
            run.exit_reason = self.decide_exit(run)
            if run.exit_reason:
                break
        self.sink.emit(RunEvent(type="loop_exit", case_id=case.case_id,
                                detail={"exit_reason": run.exit_reason, "rounds": run.round_no}))
        return self._outcome(run, started)

    def _outcome(self, run: CaseRun, started: float) -> CaseOutcome:
        return CaseOutcome(
            diagnosis=run.diagnosis,
            plan=run.plan,
            open_gaps=list(run.gaps),
            rounds=list(run.rounds),
            exit_reason=run.exit_reason,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            latency_seconds=round(time.perf_counter() - started, 3),
            backend=run.backend or getattr(self.reasoner, "name", ""),
            model=run.model or getattr(self.reasoner, "model", ""),
            recurrence=recurrence.with_departure(run.recurrence, run.diagnosis),
        )


class BoardLoopController:
    """The outer loop: work the board in priority order, to budget."""

    def __init__(self, policy: BoardPolicy | None = None, sink: EventSink | None = None) -> None:
        self.policy = policy or BoardPolicy()
        self.sink = sink or NullSink()

    def run(
        self,
        cases: list[ExceptionCase],
        process: Callable[[ExceptionCase], CaseOutcome],
    ) -> tuple[BoardRunResult, list[tuple[ExceptionCase, CaseOutcome]]]:
        """Work cases via ``process`` until the board clears or the
        shift budget is spent. Returns the run summary + outcomes."""
        ordered = board_order(cases)
        processed: list[str] = []
        exits: dict[str, str] = {}
        outcomes: list[tuple[ExceptionCase, CaseOutcome]] = []
        remaining: list[dict] = []
        tokens_spent = 0
        stop_reason = ""
        for index, case in enumerate(ordered):
            if len(processed) >= self.policy.max_cases:
                stop_reason = f"shift case budget reached ({self.policy.max_cases} cases)"
            elif tokens_spent >= self.policy.board_token_budget:
                stop_reason = f"shift token budget reached ({self.policy.board_token_budget} tokens)"
            if stop_reason:
                for leftover in ordered[index:]:
                    remaining.append({"case_id": leftover.case_id, "reason": stop_reason})
                break
            score, reason = priority_for(case)
            self.sink.emit(RunEvent(type="case_selected", case_id=case.case_id,
                                    detail={"priority": score, "reason": reason}))
            outcome = process(case)
            processed.append(case.case_id)
            exits[case.case_id] = outcome.exit_reason
            outcomes.append((case, outcome))
            tokens_spent += outcome.input_tokens + outcome.output_tokens
        status = "board_clear" if not remaining else "shift_budget_exceeded"
        in_tokens = sum(o.input_tokens for _, o in outcomes)
        out_tokens = sum(o.output_tokens for _, o in outcomes)
        from .backends import estimate_cost_usd
        from .schemas import CaseTelemetry

        model = outcomes[0][1].model if outcomes else ""
        telemetry = CaseTelemetry(
            backend=outcomes[0][1].backend if outcomes else "",
            model=model,
            rounds=sum(len(o.rounds) for _, o in outcomes),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            estimated_cost_usd=estimate_cost_usd(model, in_tokens, out_tokens) if outcomes else None,
            latency_seconds=round(sum(o.latency_seconds for _, o in outcomes), 3),
        )
        summary = self._handoff(processed, exits, remaining)
        result = BoardRunResult(
            run_id="",
            status=status,
            processed=processed,
            exits=exits,
            remaining=remaining,
            handoff_summary=summary,
            telemetry=telemetry,
        )
        self.sink.emit(RunEvent(type="board_finished", case_id="",
                                detail={"processed": len(processed), "remaining": len(remaining)}))
        return result, outcomes

    @staticmethod
    def _handoff(processed: list[str], exits: dict[str, str], remaining: list[dict]) -> str:
        lines = [f"Board run: {len(processed)} case(s) worked to the supervisor gate."]
        by_exit: dict[str, list[str]] = {}
        for case_id in processed:
            by_exit.setdefault(exits.get(case_id, ""), []).append(case_id)
        for exit_reason, ids in sorted(by_exit.items()):
            lines.append(f"  {exit_reason}: {', '.join(ids)}")
        if remaining:
            lines.append(f"{len(remaining)} case(s) NOT worked — handed to the next shift:")
            for item in remaining:
                lines.append(f"  {item['case_id']}: {item['reason']}")
        else:
            lines.append("Board clear: nothing handed over.")
        lines.append("Every worked case waits at the supervisor gate; nothing was actioned by the agent.")
        return "\n".join(lines)
