"""The loop controllers: where the agent's iteration is governed.

Two nested loops, both driven from this module — the LangGraph
graphs in ``graph.py`` wire these same step functions into nodes, so
the graph and the plain-code path can never drift apart.

**The case loop** (per exception)::

    diagnose → plan → verify ── gaps? ──→ back to diagnose (gaps only)
                        │
                        └──── exit ──→ draft → supervisor gate

The verify round merges two critics: the reasoner's critique and
the code checks in ``guardrails.verify_plan``. Code wins ties — a
model cannot talk a missing safety step out of existence. Exits are
decided in code (:meth:`CaseLoopController.decide_exit`):

- ``SAFETY_STOP`` — a safety_observation case (round 0, before any
  reasoning), or a plan step that recommends a safety bypass.
- ``BLOCKED_EXTERNAL`` — a required part is unavailable with no
  approved substitute on the shelf. Looping cannot fix a stockout;
  the case escalates to whoever can order or substitute.
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

from . import guardrails, kb
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
    RoundRecord,
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


def _gap_keys(gaps: list[Gap]) -> list[str]:
    return [f"{g.kind}:{g.ref}" for g in gaps]


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
    ) -> None:
        self.reasoner = reasoner
        self.retriever = retriever
        self.policy = policy or LoopPolicy()
        self.sink = sink or NullSink()

    # -- context ------------------------------------------------------------

    def prepare(self, case: ExceptionCase, case_text: str) -> CaseRun:
        query = f"{case.machine_id} {case.type} {case.title} {case.description}"
        chunks = self.retriever.retrieve(query, limit=5)
        ctx = CaseContext(case=case, case_text=case_text, kb_context=chunks)
        return CaseRun(case=case, ctx=ctx)

    # -- round steps (also the graph's nodes) -------------------------------

    def diagnose_round(self, run: CaseRun) -> None:
        self.sink.emit(RunEvent(type="round_started", case_id=run.case.case_id,
                                detail={"round": run.round_no, "phase": "diagnose"}))
        out = self.reasoner.diagnose(run.ctx, focus=run.gaps)
        run.diagnosis = out.payload
        self._account(run, out)

    def plan_round(self, run: CaseRun) -> None:
        self.sink.emit(RunEvent(type="round_started", case_id=run.case.case_id,
                                detail={"round": run.round_no, "phase": "plan"}))
        out = self.reasoner.draft_plan(run.ctx, run.diagnosis, focus=run.gaps)
        run.plan = out.payload
        self._account(run, out)

    def verify_round(self, run: CaseRun) -> None:
        self.sink.emit(RunEvent(type="round_started", case_id=run.case.case_id,
                                detail={"round": run.round_no, "phase": "verify"}))
        previous_keys = _gap_keys(run.gaps)
        out = self.reasoner.critique(run.ctx, run.diagnosis, run.plan)
        self._account(run, out)
        critique: Critique = out.payload
        code_gaps = guardrails.verify_plan(run.case, run.diagnosis, run.plan)
        merged = _merge_gaps(critique.gaps, code_gaps)
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
        """A part gap the loop cannot work: not stocked, or stocked
        out with no approved substitute on the shelf."""
        top = run.diagnosis.top()
        sop = kb.sop_for(run.case.machine_id, top.fault_class) if top else None
        for gap in run.gaps:
            if gap.kind != "part_unavailable":
                continue
            part = gap.ref
            if kb.part_qty(part) is None:
                return part  # not a stocked part at all — order/vendor
            substitute = kb.substitute_for(sop, part)
            if substitute and (kb.part_qty(substitute) or 0) >= 1:
                continue  # a substitute exists; the planner can work it
            if (kb.part_qty(part) or 0) >= 1:
                continue  # shelf has some; a plan revision can re-check qty
            return part
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
