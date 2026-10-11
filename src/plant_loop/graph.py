"""The loops as LangGraph state graphs.

Two graphs, mirroring the two controllers in ``loops.py`` — the
nodes call the controllers' step functions, so the graph path and
the plain-code path share one implementation of every round.

**Case graph** (one exception)::

    intake → screen → prepare ── safety? ──→ draft
                        │
                        └→ diagnose → plan → verify ── gaps ──→ diagnose (loop back)
                                                    │
                                                    └ exit → draft → supervisor gate ⏸ → record decision

- ``screen`` runs the injection screen over the case's free text;
  flagged sentences never reach a model, and the flag stays visible.
- ``prepare`` retrieves the KB context and — for a
  safety_observation — sets the SAFETY_STOP exit immediately: the
  case routes to its escalation draft without a single reasoning
  round (round 0 by rule).
- the loop-back edge is a conditional edge out of ``verify`` driven
  by :meth:`CaseLoopController.decide_exit`: another round works
  only the open gaps (they ride the state as the next round's focus).
- ``draft`` assembles the record: priority (code), guardrails
  (code), and the draft work order or escalation note (templates).
- ``supervisor_gate`` pauses with ``interrupt()``; a decision
  resumes the thread into ``record_decision``. In ``record`` gate
  mode (board runs, checkpoints off) the gate is a pass-through and
  the case simply waits as ``awaiting_supervisor``.

**Board graph** (one shift run)::

    order → select ── budget/clear ──→ handoff
              ↑          │
              └──── process (runs the case graph per case)

``select`` applies the board budget in code; ``process`` runs the
case graph in record mode and reports each finished record through
the ``on_case_result`` callback so the service persists it at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import drafts, guardrails, screening
from .backends import estimate_cost_usd
from .events import NullSink, RunEvent, format_timestamp
from .loops import BoardPolicy, CaseLoopController, CaseRun, LoopPolicy
from .ports import EventSink
from .priority import board_order, priority_for
from .schemas import (
    EXIT_CONVERGED,
    EXIT_SAFETY_STOP,
    BoardRunResult,
    CaseResult,
    CaseTelemetry,
    DraftInfo,
    ExceptionCase,
    Gap,
    RoundRecord,
    TraceStep,
)

_NODE_TITLES = {
    "intake": "Intake — case normalised",
    "screen": "Untrusted-text screen — injection patterns flagged and fenced",
    "prepare": "Prepare — KB context retrieved, loop state initialised",
    "diagnose": "Diagnose round — ranked fault hypotheses with evidence",
    "plan": "Plan round — corrective steps, parts, mandatory safety steps",
    "verify": "Verify round — critic + code checks produce the gap list",
    "draft": "Draft — record assembled, work order or escalation drafted",
    "supervisor_gate": "Supervisor gate — the record waits for a human",
    "record_decision": "Supervisor decision recorded",
}


class CaseState(TypedDict, total=False):
    case: dict
    gate_mode: str  # "pause" (interrupt at the gate) | "record"
    case_text: str
    injection_flags: list[str]
    kb_chunks: list[dict]
    session: dict
    diagnosis: dict
    plan: dict
    gaps: list[dict]
    rounds: list[dict]
    round_no: int
    input_tokens: int
    output_tokens: int
    backend: str
    model: str
    exit_reason: str
    result: dict
    decision: dict
    trace: list[dict]
    started_at: float


@dataclass
class CaseRunOutcome:
    """The outcome of starting a case run: the record + whether it waits."""

    result: CaseResult
    awaiting_supervisor: bool


def _node(state: CaseState, name: str, sink: EventSink, fn) -> dict:
    """Run one node with timing, tracing, and events (the house wrapper)."""
    case_id = state["case"]["case_id"]
    sink.emit(RunEvent(type="node_started", case_id=case_id, node=name))
    started = time.perf_counter()
    update = fn(state)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    details = update.pop("_details", []) if isinstance(update, dict) else []
    step = TraceStep(
        name=name, title=_NODE_TITLES.get(name, name), details=details, duration_ms=duration_ms
    )
    trace = list(state.get("trace", []))
    trace.append(step.model_dump())
    update["trace"] = trace
    if isinstance(update.get("result"), dict):
        merged = dict(update["result"])
        merged["trace"] = trace
        update["result"] = merged
    sink.emit(RunEvent(type="node_finished", case_id=case_id, node=name, duration_ms=duration_ms))
    return update


def _run_from_state(controller: CaseLoopController, state: CaseState) -> CaseRun:
    from .ports import CaseContext
    from .schemas import ActionPlan, Diagnosis

    case = ExceptionCase.model_validate(state["case"])
    ctx = CaseContext(
        case=case,
        case_text=state.get("case_text", ""),
        kb_context=state.get("kb_chunks", []),
        session=state.get("session", {}),
    )
    run = CaseRun(case=case, ctx=ctx)
    run.diagnosis = Diagnosis.model_validate(state["diagnosis"]) if state.get("diagnosis") else Diagnosis()
    run.plan = ActionPlan.model_validate(state["plan"]) if state.get("plan") else ActionPlan()
    run.gaps = [Gap.model_validate(g) for g in state.get("gaps", [])]
    run.rounds = [RoundRecord.model_validate(r) for r in state.get("rounds", [])]
    run.round_no = state.get("round_no", 0)
    run.input_tokens = state.get("input_tokens", 0)
    run.output_tokens = state.get("output_tokens", 0)
    run.backend = state.get("backend", "")
    run.model = state.get("model", "")
    run.exit_reason = state.get("exit_reason", "")
    return run


def _run_to_state(run: CaseRun) -> dict:
    return {
        "diagnosis": run.diagnosis.model_dump(),
        "plan": run.plan.model_dump(),
        "gaps": [g.model_dump() for g in run.gaps],
        "rounds": [r.model_dump() for r in run.rounds],
        "round_no": run.round_no,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "backend": run.backend,
        "model": run.model,
        "exit_reason": run.exit_reason,
        "session": run.ctx.session,
        "kb_chunks": run.ctx.kb_context,
    }


def build_case_graph(
    controller: CaseLoopController,
    checkpointer,
    sink: EventSink | None = None,
):
    """Compile the case graph with its controller bound."""
    sink = sink or NullSink()

    def intake(state: CaseState) -> dict:
        case = ExceptionCase.model_validate(state["case"])
        case.title = " ".join(case.title.split())
        case.description = case.description.strip()
        return {
            "case": case.model_dump(),
            "_details": [
                f"{case.type} on {case.machine_id or 'no machine'} ({case.line_id or 'no line'})",
                f"reported by {case.reporter or 'unknown'} at {case.reported_at or '—'}",
            ],
        }

    def screen(state: CaseState) -> dict:
        case = ExceptionCase.model_validate(state["case"])
        flags = screening.screen_text("description", case.description) + screening.screen_text(
            "notes", case.notes
        )
        clean_notes = screening.sanitize_text(case.notes)
        clean_description = screening.sanitize_text(case.description)
        readings = ", ".join(f"{k}={v}" for k, v in sorted(case.readings.items())) or "none"
        header = [
            f"Exception {case.case_id}: {case.title}",
            f"Type: {case.type}; machine: {case.machine_id}; line: {case.line_id}",
            f"Product/batch: {case.product}/{case.batch}; downtime so far: {case.downtime_minutes} min",
            f"Description: {clean_description}",
            f"Readings: {readings}",
        ]
        case_text = "\n".join(header)
        fenced = screening.fence(clean_notes)
        if fenced:
            case_text += f"\nOperator notes:\n{fenced}"
        return {
            "injection_flags": flags,
            "case_text": case_text,
            "_details": (
                [f"{len(flags)} injection pattern(s) flagged and removed from model state"]
                if flags
                else ["no injection patterns found"]
            ),
        }

    def prepare(state: CaseState) -> dict:
        case = ExceptionCase.model_validate(state["case"])
        run = controller.prepare(case, state.get("case_text", ""))
        update = _run_to_state(run)
        if case.type == "safety_observation":
            update["exit_reason"] = EXIT_SAFETY_STOP
            update["_details"] = [
                "safety observation — SAFETY_STOP at round 0 by rule: no diagnosis, no plan, no model call"
            ]
        else:
            update["_details"] = [f"{len(run.ctx.kb_context)} KB chunk(s) retrieved for the diagnose round"]
        return update

    def diagnose(state: CaseState) -> dict:
        run = _run_from_state(controller, state)
        run.round_no += 1
        controller.diagnose_round(run)
        update = _run_to_state(run)
        top = run.diagnosis.top()
        update["_details"] = [
            f"round {run.round_no}: top hypothesis {top.fault_class if top else '—'} "
            f"(confidence {top.confidence:.2f})" if top else "no hypothesis produced"
        ]
        return update

    def plan(state: CaseState) -> dict:
        run = _run_from_state(controller, state)
        controller.plan_round(run)
        update = _run_to_state(run)
        update["_details"] = [
            f"round {run.round_no}: {len(run.plan.steps)} step(s), "
            f"{len(run.plan.required_parts)} part requirement(s), "
            f"est. downtime {run.plan.estimated_downtime_min} min"
        ]
        return update

    def verify(state: CaseState) -> dict:
        run = _run_from_state(controller, state)
        round_in, round_out = run.input_tokens, run.output_tokens
        controller.verify_round(run)
        controller.finalize_round(run, round_in, round_out)
        exit_reason = controller.decide_exit(run)
        update = _run_to_state(run)
        update["exit_reason"] = exit_reason
        details = [
            f"round {run.round_no}: {len(run.gaps)} gap(s) open"
            + (f" — exit: {exit_reason}" if exit_reason else " — looping on the gaps")
        ]
        if run.rounds and run.rounds[-1].substitutions:
            details.append(
                "substitute search: " + "; ".join(run.rounds[-1].substitutions)
            )
        update["_details"] = details
        return update

    def route_after_prepare(state: CaseState) -> str:
        return "draft" if state.get("exit_reason") == EXIT_SAFETY_STOP else "diagnose"

    def route_after_verify(state: CaseState) -> str:
        return "draft" if state.get("exit_reason") else "diagnose"

    def draft(state: CaseState) -> dict:
        case = ExceptionCase.model_validate(state["case"])
        run = _run_from_state(controller, state)
        exit_reason = state.get("exit_reason") or run.exit_reason
        score, reason = priority_for(case)
        telemetry = CaseTelemetry(
            backend=run.backend or getattr(controller.reasoner, "name", ""),
            model=run.model or getattr(controller.reasoner, "model", ""),
            rounds=len(run.rounds),
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            estimated_cost_usd=estimate_cost_usd(run.model, run.input_tokens, run.output_tokens),
            latency_seconds=round(time.time() - state.get("started_at", time.time()), 3),
        )
        result = CaseResult(
            case_id=case.case_id,
            status="awaiting_supervisor",
            exit_reason=exit_reason,
            type=case.type,
            title=case.title,
            machine_id=case.machine_id,
            line_id=case.line_id,
            priority=score,
            priority_reason=reason,
            diagnosis=run.diagnosis,
            plan=run.plan,
            open_gaps=[] if exit_reason == EXIT_CONVERGED else list(run.gaps),
            rounds=list(run.rounds),
            injection_flags=state.get("injection_flags", []),
            telemetry=telemetry,
            reported_at=case.reported_at,
            created_at=format_timestamp(datetime.now(timezone.utc)),
        )
        if exit_reason == EXIT_CONVERGED:
            text = drafts.build_work_order(case, result)
            result.draft = DraftInfo(kind="work_order", text=text)
        else:
            text = drafts.build_escalation(case, result)
            result.draft = DraftInfo(kind="escalation", text=text)
        result.guardrails = guardrails.run_guardrails(result)
        return {
            "result": result.model_dump(),
            "_details": [
                f"exit: {exit_reason}; draft: {result.draft.kind}",
                f"priority: {score} ({reason})",
                f"guardrails: {'passed' if result.guardrails.passed else 'FAILED — see record'}",
                "status: awaiting_supervisor — nothing actioned, nothing sent anywhere",
            ],
        }

    def supervisor_gate(state: CaseState) -> dict:
        if state.get("gate_mode") == "record":
            return {"_details": ["gate mode record — the case waits as awaiting_supervisor"]}
        outcome = interrupt(
            {
                "case_id": state["case"]["case_id"],
                "status": "awaiting_supervisor",
                "prompt": "Approve the draft (work order or escalation), or reject it.",
            }
        )
        decision = outcome if isinstance(outcome, dict) else {"decision": str(outcome)}
        return {"decision": decision, "_details": ["supervisor decision received"]}

    def record_decision(state: CaseState) -> dict:
        result = CaseResult.model_validate(state["result"])
        decision = state.get("decision", {})
        verdict = str(decision.get("decision", "")).strip().lower()
        if verdict:
            result.decision = verdict
            result.decision_by = str(decision.get("by", ""))
            result.decision_reason = str(decision.get("reason", ""))
            result.status = "approved" if verdict == "approved" else "rejected"
        return {
            "result": result.model_dump(),
            "_details": [f"decision: {verdict or 'none recorded'}; status: {result.status}"],
        }

    builder = StateGraph(CaseState)
    builder.add_node("intake", lambda s: _node(s, "intake", sink, intake))
    builder.add_node("screen", lambda s: _node(s, "screen", sink, screen))
    builder.add_node("prepare", lambda s: _node(s, "prepare", sink, prepare))
    builder.add_node("diagnose", lambda s: _node(s, "diagnose", sink, diagnose))
    builder.add_node("plan", lambda s: _node(s, "plan", sink, plan))
    builder.add_node("verify", lambda s: _node(s, "verify", sink, verify))
    builder.add_node("draft", lambda s: _node(s, "draft", sink, draft))
    builder.add_node("supervisor_gate", lambda s: _node(s, "supervisor_gate", sink, supervisor_gate))
    builder.add_node("record_decision", lambda s: _node(s, "record_decision", sink, record_decision))
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "screen")
    builder.add_edge("screen", "prepare")
    builder.add_conditional_edges("prepare", route_after_prepare, {"draft": "draft", "diagnose": "diagnose"})
    builder.add_edge("diagnose", "plan")
    builder.add_edge("plan", "verify")
    builder.add_conditional_edges("verify", route_after_verify, {"draft": "draft", "diagnose": "diagnose"})
    builder.add_edge("draft", "supervisor_gate")
    builder.add_edge("supervisor_gate", "record_decision")
    builder.add_edge("record_decision", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Board graph
# ---------------------------------------------------------------------------


class BoardState(TypedDict, total=False):
    run_id: str
    queue: list[dict]
    current: dict
    processed: list[str]
    exits: dict
    remaining: list[dict]
    results: list[dict]
    tokens_spent: int
    stop_reason: str
    board_result: dict


def build_board_graph(
    case_graph,
    board_policy: BoardPolicy,
    on_case_result: Callable[[CaseResult], None] | None = None,
    sink: EventSink | None = None,
):
    """Compile the board graph. ``case_graph`` is the compiled case
    graph; finished records are reported via ``on_case_result``."""
    sink = sink or NullSink()

    def order(state: BoardState) -> dict:
        cases = [ExceptionCase.model_validate(c) for c in state.get("queue", [])]
        ordered = board_order(cases)
        return {
            "queue": [c.model_dump() for c in ordered],
            "processed": [],
            "exits": {},
            "remaining": [],
            "results": [],
            "tokens_spent": 0,
            "stop_reason": "",
        }

    def select(state: BoardState) -> dict:
        queue = list(state.get("queue", []))
        stop = ""
        if not queue:
            stop = "board clear"
        elif len(state.get("processed", [])) >= board_policy.max_cases:
            stop = f"shift case budget reached ({board_policy.max_cases} cases)"
        elif state.get("tokens_spent", 0) >= board_policy.board_token_budget:
            stop = f"shift token budget reached ({board_policy.board_token_budget} tokens)"
        if stop:
            remaining = [
                {"case_id": c["case_id"], "reason": stop if stop != "board clear" else "not reached"}
                for c in queue
            ] if stop != "board clear" else []
            return {"current": {}, "remaining": remaining, "stop_reason": stop, "queue": []}
        current = queue.pop(0)
        case = ExceptionCase.model_validate(current)
        score, reason = priority_for(case)
        sink.emit(RunEvent(type="case_selected", case_id=case.case_id,
                           detail={"priority": score, "reason": reason}))
        return {"current": current, "queue": queue, "stop_reason": ""}

    def route_after_select(state: BoardState) -> str:
        return "process" if state.get("current") else "handoff"

    def process(state: BoardState) -> dict:
        case = ExceptionCase.model_validate(state["current"])
        config = {"configurable": {"thread_id": f"{state.get('run_id', 'board')}:{case.case_id}"}}
        output = case_graph.invoke(
            {
                "case": case.model_dump(),
                "gate_mode": "record",
                "trace": [],
                "started_at": time.time(),
            },
            config=config,
        )
        result = CaseResult.model_validate(output["result"])
        if on_case_result is not None:
            on_case_result(result)
        processed = list(state.get("processed", [])) + [case.case_id]
        exits = dict(state.get("exits", {}))
        exits[case.case_id] = result.exit_reason
        results = list(state.get("results", [])) + [result.model_dump()]
        tokens = state.get("tokens_spent", 0) + result.telemetry.input_tokens + result.telemetry.output_tokens
        return {
            "processed": processed,
            "exits": exits,
            "results": results,
            "tokens_spent": tokens,
            "current": {},
        }

    def handoff(state: BoardState) -> dict:
        from .loops import BoardLoopController

        processed = state.get("processed", [])
        exits = state.get("exits", {})
        remaining = state.get("remaining", [])
        results = [CaseResult.model_validate(r) for r in state.get("results", [])]
        in_tokens = sum(r.telemetry.input_tokens for r in results)
        out_tokens = sum(r.telemetry.output_tokens for r in results)
        model = results[0].telemetry.model if results else ""
        telemetry = CaseTelemetry(
            backend=results[0].telemetry.backend if results else "",
            model=model,
            rounds=sum(len(r.rounds) for r in results),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            estimated_cost_usd=estimate_cost_usd(model, in_tokens, out_tokens) if results else None,
            latency_seconds=round(sum(r.telemetry.latency_seconds for r in results), 3),
        )
        result = BoardRunResult(
            run_id=state.get("run_id", ""),
            status="board_clear" if not remaining else "shift_budget_exceeded",
            processed=processed,
            exits=exits,
            remaining=remaining,
            handoff_summary=BoardLoopController._handoff(processed, exits, remaining),
            telemetry=telemetry,
            created_at=format_timestamp(datetime.now(timezone.utc)),
        )
        sink.emit(RunEvent(type="board_finished", case_id="",
                           detail={"processed": len(processed), "remaining": len(remaining)}))
        return {"board_result": result.model_dump()}

    builder = StateGraph(BoardState)
    builder.add_node("order", order)
    builder.add_node("select", select)
    builder.add_node("process", process)
    builder.add_node("handoff", handoff)
    builder.add_edge(START, "order")
    builder.add_edge("order", "select")
    builder.add_conditional_edges("select", route_after_select, {"process": "process", "handoff": "handoff"})
    builder.add_edge("process", "select")
    builder.add_edge("handoff", END)
    return builder.compile()
