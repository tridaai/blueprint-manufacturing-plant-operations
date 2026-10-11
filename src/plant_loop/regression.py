"""The reasoner regression gate: prove a model swap does not
regress the plant before the swapped profile is trusted.

The loop's value lives in its reasoner, and providers change models
constantly — a new Muse Spark point release, a fallback LLM swapped
in for cost, a surface change from Chat Completions to Responses.
Until now, swapping meant re-running the golden pack by eye and
hoping. This module makes the swap a gate instead:

- :func:`run_suite` drives a FIXED scenario suite — the golden
  pack's case entries, safety cases included — through one NAMED
  reasoner configuration and records what happened per case: exit,
  rounds, top diagnosis, model calls (counted at the port by
  :class:`CountingReasoner`), tokens, and the two safety pins.
- :func:`compare_runs` diffs two such records in code and returns a
  mechanical verdict against the criteria in :data:`GATE_CRITERIA`.
  Same two records in, same verdict out — the verdict is a function,
  not a judgement call, and the tests pin it.

The criteria are ordered, and the order is the point:

1. **Safety pins, first and absolute.** A safety-observation case
   that received even one model call, or a case that *converged*
   carrying a plan missing a mandatory safety step of its diagnosed
   SOP, fails the gate outright — before exits or agreement are
   even looked at. Safety is a criterion, not a metric to be
   averaged away. (Through the shipped loop these are unreachable —
   the safety stop is round 0 and the code verify cannot be talked
   out of a missing step — so the pin criteria also stand guard over
   records produced by any other harness or a future loop change.)
2. **Exit regressions.** Exits have a severity order
   (:data:`EXIT_SEVERITY`, printed on every report): a case that
   ends in a *more severe* exit under the candidate than under the
   baseline fails the gate. ``converged`` is the only exit that
   produces a work order, so losing it is the regression that
   matters most; the order ranks the rest by how much verified work
   the plant still gets.
3. **Diagnosis agreement.** The share of cases whose top fault
   class matches the baseline's must clear the authored floor.
   Below it, the candidate is not a slightly different reasoner —
   it is a different diagnoser wearing the same name.

Everything else — cases that *improved*, round and call deltas —
is reported, not gated: an FDE reads the deltas, the gate guards
the floor.

The harness observes through the existing seams only: cases run
through :class:`~plant_loop.service.PlantService` exactly as the
golden pack runs them, and calls are counted by wrapping the
reasoner at the port. No loop behaviour changes for this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import guardrails, kb
from .ports import CaseContext, RoundOutput
from .schemas import (
    EXIT_BLOCKED_EXTERNAL,
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONVERGED,
    EXIT_EXHAUSTED,
    EXIT_SAFETY_STOP,
    Diagnosis,
    ExceptionCase,
    Gap,
)

# ---------------------------------------------------------------------------
# Gate criteria — authored here, in one place, and printed on every
# report so a reader never has to take the verdict on faith.
# ---------------------------------------------------------------------------

#: Exit severity, best → worst. ``converged`` is the only exit that
#: produces a work order. ``blocked_external`` still hands the
#: supervisor a complete, verified diagnosis — the plant keeps the
#: knowledge work and loses only the fix. ``budget_exceeded`` stopped
#: mid-loop on a token bound: partial work, no verified plan.
#: ``exhausted`` spent every round with gaps open. ``safety_stop``
#: ranks last *as a regression*: for a case the baseline worked, a
#: candidate that now stops it as a safety event has refused
#: ordinary work (the dangerous direction — a safety case being
#: reasoned on — is a pin violation above, never a severity matter).
EXIT_SEVERITY: dict[str, int] = {
    EXIT_CONVERGED: 0,
    EXIT_BLOCKED_EXTERNAL: 1,
    EXIT_BUDGET_EXCEEDED: 2,
    EXIT_EXHAUSTED: 3,
    EXIT_SAFETY_STOP: 4,
}

GATE_CRITERIA = {
    "diagnosis_agreement_floor": 0.90,
}

#: The suite's identity. The suite is the golden pack's case entries
#: (golden.jsonl, kind == "case") — the same fixed scenarios the
#: golden gate checks, safety cases (G02, G18) among them.
SUITE_NAME = "golden-cases"
SUITE_VERSION = 1

RECORD_KIND = "reasoner-gate-run"


# ---------------------------------------------------------------------------
# Configuration stamp: how a run names the reasoner it drove
# ---------------------------------------------------------------------------


def config_stamp(reasoner, label: str = "") -> dict:
    """Name a reasoner configuration: backend + model + surface.

    The surface matters for Muse Spark: the same model reasons
    differently when its round-1 thread is carried into round 2
    (Responses) than when each round starts cold (Chat
    Completions), so the two adapters are different configurations
    for gate purposes even at the same model name. The backend's
    ``name`` already distinguishes them; the surface field spells
    it out on the record.
    """
    backend = getattr(reasoner, "name", "") or ""
    surface = ""
    if backend == "muse":
        surface = "chat"
    elif backend == "muse-responses":
        surface = "responses"
    return {
        "label": label or backend,
        "backend": backend,
        "model": getattr(reasoner, "model", "") or "",
        "surface": surface,
    }


# ---------------------------------------------------------------------------
# Call counting at the port
# ---------------------------------------------------------------------------


class CountingReasoner:
    """Wrap a reasoner and count its calls, per case.

    The loop owns the per-case context; the harness owns the count.
    Delegation is total — payloads pass through untouched, so the
    loop's behaviour under a counted reasoner is the behaviour
    being measured, not a rehearsal of it.
    """

    name = ""
    model = ""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "")
        self.model = getattr(inner, "model", "")
        self.calls = 0
        self.calls_by_phase: dict[str, int] = {"diagnose": 0, "draft_plan": 0, "critique": 0}

    def _counted(self, phase: str, out: RoundOutput) -> RoundOutput:
        self.calls += 1
        self.calls_by_phase[phase] += 1
        return out

    def diagnose(self, ctx: CaseContext, focus: list[Gap]) -> RoundOutput:
        return self._counted("diagnose", self.inner.diagnose(ctx, focus))

    def draft_plan(self, ctx: CaseContext, diagnosis: Diagnosis, focus: list[Gap]) -> RoundOutput:
        return self._counted("draft_plan", self.inner.draft_plan(ctx, diagnosis, focus))

    def critique(self, ctx: CaseContext, diagnosis: Diagnosis, plan) -> RoundOutput:
        return self._counted("critique", self.inner.critique(ctx, diagnosis, plan))


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def load_suite(repo_root: Path | None = None) -> list[ExceptionCase]:
    """The fixed scenario suite: the golden pack's case entries.

    Loaded from ``evals/golden.jsonl`` so the gate and the golden
    pack can never drift onto different scenarios. Only ``case``
    entries are individual loop scenarios (board / outcome / pm
    entries drive multi-case choreography the golden pack itself
    already gates).
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    path = root / "evals" / "golden.jsonl"
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("kind") == "case":
            cases.append(ExceptionCase.model_validate(entry["case"]))
    return cases


def _mandatory_steps_covered(case: ExceptionCase, result) -> bool | None:
    """Does the final plan carry every mandatory safety step of
    the diagnosed SOP? Checked with the guardrails' own verify —
    the same step-matching the loop's verify round applies — so
    the pin reads the plan exactly as the loop judged it. ``None``
    when the case never reached a plan (a round-0 safety stop has
    no plan to judge; absence is not a violation)."""
    top = result.diagnosis.top()
    if top is None or not result.plan.steps:
        return None
    gaps = guardrails.verify_plan(case, result.diagnosis, result.plan)
    return not any(g.kind == "missing_safety_step" for g in gaps)


def run_suite(reasoner, cases: list[ExceptionCase] | None = None, *, label: str = "") -> dict:
    """Drive the fixed suite through one reasoner configuration.

    Each case runs through a FRESH service — the golden pack's own
    discipline — so no case inherits another's history, claims, or
    outcomes: the record measures the reasoner, nothing else.
    """
    from .embeddings import HashingEmbedder, InMemoryVectorIndex
    from .kb import KBRetriever
    from .objectstore import InMemoryObjectStore
    from .service import PlantService
    from .store import InMemoryStore

    suite = cases if cases is not None else load_suite()
    counted = CountingReasoner(reasoner)
    records = []
    for case in suite:
        service = PlantService(
            reasoner=counted,
            retriever=KBRetriever(HashingEmbedder(), InMemoryVectorIndex()),
            store=InMemoryStore(),
            objects=InMemoryObjectStore(),
        )
        before = counted.calls
        result = service.run_case(case).result
        calls = counted.calls - before
        top = result.diagnosis.top()
        is_safety_case = case.type == "safety_observation"
        records.append(
            {
                "case_id": case.case_id,
                "type": case.type,
                "exit": result.exit_reason,
                "rounds": len(result.rounds),
                "diagnosis": top.fault_class if top else "",
                "model_calls": calls,
                "input_tokens": result.telemetry.input_tokens,
                "output_tokens": result.telemetry.output_tokens,
                "pins": {
                    # The pin, stated for safety cases only: zero
                    # model calls. For any other case: not applicable.
                    "safety_zero_calls": (calls == 0) if is_safety_case else None,
                    "mandatory_steps_covered": _mandatory_steps_covered(case, result),
                },
            }
        )
    return {
        "kind": RECORD_KIND,
        "version": 1,
        "suite": SUITE_NAME,
        "suite_version": SUITE_VERSION,
        "config": config_stamp(reasoner, label),
        "cases_total": len(records),
        "totals": {
            "model_calls": sum(r["model_calls"] for r in records),
            "rounds": sum(r["rounds"] for r in records),
            "input_tokens": sum(r["input_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
        },
        "cases": records,
    }


# ---------------------------------------------------------------------------
# Safety pins, read off a record
# ---------------------------------------------------------------------------


def pin_violations(record: dict) -> list[dict]:
    """Every safety-pin violation in one run record.

    - a safety-observation case that received any model call;
    - a case that CONVERGED while its plan missed a mandatory
      safety step (a non-converged case with an incomplete plan is
      the loop working as designed — the verify round held it).
    """
    violations = []
    for case in record.get("cases", []):
        pins = case.get("pins", {})
        if pins.get("safety_zero_calls") is False:
            violations.append(
                {
                    "case_id": case["case_id"],
                    "pin": "safety_zero_calls",
                    "detail": (
                        f"safety case received {case['model_calls']} model call(s); "
                        "a safety case must be stopped in code at round 0, unreasoned"
                    ),
                }
            )
        if case.get("exit") == EXIT_CONVERGED and pins.get("mandatory_steps_covered") is False:
            violations.append(
                {
                    "case_id": case["case_id"],
                    "pin": "mandatory_steps_covered",
                    "detail": (
                        "case converged with a plan missing a mandatory safety step "
                        "of its diagnosed SOP; a plan like that must never converge"
                    ),
                }
            )
    return violations


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def compare_runs(baseline: dict, candidate: dict) -> dict:
    """Diff two run records and return the gate verdict.

    Mechanical by construction: criteria are evaluated in the
    authored order (suite completeness → safety pins → exit
    regressions → diagnosis agreement) and the verdict is PASS
    only when every criterion passes. Improvements and cost deltas
    are reported alongside, never gated.
    """
    base_cases = {c["case_id"]: c for c in baseline.get("cases", [])}
    cand_cases = {c["case_id"]: c for c in candidate.get("cases", [])}

    criteria: list[dict] = []

    # 0 · the same suite, whole — a diff over different scenarios
    # is not a diff.
    missing = sorted(set(base_cases) - set(cand_cases))
    extra = sorted(set(cand_cases) - set(base_cases))
    criteria.append(
        {
            "name": "suite_completeness",
            "passed": not missing and not extra,
            "detail": (
                "candidate record covers the baseline suite exactly"
                if not missing and not extra
                else f"missing cases: {missing or '—'}; unexpected cases: {extra or '—'}"
            ),
        }
    )
    shared = sorted(set(base_cases) & set(cand_cases))

    # 1 · safety pins — first, absolute, on the CANDIDATE's record
    # (the baseline is the trusted reference; its own pins ride
    # along in the report for the reader).
    violations = pin_violations(candidate)
    criteria.append(
        {
            "name": "safety_pins",
            "passed": not violations,
            "detail": (
                "no safety-pin violations in the candidate record"
                if not violations
                else "; ".join(f"{v['case_id']}: {v['detail']}" for v in violations)
            ),
        }
    )

    # 2 · exit regressions against the severity order.
    exit_changes = []
    regressions = []
    for case_id in shared:
        before, after = base_cases[case_id]["exit"], cand_cases[case_id]["exit"]
        if before == after:
            continue
        regressed = EXIT_SEVERITY.get(after, 99) > EXIT_SEVERITY.get(before, 99)
        change = {
            "case_id": case_id,
            "baseline": before,
            "candidate": after,
            "regression": regressed,
        }
        exit_changes.append(change)
        if regressed:
            regressions.append(change)
    criteria.append(
        {
            "name": "exit_regressions",
            "passed": not regressions,
            "detail": (
                "no case ends in a more severe exit than the baseline"
                if not regressions
                else "; ".join(
                    f"{c['case_id']}: {c['baseline']} → {c['candidate']}" for c in regressions
                )
            ),
        }
    )

    # 3 · diagnosis agreement against the authored floor.
    differences = [
        {
            "case_id": case_id,
            "baseline": base_cases[case_id]["diagnosis"],
            "candidate": cand_cases[case_id]["diagnosis"],
        }
        for case_id in shared
        if base_cases[case_id]["diagnosis"] != cand_cases[case_id]["diagnosis"]
    ]
    compared = len(shared)
    agreement = (compared - len(differences)) / compared if compared else 0.0
    floor = GATE_CRITERIA["diagnosis_agreement_floor"]
    criteria.append(
        {
            "name": "diagnosis_agreement",
            "passed": agreement >= floor,
            "detail": (
                f"top fault class agrees on {compared - len(differences)}/{compared} cases "
                f"({agreement:.1%}); floor {floor:.0%}"
            ),
        }
    )

    verdict = "PASS" if all(c["passed"] for c in criteria) else "FAIL"
    return {
        "verdict": verdict,
        "baseline_config": baseline.get("config", {}),
        "candidate_config": candidate.get("config", {}),
        "criteria": criteria,
        "severity_order": sorted(EXIT_SEVERITY, key=EXIT_SEVERITY.get),
        "safety_pin_violations": violations,
        "baseline_pin_violations": pin_violations(baseline),
        "exit_changes": exit_changes,
        "diagnosis": {
            "compared": compared,
            "agreement": round(agreement, 4),
            "floor": floor,
            "differences": differences,
        },
        "deltas": {
            "rounds": {
                "baseline": baseline.get("totals", {}).get("rounds", 0),
                "candidate": candidate.get("totals", {}).get("rounds", 0),
            },
            "model_calls": {
                "baseline": baseline.get("totals", {}).get("model_calls", 0),
                "candidate": candidate.get("totals", {}).get("model_calls", 0),
            },
            "per_case": [
                {
                    "case_id": case_id,
                    "rounds_baseline": base_cases[case_id]["rounds"],
                    "rounds_candidate": cand_cases[case_id]["rounds"],
                    "calls_baseline": base_cases[case_id]["model_calls"],
                    "calls_candidate": cand_cases[case_id]["model_calls"],
                }
                for case_id in shared
                if base_cases[case_id]["rounds"] != cand_cases[case_id]["rounds"]
                or base_cases[case_id]["model_calls"] != cand_cases[case_id]["model_calls"]
            ],
        },
    }


def format_report(report: dict) -> str:
    """The comparison as plain text — CLI output and test fodder."""
    lines = [
        "reasoner regression gate",
        f"  baseline:  {report['baseline_config'].get('label', '')} "
        f"({report['baseline_config'].get('backend', '')}/{report['baseline_config'].get('model', '')}"
        + (f", surface {report['baseline_config']['surface']}" if report["baseline_config"].get("surface") else "")
        + ")",
        f"  candidate: {report['candidate_config'].get('label', '')} "
        f"({report['candidate_config'].get('backend', '')}/{report['candidate_config'].get('model', '')}"
        + (f", surface {report['candidate_config']['surface']}" if report["candidate_config"].get("surface") else "")
        + ")",
        f"  exit severity (best → worst): {' > '.join(report['severity_order'])}",
        "  criteria:",
    ]
    for criterion in report["criteria"]:
        mark = "PASS" if criterion["passed"] else "FAIL"
        lines.append(f"    [{mark}] {criterion['name']} — {criterion['detail']}")
    if report["exit_changes"]:
        lines.append("  exit changes:")
        for change in report["exit_changes"]:
            tag = "REGRESSION" if change["regression"] else "improvement"
            lines.append(
                f"    {change['case_id']}: {change['baseline']} → {change['candidate']} ({tag})"
            )
    if report["diagnosis"]["differences"]:
        lines.append("  diagnosis differences:")
        for diff in report["diagnosis"]["differences"]:
            lines.append(
                f"    {diff['case_id']}: {diff['baseline'] or '—'} → {diff['candidate'] or '—'}"
            )
    rounds = report["deltas"]["rounds"]
    calls = report["deltas"]["model_calls"]
    lines.append(
        f"  totals: rounds {rounds['baseline']} → {rounds['candidate']}, "
        f"model calls {calls['baseline']} → {calls['candidate']}"
    )
    lines.append(f"VERDICT: {report['verdict']}")
    return "\n".join(lines)
