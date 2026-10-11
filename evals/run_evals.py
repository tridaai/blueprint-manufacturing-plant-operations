"""Golden loop eval: known cases in, loop behaviour checked.

Twenty-eight entries — twenty cases with authored expectations
(per-case diagnosis class, exit reason, rounds used, draft kind,
open gaps, verify-stage substitution records), three *defect* cases
(a deliberately defective planner that drops safety steps, names an
unstocked part, and proposes a banned action), three board cases
(priority order, case budget, token budget), and two *outcome*
entries (cases run end to end, supervisor outcomes recorded, the
live-accuracy stats and the recurring-fault watch asserted — the
outcome loop measuring itself). Every case runs the full pipeline
through a fresh service on the deterministic offline backend, so
the gate is hermetic: ``uv run python evals/run_evals.py`` exits
non-zero below the gate.

These expectations encode the intended behaviour of the loops as
designed — a regression gate, not an accuracy claim about any model.

Results are written to ``evals/results.json`` and to the packaged
``src/plant_loop/data/eval_results.json`` the console reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_loop.backends import OfflineReasoner  # noqa: E402
from plant_loop.embeddings import HashingEmbedder, InMemoryVectorIndex  # noqa: E402
from plant_loop.kb import KBRetriever  # noqa: E402
from plant_loop.loops import BoardPolicy  # noqa: E402
from plant_loop.objectstore import InMemoryObjectStore  # noqa: E402
from plant_loop.schemas import (  # noqa: E402
    ExceptionCase,
    PartRequirement,
    PlanStep,
    SupervisorOutcome,
)
from plant_loop.service import PlantService  # noqa: E402
from plant_loop.stats import diagnosis_stats  # noqa: E402
from plant_loop.store import InMemoryStore  # noqa: E402

GATE = 0.95


def load_cases() -> list[dict]:
    path = Path(__file__).resolve().parent / "golden.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fresh_service(reasoner=None) -> PlantService:
    embedder = HashingEmbedder()
    return PlantService(
        reasoner=reasoner or OfflineReasoner(),
        retriever=KBRetriever(embedder, InMemoryVectorIndex()),
        store=InMemoryStore(),
        objects=InMemoryObjectStore(),
    )


# ---------------------------------------------------------------------------
# Defective planners: the verify round exists for components like these
# ---------------------------------------------------------------------------


class DropSafetyPlanner(OfflineReasoner):
    """Plans correctly, then strips every safety step."""

    def draft_plan(self, ctx, diagnosis, focus):
        out = super().draft_plan(ctx, diagnosis, focus)
        out.payload.steps = [s for s in out.payload.steps if s.kind != "safety"]
        return out


class WrongPartPlanner(OfflineReasoner):
    """Insists on a generic bearing the plant does not stock."""

    def draft_plan(self, ctx, diagnosis, focus):
        out = super().draft_plan(ctx, diagnosis, focus)
        out.payload.steps.append(
            PlanStep(step_id="fit-generic", text="Fit the generic bearing from the van stock.",
                     part_id="GEN-BEARING", qty=1)
        )
        out.payload.required_parts = [
            PartRequirement(part_id="GEN-BEARING", qty=1, available_qty=0, satisfied=False)
        ]
        return out


class BannedStepPlanner(OfflineReasoner):
    """Adds a step that defeats a safety device 'to save time'."""

    def draft_plan(self, ctx, diagnosis, focus):
        out = super().draft_plan(ctx, diagnosis, focus)
        out.payload.steps.append(
            PlanStep(step_id="test-hot", text="Bypass the interlock and run one test cycle to check the seal under pressure.")
        )
        return out


_DEFECTS = {
    "drop_safety_steps": DropSafetyPlanner,
    "wrong_part": WrongPartPlanner,
    "banned_step": BannedStepPlanner,
}


# ---------------------------------------------------------------------------
# Case checks
# ---------------------------------------------------------------------------


def check_result(case: dict, result) -> list[str]:
    expected = case["expected"]
    failures: list[str] = []
    if result.exit_reason != expected["exit"]:
        failures.append(f"exit: expected {expected['exit']}, got {result.exit_reason}")
    if "rounds" in expected and len(result.rounds) != expected["rounds"]:
        failures.append(f"rounds: expected {expected['rounds']}, got {len(result.rounds)}")
    if "diagnosis" in expected:
        top = result.diagnosis.top()
        actual = top.fault_class if top else ""
        if actual != expected["diagnosis"]:
            failures.append(f"diagnosis: expected {expected['diagnosis']}, got {actual}")
    if "draft" in expected and result.draft.kind != expected["draft"]:
        failures.append(f"draft: expected {expected['draft']}, got {result.draft.kind}")
    if expected.get("injection_flagged") and not result.injection_flags:
        failures.append("injection: expected at least one flag on the record")
    if "open_gap_kinds" in expected:
        kinds = {g.kind for g in result.open_gaps}
        for kind in expected["open_gap_kinds"]:
            if kind not in kinds:
                failures.append(f"gaps: expected open gap {kind}, got {sorted(kinds)}")
    if "plan_substitute" in expected:
        subs = {r.substitute_part_id for r in result.plan.required_parts}
        if expected["plan_substitute"] not in subs:
            failures.append(f"parts: expected substitute {expected['plan_substitute']}, got {sorted(subs)}")
    if "substitution_applied" in expected:
        applied = {s.substitute_part_id for s in result.plan.substitutions if s.status == "applied"}
        if expected["substitution_applied"] not in applied:
            failures.append(
                f"substitutions: expected an applied record for {expected['substitution_applied']}, "
                f"got {sorted(applied)}"
            )
        record = next(
            (s for s in result.plan.substitutions if s.substitute_part_id == expected["substitution_applied"]),
            None,
        )
        if record is not None and not (record.basis and record.spec_deltas):
            failures.append("substitutions: the applied record carries no basis/spec deltas")
    if "substitution_rejected" in expected:
        rejected = {
            s.substitute_part_id: s
            for s in result.plan.substitutions
            if s.status == "rejected"
        }
        record = rejected.get(expected["substitution_rejected"])
        if record is None:
            failures.append(
                f"substitutions: expected a rejected record for {expected['substitution_rejected']}, "
                f"got {sorted(rejected)}"
            )
        elif not record.reason:
            failures.append("substitutions: the rejected record names no reason")
    if expected.get("substitution_flag_in_draft") and "SUBSTITUTION" not in result.draft.text:
        failures.append("draft: expected the SUBSTITUTION flag on the work order")
    if expected.get("never_converged") and result.exit_reason == "converged":
        failures.append("defect case converged — the verify round failed to catch the defect")
    if result.status != "awaiting_supervisor":
        failures.append(f"status: expected awaiting_supervisor, got {result.status}")
    return failures


def run_case_entry(entry: dict) -> list[str]:
    reasoner = _DEFECTS[entry["defect"]]() if entry["kind"] == "defect" else None
    service = fresh_service(reasoner)
    case = ExceptionCase.model_validate(entry["case"])
    result = service.run_case(case).result
    return check_result(entry, result)


def run_board_entry(entry: dict) -> list[str]:
    service = fresh_service()
    cases = [ExceptionCase.model_validate(c) for c in entry["cases"]]
    policy = None
    if "board_policy" in entry:
        raw = entry["board_policy"]
        policy = BoardPolicy(
            max_cases=raw.get("max_cases", 20),
            board_token_budget=raw.get("max_tokens", 1_000_000),
        )
    result = service.run_board(cases, board_policy=policy)
    expected = entry["expected"]
    failures: list[str] = []
    if result.processed != expected["order"]:
        failures.append(f"order: expected {expected['order']}, got {result.processed}")
    if result.status != expected["status"]:
        failures.append(f"status: expected {expected['status']}, got {result.status}")
    remaining_ids = [item["case_id"] for item in result.remaining]
    if remaining_ids != expected.get("remaining", []):
        failures.append(f"remaining: expected {expected.get('remaining', [])}, got {remaining_ids}")
    for case_id, exit_reason in expected.get("exits", {}).items():
        if result.exits.get(case_id) != exit_reason:
            failures.append(f"exit[{case_id}]: expected {exit_reason}, got {result.exits.get(case_id)}")
    return failures


def run_outcome_entry(entry: dict) -> list[str]:
    """Run cases, record authored outcomes, assert the stats math
    and the watch — the outcome loop, gated like everything else."""
    service = fresh_service()
    for raw in entry["cases"]:
        service.run_case(ExceptionCase.model_validate(raw))
    for case_id, payload in entry.get("outcomes", {}).items():
        service.record_outcome(case_id, SupervisorOutcome.model_validate(payload))
    failures: list[str] = []
    stats = diagnosis_stats(service.store.all_cases())
    for key, want in entry.get("expected_stats", {}).items():
        got = stats.get(key)
        ok = abs(got - want) < 1e-3 if isinstance(want, float) else got == want
        if not ok:
            failures.append(f"stats.{key}: expected {want}, got {got}")
    if "expected_reliability_items" in entry:
        open_items = [i for i in service.store.all_board_items() if i.status == "open"]
        if len(open_items) != entry["expected_reliability_items"]:
            failures.append(
                f"reliability items: expected {entry['expected_reliability_items']}, "
                f"got {len(open_items)}"
            )
    return failures


def main() -> int:
    entries = load_cases()
    results = []
    passed = 0
    for entry in entries:
        if entry["kind"] == "board":
            failures = run_board_entry(entry)
        elif entry["kind"] == "outcome":
            failures = run_outcome_entry(entry)
        else:
            failures = run_case_entry(entry)
        ok = not failures
        passed += ok
        results.append({"id": entry["id"], "kind": entry["kind"], "passed": ok, "failures": failures})
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {entry['id']} ({entry['kind']})" + (f" — {'; '.join(failures)}" if failures else ""))
    total = len(entries)
    rate = passed / total if total else 0.0
    gate_passed = rate >= GATE
    print(f"\ngolden loop eval (backend=offline): {passed}/{total} passed ({rate:.1%})")
    print(f"gate: {'PASS' if gate_passed else 'FAIL'} (>= {GATE:.2f})")
    payload = {
        "total": total,
        "passed": passed,
        "rate": round(rate, 4),
        "gate": GATE,
        "gate_passed": gate_passed,
        "backend": "offline",
        "results": results,
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    packaged = Path(__file__).resolve().parents[1] / "src" / "plant_loop" / "data" / "eval_results.json"
    packaged.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
