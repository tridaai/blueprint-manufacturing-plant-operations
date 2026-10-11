"""Golden loop eval: known cases in, loop behaviour checked.

Thirty-eight entries — twenty cases with authored expectations
(per-case diagnosis class, exit reason, rounds used, draft kind,
open gaps, verify-stage substitution records), three *defect* cases
(a deliberately defective planner that drops safety steps, names an
unstocked part, and proposes a banned action), three board cases
(priority order, case budget, token budget), two *outcome*
entries (cases run end to end, supervisor outcomes recorded, the
live-accuracy stats and the recurring-fault watch asserted — the
outcome loop measuring itself), two *learning* entries (a case
run, a supervisor outcome recorded, a similar case run against the
SAME store — the confirmation written back to the machine's
history is the only difference between the two halves: one entry
proves the loop closes on what the plant learned, the other proves
an outcome that confirms nothing teaches nothing), and two
*allocation* entries (the board sharing one shelf: two cases want
the shelf's only belt — the higher-priority case converges and
claims it, the lower blocks with the claim named; the supervisor
rejects the winner at the gate, the claim is released, and a fresh
run lets the lower case converge — while a lone case with no
contention claims its substitute and behaves exactly as before),
and four *economics* entries (the board priced at the plant
model's line rates: exposure maths over a two-case board, a
minutes-only board on the unrated Utilities line, a dollar-heavy
maintenance alert that must still work *after* a cheap line-down
case — exposure never reorders the board — and cost accuracy
against recorded actuals, per line and plant-wide), and two
*carryover* entries (the next shift inheriting the board: a
budget-cut run whose blocked case resumes with its rounds intact
and takes the belt a decided-and-done case's released claim
freed, a waiting case's claim renewed with its provenance
stamped — and a carried case dropped at acceptance, audited,
its claim released, the case it blocked converging on the freed
part). Every case
runs the full pipeline through a fresh service on the
deterministic offline backend, so the gate is hermetic:
``uv run python evals/run_evals.py`` exits non-zero below the gate.

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


def run_learning_entry(entry: dict) -> list[str]:
    """Baseline case → supervisor outcome → follow-up case, one store.

    The baseline run must show the un-learned behaviour (its own
    expectations block); the outcome is then recorded (which writes
    the confirmation back to the machine's history — the entry
    count is asserted); the follow-up run is checked against the
    entry's expectations. Both runs share the service's store, so
    the write-back is the only difference between the halves.
    """
    service = fresh_service()
    baseline = service.run_case(ExceptionCase.model_validate(entry["case"])).result
    failures = [f"baseline: {f}" for f in check_result({"expected": entry["baseline"]}, baseline)]
    if "runner_up" in entry["baseline"]:
        hyps = baseline.diagnosis.hypotheses
        actual = hyps[1].fault_class if len(hyps) > 1 else ""
        if actual != entry["baseline"]["runner_up"]:
            failures.append(
                f"baseline: runner-up: expected {entry['baseline']['runner_up']}, got {actual}"
            )
    if "outcome" in entry:
        service.record_outcome(
            entry["case"]["case_id"], SupervisorOutcome.model_validate(entry["outcome"])
        )
    written = len(service.store.all_history_entries())
    if "written_back" in entry["expected"] and written != entry["expected"]["written_back"]:
        failures.append(
            f"write-back: expected {entry['expected']['written_back']} history "
            f"entr(ies), got {written}"
        )
    follow = service.run_case(ExceptionCase.model_validate(entry["followup"])).result
    failures += [f"followup: {f}" for f in check_result(entry, follow)]
    if entry["expected"].get("evidence_marks_confirmed"):
        top = follow.diagnosis.top()
        if not top or not any("supervisor-confirmed" in e for e in top.evidence):
            failures.append(
                "followup: the top hypothesis cites no supervisor-confirmed history"
            )
    return failures


def run_allocation_entry(entry: dict) -> list[str]:
    """Board-level parts allocation, proven on the seeded shelf.

    Two shapes. A *board* entry runs its cases in one run and
    asserts the exits, the claims the run recorded, and the claim
    named in the blocked case's gap; when it names a ``reject_case``
    the gate rejection must release that case's claim (stamped
    ``released_by_gate_rejection``) without rewriting the finished
    run's records, and its ``followup_case`` — run as a FRESH board
    run — must take the freed part. A *single-case* entry runs one
    case alone: no contention, so the exit and rounds are the
    pre-allocation behaviour, with the claim recorded on top.
    """
    service = fresh_service()
    expected = entry["expected"]
    failures: list[str] = []

    def _claims_mismatch(got, want, label):
        got_sorted = sorted((c.part_id, c.case_id, c.status) for c in got)
        want_sorted = sorted(tuple(w) for w in want)
        if got_sorted != want_sorted:
            return [f"{label}: expected {want_sorted}, got {got_sorted}"]
        return []

    if "single_case" in entry:
        case = ExceptionCase.model_validate(entry["single_case"])
        result = service.run_case(case).result
        if result.exit_reason != expected["exit"]:
            failures.append(f"exit: expected {expected['exit']}, got {result.exit_reason}")
        if len(result.rounds) != expected["rounds"]:
            failures.append(
                f"rounds: expected {expected['rounds']}, got {len(result.rounds)}"
            )
        failures += _claims_mismatch(
            service.allocation.all_claims(), expected.get("claims", []), "claims"
        )
        if "claim_original_part" in expected:
            claims = service.allocation.all_claims()
            if not claims or claims[0].original_part_id != expected["claim_original_part"]:
                failures.append(
                    "claims: expected the claim to record the requirement it "
                    f"satisfies ({expected['claim_original_part']})"
                )
        return failures

    cases = [ExceptionCase.model_validate(c) for c in entry["cases"]]
    result = service.run_board(cases)
    for case_id, exit_reason in expected.get("exits", {}).items():
        if result.exits.get(case_id) != exit_reason:
            failures.append(
                f"exit[{case_id}]: expected {exit_reason}, got {result.exits.get(case_id)}"
            )
    failures += _claims_mismatch(result.allocations, expected.get("claims", []), "run claims")
    blocked_id = expected.get("blocked_case", "")
    if blocked_id:
        record = service.store.get_case(blocked_id)
        phrase = expected.get("gap_claim_phrase", "")
        if not any(phrase in gap.detail for gap in record.open_gaps):
            failures.append(
                f"gap[{blocked_id}]: expected the claim named ({phrase!r}), "
                f"got {[g.detail for g in record.open_gaps]}"
            )
    reject_id = entry.get("reject_case", "")
    if reject_id:
        service.record_decision(
            reject_id, "rejected", by="Eval Supervisor", reason="eval: gate release"
        )
        claims = service.allocation.claims_for_case(reject_id)
        if len(claims) != 1 or claims[0].status != expected.get(
            "claim_after_reject", "released"
        ):
            failures.append(
                f"release: expected {reject_id}'s claim released, got "
                f"{[(c.part_id, c.status) for c in claims]}"
            )
        elif claims[0].release_reason != "released_by_gate_rejection":
            failures.append(
                f"release: expected the released_by_gate_rejection stamp, "
                f"got {claims[0].release_reason!r}"
            )
        if blocked_id:
            record = service.store.get_case(blocked_id)
            if record.exit_reason != "blocked_external":
                failures.append(
                    "release rewrote the finished run: the blocked case's "
                    f"record changed to {record.exit_reason}"
                )
    if "followup_case" in entry:
        follow = ExceptionCase.model_validate(entry["followup_case"])
        fresh = service.run_board([follow])
        if fresh.exits.get(follow.case_id) != expected.get("followup_exit"):
            failures.append(
                f"followup exit: expected {expected.get('followup_exit')}, "
                f"got {fresh.exits.get(follow.case_id)}"
            )
        failures += _claims_mismatch(
            fresh.allocations, expected.get("followup_claims", []), "followup claims"
        )
    return failures


def _dig(payload, path: str):
    node = payload
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _matches(got, want) -> bool:
    if want is None:
        return got is None
    if isinstance(want, float):
        return got is not None and abs(got - want) < 1e-6
    return got == want


def run_economics_entry(entry: dict) -> list[str]:
    """Downtime economics, proven against the plant model's rates.

    An entry runs its ``board_cases`` in one board run and/or its
    ``cases`` singly, records any ``outcomes``, then checks the
    expectations block: per-record economics fields (rate applied,
    minutes-only nulls on the unrated line), the run's totals, and
    dotted paths into the diagnosis stats' economics section.
    Expected dollar figures are hand-computed from the KB rates —
    line-a $2,400/hr, line-b $1,800/hr, line-c $950/hr, line-d
    $720/hr, line-e unrated — so the gate checks the pricing, not
    a snapshot of it.
    """
    service = fresh_service()
    expected = entry["expected"]
    failures: list[str] = []
    run_result = None
    if "board_cases" in entry:
        run_result = service.run_board(
            [ExceptionCase.model_validate(c) for c in entry["board_cases"]]
        )
    for raw in entry.get("cases", []):
        service.run_case(ExceptionCase.model_validate(raw))
    for case_id, payload in entry.get("outcomes", {}).items():
        service.record_outcome(case_id, SupervisorOutcome.model_validate(payload))
    if "order" in expected:
        if run_result is None or run_result.processed != expected["order"]:
            got = run_result.processed if run_result else None
            failures.append(f"order: expected {expected['order']}, got {got}")
    for case_id, fields in expected.get("records", {}).items():
        record = service.store.get_case(case_id)
        econ = record.economics.model_dump() if record and record.economics else {}
        for field, want in fields.items():
            got = econ.get(field)
            if not _matches(got, want):
                failures.append(
                    f"record[{case_id}].{field}: expected {want}, got {got}"
                )
    if "run" in expected:
        econ = run_result.economics.model_dump() if run_result and run_result.economics else {}
        for field, want in expected["run"].items():
            got = econ.get(field)
            if not _matches(got, want):
                failures.append(f"run.{field}: expected {want}, got {got}")
    if "stats" in expected:
        stats = diagnosis_stats(service.store.all_cases())
        for path, want in expected["stats"].items():
            got = _dig(stats, path)
            if not _matches(got, want):
                failures.append(f"stats.{path}: expected {want}, got {got}")
    return failures


def run_carryover_entry(entry: dict) -> list[str]:
    """Shift carry-over, proven across two board runs.

    Run 1 works ``board_cases`` (optionally under a case budget);
    the entry's gate ``decisions`` and ``outcomes`` then land, and
    run 2 accepts run 1's board by ``carry_over_from`` (with any
    ``drop`` entries). The checks read run 1's computed manifest,
    run 2's settlement — blocked cases re-worked with their round
    counts intact, gate cases inherited standing, drops audited —
    the whole claim ledger after the turnover (renewed claims
    stamped with their provenance, finished cases' claims
    released with ``released_run_ended``), and the plant-wide
    case count: a case worked on two boards is one case in the
    stats, never two.
    """
    service = fresh_service()
    expected = entry["expected"]
    failures: list[str] = []
    policy_spec = entry.get("board_policy") or {}
    policy = (
        BoardPolicy(max_cases=policy_spec["max_cases"])
        if "max_cases" in policy_spec
        else None
    )
    run1 = service.run_board(
        [ExceptionCase.model_validate(c) for c in entry["board_cases"]],
        board_policy=policy,
    )
    for case_id, decision in (entry.get("decisions") or {}).items():
        service.record_decision(case_id, decision, by="Eval Supervisor")
    for case_id, payload in (entry.get("outcomes") or {}).items():
        service.record_outcome(case_id, SupervisorOutcome.model_validate(payload))
    second = entry["second_run"]
    run2 = service.run_board(
        [ExceptionCase.model_validate(c) for c in second.get("cases", [])],
        carry_over_from=run1.run_id if second.get("carry_over_from") else "",
        drop=second.get("drop") or None,
    )
    if "run1_order" in expected and run1.processed != expected["run1_order"]:
        failures.append(
            f"run1_order: expected {expected['run1_order']}, got {run1.processed}"
        )
    if "run1_exits" in expected and run1.exits != expected["run1_exits"]:
        failures.append(
            f"run1_exits: expected {expected['run1_exits']}, got {run1.exits}"
        )
    if "run1_remaining" in expected:
        got = [item["case_id"] for item in run1.remaining]
        if got != expected["run1_remaining"]:
            failures.append(
                f"run1_remaining: expected {expected['run1_remaining']}, got {got}"
            )
    if "manifest" in expected:
        got = {e["case_id"]: e["kind"] for e in run1.carry_over}
        if got != expected["manifest"]:
            failures.append(f"manifest: expected {expected['manifest']}, got {got}")
    if "settled" in expected:
        for key, want in expected["settled"].items():
            got = sorted(run2.carry_over_settled.get(key, []))
            if got != sorted(want):
                failures.append(f"settled.{key}: expected {want}, got {got}")
    if "run2_exits" in expected and run2.exits != expected["run2_exits"]:
        failures.append(
            f"run2_exits: expected {expected['run2_exits']}, got {run2.exits}"
        )
    for case_id, want in expected.get("record_rounds", {}).items():
        record = service.store.get_case(case_id)
        got = len(record.rounds) if record else None
        if got != want:
            failures.append(f"record[{case_id}].rounds: expected {want}, got {got}")
    for case_id, want in expected.get("record_status", {}).items():
        record = service.store.get_case(case_id)
        got = record.status if record else None
        if got != want:
            failures.append(f"record[{case_id}].status: expected {want}, got {got}")
    if "claims" in expected:
        got = sorted(
            [c.part_id, c.case_id, c.status]
            for c in service.allocation.all_claims()
        )
        if got != sorted(expected["claims"]):
            failures.append(f"claims: expected {expected['claims']}, got {got}")
    if "claim_reasons" in expected:
        got = {
            c.claim_id: c.release_reason
            for c in service.allocation.all_claims()
            if c.release_reason
        }
        if got != expected["claim_reasons"]:
            failures.append(
                f"claim_reasons: expected {expected['claim_reasons']}, got {got}"
            )
    if "claim_carried_from" in expected:
        got = {
            c.case_id: (
                "run1" if c.carried_from_run == run1.run_id else c.carried_from_run
            )
            for c in service.allocation.all_claims()
            if c.carried_from_run
        }
        if got != expected["claim_carried_from"]:
            failures.append(
                f"claim_carried_from: expected {expected['claim_carried_from']}, got {got}"
            )
    if "stats_cases_total" in expected:
        got = diagnosis_stats(service.store.all_cases())["cases_total"]
        if got != expected["stats_cases_total"]:
            failures.append(
                f"stats.cases_total: expected {expected['stats_cases_total']}, got {got}"
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
        elif entry["kind"] == "learning":
            failures = run_learning_entry(entry)
        elif entry["kind"] == "allocation":
            failures = run_allocation_entry(entry)
        elif entry["kind"] == "economics":
            failures = run_economics_entry(entry)
        elif entry["kind"] == "carryover":
            failures = run_carryover_entry(entry)
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
