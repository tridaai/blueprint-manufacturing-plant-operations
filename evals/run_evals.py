"""Golden loop eval: known cases in, loop behaviour checked.

Forty-eight entries — twenty cases with authored expectations
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
part), and three *pm* entries (from watch to work: a machine
crossing the recurrence threshold drafts a PM proposal citing its
evidence with the interval arithmetic on the record; approval at
an edited interval creates the obligation, a later board surfaces
its cycle case, and the supervisor's outcome closes the cycle
with the next due point computed — while a rejected proposal's
evidence is spent (the watermark) until a fresh threshold of
faults re-earns a draft, and two faults draft nothing at all),
and one *gate* entry (the reasoner regression gate exercised end
to end: the offline baseline and the shipped stub candidate drive
the fixed suite — this pack's own case entries — and the
comparison must PASS with exactly the stub's two deliberate
differences showing, one diagnosis difference and one extra round,
and no safety-pin violation on either record), and one
*retirement* entry (machines and obligations end: two obligations
are earned the honest way — cases, a drafted proposal, an
approval — one is retired by hand with its own reason and a board
past its due point stays silent, the other's machine is
decommissioned and its obligation retires automatically with the
fixed reason and the decommission linked; the machine's open
cases are disposed differently — one closed as moot with its
claims released, one kept and worked through the gate, a safety
case kept and still a safety stop — new intake for the dead
machine is refused with the state named, and the handoff carries
both retirements with their reasons), and one
*outlook* entry (the board looking ahead: real cases claim the
  shelf, two PM obligations are earned the honest way — one part's
  projected demand is tipped past the shelf by the obligation's
  due cycle inside the horizon, its bites date the due date
  itself; a part two cases demand stays covered; a part at
  exactly zero shortfall is not a risk row; and a retired
  obligation contributes nothing to a part whose shelf is empty —
  while the store's records stand byte-identical across the
  projection, because it is a read model), and one
*gaps* entry (what the plant doesn't know: two feeder cases on a
  machine the plant model does not even register exhaust at
  ``unknown_fault`` — the repeat-gap story, one decided at the
  gate, one still open — while an exhausted case whose fault HAS
  an SOP and a case blocked on a part contribute no row, because
  neither block is missing knowledge; the report must carry
  exactly the one hand-derived row, and the store stands
  byte-identical across the build), and one
*amendment* entry (the supervisor edits, the machine re-checks:
  a converged case's part swap and step strike are re-verified in
  code and approved, its claim moving with the swap; the seeded
  incompatible fan swap fails with its spec delta on the record
  and approval is refused over the failure; a swap onto a part
  another case claims fails naming the holder and passes once
  the holder is rejected; recording refusals — a mandatory
  safety strike, a free-text part, an amendment on a decided
  case — write nothing, and the loop's original plan is
  byte-identical at the end), and one
*recurrence* entry (the fault that came back: a case closes
  with a confirming outcome and its follow-up, six days later,
  links to it in code — prior, spacing and outcome on the new
  record, the prior episode in the reasoner's later rounds as a
  labelled chunk — and agrees; a departing case links, settles
  elsewhere, is marked departed and still approves at the gate;
  a case five weeks out links to nothing; a prior closed as a
  misdiagnosis seeds no link; the stats count the links and the
  agreement, the handoff names the open ones, and the prior
  record stands byte-identical at the end), and one
*windows* entry (Saturday's four hours, chosen in code: three
  work-ready cases pack a 180-minute window exactly in board
  order; a due PM obligation defers on the same priority footing,
  a longer one is oversized, one with no SOP behind it is
  unestimated and never numbered, and one not yet due — like the
  approved case outside the scope — appears nowhere; an outcome
  landing on a packed member after computation makes
  confirmation refuse, naming the drift, the re-plan drops the
  member and confirms, and an overlapping second window plans
  around the booked members — while every case record stands
  byte-identical across planning and confirmation). Every case
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

from plant_loop import outlook as outlook_module  # noqa: E402
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
from plant_loop.reliability import parse_timestamp  # noqa: E402
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


def run_pm_entry(entry: dict) -> list[str]:
    """Preventive maintenance, proven end to end (pm.py).

    Steps run in order against one fresh service: ``run_cases``
    works cases through their loops; ``decide`` records the
    supervisor's decision on the pair's open proposal; ``board``
    runs an empty board (any case it works is one an obligation
    sent); ``outcome_on_pm_case`` records the supervisor's
    outcome on the cycle case the last board surfaced. After any
    step, ``check`` asserts against the PM records as they stand:
    proposal counts, the latest proposal's cited evidence /
    interval / status, the standing obligation's interval / due
    point / basis / cycle state, and the last board's PM cases.
    """
    service = fresh_service()
    failures: list[str] = []
    state: dict = {"board": None, "pm_case_id": ""}

    def check(spec: dict, where: str) -> None:
        proposals = service.pm_proposals()
        proposal = proposals[-1] if proposals else None
        obligations = service.pm_obligations()
        obligation = obligations[-1] if obligations else None
        for key, want in spec.items():
            if key == "proposals_total":
                got = len(proposals)
            elif key == "proposals_proposed":
                got = len([p for p in proposals if p.status == "proposed"])
            elif key == "obligations_total":
                got = len(obligations)
            elif key == "reliability_items":
                got = len([i for i in service.board_items() if i.status == "open"])
            elif key == "proposal_evidence":
                got = [e["case_id"] for e in proposal.evidence] if proposal else None
            elif key == "proposal_interval":
                got = proposal.proposed_interval_days if proposal else None
            elif key == "proposal_clamped":
                got = proposal.clamped if proposal else None
            elif key == "proposal_status":
                got = proposal.status if proposal else None
            elif key == "proposal_board_item_linked":
                items = {i.item_id for i in service.board_items() if i.status == "open"}
                got = bool(proposal and proposal.board_item_id in items)
            elif key == "obligation_interval":
                got = obligation.interval_days if obligation else None
            elif key == "obligation_next_due":
                got = obligation.next_due_at if obligation else None
            elif key == "obligation_basis":
                got = obligation.clock_basis if obligation else None
            elif key == "obligation_cycles":
                got = obligation.cycles_completed if obligation else None
            elif key == "obligation_cycle_open":
                got = bool(obligation and obligation.cycle_case_id)
            elif key == "obligation_due_days_after_performed":
                if obligation and obligation.last_performed_at and obligation.next_due_at:
                    got = (
                        parse_timestamp(obligation.next_due_at)
                        - parse_timestamp(obligation.last_performed_at)
                    ).days
                else:
                    got = None
            elif key == "board_pm_cases":
                got = len(state["board"].pm_cases) if state["board"] else None
            elif key in ("pm_case_type", "pm_case_status", "pm_case_exit"):
                record = (
                    service.get_record(state["pm_case_id"]) if state["pm_case_id"] else None
                )
                got = (
                    {
                        "pm_case_type": record.type,
                        "pm_case_status": record.status,
                        "pm_case_exit": record.exit_reason,
                    }[key]
                    if record
                    else None
                )
            else:
                failures.append(f"{where}: unknown check {key!r}")
                continue
            if got != want:
                failures.append(f"{where}.{key}: expected {want!r}, got {got!r}")

    for index, step in enumerate(entry["steps"]):
        where = f"step {index + 1}"
        if "run_cases" in step:
            for payload in step["run_cases"]:
                service.run_case(ExceptionCase.model_validate(payload))
        if "decide" in step:
            spec = step["decide"]
            open_proposals = [p for p in service.pm_proposals() if p.status == "proposed"]
            if not open_proposals:
                failures.append(f"{where}: decide with no open proposal")
            else:
                service.decide_pm_proposal(
                    open_proposals[-1].proposal_id,
                    spec["decision"],
                    by=spec.get("by", ""),
                    reason=spec.get("reason", ""),
                    interval_days=spec.get("interval_days"),
                    last_performed_at=spec.get("last_performed_at", ""),
                )
        if "board" in step:
            state["board"] = service.run_board([])
            if state["board"].pm_cases:
                state["pm_case_id"] = state["board"].pm_cases[0]
        if "outcome_on_pm_case" in step:
            if not state["pm_case_id"]:
                failures.append(f"{where}: outcome with no PM case surfaced")
            else:
                service.record_outcome(
                    state["pm_case_id"],
                    SupervisorOutcome.model_validate(step["outcome_on_pm_case"]),
                )
        if "check" in step:
            check(step["check"], where)
    return failures


def run_retirement_entry(entry: dict) -> list[str]:
    """Retirement and decommission, end to end (lifecycle.py).

    Both obligations are earned the honest way — cases run, the
    watch drafts, a supervisor approves. One is then retired by
    hand, with its own reason, and a board run past its due point
    must stay silent. The other's machine is decommissioned: the
    obligation retires automatically (fixed reason, decommission
    linked), the machine's open cases are listed on the record
    and disposed one by one — one closed as moot (its claims
    released), one kept and still worked through the gate, a
    safety case kept and still a safety stop — new intake for
    the dead machine is refused with the state named, and the
    handoff's PM section carries both endings truthfully.
    """
    from plant_loop.handoff import build_handoff

    spec = entry["retirement"]
    expected = entry["expected"]
    failures: list[str] = []
    service = fresh_service()

    def check(key: str, actual) -> None:
        if key not in expected:
            failures.append(f"check {key}: not in expected")
        elif actual != expected[key]:
            failures.append(f"check {key}: expected {expected[key]!r}, got {actual!r}")

    def proposal_for(machine_id: str, fault_class: str):
        for proposal in service.pm_proposals():
            if (
                proposal.machine_id == machine_id
                and proposal.fault_class == fault_class
                and proposal.status == "proposed"
            ):
                return proposal
        return None

    def approve(machine_id: str, fault_class: str, decision_spec: dict):
        proposal = proposal_for(machine_id, fault_class)
        if proposal is None:
            failures.append(f"proposal for {machine_id}/{fault_class}: none drafted")
            return None
        _proposal, obligation = service.decide_pm_proposal(
            proposal.proposal_id,
            "approved",
            by=decision_spec["by"],
            interval_days=decision_spec["interval_days"],
            last_performed_at=decision_spec["last_performed_at"],
        )
        if obligation is None:
            failures.append(f"obligation for {machine_id}: approval created none")
        return obligation

    # WELD-4: earn the obligation, retire it by hand, board silent.
    for payload in spec["weld_cases"]:
        service.run_case(ExceptionCase.model_validate(payload))
    weld_obligation = approve("WELD-4", "weld_quality_drift", spec["weld_decision"])
    if weld_obligation is None:
        return failures
    service.retire_pm_obligation(
        weld_obligation.obligation_id,
        by=spec["manual_retirement"]["by"],
        reason=spec["manual_retirement"]["reason"],
    )
    board = service.run_board([])
    check("board_pm_cases_after_retire", len(board.pm_cases))

    # CNC-3: earn the obligation, then open the cases the
    # decommission will find on the board.
    for payload in spec["cnc_cases"]:
        service.run_case(ExceptionCase.model_validate(payload))
    cnc_obligation = approve("CNC-3", "spindle_bearing_wear", spec["cnc_decision"])
    if cnc_obligation is None:
        return failures
    for payload in spec["open_cases"]:
        service.run_case(ExceptionCase.model_validate(payload))

    dec = spec["decommission"]
    record = service.decommission_machine(
        dec["machine_id"],
        by=dec["by"],
        reason=dec["reason"],
        effective_at=dec["effective_at"],
    )
    check("open_cases_at_decommission", sorted(record.open_case_ids))
    cnc_after = next(
        o for o in service.pm_obligations() if o.obligation_id == cnc_obligation.obligation_id
    )
    check("cnc_obligation_status", cnc_after.status)
    check("cnc_retirement_kind", cnc_after.retirement_kind)
    check("cnc_retirement_reason", cnc_after.retirement_reason)
    check("cnc_decommission_link", cnc_after.decommission_id == record.record_id)

    for disposition in spec["dispositions"]:
        service.dispose_machine_case(
            dec["machine_id"],
            disposition["case_id"],
            disposition["disposition"],
            by=disposition["by"],
        )
    check("moot_case_status", service.get_record(spec["moot_case"]).status)
    moot_claims = service.allocation.claims_for_case(spec["moot_case"])
    check(
        "moot_claims_released",
        bool(moot_claims) and all(c.status == "released" for c in moot_claims),
    )
    safety = service.get_record(spec["safety_case"])
    check("kept_safety_status", safety.status)
    check("kept_safety_exit", safety.exit_reason)

    # The kept case is still worked: decision and outcome land,
    # and the outcome drafts no new proposal for a dead machine.
    work = spec["work_kept_case"]
    service.record_decision(work["case_id"], "approved", by=work["by"])
    worked = service.record_outcome(
        work["case_id"],
        SupervisorOutcome(
            decision="accepted",
            confirmed_fault=work["confirmed_fault"],
            work_performed=work["work_performed"],
            recorded_at=work["recorded_at"],
        ),
    )
    check("kept_case_outcome", worked.outcome.decision if worked.outcome else None)
    check(
        "cnc_proposals_after_outcome",
        len([p for p in service.pm_proposals() if p.machine_id == "CNC-3"]),
    )

    intake_error = ""
    try:
        service.run_case(ExceptionCase.model_validate(spec["intake_case"]))
    except ValueError as exc:
        intake_error = str(exc)
    check("intake_refused", "DECOMMISSIONED" in intake_error)

    final_board = service.run_board([])
    check("final_board_pm_cases", len(final_board.pm_cases))

    handoff = build_handoff(service.store)
    check(
        "handoff_retired",
        sorted([r["machine_id"], r["retirement_reason"]] for r in handoff["pm"]["retired"]),
    )
    return failures


def run_outlook_entry(entry: dict) -> list[str]:
    """The parts outlook, proven on a staged book (outlook.py).

    Real cases run singly — their converged plans claim the
    shelf — and two obligations are earned the honest way (the
    watch drafts, a supervisor approves at a chosen clock), one
    of them retired before the projection is built at the
    entry's fixed moment. The expectations are hand-computed
    against the seeded shelf (data/kb.json: HYD-SEAL 3, AIR-FLT
    5, LINER-045 2, CLNT-FLT 0, CLNT-FLT-E 4): PRESS-1's three
    hydraulic cases claim all 3 seal kits, so the standing
    obligation's cycle due 2026-10-18 tips HYD-SEAL to a
    shortfall of 1 that bites on the due date; two COMP-7 cases
    demand AIR-FLT and stay covered; two wire-feed cases take
    LINER-045 to exactly the shelf — covered, not a risk; the
    coolant cases claim the substitute CLNT-FLT-E, and the
    retired coolant obligation contributes nothing to CLNT-FLT,
    whose empty shelf would flag any counted demand. The
    store's records must stand byte-identical across the build.
    """
    service = fresh_service()
    expected = entry["expected"]
    failures: list[str] = []
    for raw in entry["cases"]:
        service.run_case(ExceptionCase.model_validate(raw))

    def approve(machine_id: str, fault_class: str, decision_spec: dict):
        proposal = next(
            (
                p
                for p in service.pm_proposals()
                if p.machine_id == machine_id
                and p.fault_class == fault_class
                and p.status == "proposed"
            ),
            None,
        )
        if proposal is None:
            failures.append(f"proposal for {machine_id}/{fault_class}: none drafted")
            return None
        _proposal, obligation = service.decide_pm_proposal(
            proposal.proposal_id,
            "approved",
            by=decision_spec["by"],
            interval_days=decision_spec["interval_days"],
            last_performed_at=decision_spec["last_performed_at"],
        )
        if obligation is None:
            failures.append(f"obligation for {machine_id}: approval created none")
        return obligation

    spec = entry["outlook"]
    approve("PRESS-1", "hydraulic_pressure_loss", spec["press_decision"])
    coolant = approve("CNC-3", "coolant_system_fault", spec["coolant_decision"])
    if coolant is not None:
        service.retire_pm_obligation(
            coolant.obligation_id,
            by=spec["retirement"]["by"],
            reason=spec["retirement"]["reason"],
        )

    def snapshot() -> str:
        return json.dumps(
            {
                "cases": [r.model_dump() for r in service.store.all_cases()],
                "claims": [c.model_dump() for c in service.store.all_part_claims()],
                "pm": [r.model_dump() for r in service.store.all_pm_records()],
            },
            sort_keys=True,
            default=str,
        )

    before = snapshot()
    outlook = outlook_module.build_outlook(
        service.store,
        now=parse_timestamp(spec["as_of"]),
        horizon_days=spec["horizon_days"],
    )
    if snapshot() != before:
        failures.append("purity: the projection wrote to the store — it is a read model")
    got_shortages = [
        {"part_id": p.part_id, "shortfall": p.shortfall, "bites_at": p.bites_at}
        for p in outlook.shortages
    ]
    if got_shortages != expected["shortages"]:
        failures.append(f"shortages: expected {expected['shortages']}, got {got_shortages}")
    parts = {p.part_id: p for p in outlook.parts}
    for part_id, fields in expected.get("parts", {}).items():
        part = parts.get(part_id)
        if part is None:
            failures.append(f"part[{part_id}]: missing from the projection")
            continue
        for field, want in fields.items():
            got = getattr(part, field)
            if got != want:
                failures.append(f"part[{part_id}].{field}: expected {want!r}, got {got!r}")
    for part_id in expected.get("absent_parts", []):
        if part_id in parts:
            failures.append(
                f"part[{part_id}]: expected no projection lines, got "
                f"{parts[part_id].projected_demand} projected"
            )
    return failures


def run_gaps_entry(entry: dict) -> list[str]:
    """The knowledge-gap report, proven on a staged board (gaps.py).

    Four real cases run singly on one service. Two feeder cases
    (the golden pack's G16 shape: FEED-9, a machine the plant
    model does not even register, no readings) exhaust at
    ``unknown_fault`` — the first is then rejected at the gate, so
    the row carries history as well as a live block. Between them
    run the two boundaries: a conveyor case that exhausts on an
    unseparated pair whose fault HAS an SOP (not knowledge), and
    a compressor case that blocks on a part (the outside world,
    not knowledge). The expectations are hand-derived from those
    records: exactly one row — FEED-9 / unknown_fault, 2 cases,
    6 rounds burned (3 each), the second case open now — and the
    store's records byte-identical across the build.
    """
    from plant_loop import gaps as gaps_module

    service = fresh_service()
    expected = entry["expected"]
    failures: list[str] = []
    for raw in entry["cases"]:
        service.run_case(ExceptionCase.model_validate(raw))
    for case_id, exit_reason in expected.get("exits", {}).items():
        record = service.store.get_case(case_id)
        got = record.exit_reason if record else None
        if got != exit_reason:
            failures.append(f"exit[{case_id}]: expected {exit_reason}, got {got}")
    for case_id, decision in (entry.get("decisions") or {}).items():
        service.record_decision(case_id, decision, by="Eval Supervisor")

    def snapshot() -> str:
        return json.dumps(
            {
                "cases": [r.model_dump() for r in service.store.all_cases()],
                "claims": [c.model_dump() for c in service.store.all_part_claims()],
                "pm": [r.model_dump() for r in service.store.all_pm_records()],
            },
            sort_keys=True,
            default=str,
        )

    before = snapshot()
    report = gaps_module.build_gap_report(
        service.store, now=parse_timestamp("2026-10-12T00:00:00Z")
    )
    if snapshot() != before:
        failures.append("purity: the report wrote to the store — it is a read model")
    if report.cases_examined != expected["cases_examined"]:
        failures.append(
            f"cases_examined: expected {expected['cases_examined']}, "
            f"got {report.cases_examined}"
        )
    if report.blocked_cases != expected["blocked_cases"]:
        failures.append(
            f"blocked_cases: expected {expected['blocked_cases']}, "
            f"got {report.blocked_cases}"
        )
    got_gaps = [g.model_dump() for g in report.gaps]
    want_gaps = expected["gaps"]
    if len(got_gaps) != len(want_gaps):
        failures.append(f"gaps: expected {len(want_gaps)} row(s), got {got_gaps}")
    else:
        for got, want in zip(got_gaps, want_gaps):
            for field, want_value in want.items():
                if got.get(field) != want_value:
                    failures.append(
                        f"gap[{want['machine_id']}].{field}: expected "
                        f"{want_value!r}, got {got.get(field)!r}"
                    )
    return failures


def run_amendment_entry(entry: dict) -> list[str]:
    """Gate amendments, proven end to end on four gated cases.

    The choreography (the entry's cases run as single runs on one
    service): a converged weld case is amended twice — a part swap
    onto a stocked shelf part, then a step strike — both
    re-verified in code, its claim moving with the swap, its
    packet printing the amended plan with the swap marked, and
    approval following. The blocked fan case takes the seeded
    incompatible swap (the generic fan, 380 against the SOP's 420
    floor): AMENDMENT_FAILED with the delta on the record,
    approval refused over the failure, rejection still open. A
    seal case's swap onto the printer head another case claims
    fails naming the holder, and passes once that case is
    rejected at its own gate. Recording refusals (a mandatory
    safety strike, a free-text part, an amendment on a decided
    case) write nothing. The original plan of the amended case is
    byte-identical at the end.
    """
    service = fresh_service()
    expected = entry["expected"]
    failures: list[str] = []

    def check(label, condition):
        if not condition:
            failures.append(label)

    cases = {
        cid: ExceptionCase.model_validate(raw)
        for cid, raw in entry["cases"].items()
    }
    for case_id, case in cases.items():
        result = service.run_case(case).result
        if result.exit_reason != expected["exits"][case_id]:
            failures.append(
                f"exit[{case_id}]: expected {expected['exits'][case_id]}, "
                f"got {result.exit_reason}"
            )

    def amend(case_id, **kwargs):
        return service.amend_case(case_id, by="Golden Supervisor", **kwargs)

    original_plan = service.get_record("AM1").plan.model_dump_json()

    # A mandatory safety step cannot be struck — refused, nothing written.
    try:
        amend("AM1", kind="step_strike", reason="skip the isolation",
              step_id="safety-isolate-weld4")
        failures.append("mandatory strike: expected a refusal, got a record")
    except ValueError as exc:
        check("mandatory strike refusal names the step", "mandatory safety step" in str(exc))
    check("mandatory strike wrote nothing", service.get_record("AM1").amendments == [])

    # The claim conflict: AM4 wants the printer head AM3 claims.
    record = amend("AM4", kind="part_swap", reason="the seal bar is on back order",
                   from_part_id="SEAL-BAR", to_part_id="PRT-HEAD")
    amendment = record.amendments[-1]
    check("conflict swap failed", amendment.verification.status == "failed")
    check("conflict names the holder", "AM3" in amendment.verification.detail)

    # The incompatible swap on the blocked fan case.
    record = amend("AM2", kind="part_swap", reason="try the generic fan",
                   from_part_id="FAN-C7", to_part_id="FAN-C7-G")
    amendment = record.amendments[-1]
    check("fan swap failed", amendment.verification.status == "failed")
    check(
        "fan swap gap is the constraint",
        any(g.kind == "incompatible_part" for g in amendment.verification.gaps),
    )
    deltas = {d.attribute: d for d in amendment.spec_deltas}
    check(
        "fan swap delta is 450 -> 380",
        deltas.get("airflow_m3h") is not None
        and deltas["airflow_m3h"].original == 450
        and deltas["airflow_m3h"].substitute == 380,
    )
    try:
        service.record_decision("AM2", "approved", by="Golden Supervisor")
        failures.append("approve over a failed amendment: expected a refusal")
    except ValueError:
        pass
    record = service.record_decision(
        "AM2", "rejected", by="Golden Supervisor", reason="no compliant fan"
    )
    check("fan case rejected after failure", record.status == "rejected")

    # The passing swap + strike on the converged weld case.
    record = amend("AM1", kind="part_swap",
                   reason="tips inspected — the liner is the worn consumable",
                   from_part_id="TIP-08", to_part_id="LINER-045")
    check("weld swap passed", record.amendments[-1].verification.status == "passed")
    claims = {c.part_id: c for c in service.allocation.claims_for_case("AM1")}
    check(
        "old claim released by the amendment",
        claims["TIP-08"].status == "released"
        and claims["TIP-08"].release_reason == "released_by_amendment",
    )
    check("new part claimed", claims["LINER-045"].status == "claimed")
    record = amend("AM1", kind="step_strike",
                   reason="gas flow was verified at the last PM",
                   step_id="gas-check")
    check("strike passed", record.amendments[-1].verification.status == "passed")
    packet_text = service.case_packet("AM1")["text"]
    check("packet marks the swap", "AMENDED AT THE GATE" in packet_text)
    check(
        "packet names the drawn part",
        "LINER-045 is fitted in place of TIP-08" in packet_text,
    )
    record = service.record_decision("AM1", "approved", by="Golden Supervisor")
    check("amended case approved", record.status == "approved")
    check("amendments survive approval", len(record.amendments) == 2)

    # A decided case takes no amendment; a free-text part neither.
    try:
        amend("AM1", kind="step_strike", reason="too late", step_id="test-welds")
        failures.append("amendment on a decided case: expected a refusal")
    except ValueError:
        pass
    try:
        amend("AM4", kind="part_swap", reason="a part from the van",
              from_part_id="SEAL-BAR", to_part_id="UNOBTAINIUM-9")
        failures.append("free-text part: expected a refusal")
    except ValueError:
        pass

    # The holder's rejection frees the head; AM4's swap now passes.
    service.record_decision("AM3", "rejected", by="Golden Supervisor",
                            reason="the printer is being replaced whole")
    record = amend("AM4", kind="part_swap", reason="the seal bar is on back order",
                   from_part_id="SEAL-BAR", to_part_id="PRT-HEAD")
    check("conflict swap passes once freed",
          record.amendments[-1].verification.status == "passed")

    # The loop's original plan was never edited, and the stats
    # count the gate's work as its own facts.
    check(
        "original plan byte-identical",
        service.get_record("AM1").plan.model_dump_json() == original_plan,
    )
    stats = service.diagnosis_stats()["amendments"]
    for key, want in expected["stats"].items():
        if stats.get(key) != want:
            failures.append(f"stats[{key}]: expected {want}, got {stats.get(key)}")
    check(
        "handoff names the amended cases",
        "AMENDED AT THE GATE" in service.handoff()["text"],
    )
    return failures


def run_gate_entry(entry: dict) -> list[str]:
    """The reasoner regression gate, exercised end to end.

    The offline baseline and the shipped stub candidate
    (``plant_loop/stubs.py``) both drive the gate's fixed suite —
    this pack's own case entries — and the comparison must come
    back PASS with exactly the stub's two deliberate differences
    showing: one diagnosis difference (G14) and one extra round
    (G20). Neither record may carry a safety-pin violation. The
    FAIL paths of the comparison are pinned in the unit tests,
    where pin-breaking candidates can be constructed without
    shipping one.
    """
    from plant_loop import regression
    from plant_loop.stubs import StubCandidateReasoner

    expected = entry["expected"]
    failures: list[str] = []
    baseline = regression.run_suite(OfflineReasoner(), label="offline-baseline")
    candidate = regression.run_suite(StubCandidateReasoner(), label="stub-candidate")
    report = regression.compare_runs(baseline, candidate)
    if report["verdict"] != expected["verdict"]:
        failures.append(f"verdict: expected {expected['verdict']}, got {report['verdict']}")
    differences = sorted(d["case_id"] for d in report["diagnosis"]["differences"])
    if differences != sorted(expected["diagnosis_differences"]):
        failures.append(
            f"diagnosis differences: expected {expected['diagnosis_differences']}, got {differences}"
        )
    extra_rounds = sorted(d["case_id"] for d in report["deltas"]["per_case"] if d["rounds_candidate"] > d["rounds_baseline"])
    if extra_rounds != sorted(expected["extra_round_cases"]):
        failures.append(
            f"extra-round cases: expected {expected['extra_round_cases']}, got {extra_rounds}"
        )
    if len(report["baseline_pin_violations"]) != expected["baseline_pin_violations"]:
        failures.append(
            f"baseline pin violations: expected {expected['baseline_pin_violations']}, "
            f"got {report['baseline_pin_violations']}"
        )
    if len(report["safety_pin_violations"]) != expected["candidate_pin_violations"]:
        failures.append(
            f"candidate pin violations: expected {expected['candidate_pin_violations']}, "
            f"got {report['safety_pin_violations']}"
        )
    return failures


class RecurrenceProbeReasoner(OfflineReasoner):
    """Offline rules, instrumented for the recurrence entry.

    Captures a snapshot of the chunk sources in every diagnose
    context (so the entry can prove the recurrence chunk reached
    the reasoner only from the round after the link formed), and
    — for the entry's departing case, from its second diagnose
    call — settles on the machine's coolant fault instead: a
    reasoner disagreeing with the plant's confirmed past, the
    departure the round exists to mark.
    """

    DEPARTING_CASE = "REC-B"
    DEPARTS_TO = "coolant_system_fault"

    def __init__(self) -> None:
        self.contexts: dict[str, list[list[str]]] = {}
        self._calls: dict[str, int] = {}

    def diagnose(self, ctx, focus):
        import dataclasses

        case_id = ctx.case.case_id
        snapshot = dataclasses.replace(
            ctx, kb_context=[dict(c) for c in ctx.kb_context]
        )
        self.contexts.setdefault(case_id, []).append(
            [c.get("source", "") for c in snapshot.kb_context]
        )
        self._calls[case_id] = self._calls.get(case_id, 0) + 1
        out = super().diagnose(ctx, focus)
        if case_id == self.DEPARTING_CASE and self._calls[case_id] >= 2:
            from plant_loop import kb as kb_module
            from plant_loop.ports import RoundOutput
            from plant_loop.schemas import Diagnosis, Hypothesis

            sop = kb_module.sop_for(ctx.case.machine_id, self.DEPARTS_TO) or {}
            diagnosis = Diagnosis(
                hypotheses=[
                    Hypothesis(
                        fault_class=self.DEPARTS_TO,
                        statement=(
                            f"{sop.get('title', self.DEPARTS_TO)} fits the "
                            "readings on the second look."
                        ),
                        confidence=0.7,
                        evidence=["probe: the second look separates it"],
                        sop_id=sop.get("sop_id", ""),
                    )
                ],
                summary=f"Top hypothesis: {self.DEPARTS_TO} at 0.70 confidence.",
            )
            return RoundOutput(
                payload=diagnosis,
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                latency_ms=out.latency_ms,
                backend=out.backend,
                model=out.model,
            )
        return out


def run_recurrence_entry(entry: dict) -> list[str]:
    """Recurrence, proven end to end on one store (recurrence.py).

    The prior case runs cold and closes with a confirming
    outcome; the follow-up — same machine, same signature, six
    days later — must link to it (prior, spacing, outcome and
    all), meet the prior episode as a labelled chunk in its later
    rounds, and agree. The departing case links too, settles on
    another fault, is marked DIAGNOSIS_DEPARTED_FROM_PRIOR, and
    still approves at the gate on its own merits. A case five
    weeks out links to nothing. A second prior, closed as a
    misdiagnosis (an outcome confirming nothing), seeds no link
    for its follow-up. The stats count the two links and the
    one-in-two agreement; the handoff names both open links; the
    prior record stands byte-identical at the end.
    """
    from plant_loop.handoff import build_handoff
    from plant_loop.stats import diagnosis_stats as stats_of

    expected = entry["expected"]
    failures: list[str] = []

    def check(label, condition):
        if not condition:
            failures.append(label)

    reasoner = RecurrenceProbeReasoner()
    service = fresh_service(reasoner)

    prior = service.run_case(ExceptionCase.model_validate(entry["prior"])).result
    failures += [
        f"prior: {f}"
        for f in check_result({"expected": entry["prior_expected"]}, prior)
    ]
    service.record_outcome(
        entry["prior"]["case_id"],
        SupervisorOutcome.model_validate(entry["prior_outcome"]),
    )
    prior_before = service.store.get_case(entry["prior"]["case_id"]).model_dump_json()

    follow = service.run_case(
        ExceptionCase.model_validate(entry["followup"])
    ).result
    failures += [f"followup: {f}" for f in check_result(entry, follow)]
    link = follow.recurrence
    check("followup: expected a recurrence link", link is not None)
    if link is not None:
        check(
            "followup: link names the prior",
            link.prior_case_id == expected["link_prior"],
        )
        check(
            "followup: spacing",
            link.days_between == expected["link_days_between"],
        )
        check(
            "followup: the prior's outcome rides the link",
            link.outcome_decision == expected["link_outcome_decision"]
            and link.outcome_resolved == expected["link_outcome_resolved"],
        )
        check("followup: agreement is a clean face", link.departed is False)
    seen = reasoner.contexts.get(entry["followup"]["case_id"], [])
    check(
        "followup: the recurrence chunk reached the reasoner after round 1",
        len(seen) > 1
        and "recurrence" not in seen[0]
        and all("recurrence" in sources for sources in seen[1:]),
    )

    departing = service.run_case(
        ExceptionCase.model_validate(entry["departing"])
    ).result
    dlink = departing.recurrence
    check("departing: expected a recurrence link", dlink is not None)
    if dlink is not None:
        check(
            "departing: link names the prior",
            dlink.prior_case_id == expected["link_prior"],
        )
        check("departing: marked departed", dlink.departed is True)
    check(
        "departing: settled diagnosis is the departed fault",
        departing.diagnosis.top() is not None
        and departing.diagnosis.top().fault_class == expected["departed_to"],
    )
    check(
        "departing: converged on its own merits",
        departing.exit_reason == expected["departed_exit"],
    )

    handoff = build_handoff(service.store)
    check(
        "handoff: names both open recurrence cases",
        sorted(e["case_id"] for e in handoff["recurrences"])
        == sorted(expected["handoff_open"]),
    )
    decided = service.record_decision(
        entry["departing"]["case_id"], "approved", by="Eval Supervisor"
    )
    check(
        "departing: approval is not blocked by the departure",
        decided.status == "approved",
    )

    outside = service.run_case(
        ExceptionCase.model_validate(entry["outside"])
    ).result
    check("outside: no link beyond the window", outside.recurrence is None)
    check(
        "outside: diagnosis unaffected",
        outside.diagnosis.top() is not None
        and outside.diagnosis.top().fault_class == expected["outside_diagnosis"],
    )

    cold_prior = service.run_case(
        ExceptionCase.model_validate(entry["misdiagnosed_prior"])
    ).result
    service.record_outcome(
        entry["misdiagnosed_prior"]["case_id"],
        SupervisorOutcome.model_validate(entry["misdiagnosed_outcome"]),
    )
    cold = service.run_case(
        ExceptionCase.model_validate(entry["misdiagnosed_followup"])
    ).result
    check(
        "misdiagnosed: the follow-up diagnoses the same fault",
        cold.diagnosis.top() is not None
        and cold_prior.diagnosis.top() is not None
        and cold.diagnosis.top().fault_class
        == cold_prior.diagnosis.top().fault_class,
    )
    check(
        "misdiagnosed: a close confirming nothing seeds no link",
        cold.recurrence is None,
    )

    section = stats_of(service.store.all_cases())["recurrence"]
    for key, want in expected["stats"].items():
        got = section.get(key)
        ok = abs(got - want) < 1e-3 if isinstance(want, float) else got == want
        if not ok:
            failures.append(f"stats.recurrence.{key}: expected {want}, got {got}")
    check(
        "prior: byte-identical at the end",
        service.store.get_case(entry["prior"]["case_id"]).model_dump_json()
        == prior_before,
    )
    return failures


def run_windows_entry(entry: dict) -> list[str]:
    """One downtime-window entry (windows.py): the staged board —
    three work-ready cases in scope packing a 180-minute window
    exactly (120 + 40 + 20, in board priority order), a fourth
    approved out of scope, and four PM obligations: one deferred
    by the minutes, one oversized, one unestimated, one not yet
    due. Then the confirmation choreography: an outcome recorded
    on a packed member after computation makes confirmation
    refuse, naming the drift; re-planning drops the member and
    confirms; a second, overlapping window then plans around the
    booked members. Case records are compared byte-for-byte
    across planning and confirmation — the window is an overlay
    and moves nothing. Expectations were derived from the SOP
    downtime figures and the priority formula on paper, then
    pinned (the obligations are staged as records, as a
    supervisor-approved obligation stands; their approval path
    is the PM entries' own subject)."""
    from datetime import datetime, timezone

    from plant_loop.handoff import build_handoff
    from plant_loop.schemas import ExceptionCase, PmObligation, SupervisorOutcome

    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        if not condition:
            failures.append(f"{entry['id']}: {label}")

    service = fresh_service()

    def run_approved(payload: dict) -> None:
        result = service.run_case(ExceptionCase.model_validate(payload)).result
        check(f"{payload['case_id']} converged", result.exit_reason == "converged")
        service.record_decision(payload["case_id"], "approved", by="Eval Supervisor")

    run_approved({
        "case_id": "WIN-A", "type": "machine_down",
        "title": "CNC-3 spindle vibration high, line stopped",
        "description": "Spindle vibration alarm tripped at 06:10 and the machine was stopped by the operator. Audible rumble from the spindle head at speed.",
        "machine_id": "CNC-3", "line_id": "line-b",
        "product": "Gear blank GB-220", "batch": "B-4471",
        "readings": {"spindle_vibration_mm_s": 8.4, "spindle_temp_c": 86, "tool_runout_mm": 0.008},
        "notes": "Operator: rumble gets worse above 6000 rpm. No recent tool change.",
        "reporter": "J. Okafor (operator)", "reported_at": "2026-10-15T06:12:00Z",
        "downtime_minutes": 0,
    })
    run_approved({
        "case_id": "WIN-B", "type": "quality_nonconformance",
        "title": "Porosity above limit on bracket welds",
        "description": "QC inspection found porosity in 6.2 percent of welds on the morning bracket batch; batch held at inspection.",
        "machine_id": "WELD-4", "line_id": "line-c",
        "product": "Bracket BR-118", "batch": "B-4470",
        "readings": {"qc_porosity_pct": 6.2, "gas_flow_l_min": 9.5},
        "notes": "Cylinder was changed on night shift.",
        "reporter": "S. Patel (quality)", "reported_at": "2026-10-15T05:40:00Z",
        "downtime_minutes": 0,
    })
    run_approved({
        "case_id": "WIN-C", "type": "material_shortage",
        "title": "Packing film out of stock at PACK-6",
        "description": "Last roll of 450 mm packing film used at 06:30; stores show zero rolls. Packing can continue on the approved grade B film if confirmed.",
        "machine_id": "PACK-6", "line_id": "line-d",
        "product": "Retail carton RC-9", "batch": "B-4472",
        "readings": {"material": "FILM-PKG", "rolls_remaining": 0},
        "notes": "", "reporter": "T. Nguyen (stores)",
        "reported_at": "2026-10-15T06:31:00Z", "downtime_minutes": 0,
    })
    run_approved({
        "case_id": "WIN-E", "type": "machine_down",
        "title": "CONV-5 belt tracking fault, line stopped",
        "description": "Belt drifting to one side, edge fraying past the wear mark.",
        "machine_id": "CONV-5", "line_id": "line-d",
        "product": "Assembly kit AK-12", "batch": "B-4621",
        "readings": {"belt_drift_mm": 9}, "notes": "",
        "reporter": "floor-lead", "reported_at": "2026-10-15T06:00:00Z",
        "downtime_minutes": 0,
    })

    def obligation(oid: str, machine: str, fault: str, line: str, due: str) -> None:
        service.store.save_pm_record(PmObligation(
            obligation_id=oid, proposal_id="PMP-EVAL", machine_id=machine,
            fault_class=fault, line_id=line, interval_days=30,
            approved_by="Eval Supervisor", approved_at="2026-10-01T00:00:00Z",
            clock_basis="approval", next_due_at=due,
            created_at="2026-10-01T00:00:00Z"))

    obligation("PMO-W1", "PRESS-1", "hydraulic_pressure_loss", "line-a", "2026-10-17T06:00:00Z")
    obligation("PMO-W2", "CNC-2", "spindle_cartridge_failure", "line-b", "2026-10-17T09:00:00Z")
    obligation("PMO-W3", "CNC-3", "bearing_seat_fretting", "line-b", "2026-10-17T09:30:00Z")
    obligation("PMO-W4", "WELD-4", "wire_feed_fault", "line-c", "2026-11-05T09:00:00Z")

    scope = dict(line_ids=["line-b"], machine_ids=["CNC-3", "WELD-4", "PACK-6", "PRESS-1"])
    window = service.declare_window(
        by="Eval Supervisor", starts_at="2026-10-17T08:00:00Z",
        length_min=180, **scope)
    check("the plan packs A, B, C in board priority order",
          [i.item_id for i in window.packed] == ["WIN-A", "WIN-B", "WIN-C"])
    check("the packing fills the window exactly (120 + 40 + 20 = 180)",
          [(i.start_offset_min, i.end_offset_min) for i in window.packed]
          == [(0, 120), (120, 160), (160, 180)]
          and window.minutes_used == 180 and window.minutes_left == 0)
    check("the obligation interleaves on the board's own footing and defers",
          [i.item_id for i in window.deferred] == ["PMO-W1"]
          and window.deferred[0].priority == 225.0)
    check("the oversized obligation is listed, never squeezed",
          [i.item_id for i in window.oversized] == ["PMO-W2"]
          and window.oversized[0].estimate_min == 240)
    check("the unestimated obligation is listed, never given a number",
          [i.item_id for i in window.unestimated] == ["PMO-W3"]
          and window.unestimated[0].estimate_min is None)
    section_ids = [i.item_id for s in (window.packed, window.deferred,
                                       window.oversized, window.unestimated)
                   for i in s]
    check("the not-yet-due obligation and the out-of-scope case appear nowhere",
          "PMO-W4" not in section_ids and "WIN-E" not in section_ids)

    # Drift: an outcome lands on a packed member after computation.
    service.record_outcome("WIN-B", SupervisorOutcome(
        decision="accepted", confirmed_fault="weld_quality_drift",
        work_performed="nozzle changed, welds clean",
        recorded_by="Eval Supervisor"))
    refusal = ""
    try:
        service.confirm_window(window.window_id, by="Eval Supervisor")
    except ValueError as exc:
        refusal = str(exc)
    check("confirmation refuses on the drift, naming the member",
          "WIN-B" in refusal and "Re-plan" in refusal)

    snapshots = {c.case_id: c.model_dump_json() for c in service.store.all_cases()}
    window = service.plan_window(window.window_id)
    check("the re-plan drops the outcomed member and re-reads the board",
          [i.item_id for i in window.packed] == ["WIN-A", "WIN-C"]
          and window.minutes_used == 140 and window.minutes_left == 40
          and [i.item_id for i in window.deferred] == ["PMO-W1"])
    window = service.confirm_window(window.window_id, by="Eval Supervisor")
    check("the fresh plan confirms", window.status == "confirmed")

    second = service.declare_window(
        by="Eval Supervisor", starts_at="2026-10-17T10:30:00Z",
        length_min=120, **scope)
    check("the overlapping window plans around the booked members",
          [i.item_id for i in second.booked_elsewhere] == ["WIN-A", "WIN-C"]
          and not second.packed)
    check("the deferred obligation reads oversized against the shorter window",
          [i.item_id for i in second.oversized] == ["PMO-W1", "PMO-W2"])
    check("planning and confirming moved no case record, byte for byte",
          snapshots == {c.case_id: c.model_dump_json()
                        for c in service.store.all_cases()})

    handoff = build_handoff(
        service.store, now=datetime(2026, 10, 17, 7, 0, tzinfo=timezone.utc))
    check("the handoff names the confirmed window still ahead",
          [w["window_id"] for w in handoff["windows"]] == [window.window_id]
          and handoff["windows"][0]["packed_cases"] == ["WIN-A", "WIN-C"])
    check("the counts are counts", service.window_counts()
          == {"planned": 2, "proposed": 1, "confirmed": 1})
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
        elif entry["kind"] == "pm":
            failures = run_pm_entry(entry)
        elif entry["kind"] == "retirement":
            failures = run_retirement_entry(entry)
        elif entry["kind"] == "outlook":
            failures = run_outlook_entry(entry)
        elif entry["kind"] == "gaps":
            failures = run_gaps_entry(entry)
        elif entry["kind"] == "amendment":
            failures = run_amendment_entry(entry)
        elif entry["kind"] == "recurrence":
            failures = run_recurrence_entry(entry)
        elif entry["kind"] == "windows":
            failures = run_windows_entry(entry)
        elif entry["kind"] == "gate":
            failures = run_gate_entry(entry)
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
