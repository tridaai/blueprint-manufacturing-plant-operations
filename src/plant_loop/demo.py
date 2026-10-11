"""Demo: work the seeded shift board end to end, narrated.

Default mode needs no keys and no network — the offline reasoning
backend, in-memory stores — and prints the board run as it happens:
the priority order the board loop works in, each case's loop (rounds,
gaps, exit), the two escalations and the safety stop, and the shift
handoff. The spotlight section then opens one case record — EXC-101,
the two-round convergence — and shows its loop timeline and draft.

With ``REASONING_BACKEND`` set to a real backend (muse / llm / a
provider alias) the same board is worked by that engine instead.
``--decision approved`` also records a supervisor's approval for the
spotlight case, to show the gate's other side.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_dotenv, silence_langchain_deprecation_warnings

silence_langchain_deprecation_warnings()

from .events import PrintingSink  # noqa: E402
from .kb import plant_name  # noqa: E402
from .priority import priority_for  # noqa: E402
from .samples import load_sample_cases  # noqa: E402
from .service import PlantService  # noqa: E402
from .wiring import build_service  # noqa: E402

SPOTLIGHT = "EXC-101"


def describe_backend(service: PlantService) -> str:
    reasoner = service.reasoner
    if reasoner.name == "offline":
        return (
            "REASONING BACKEND: offline rules (REASONING_BACKEND=offline) — deterministic\n"
            "SOP-threshold rules over the readings, the labelled plumbing backend.\n"
            "Set REASONING_BACKEND=muse (with MODEL_API_KEY) or =llm in .env for a live model."
        )
    if reasoner.name.startswith("muse"):
        return f"REASONING BACKEND: {reasoner.name} — Muse Spark (model: {reasoner.model}) via Meta Model API"
    return f"REASONING BACKEND: {reasoner.name} (model: {getattr(reasoner, 'model', '')})"


def print_board(service: PlantService, cases, result) -> None:
    print()
    print("[BOARD] worked in priority order (computed in code, never by the model):")
    records = {cid: service.get_record(cid) for cid in result.processed}
    for case_id in result.processed:
        record = records[case_id]
        top = record.diagnosis.top()
        print(
            f"  {case_id}  priority {record.priority:7.1f}  exit {record.exit_reason:<16} "
            f"rounds {len(record.rounds)}  "
            f"{(top.fault_class if top else '—')}"
        )
        if record.injection_flags:
            for flag in record.injection_flags:
                print(f"      injection flagged: {flag}")
    for leftover in result.remaining:
        print(f"  {leftover['case_id']}  NOT WORKED — {leftover['reason']}")
    print()
    print("[HANDOFF]")
    for line in result.handoff_summary.splitlines():
        print(f"  {line}")
    telemetry = result.telemetry
    print()
    print(
        f"[TELEMETRY] backend {telemetry.backend} · rounds {telemetry.rounds} · "
        f"tokens {telemetry.input_tokens} in / {telemetry.output_tokens} out · "
        f"est. cost ${telemetry.estimated_cost_usd if telemetry.estimated_cost_usd is not None else '—'}"
    )


def print_spotlight(service: PlantService) -> None:
    record = service.get_record(SPOTLIGHT)
    if record is None:
        return
    case = next(c for c in load_sample_cases() if c.case_id == SPOTLIGHT)
    score, reason = priority_for(case)
    print()
    print(f"[SPOTLIGHT] {SPOTLIGHT} — {record.title}")
    print(f"  priority {score}: {reason}")
    print("  loop timeline:")
    if not record.rounds:
        print("    (no rounds — the case exited at round 0)")
    for entry in record.rounds:
        print(
            f"    round {entry.round}: top {entry.top_fault_class} · {entry.steps} plan step(s) · "
            f"gaps opened [{', '.join(entry.gaps_opened) or '—'}] · "
            f"closed [{', '.join(entry.gaps_closed) or '—'}] · "
            f"tokens {entry.input_tokens}+{entry.output_tokens}"
        )
    print(f"  exit: {record.exit_reason} · status: {record.status}")
    top = record.diagnosis.top()
    if top is not None:
        print(f"  diagnosis: {top.fault_class} (confidence {top.confidence:.2f})")
        for item in top.evidence:
            print(f"    evidence: {item}")
    print(f"  guardrails: {'passed' if record.guardrails.passed else 'FAILED'}")
    print()
    print("[DRAFT — first lines]")
    for line in record.draft.text.splitlines()[:10]:
        print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plant-loop-demo", description=__doc__)
    parser.add_argument("--decision", choices=["approved", "rejected"], default="",
                        help="also record this supervisor decision for the spotlight case")
    args = parser.parse_args(argv)
    load_dotenv()
    service = build_service()
    cases = load_sample_cases()
    print(f"PLANT OPERATIONS LOOP AGENT — {plant_name()}")
    print(describe_backend(service))
    print()
    print(f"[BOARD] {len(cases)} cases on the shift board. Working the board…")
    result = service.run_board(cases, sink=PrintingSink())
    print_board(service, cases, result)
    print_spotlight(service)
    if args.decision:
        record = service.record_decision(SPOTLIGHT, args.decision, by="demo supervisor")
        print()
        print(f"[GATE] supervisor decision recorded for {SPOTLIGHT}: {record.decision} → status {record.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
