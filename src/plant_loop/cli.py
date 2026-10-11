"""Command-line surface: plant-loop.

    plant-loop analyze case.json          run one case to the gate
    plant-loop board [--max-cases N]      work the bundled sample board
    plant-loop show EXC-101               print a stored record (JSON)
    plant-loop decide EXC-101 approved   record a supervisor decision
    plant-loop pm-proposals               PM proposals the watch earned
    plant-loop pm-obligations             PM obligations (standing + retired)
    plant-loop pm-decide PMP-x approved   decide a PM proposal (pm.py)
    plant-loop pm-retire PMO-x            retire a PM obligation (pm.py)
    plant-loop machines                   machines + lifecycle state
    plant-loop machine-decommission CNC-2 decommission a machine (lifecycle.py)
    plant-loop machine-dispose CNC-2 C-1  dispose one open case of it
    plant-loop metrics                    summarise the stored runs

State lives wherever the service is wired to keep it (PostgreSQL with
DATABASE_URL, in-memory otherwise — an in-memory CLI run is a look,
not a work queue).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_dotenv, silence_langchain_deprecation_warnings

silence_langchain_deprecation_warnings()

from .events import PrintingSink  # noqa: E402
from .loops import BoardPolicy  # noqa: E402
from .samples import load_sample_cases  # noqa: E402
from .schemas import ExceptionCase  # noqa: E402
from .wiring import build_service  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plant-loop", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="run one case JSON file to the gate")
    analyze.add_argument("path")

    board = sub.add_parser("board", help="work the bundled sample board")
    board.add_argument("--max-cases", type=int, default=None)
    board.add_argument(
        "--carry-over-from",
        default="",
        help="a finished board run id whose board this run inherits (carryover.py)",
    )
    board.add_argument(
        "--drop",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="drop a carried case at acceptance (repeatable; pair with --by/--reason)",
    )
    board.add_argument("--by", default="", help="who is dropping (with --drop)")
    board.add_argument("--reason", default="", help="why (with --drop)")

    show = sub.add_parser("show", help="print a stored case record")
    show.add_argument("case_id")

    decide = sub.add_parser("decide", help="record a supervisor decision")
    decide.add_argument("case_id")
    decide.add_argument("decision", choices=["approved", "rejected"])
    decide.add_argument("--by", default="")
    decide.add_argument("--reason", default="")

    sub.add_parser("pm-proposals", help="PM proposals the watch's pattern earned (pm.py)")

    sub.add_parser("pm-obligations", help="standing PM obligations (pm.py)")

    pm_decide = sub.add_parser("pm-decide", help="record the supervisor's decision on a PM proposal")
    pm_decide.add_argument("proposal_id")
    pm_decide.add_argument("decision", choices=["approved", "rejected"])
    pm_decide.add_argument("--by", default="")
    pm_decide.add_argument("--reason", default="")
    pm_decide.add_argument("--interval-days", type=int, default=None)
    pm_decide.add_argument("--last-performed-at", default="")

    pm_retire = sub.add_parser("pm-retire", help="retire a standing PM obligation — one-way (pm.py)")
    pm_retire.add_argument("obligation_id")
    pm_retire.add_argument("--by", default="", help="who is retiring it (required)")
    pm_retire.add_argument("--reason", default="", help="why (required)")

    sub.add_parser("machines", help="the plant's machines and their lifecycle state (lifecycle.py)")

    machine_decommission = sub.add_parser(
        "machine-decommission", help="decommission a machine (lifecycle.py)"
    )
    machine_decommission.add_argument("machine_id")
    machine_decommission.add_argument("--by", default="", help="who is decommissioning it (required)")
    machine_decommission.add_argument("--reason", default="", help="why (required)")
    machine_decommission.add_argument(
        "--effective-at", default="", help="record timestamp it takes effect (default: now)"
    )

    machine_dispose = sub.add_parser(
        "machine-dispose",
        help="record the disposition of one open case of a decommissioned machine",
    )
    machine_dispose.add_argument("machine_id")
    machine_dispose.add_argument("case_id")
    machine_dispose.add_argument("disposition", choices=["keep", "close_moot"])
    machine_dispose.add_argument("--by", default="", help="whose choice this is (required)")

    sub.add_parser("metrics", help="summarise the stored runs")

    args = parser.parse_args(argv)
    load_dotenv()
    service = build_service()
    try:
        if args.command == "analyze":
            payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
            case = ExceptionCase.model_validate(payload)
            run = service.run_case(case, sink=PrintingSink())
            print(run.result.model_dump_json(indent=1))
        elif args.command == "board":
            policy = BoardPolicy(
                max_cases=args.max_cases if args.max_cases is not None else service.board_policy.max_cases,
                board_token_budget=service.board_policy.board_token_budget,
            )
            drops = [
                {"case_id": case_id, "by": args.by, "reason": args.reason}
                for case_id in args.drop
            ]
            result = service.run_board(
                load_sample_cases(),
                sink=PrintingSink(),
                board_policy=policy,
                carry_over_from=args.carry_over_from or "",
                drop=drops or None,
            )
            print(result.model_dump_json(indent=1))
        elif args.command == "show":
            record = service.get_record(args.case_id)
            if record is None:
                print(f"error: unknown case {args.case_id}", file=sys.stderr)
                return 1
            print(record.model_dump_json(indent=1))
        elif args.command == "decide":
            record = service.record_decision(args.case_id, args.decision, by=args.by, reason=args.reason)
            print(f"{record.case_id}: {record.decision} by {record.decision_by or 'supervisor'} → {record.status}")
        elif args.command == "pm-proposals":
            proposals = service.pm_proposals()
            if not proposals:
                print("No PM proposals.")
            for p in proposals:
                print(f"{p.proposal_id}  {p.status.upper()}  {p.machine_id} / {p.fault_class}")
                print(f"  {p.rationale}")
                if p.status != "proposed":
                    print(
                        f"  decided by {p.decided_by or '-'} at {p.decided_at}"
                        + (f" — {p.decision_reason}" if p.decision_reason else "")
                    )
        elif args.command == "pm-obligations":
            obligations = service.pm_obligations()
            if not obligations:
                print("No PM obligations.")
            for o in obligations:
                print(
                    f"{o.obligation_id}  {o.status.upper()}  {o.machine_id} / {o.fault_class}  "
                    f"every {o.interval_days} days  next due {o.next_due_at}"
                )
                print(
                    f"  approved {o.approved_at} by {o.approved_by or '-'} (proposal "
                    f"{o.proposal_id}); last performed {o.last_performed_at or 'never'}; "
                    f"cycles completed {o.cycles_completed}; "
                    f"open cycle case {o.cycle_case_id or 'none'}; deferrals {len(o.deferrals)}"
                )
                if o.status == "retired":
                    print(
                        f"  retired {o.retired_at} by {o.retired_by or '-'} "
                        f"({o.retirement_kind}) — {o.retirement_reason}"
                        + (f" · via {o.decommission_id}" if o.decommission_id else "")
                    )
        elif args.command == "pm-retire":
            obligation = service.retire_pm_obligation(
                args.obligation_id, by=args.by, reason=args.reason
            )
            print(
                f"{obligation.obligation_id}: RETIRED by {obligation.retired_by} "
                f"— {obligation.retirement_reason}"
            )
        elif args.command == "machines":
            records = {r.machine_id: r for r in service.machine_records()}
            from .kb import machines as kb_machines

            for machine_id, machine in sorted(kb_machines().items()):
                record = records.get(machine_id)
                state = "DECOMMISSIONED" if record else "in service"
                print(f"{machine_id}  {machine.get('name', '')}  {machine.get('line_id', '')}  {state}")
                if record:
                    print(
                        f"  {record.record_id}: {record.reason} (by "
                        f"{record.decided_by}, effective {record.effective_at}); "
                        f"obligations retired: {', '.join(record.retired_obligation_ids) or 'none'}; "
                        f"open cases listed: {', '.join(record.open_case_ids) or 'none'}"
                    )
                    for d in record.dispositions:
                        print(f"  disposition: {d['case_id']} → {d['disposition']} by {d['by']}")
        elif args.command == "machine-decommission":
            record = service.decommission_machine(
                args.machine_id,
                by=args.by,
                reason=args.reason,
                effective_at=args.effective_at,
            )
            print(
                f"{record.machine_id}: DECOMMISSIONED ({record.record_id}) by "
                f"{record.decided_by}, effective {record.effective_at} — {record.reason}"
            )
            print(
                f"  obligations retired: {', '.join(record.retired_obligation_ids) or 'none'}"
            )
            print(
                f"  open cases awaiting disposition: "
                f"{', '.join(record.open_case_ids) or 'none'}"
            )
        elif args.command == "machine-dispose":
            record = service.dispose_machine_case(
                args.machine_id, args.case_id, args.disposition, by=args.by
            )
            print(
                f"{args.case_id}: {args.disposition} (recorded on {record.record_id} "
                f"by {args.by})"
            )
            case = service.get_record(args.case_id)
            if case is not None:
                print(f"  case status now: {case.status}")
        elif args.command == "pm-decide":
            proposal, obligation = service.decide_pm_proposal(
                args.proposal_id,
                args.decision,
                by=args.by,
                reason=args.reason,
                interval_days=args.interval_days,
                last_performed_at=args.last_performed_at,
            )
            print(f"{proposal.proposal_id}: {proposal.status} by {proposal.decided_by or 'supervisor'}")
            if obligation is not None:
                print(
                    f"obligation {obligation.obligation_id}: every "
                    f"{obligation.interval_days} days, next due {obligation.next_due_at} "
                    f"(clock from {obligation.clock_basis})"
                )
        elif args.command == "metrics":
            print(json.dumps(service.metrics(), indent=1))
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
