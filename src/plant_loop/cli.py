"""Command-line surface: plant-loop.

    plant-loop analyze case.json          run one case to the gate
    plant-loop board [--max-cases N]      work the bundled sample board
    plant-loop show EXC-101               print a stored record (JSON)
    plant-loop decide EXC-101 approved   record a supervisor decision
    plant-loop pm-proposals               PM proposals the watch earned
    plant-loop pm-obligations             standing PM obligations
    plant-loop pm-decide PMP-x approved   decide a PM proposal (pm.py)
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
                print("No standing PM obligations.")
            for o in obligations:
                print(
                    f"{o.obligation_id}  {o.machine_id} / {o.fault_class}  "
                    f"every {o.interval_days} days  next due {o.next_due_at}"
                )
                print(
                    f"  approved {o.approved_at} by {o.approved_by or '-'} (proposal "
                    f"{o.proposal_id}); last performed {o.last_performed_at or 'never'}; "
                    f"cycles completed {o.cycles_completed}; "
                    f"open cycle case {o.cycle_case_id or 'none'}; deferrals {len(o.deferrals)}"
                )
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
