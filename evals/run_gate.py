"""Reasoner regression gate: run a configuration, compare records.

    uv run python evals/run_gate.py run --backend offline --label offline-baseline
    uv run python evals/run_gate.py run --backend stub --out evals/gate/candidate.json
    uv run python evals/run_gate.py compare evals/gate/baseline-offline.json evals/gate/candidate.json
    uv run python evals/run_gate.py check --backend stub

``run`` drives the fixed suite (the golden pack's case entries,
safety cases included) through one reasoner configuration and
writes the run record as JSON. ``--backend`` takes any name
``REASONING_BACKEND`` takes (offline | muse | llm | openai |
anthropic | ollama — a live backend needs its key, exactly as the
service does) plus ``stub``, the shipped stub candidate
(``plant_loop/stubs.py``): the gate's exercise partner, differing
from the offline baseline by one diagnosis and one extra round.

``compare`` diffs two records and prints the verdict; exit code is
0 on PASS, 1 on FAIL. ``check`` is the one-command form: run the
named configuration now and compare it against the committed
baseline (``evals/gate/baseline-offline.json``).

The committed baseline is the offline backend's record, regenerated
by ``run --backend offline --label offline-baseline`` (the default
output path IS the committed baseline — regenerating it is how a
deliberate baseline change is made, in the open, in a commit). A
packaged copy rides in ``src/plant_loop/data/gate_baseline.json``
for the API's ``/evals/gate-baseline`` view, mirroring how the
golden results are packaged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_loop import regression  # noqa: E402
from plant_loop.backends import get_reasoner  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "evals" / "gate" / "baseline-offline.json"
PACKAGED_PATH = REPO_ROOT / "src" / "plant_loop" / "data" / "gate_baseline.json"


def build_reasoner(backend: str):
    if backend == "stub":
        from plant_loop.stubs import StubCandidateReasoner

        return StubCandidateReasoner()
    return get_reasoner(backend)


def write_record(record: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"wrote {out} — {record['cases_total']} cases, "
        f"{record['totals']['model_calls']} model calls, {record['totals']['rounds']} rounds"
    )


def load_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_gate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="drive the suite through one configuration, write the record")
    run.add_argument("--backend", default="offline")
    run.add_argument("--label", default="")
    run.add_argument("--out", type=Path, default=None)

    compare = sub.add_parser("compare", help="diff two run records, print the verdict")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)

    check = sub.add_parser("check", help="run a configuration now, compare against the committed baseline")
    check.add_argument("--backend", default="stub")
    check.add_argument("--label", default="")

    args = parser.parse_args(argv)

    if args.command == "run":
        reasoner = build_reasoner(args.backend)
        record = regression.run_suite(reasoner, label=args.label)
        out = args.out or BASELINE_PATH
        write_record(record, out)
        if out.resolve() == BASELINE_PATH.resolve():
            write_record(record, PACKAGED_PATH)
        return 0

    if args.command == "compare":
        report = regression.compare_runs(load_record(args.baseline), load_record(args.candidate))
        print(regression.format_report(report))
        return 0 if report["verdict"] == "PASS" else 1

    if args.command == "check":
        reasoner = build_reasoner(args.backend)
        candidate = regression.run_suite(reasoner, label=args.label)
        report = regression.compare_runs(load_record(BASELINE_PATH), candidate)
        print(regression.format_report(report))
        return 0 if report["verdict"] == "PASS" else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
