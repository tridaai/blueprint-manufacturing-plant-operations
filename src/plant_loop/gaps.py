"""The knowledge-gap report: what the plant doesn't know, priced.

Every round before this one handled a blocked case honestly and
individually. What nobody could see was the pattern: the same
missing knowledge blocking case after case, shift after shift.
The record already contains the pattern — it is just never added
up:

- A case whose diagnosed fault has **no SOP on record** for its
  machine cannot be planned against the plant's own procedure,
  because there isn't one. Offline, the diagnosis derives its
  hypotheses from SOPs, so the shape on the record is a top fault
  of ``unknown_fault`` with no evidence, an
  ``unevidenced_hypothesis`` gap that no round can close, and an
  ``exhausted`` exit after every round is burned (the golden
  pack's G16: a feeder unit that is not even on the machine
  register). The verify stage's safety check reads the SOP's
  mandatory steps — with no SOP there is nothing to check, and
  until now nothing anywhere *named* the absence. The handoff
  files such a case as a generic escalation; the diagnosis stats
  do not break exits down at all.
- The parts outlook (round 11) owns the one existing
  missing-knowledge list: its ``unquantified_pm`` names standing
  PM obligations due inside its horizon whose machine + fault
  class fails the same test — ``kb.sop_for(...) is None``. This
  report applies **that test, unchanged**, to case records: the
  definition is referenced, not forked. (The outlook's window is
  a fortnight because it prices parts; a knowledge gap matters
  from the first blocked case, so this report has no horizon.)

This module is the aggregation, in code, over the stored case
records (open and closed alike — a gap whose cases were all
decided at the gate is still a hole in the plant's knowledge;
it just isn't biting *right now*).

**The advisory boundary — the point of the round.** The report
advises the knowledge base's maintainer. It generates NOTHING
adoptable: no draft SOPs, no suggested safety steps, no proposed
parts lists. The schema carries identifiers, counts and
timestamps only — there is no field that could hold SOP content,
by construction, because a machine must never author the safety
content a plant expert has not written. Authoring the SOP is a
human's act, through the plant model's own workflow (an authored
edit to ``data/kb.json``, kept honest by the KB shape test:
registered machine, mandatory safety steps present). And nothing
here unblocks a case: a blocked case moves only through the
existing flows — a supervisor's gate decision, or a fresh run
that re-plans it (carry-over re-works the ``blocked_external``
family; an exhausted case carries as ``at_gate``, inherited
standing). What changes when the SOP lands is the *future*: new
cases on the pair stop blocking, which this report shows as a
row that stops growing — and, once every contributing record
has been superseded by a re-run that converged, a row that is
gone. History is never rewritten to make a gap look healed.

Deliberately NOT gap categories, from the survey:

- **An SOP that names no parts.** It cannot block a case (the
  plan needs nothing) and the outlook already ruled it a
  consulted fact, not a hole: SOP-CONV5-MTR names no parts and
  its cases converge. Zero is a quantity, not a gap.
- **A machine with no criticality on record.** Machines carry no
  criticality in the plant model — criticality is a *line*
  attribute, every line carries one, and an unknown line defaults
  silently in the priority formula. Nothing blocks on it and no
  record distinguishes it; the engine cannot see this category.
- **A part absent from the inventory register** (``part_qty``
  is None rather than 0). The engine can see it, but the record
  cannot attribute it: the flagship instance is a defective
  planner inventing a part the plant never stocked (the golden
  pack's G22) — a planner's error, not the plant's ignorance.
  Parts blocks stay in the outside-world family, where the
  handoff already names them.
- **A rival fault with no SOP.** An exhausted case's open
  ``competing_hypothesis`` gap can name a fault class that has
  no SOP for the machine, but the block is jointly caused by an
  unseparated pair; charging it to one missing SOP would
  overstate the evidence. (Offline it cannot occur at all:
  rival hypotheses derive from SOPs.)
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import kb
from .events import format_timestamp
from .ports import Store
from .schemas import (
    EXIT_CONVERGED,
    EXIT_SAFETY_STOP,
    KnowledgeGap,
    KnowledgeGapReport,
    PmObligation,
)

#: A case is *open* for gap purposes while it still waits at the
#: gate undecided — the gap is biting right now. Decided, dropped
#: and moot-closed records are the gap's history, not its present.
OPEN_STATUS = "awaiting_supervisor"

BASIS = (
    "A case counts toward a gap when its diagnosed fault has no SOP on "
    "record for its machine — the same sop_for test the loop and the "
    "parts outlook apply — and the case ended without a verified plan "
    "(safety stops are excluded: they end by rule, not by missing "
    "knowledge). The report advises the knowledge base's maintainer: "
    "it names the missing SOPs and what their absence has cost, and "
    "it drafts nothing — no field in it could carry SOP content, "
    "because a machine must not author the safety content a plant "
    "expert has not written. Authoring the SOP into the plant model "
    "(data/kb.json) is a human's act; a blocked case is never "
    "unblocked by this report, only by the gate or a fresh run."
)


def _stamp(record) -> str:
    """The record's own moment: when the loop produced it, else
    when the floor reported it (the stats module's chronology —
    created, then case id)."""
    return record.created_at or record.reported_at or ""


def build_gap_report(
    store: Store,
    *,
    now: datetime | None = None,
) -> KnowledgeGapReport:
    """Aggregate knowledge-caused blocks by the missing SOP.

    Pure over the store and the plant model: the same records
    produce the same report, byte for byte, and building it
    changes no record, calls no reasoner, and creates nothing —
    the pins the tests hold it to.
    """
    moment = now or datetime.now(timezone.utc)
    records = store.all_cases()

    # Standing obligations standing on a hole, by the outlook's
    # own test — referenced, not forked: sop_for is None for the
    # obligation's machine + fault class. (The outlook applies it
    # inside its horizon to price parts; here it annotates the
    # case rows, because the SOP that heals the cases is the same
    # one that would let the obligation's cycles be priced and
    # worked.)
    obligations_by_pair: dict[tuple[str, str], list[str]] = {}
    for record in store.all_pm_records():
        if not isinstance(record, PmObligation) or record.status != "standing":
            continue
        if kb.sop_for(record.machine_id, record.fault_class) is None:
            obligations_by_pair.setdefault(
                (record.machine_id, record.fault_class), []
            ).append(record.obligation_id)

    grouped: dict[tuple[str, str], list] = {}
    for record in records:
        top = record.diagnosis.top()
        if top is None:
            continue  # no diagnosis: a safety stop — ends by rule
        if not record.exit_reason or record.exit_reason in (
            EXIT_CONVERGED,
            EXIT_SAFETY_STOP,
        ):
            continue  # it reached a verified plan, or stopped on safety
        if kb.sop_for(record.machine_id, top.fault_class) is not None:
            continue  # the plant has the procedure; the block is elsewhere
        grouped.setdefault((record.machine_id, top.fault_class), []).append(record)

    gaps: list[KnowledgeGap] = []
    for (machine_id, fault_class), members in grouped.items():
        members.sort(key=lambda r: (_stamp(r), r.case_id))
        exits: dict[str, int] = {}
        for record in members:
            exits[record.exit_reason] = exits.get(record.exit_reason, 0) + 1
        open_ids = [r.case_id for r in members if r.status == OPEN_STATUS]
        gaps.append(
            KnowledgeGap(
                machine_id=machine_id,
                fault_class=fault_class,
                machine_on_register=machine_id in kb.machines(),
                blocked_cases=len(members),
                case_ids=[r.case_id for r in members],
                first_case_id=members[0].case_id,
                latest_case_id=members[-1].case_id,
                first_seen_at=_stamp(members[0]),
                latest_seen_at=_stamp(members[-1]),
                rounds_burned=sum(len(r.rounds) for r in members),
                exits=dict(sorted(exits.items())),
                open_now=bool(open_ids),
                open_case_ids=open_ids,
                standing_obligation_ids=sorted(
                    obligations_by_pair.get((machine_id, fault_class), [])
                ),
            )
        )
    # Open-now first (the gap is biting), then the count of cases
    # it has blocked, then the rounds it has burned; the asset
    # name settles ties so the order is the records', not a dict's.
    gaps.sort(
        key=lambda g: (
            not g.open_now,
            -g.blocked_cases,
            -g.rounds_burned,
            g.machine_id,
            g.fault_class,
        )
    )
    return KnowledgeGapReport(
        as_of=format_timestamp(moment),
        basis=BASIS,
        cases_examined=len(records),
        blocked_cases=sum(g.blocked_cases for g in gaps),
        gaps=gaps,
    )


def render_text(report: KnowledgeGapReport) -> str:
    """The report as plain text — the CLI's and the shift log's form."""
    lines = [
        f"KNOWLEDGE GAPS — as of {report.as_of}",
        report.basis,
        "",
        f"{report.cases_examined} case record(s) examined · "
        f"{report.blocked_cases} blocked on missing knowledge · "
        f"{len(report.gaps)} missing SOP(s)",
        "",
    ]
    if report.gaps:
        for gap in report.gaps:
            register = (
                "on the machine register"
                if gap.machine_on_register
                else "NOT on the machine register"
            )
            lines.append(
                f"  - {gap.machine_id} · {gap.fault_class} ({register}) — "
                f"{gap.blocked_cases} case(s) blocked, "
                f"{gap.rounds_burned} round(s) burned · "
                + (
                    f"BLOCKED NOW: {', '.join(gap.open_case_ids)}"
                    if gap.open_now
                    else "not blocking now — history"
                )
            )
            lines.append(
                f"      cases: {', '.join(gap.case_ids)} · first "
                f"{gap.first_case_id} ({gap.first_seen_at or '—'}) · latest "
                f"{gap.latest_case_id} ({gap.latest_seen_at or '—'}) · exits: "
                + ", ".join(f"{k} x{v}" for k, v in gap.exits.items())
            )
            if gap.standing_obligation_ids:
                lines.append(
                    "      standing PM obligation(s) on the same hole: "
                    + ", ".join(gap.standing_obligation_ids)
                )
    else:
        lines.append(
            "  (none — every non-converged case's diagnosed fault has "
            "an SOP on record)"
        )
    return "\n".join(lines) + "\n"
