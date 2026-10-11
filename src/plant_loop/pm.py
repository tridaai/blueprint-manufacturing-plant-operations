"""From watch to work: preventive-maintenance proposals.

The recurring-fault watch (reliability.py) notices a pattern and
puts an item on the board. An item changes nothing — a pattern the
plant has seen three times gets the same fix a fourth. This layer
is what the watch earns when its rule fires:

- a **proposal**, drafted in code from the watch's own
  contributing set (the recurrence is defined once, in the watch;
  this module cites its cases, it does not recount them). The
  proposal carries the evidence — case ids and the moment each
  entered the record — and a proposed inspection interval derived
  from the evidence's spacing by the plant model's authored rule:
  ``interval_fraction`` × the mean gap between consecutive cases,
  clamped to the model's [min, max] bounds. The arithmetic is on
  the record, so the supervisor decides on numbers, not on a
  summary of numbers.
- a supervisor **decision**. Approving creates the standing
  **obligation** — at the interval the supervisor approves, which
  may be edited at the gate; the approved interval is the one
  that stands, and both values stay on the proposal. Rejecting
  requires a reason and moves the **evidence watermark**: the
  cases the rejected proposal cited are spent, and a new proposal
  for the same machine + fault needs a fresh threshold of cases
  beyond them. A watch that re-nags the same evidence is a watch
  people mute.
- a **cycle** on the board. When a board runs at or after an
  obligation's due point, the obligation sends an ordinary
  ``maintenance_alert`` case — the existing machinery works it,
  the existing priority formula ranks it (maintenance base), the
  supervisor decides at the existing gate. The supervisor's
  *outcome* on that case is what closes the cycle: accepted means
  the inspection was performed — and only that outcome ever marks
  a PM performed; this module stamps the performed date from the
  outcome, never from its own bookkeeping. A rejected outcome is
  a deferral: the supervisor's reason rides in the outcome's
  work-performed note (the service refuses a reasonless one), and
  the due point rolls forward by the plant model's deferral step
  — soon, not a full interval, because the pattern that earned
  the obligation has not gone away.

Two exclusions keep the evidence honest. A PM cycle case is an
inspection the system itself scheduled, so it never counts as a
recurrence — neither as an evaluation trigger nor in a proposal's
evidence (the watch still counts case records exactly as it
always has; the proposal layer cites faults, not inspections).
And while a proposal waits undecided, or an obligation stands,
for a machine + fault class, no second proposal is drafted for
that pair — the plant is already answering the pattern.

An obligation whose cycle case is dropped when the next shift
accepts the board (carryover.py) rolls like a deferral, with the
drop's own by/reason on the deferral entry — the supervisor said
"not this shift" out loud, and the obligation believes them.

Storage follows the house precedent (migration 0006): proposals
and obligations are PM records in their own table, because a
decision may land days after the draft and a due point weeks
after the approval — later requests must find them where they
live, not inside some case's payload.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

from . import kb
from .events import format_timestamp
from .ports import Store
from .reliability import ReliabilityWatch, case_moment, effective_fault_class, parse_timestamp
from .schemas import (
    CaseResult,
    ExceptionCase,
    PmObligation,
    PmProposal,
    SupervisorOutcome,
)

# Outcome decisions that mean the work happened (the plan the gate
# approved was the plan the floor executed, possibly with changes).
_PERFORMED_DECISIONS = ("accepted", "accepted_with_changes")


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def interval_arithmetic(moments: list[datetime], policy: dict) -> dict:
    """The proposed interval for a set of evidence moments.

    Gaps between consecutive cases, their mean, the plant model's
    fraction of it, rounded half-up to whole days and clamped to
    the model's bounds. Every intermediate value is returned —
    the proposal prints them all, so the rule can be re-derived
    from the record alone.
    """
    gaps = [
        (later - earlier).total_seconds() / 86400.0
        for earlier, later in zip(moments, moments[1:])
    ]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    raw = float(policy["interval_fraction"]) * mean_gap
    rounded = _round_half_up(raw)
    proposed = min(
        max(rounded, int(policy["min_interval_days"])),
        int(policy["max_interval_days"]),
    )
    return {
        "gaps_days": [round(g, 2) for g in gaps],
        "mean_gap_days": round(mean_gap, 2),
        "raw_interval_days": round(raw, 2),
        "proposed_interval_days": proposed,
        "clamped": proposed != rounded,
    }


class PmManager:
    """Proposals, decisions, obligations, cycles — over the store."""

    def __init__(
        self,
        store: Store,
        watch: ReliabilityWatch,
        policy: dict | None = None,
    ) -> None:
        self.store = store
        self.watch = watch
        # The rule's parameters are plant-model data (kb.pm_policy);
        # a caller (a test, a deployment harness) may pin its own.
        self.policy = policy or kb.pm_policy()

    # -- reads -------------------------------------------------------------

    def proposals(self) -> list[PmProposal]:
        return [r for r in self.store.all_pm_records() if isinstance(r, PmProposal)]

    def obligations(self) -> list[PmObligation]:
        return [r for r in self.store.all_pm_records() if isinstance(r, PmObligation)]

    def obligation_for_case(self, case_id: str) -> PmObligation | None:
        """The obligation whose open cycle case this is, if any."""
        for obligation in self.obligations():
            if obligation.cycle_case_id == case_id:
                return obligation
        return None

    def _pm_case_ids(self) -> set[str]:
        """Every cycle case ever surfaced — the evidence exclusion."""
        return {cid for o in self.obligations() for cid in o.cycle_case_ids}

    # -- drafting ------------------------------------------------------------

    def evaluate(
        self, trigger: CaseResult, *, now: datetime | None = None
    ) -> PmProposal | None:
        """Draft (and return) a proposal if the trigger's machine +
        fault class has earned one; else None.

        Runs wherever the watch is checked (a case persisting at
        the gate, an outcome landing). "Earned" is the watch's own
        rule — its threshold over its contributing set — read
        against what is already spoken for: an undecided proposal
        or a standing obligation for the pair suppresses a draft,
        and cases a rejected proposal already cited (the watermark)
        or that are PM cycle cases themselves do not count toward
        the fresh threshold.
        """
        machine_id = trigger.machine_id
        fault_class = effective_fault_class(trigger)
        if not machine_id or not fault_class:
            return None
        pm_case_ids = self._pm_case_ids()
        if trigger.case_id in pm_case_ids:
            return None  # an inspection is not a recurrence
        moment = now or datetime.now(timezone.utc)
        proposals = [
            p
            for p in self.proposals()
            if p.machine_id == machine_id and p.fault_class == fault_class
        ]
        if any(p.status == "proposed" for p in proposals):
            return None  # one open proposal per pair at a time
        if any(
            o.machine_id == machine_id and o.fault_class == fault_class
            for o in self.obligations()
        ):
            return None  # the plant is already acting on this pattern
        watermark = {
            entry["case_id"]
            for p in proposals
            if p.status == "rejected"
            for entry in p.evidence
        }
        contributing = self.watch.contributing_cases(machine_id, fault_class, now=moment)
        eligible = [
            r
            for r in contributing
            if r.case_id not in watermark and r.case_id not in pm_case_ids
        ]
        if len(eligible) < self.watch.policy.threshold:
            return None
        moments = [case_moment(r) or moment for r in eligible]
        math_out = interval_arithmetic(moments, self.policy)
        board_item_id = ""
        for item in self.store.all_board_items():
            if (
                item.kind == "reliability_review"
                and item.status == "open"
                and item.machine_id == machine_id
                and item.fault_class == fault_class
            ):
                board_item_id = item.item_id
                break
        case_list = ", ".join(
            f"{r.case_id} ({format_timestamp(m)})" for r, m in zip(eligible, moments)
        )
        gaps_text = ", ".join(str(g) for g in math_out["gaps_days"]) or "—"
        rationale = (
            f"{len(eligible)} cases of {fault_class} on {machine_id} reached the "
            f"gate within {self.watch.policy.window_days} days: {case_list}. "
            f"Gaps between consecutive cases: {gaps_text} days — mean gap "
            f"{math_out['mean_gap_days']} days. Proposed interval = "
            f"{self.policy['interval_fraction']} x mean gap = "
            f"{math_out['raw_interval_days']} days"
        )
        if math_out["clamped"]:
            rationale += (
                f", clamped to the plant model's "
                f"{self.policy['min_interval_days']}–{self.policy['max_interval_days']} "
                f"day bounds"
            )
        rationale += f" -> {math_out['proposed_interval_days']} days."
        proposal = PmProposal(
            proposal_id=f"PMP-{uuid.uuid4().hex[:8]}",
            status="proposed",
            machine_id=machine_id,
            fault_class=fault_class,
            line_id=kb.machines().get(machine_id, {}).get("line_id", ""),
            board_item_id=board_item_id,
            evidence=[
                {"case_id": r.case_id, "at": format_timestamp(m)}
                for r, m in zip(eligible, moments)
            ],
            fault_count=len(eligible),
            window_days=self.watch.policy.window_days,
            threshold=self.watch.policy.threshold,
            gaps_days=math_out["gaps_days"],
            mean_gap_days=math_out["mean_gap_days"],
            interval_fraction=float(self.policy["interval_fraction"]),
            min_interval_days=int(self.policy["min_interval_days"]),
            max_interval_days=int(self.policy["max_interval_days"]),
            raw_interval_days=math_out["raw_interval_days"],
            clamped=math_out["clamped"],
            proposed_interval_days=math_out["proposed_interval_days"],
            rationale=rationale,
            created_at=format_timestamp(moment),
        )
        self.store.save_pm_record(proposal)
        return proposal

    # -- the decision ----------------------------------------------------------

    def decide(
        self,
        proposal_id: str,
        decision: str,
        *,
        by: str = "",
        reason: str = "",
        interval_days: int | None = None,
        last_performed_at: str = "",
        now: datetime | None = None,
    ) -> tuple[PmProposal, PmObligation | None]:
        """Record the supervisor's decision on a proposal.

        Approval creates the standing obligation at the approved
        interval (the proposal's own, unless edited here — the
        approved value is the one that stands). Its clock runs
        from the last performed inspection when the supervisor
        declares that date at approval (the plant knows when it
        last did this inspection; the declaration is the
        supervisor's fact, recorded as theirs), else from the
        approval itself. Rejection requires the reason: it moves
        the evidence watermark, and the record must say why the
        pattern did not earn prevention.
        """
        record = self.store.get_pm_record(proposal_id)
        if record is None or not isinstance(record, PmProposal):
            raise KeyError(f"unknown PM proposal: {proposal_id}")
        proposal = record
        if proposal.status != "proposed":
            raise ValueError(
                f"proposal {proposal_id} is already {proposal.status} — "
                "a decision is recorded once"
            )
        moment = now or datetime.now(timezone.utc)
        stamp = format_timestamp(moment)
        if decision == "rejected":
            if not reason.strip():
                raise ValueError(
                    "a rejected PM proposal needs the supervisor's reason — "
                    "the rejection spends the cited evidence, and the record "
                    "must say why"
                )
            proposal.status = "rejected"
            proposal.decided_by = by
            proposal.decided_at = stamp
            proposal.decision_reason = reason
            self.store.save_pm_record(proposal)
            return proposal, None
        if decision != "approved":
            raise ValueError(f"unknown PM decision: {decision!r} (approved | rejected)")
        interval = (
            interval_days
            if interval_days is not None
            else proposal.proposed_interval_days
        )
        if interval < 1:
            raise ValueError("the approved interval must be at least 1 day")
        declared: datetime | None = None
        if last_performed_at:
            declared = parse_timestamp(last_performed_at)
            if declared is None:
                raise ValueError(
                    f"last_performed_at is not a record timestamp: {last_performed_at!r}"
                )
        basis = declared or moment
        obligation = PmObligation(
            obligation_id=f"PMO-{uuid.uuid4().hex[:8]}",
            proposal_id=proposal.proposal_id,
            machine_id=proposal.machine_id,
            fault_class=proposal.fault_class,
            line_id=proposal.line_id,
            interval_days=int(interval),
            approved_by=by,
            approved_at=stamp,
            clock_basis="last_performed" if declared else "approval",
            last_performed_at=format_timestamp(declared) if declared else "",
            next_due_at=format_timestamp(basis + timedelta(days=int(interval))),
            created_at=stamp,
        )
        proposal.status = "approved"
        proposal.decided_by = by
        proposal.decided_at = stamp
        proposal.decision_reason = reason
        proposal.approved_interval_days = int(interval)
        self.store.save_pm_record(proposal)
        self.store.save_pm_record(obligation)
        return proposal, obligation

    # -- the board remembering the date -----------------------------------------

    def due_cases(self, *, now: datetime | None = None) -> list[ExceptionCase]:
        """The PM cases this moment's boards must carry.

        Every standing obligation at or past its due point, with
        no cycle already open, sends one maintenance_alert case.
        The cycle opens on the obligation as the case is built —
        so two boards, or one board read twice, never surface the
        same cycle twice.
        """
        moment = now or datetime.now(timezone.utc)
        cases: list[ExceptionCase] = []
        for obligation in sorted(
            self.obligations(), key=lambda o: (o.next_due_at, o.obligation_id)
        ):
            if obligation.cycle_case_id:
                continue  # a cycle is already on a board somewhere
            due = parse_timestamp(obligation.next_due_at)
            if due is None or due > moment:
                continue
            cycle_no = obligation.cycles_completed + len(obligation.deferrals) + 1
            case_id = f"{obligation.obligation_id}-{cycle_no}"
            cases.append(self._pm_case(obligation, case_id))
            obligation.cycle_case_id = case_id
            obligation.cycle_case_ids.append(case_id)
            self.store.save_pm_record(obligation)
        return cases

    def _pm_case(self, obligation: PmObligation, case_id: str) -> ExceptionCase:
        """The board case for one due cycle of an obligation."""
        sop = kb.sop_for(obligation.machine_id, obligation.fault_class)
        sop_line = (
            f" Work the inspection under {sop['sop_id']} ({sop['title']})." if sop else ""
        )
        last = obligation.last_performed_at or "never — this is the first cycle"
        description = (
            f"Standing PM obligation {obligation.obligation_id} (proposal "
            f"{obligation.proposal_id}, approved {obligation.approved_at} by "
            f"{obligation.approved_by or 'a supervisor'}) set a "
            f"{obligation.interval_days}-day inspection interval for the recurring "
            f"{obligation.fault_class} pattern on {obligation.machine_id}. Last "
            f"performed: {last}. This cycle fell due {obligation.next_due_at}."
            f"{sop_line} Record the outcome on this case: accepted when the "
            "inspection has been performed; rejected — with the reason as the "
            "work-performed note — when it must defer. The outcome is what "
            "closes or rolls the cycle; nothing else marks a PM performed."
        )
        return ExceptionCase(
            case_id=case_id,
            type="maintenance_alert",
            title=(
                f"Preventive maintenance due: {obligation.fault_class} "
                f"inspection on {obligation.machine_id}"
            ),
            description=description,
            machine_id=obligation.machine_id,
            line_id=obligation.line_id,
            reported_at=obligation.next_due_at,
            reporter=f"PM obligation {obligation.obligation_id}",
            downtime_minutes=0,
        )

    # -- closing and rolling the cycle -------------------------------------------

    def close_cycle(
        self, record: CaseResult, outcome: SupervisorOutcome
    ) -> PmObligation | None:
        """Close or roll the cycle a supervisor's outcome settles.

        The outcome is the only pen that writes here: an accepted
        outcome stamps the PM performed at the outcome's own
        recorded moment and sets the next due point a full
        approved interval later; a rejected outcome is a deferral —
        its reason is the outcome's work-performed note (the
        service refuses a reasonless one before anything saves) —
        and the due point rolls forward by the plant model's
        deferral step from the deferral.
        """
        obligation = self.obligation_for_case(record.case_id)
        if obligation is None:
            return None
        at = parse_timestamp(outcome.recorded_at) or datetime.now(timezone.utc)
        if outcome.decision in _PERFORMED_DECISIONS:
            obligation.last_performed_at = outcome.recorded_at
            obligation.next_due_at = format_timestamp(
                at + timedelta(days=obligation.interval_days)
            )
            obligation.cycles_completed += 1
            obligation.last_cycle = {
                "case_id": record.case_id,
                "result": "performed",
                "at": outcome.recorded_at,
                "by": outcome.recorded_by,
            }
        elif outcome.decision == "rejected":
            rolled = at + timedelta(days=int(self.policy["deferral_roll_days"]))
            obligation.deferrals.append(
                {
                    "case_id": record.case_id,
                    "at": outcome.recorded_at,
                    "by": outcome.recorded_by,
                    "reason": outcome.work_performed.strip(),
                    "kind": "outcome",
                    "previous_due_at": obligation.next_due_at,
                    "rolled_to": format_timestamp(rolled),
                }
            )
            obligation.next_due_at = format_timestamp(rolled)
            obligation.last_cycle = {
                "case_id": record.case_id,
                "result": "deferred",
                "at": outcome.recorded_at,
                "by": outcome.recorded_by,
            }
        obligation.cycle_case_id = ""
        self.store.save_pm_record(obligation)
        return obligation

    def note_board_drop(
        self, case_id: str, *, by: str = "", reason: str = "", now: datetime | None = None
    ) -> PmObligation | None:
        """Roll a cycle whose case was dropped at board acceptance.

        A supervisor dropping the PM case when the next shift
        accepts the board (carryover.py) has deferred it in every
        way that matters — the obligation rolls exactly as an
        outcome deferral does, with the drop's by/reason kept on
        the deferral entry so the record says who said "not this
        shift" and why.
        """
        obligation = self.obligation_for_case(case_id)
        if obligation is None:
            return None
        moment = now or datetime.now(timezone.utc)
        rolled = moment + timedelta(days=int(self.policy["deferral_roll_days"]))
        obligation.deferrals.append(
            {
                "case_id": case_id,
                "at": format_timestamp(moment),
                "by": by,
                "reason": reason or "dropped at board acceptance",
                "kind": "board_drop",
                "previous_due_at": obligation.next_due_at,
                "rolled_to": format_timestamp(rolled),
            }
        )
        obligation.next_due_at = format_timestamp(rolled)
        obligation.last_cycle = {
            "case_id": case_id,
            "result": "dropped_at_acceptance",
            "at": format_timestamp(moment),
            "by": by,
        }
        obligation.cycle_case_id = ""
        self.store.save_pm_record(obligation)
        return obligation
