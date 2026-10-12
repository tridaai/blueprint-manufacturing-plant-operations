"""Downtime window planning: Saturday's four hours, chosen in code.

The board loop ranks cases, and the shift budget (loops.py) rations
the *agent's* work — so many cases, so many tokens. Neither answers
the floor's question: the line stops Saturday 08:00–12:00; which
work goes into those hours, in what order, and what is left out?
This module is that answer, computed in code like everything else
on the board:

- a supervisor **declares** a window: a start, a length in
  minutes, and a scope in the plant model's own ids — named lines
  and/or named machines. Declaring is an act: the actor is
  required and recorded, and an unknown line or machine is
  refused by name rather than silently narrowing the window.
- the **plan** is computed over work-ready cases and due PM
  obligations, packed in the board's own priority order against
  the window's minutes. Nothing is asked of a model: readiness,
  estimates, priority and packing are all read off records and
  the plant model, under the rules printed on the plan itself.
- the plan is a **proposal** until a supervisor **confirms** it.
  Confirmation re-reads the state the plan was computed over (a
  fingerprint over every candidate, taken at computation):
  a member that has since been decided, outcomed, dropped or
  stripped of its claims refuses the confirmation, naming the
  drift — the remedy is to re-plan, not to argue with the hash.
  A confirmed window's packed cases are *booked*: a second
  window overlapping it plans around them, listing them as
  booked elsewhere rather than double-booking the same work.

WORK-READY, precisely. A case is work-ready for a window when:

- its plan **converged** (``exit_reason == converged``) — the only
  exit whose plan the verify stage has cleared. A safety stop is
  never work-ready: it was escalated at round 0 precisely because
  it is not work to schedule, and it claims nothing;
- a supervisor has **approved** it at the gate (status
  ``approved``). A case still awaiting its decision is not
  ready — the window plans work a supervisor has already said
  yes to, it does not pre-empt the gate;
- **no outcome** is recorded — the work is still ahead of the
  floor, not already done and being written up;
- its latest gate amendment, if any, did not fail
  re-verification (an approval over a failed amendment is
  refused at the gate, so this state should not arise — the
  check stands anyway, because readiness is read here, not
  assumed from another module's discipline);
- the allocation ledger's standing claims **cover** the plan as
  it stands at the gate (the original plan with its passing
  amendments replayed, amendments.py): every satisfied
  requirement's drawn part is claimed for this case, in full
  quantity. A claim released since the approval — a board
  turnover, a moot close — is read here as it stands in the
  ledger, never assumed from the approval.

ESTIMATES. A case's estimate is its plan's own
``estimated_downtime_min`` — the downtime the plan itself states
(the offline planner takes it from the SOP for the diagnosed
fault; a live planner states its own). A PM obligation's
estimate is the SOP's ``estimated_downtime_min`` for its machine
+ fault class — the same figure the parts outlook prices its
cycles from. Both are *authored planning figures*, carried on
records and in the plant model; nothing here measures a duration
or learns one from outcomes. A candidate with no estimate on
record (a plan that states 0, an obligation whose fault has no
SOP) is UNESTIMATED: listed by name in its own section, packed
never — a window plan does not invent the number it would need.

PRIORITY. Cases pack in board order: the score stamped on the
record at the gate, which is priority.py's formula. An obligation
competes on the same footing its cycle case would: the formula
read as a ``maintenance_alert`` on the obligation's line, due
moment as its reported time — the score and its reason are
computed by priority.py itself and printed on the item, so an
obligation interleaves with cases by the plant's one ranking,
never by a second one invented here. Packing is greedy in that
order and never reorders to fill gaps: an item that does not fit
in the minutes remaining is DEFERRED (with its shortfall), and
the next item is considered against the same remainder — the
order is the plant's priority, and the plan shows the cost of
honouring it. An item longer than the window itself is
OVERSIZED: listed, never squeezed, never split.

The window is an overlay. Planning and confirming write the
window record and nothing else: no case record changes, no claim
moves, no obligation's cycle opens (an obligation is packed as an
obligation — its cycle case surfaces when a board runs, as
always). A case in a confirmed window still ends at its own
gate; the window is the floor's schedule for the downtime, not a
second decision about the work.

THE REVIEW. A confirmed window is a promise, and once its span
has closed the plant can say what the promise came to — the
review at the foot of this module. It is computed in code from
the same records, per packed item: PERFORMED (the item's outcome
says the work happened), DEFERRED (the outcome refused it, or a
PM cycle rolled), or NOT WORKED (no outcome — the item still
open, or closed without the work). The review is refused while
the span is still open: the states only mean something once the
window has closed, and naming work unworked mid-window would
libel work in progress. The minutes it restates are the plan's
own estimates — planned minutes performed, never "time worked":
an estimate is a planning figure, and the review does not
relabel it as a measurement after the fact. Like the plan, the
review writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone

from . import kb
from .allocation import AllocationLedger
from .amendments import (
    apply_passing_amendments,
    drawn_part,
    latest_failed_amendment,
)
from .events import format_timestamp
from .ports import Store
from .priority import priority_for
from .reliability import parse_timestamp
from .schemas import (
    EXIT_CONVERGED,
    EXIT_SAFETY_STOP,
    CaseResult,
    DowntimeWindow,
    ExceptionCase,
    PmObligation,
    WindowPlanItem,
)

STATUS_PROPOSED = "proposed"
STATUS_CONFIRMED = "confirmed"

PLAN_BASIS = (
    "Packed in board priority order (priority.py): a case reads its "
    "score as stamped on its record at the gate; a PM obligation reads "
    "the same formula as the maintenance_alert its cycle case would "
    "be. Work-ready means: plan converged, supervisor-approved, no "
    "outcome recorded, and the allocation ledger's standing claims "
    "covering the plan as it stands at the gate. Estimates are the "
    "plans' and SOPs' own stated downtime — authored planning "
    "figures, not measurements. Packing is greedy in priority order "
    "and never reorders to fill a gap."
)


def parse_moment(value: str) -> datetime | None:
    """Parse a window timestamp: the house record format, or an
    ISO-8601 moment with an offset (converted to UTC)."""
    parsed = parse_timestamp(value or "")
    if parsed is not None:
        return parsed
    try:
        moment = datetime.fromisoformat((value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _in_scope(window: DowntimeWindow, machine_id: str, line_id: str) -> bool:
    return machine_id in window.machine_ids or line_id in window.line_ids


def overlaps(a: DowntimeWindow, b: DowntimeWindow) -> bool:
    """Whether two windows' spans intersect (half-open: a window
    starting exactly when another ends does not overlap it)."""
    return a.starts_at < b.ends_at and b.starts_at < a.ends_at


# -- readiness -----------------------------------------------------------------


def case_readiness(
    record: CaseResult, ledger: AllocationLedger
) -> tuple[bool, str]:
    """Whether a case is work-ready, and if not, the named reason.

    The definition at the head of this module, applied in order —
    the first failing clause is the reason printed for the case.
    """
    if record.exit_reason == EXIT_SAFETY_STOP:
        return False, (
            "a safety stop — escalated at round 0, never work to schedule"
        )
    if record.exit_reason != EXIT_CONVERGED:
        return False, f"the plan did not converge ({record.exit_reason})"
    if record.status != "approved":
        return False, (
            f"not approved at the gate (status: {record.status}) — "
            "a case awaiting its decision is not ready"
        )
    if record.outcome is not None:
        return False, "an outcome is already recorded — the work is not ahead"
    if latest_failed_amendment(record) is not None:
        return False, "the latest gate amendment failed re-verification"
    plan = apply_passing_amendments(record)
    standing: dict[str, int] = {}
    for claim in ledger.claims_for_case(record.case_id):
        if claim.status == "claimed":
            standing[claim.part_id] = standing.get(claim.part_id, 0) + claim.qty
    for req in plan.required_parts:
        if not req.satisfied:
            return False, f"requirement {req.part_id} is not satisfied"
        drawn = drawn_part(req)
        if standing.get(drawn, 0) < req.qty:
            return False, (
                f"the ledger's standing claims do not cover {drawn} "
                f"(needs {req.qty}, claims {standing.get(drawn, 0)}) — "
                "readiness is read from the ledger, never assumed"
            )
    return True, ""


def _case_item(record: CaseResult) -> WindowPlanItem:
    estimate = record.plan.estimated_downtime_min or None
    return WindowPlanItem(
        kind="case",
        item_id=record.case_id,
        machine_id=record.machine_id,
        line_id=record.line_id,
        title=record.title,
        priority=record.priority,
        priority_reason=record.priority_reason,
        estimate_min=estimate,
        estimate_basis=(
            "the plan's own estimated downtime (the SOP's figure for "
            "the diagnosed fault)"
            if estimate
            else ""
        ),
    )


def _obligation_item(obligation: PmObligation) -> WindowPlanItem:
    """An obligation as a packing candidate: its priority is the
    board formula read as the maintenance_alert its cycle case
    would be (priority.py computes it — no second ranking), and
    its estimate is its SOP's stated downtime, if it has one."""
    probe = ExceptionCase(
        case_id=obligation.obligation_id,
        type="maintenance_alert",
        title=(
            f"Preventive maintenance due: {obligation.fault_class} "
            f"inspection on {obligation.machine_id}"
        ),
        machine_id=obligation.machine_id,
        line_id=obligation.line_id,
        reported_at=obligation.next_due_at,
        downtime_minutes=0,
    )
    score, reason = priority_for(probe)
    sop = kb.sop_for(obligation.machine_id, obligation.fault_class)
    estimate = (sop or {}).get("estimated_downtime_min") or None
    return WindowPlanItem(
        kind="pm_obligation",
        item_id=obligation.obligation_id,
        machine_id=obligation.machine_id,
        line_id=obligation.line_id,
        title=probe.title,
        priority=score,
        priority_reason=reason,
        estimate_min=int(estimate) if estimate else None,
        estimate_basis=(
            f"the SOP's stated downtime ({sop['sop_id']})" if estimate else ""
        ),
    )


# -- the fingerprint -------------------------------------------------------------


def _snapshot(
    store: Store,
    ledger: AllocationLedger,
    window: DowntimeWindow,
) -> list[dict]:
    """The readiness-relevant state the plan reads, as small dicts.

    One entry per candidate: every in-scope case (ready or not —
    a case becoming ready is drift too), every candidate
    obligation, and the confirmed bookings the plan must respect.
    The fingerprint is a hash of this list; the list itself rides
    the window record so a confirmation refusal can name what
    drifted instead of reporting a bare hash change.
    """
    entries: list[dict] = []
    for record in sorted(store.all_cases(), key=lambda r: r.case_id):
        if not _in_scope(window, record.machine_id, record.line_id):
            continue
        ready, _why = case_readiness(record, ledger)
        plan = apply_passing_amendments(record)
        entries.append(
            {
                "key": f"case:{record.case_id}",
                "status": record.status,
                "decision": record.decision,
                "exit_reason": record.exit_reason,
                "outcome_recorded": record.outcome is not None,
                "estimate_min": record.plan.estimated_downtime_min,
                "requirements": [
                    [drawn_part(req), req.qty, req.satisfied]
                    for req in plan.required_parts
                ],
                "claims": sorted(
                    [c.part_id, c.qty]
                    for c in ledger.claims_for_case(record.case_id)
                    if c.status == "claimed"
                ),
                "amendments": len(record.amendments),
                "ready": ready,
            }
        )
    window_end = parse_moment(window.ends_at)
    for obligation in sorted(
        store.all_pm_records(), key=lambda o: getattr(o, "obligation_id", "")
    ):
        if not isinstance(obligation, PmObligation):
            continue
        if obligation.status != "standing":
            continue
        if not _in_scope(window, obligation.machine_id, obligation.line_id):
            continue
        due = parse_timestamp(obligation.next_due_at)
        entries.append(
            {
                "key": f"pm:{obligation.obligation_id}",
                "status": obligation.status,
                "next_due_at": obligation.next_due_at,
                "cycle_case_id": obligation.cycle_case_id,
                "candidate": bool(
                    due is not None and window_end is not None and due <= window_end
                ),
            }
        )
    for other in sorted(store.all_window_records(), key=lambda w: w.window_id):
        if other.window_id == window.window_id or other.status != STATUS_CONFIRMED:
            continue
        if not overlaps(window, other):
            continue
        entries.append(
            {
                "key": f"window:{other.window_id}",
                "packed": sorted(
                    f"{item.kind}:{item.item_id}" for item in other.packed
                ),
            }
        )
    return entries


def _fingerprint(window: DowntimeWindow, snapshot: list[dict]) -> str:
    payload = {
        "scope": {
            "starts_at": window.starts_at,
            "ends_at": window.ends_at,
            "length_min": window.length_min,
            "line_ids": sorted(window.line_ids),
            "machine_ids": sorted(window.machine_ids),
        },
        "snapshot": snapshot,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _drift_lines(stored: list[dict], current: list[dict]) -> list[str]:
    """Name the difference between two snapshots, for a refusal."""
    before = {e["key"]: e for e in stored}
    now = {e["key"]: e for e in current}
    lines: list[str] = []
    for key in sorted(set(before) | set(now)):
        if key not in now:
            lines.append(f"{key} no longer stands as it did")
        elif key not in before:
            lines.append(f"{key} stands newly since the plan was computed")
        elif before[key] != now[key]:
            lines.append(f"{key} changed since the plan was computed")
    return lines


# -- the planner -----------------------------------------------------------------


class WindowPlanner:
    """Declare, plan, confirm — over the store and the ledger."""

    def __init__(self, store: Store, ledger: AllocationLedger) -> None:
        self.store = store
        self.ledger = ledger

    # -- reads -----------------------------------------------------------------

    def windows(self) -> list[DowntimeWindow]:
        return sorted(
            self.store.all_window_records(),
            key=lambda w: (w.starts_at, w.window_id),
        )

    def booked_items(self, window: DowntimeWindow) -> dict[str, str]:
        """{(kind, item_id): window_id} packed into other CONFIRMED
        windows overlapping this one — the double-booking register
        a plan reads. Proposed windows book nothing."""
        booked: dict[str, str] = {}
        for other in self.store.all_window_records():
            if other.window_id == window.window_id:
                continue
            if other.status != STATUS_CONFIRMED:
                continue
            if not overlaps(window, other):
                continue
            for item in other.packed:
                booked[(item.kind, item.item_id)] = other.window_id
        return booked

    # -- declare -----------------------------------------------------------------

    def declare(
        self,
        *,
        by: str = "",
        starts_at: str = "",
        length_min: int = 0,
        line_ids: list[str] | None = None,
        machine_ids: list[str] | None = None,
        now: datetime | None = None,
    ) -> DowntimeWindow:
        """Declare a window and compute its first plan.

        A supervisor's act: the actor is required; the start must
        be a real moment; the length at least a minute; the scope
        must name at least one line or machine, and every id must
        be the plant model's own — an unknown id is refused by
        name rather than silently narrowing the window.
        """
        if not by.strip():
            raise ValueError(
                "declaring a window needs the supervisor's name — "
                "the record must say whose window this is"
            )
        start = parse_moment(starts_at)
        if start is None:
            raise ValueError(
                f"starts_at is not a record timestamp: {starts_at!r}"
            )
        if length_min < 1:
            raise ValueError("a window's length must be at least 1 minute")
        lines = list(line_ids or [])
        machines = list(machine_ids or [])
        if not lines and not machines:
            raise ValueError(
                "a window must name its scope: at least one line or machine"
            )
        for line_id in lines:
            if line_id not in kb.lines():
                raise ValueError(
                    f"unknown line in the window's scope: {line_id!r} "
                    "(the plant model names the lines)"
                )
        for machine_id in machines:
            if machine_id not in kb.machines():
                raise ValueError(
                    f"unknown machine in the window's scope: {machine_id!r} "
                    "(the plant model names the machines)"
                )
        moment = now or datetime.now(timezone.utc)
        window = DowntimeWindow(
            window_id=f"WIN-{uuid.uuid4().hex[:8]}",
            status=STATUS_PROPOSED,
            declared_by=by,
            declared_at=format_timestamp(moment),
            starts_at=format_timestamp(start),
            ends_at=format_timestamp(start + timedelta(minutes=int(length_min))),
            length_min=int(length_min),
            line_ids=lines,
            machine_ids=machines,
            created_at=format_timestamp(moment),
        )
        self.store.save_window_record(window)
        return self.plan(window.window_id, now=moment)

    # -- plan --------------------------------------------------------------------

    def plan(
        self, window_id: str, *, now: datetime | None = None
    ) -> DowntimeWindow:
        """(Re)compute a proposed window's plan from the records
        as they stand. A confirmed window is never re-planned —
        its plan is the record the floor is working to; declaring
        a new window is how a changed board gets a fresh plan."""
        window = self.store.get_window_record(window_id)
        if window is None:
            raise KeyError(f"unknown window: {window_id}")
        if window.status == STATUS_CONFIRMED:
            raise ValueError(
                f"window {window_id} is confirmed — a confirmed plan is "
                "not re-planned; declare a new window for a fresh plan"
            )
        moment = now or datetime.now(timezone.utc)
        booked = self.booked_items(window)
        window_end = parse_moment(window.ends_at)

        candidates: list[tuple[float, str, str, WindowPlanItem]] = []
        notes: list[str] = []
        for record in self.store.all_cases():
            if not _in_scope(window, record.machine_id, record.line_id):
                continue
            ready, _why = case_readiness(record, self.ledger)
            if ready:
                item = _case_item(record)
                candidates.append(
                    (item.priority, record.reported_at or "", item.item_id, item)
                )
        for record in self.store.all_pm_records():
            if not isinstance(record, PmObligation):
                continue
            if record.status != "standing":
                continue
            if not _in_scope(window, record.machine_id, record.line_id):
                continue
            due = parse_timestamp(record.next_due_at)
            if due is None or window_end is None or due > window_end:
                continue  # not due within the window's span
            if record.cycle_case_id:
                notes.append(
                    f"obligation {record.obligation_id} has its cycle case "
                    f"{record.cycle_case_id} open on the board — it is "
                    "planned through that case, not as an obligation"
                )
                continue
            item = _obligation_item(record)
            candidates.append(
                (item.priority, record.next_due_at or "", item.item_id, item)
            )
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        packed: list[WindowPlanItem] = []
        deferred: list[WindowPlanItem] = []
        oversized: list[WindowPlanItem] = []
        unestimated: list[WindowPlanItem] = []
        booked_elsewhere: list[WindowPlanItem] = []
        used = 0
        for _score, _moment, _id, item in candidates:
            holder = booked.get((item.kind, item.item_id))
            if holder is not None:
                item.note = f"booked into confirmed window {holder}"
                booked_elsewhere.append(item)
                continue
            if item.estimate_min is None:
                unestimated.append(item)
                continue
            if item.estimate_min > window.length_min:
                item.note = (
                    f"its estimate exceeds the window's "
                    f"{window.length_min} min by "
                    f"{item.estimate_min - window.length_min} min"
                )
                oversized.append(item)
                continue
            if used + item.estimate_min <= window.length_min:
                item.start_offset_min = used
                item.end_offset_min = used + item.estimate_min
                used += item.estimate_min
                packed.append(item)
            else:
                item.note = (
                    f"needs {item.estimate_min} min with "
                    f"{window.length_min - used} min left — short by "
                    f"{used + item.estimate_min - window.length_min} min"
                )
                deferred.append(item)

        snapshot = _snapshot(self.store, self.ledger, window)
        window.packed = packed
        window.deferred = deferred
        window.oversized = oversized
        window.unestimated = unestimated
        window.booked_elsewhere = booked_elsewhere
        window.minutes_used = used
        window.minutes_left = window.length_min - used
        window.plan_basis = PLAN_BASIS
        window.notes = notes
        window.fingerprint_detail = snapshot
        window.fingerprint = _fingerprint(window, snapshot)
        window.planned_at = format_timestamp(moment)
        self.store.save_window_record(window)
        return window

    # -- confirm -------------------------------------------------------------------

    def confirm(
        self, window_id: str, *, by: str = "", now: datetime | None = None
    ) -> DowntimeWindow:
        """Confirm a proposed window's plan: a supervisor's act.

        The state the plan was computed over is re-read and its
        fingerprint recomputed: any drift refuses the
        confirmation, naming what changed — a member decided,
        outcomed, dropped, or stripped of its claims since the
        computation means the plan no longer describes the board,
        and re-planning is the remedy. Confirmation moves no case,
        no claim, and no obligation: it marks the window's plan
        as the floor's schedule, and books its packed items
        against overlapping windows from this moment.
        """
        if not by.strip():
            raise ValueError(
                "confirming a window needs the supervisor's name — "
                "the record must say who took the plan as the schedule"
            )
        window = self.store.get_window_record(window_id)
        if window is None:
            raise KeyError(f"unknown window: {window_id}")
        if window.status == STATUS_CONFIRMED:
            raise ValueError(
                f"window {window_id} is already confirmed — "
                "confirmation is recorded once"
            )
        current = _snapshot(self.store, self.ledger, window)
        if _fingerprint(window, current) != window.fingerprint:
            drift = _drift_lines(window.fingerprint_detail, current)
            raise ValueError(
                f"window {window_id} cannot be confirmed: the board has "
                "drifted since its plan was computed — "
                + ("; ".join(drift) if drift else "the plan's basis changed")
                + ". Re-plan the window and confirm the fresh plan."
            )
        moment = now or datetime.now(timezone.utc)
        window.status = STATUS_CONFIRMED
        window.confirmed_by = by
        window.confirmed_at = format_timestamp(moment)
        self.store.save_window_record(window)
        return window


def window_counts(windows: list[DowntimeWindow]) -> dict:
    """The window statistics: counts, and only counts.

    Planned (declared) and confirmed — the two facts about the
    window book a reader needs. There is deliberately no
    utilisation figure here: the minutes arithmetic lives on each
    plan, where it is shown next to the items it summarises, and
    a plant-wide percentage would be a KPI invented from it.
    """
    return {
        "planned": len(windows),
        "proposed": sum(1 for w in windows if w.status == STATUS_PROPOSED),
        "confirmed": sum(1 for w in windows if w.status == STATUS_CONFIRMED),
    }


def upcoming_confirmed(store: Store, *, now: datetime | None = None) -> list[dict]:
    """Confirmed windows not yet over, for the shift handoff: the
    schedules the incoming shift inherits. Read-only — composed
    from the window records, in start order."""
    moment = now or datetime.now(timezone.utc)
    stamp = format_timestamp(moment)
    rows = []
    for window in store.all_window_records():
        if window.status != STATUS_CONFIRMED:
            continue
        if window.ends_at <= stamp:
            continue  # the window is over; it is history, not a handover
        rows.append(
            {
                "window_id": window.window_id,
                "starts_at": window.starts_at,
                "ends_at": window.ends_at,
                "length_min": window.length_min,
                "line_ids": window.line_ids,
                "machine_ids": window.machine_ids,
                "declared_by": window.declared_by,
                "confirmed_by": window.confirmed_by,
                "packed_cases": [
                    i.item_id for i in window.packed if i.kind == "case"
                ],
                "packed_obligations": [
                    i.item_id for i in window.packed if i.kind == "pm_obligation"
                ],
                "minutes_used": window.minutes_used,
                "minutes_left": window.minutes_left,
            }
        )
    rows.sort(key=lambda r: (r["starts_at"], r["window_id"]))
    return rows


# -- the review ------------------------------------------------------------------


# The outcome vocabulary's performed set (schemas.OUTCOME_DECISIONS):
# the decisions under which the floor did the work — the same set
# pm.py closes a cycle on. A "rejected" outcome is the floor
# declining the work, with the reason in its work-performed note.
PERFORMED_DECISIONS = ("accepted", "accepted_with_changes")

REVIEW_BASIS = (
    "Read from the records as they stand at the review's moment: a "
    "packed case is performed when its recorded outcome is accepted "
    "(with or without changes), deferred when its outcome was "
    "rejected, and not worked when no outcome stands — still open, "
    "or closed without the work (dropped, or closed as moot). A "
    "packed PM obligation is judged by the cycle case that answered "
    "the due point the plan packed: its outcome decides the same "
    "way, and a cycle case dropped at a board acceptance is a "
    "deferral — the obligation rolled, as it rolls on a rejection. "
    "The window promised the case, not the plan text: a case "
    "amended at the gate is read as it stands, amendments and all. "
    "Minutes are the plan's own estimates restated — planned "
    "minutes performed, never time worked."
)


def _planned_due(window: DowntimeWindow, obligation_id: str) -> str:
    """The due point the plan packed for an obligation, read off
    the window's own fingerprint snapshot — the promise as it was
    made, not the obligation's clock as it stands now (cycles
    answered since have advanced it)."""
    for entry in window.fingerprint_detail:
        if entry.get("key") == f"pm:{obligation_id}":
            return entry.get("next_due_at") or ""
    return ""


def _case_review(store: Store, item: WindowPlanItem) -> dict:
    record = store.get_case(item.item_id)
    row = {
        "kind": item.kind,
        "item_id": item.item_id,
        "machine_id": item.machine_id,
        "title": item.title,
        "estimate_min": item.estimate_min,
        "state": "not_worked",
        "still_open": False,
        "detail": "",
    }
    if record is None:
        row["detail"] = "the case is no longer on the record"
        return row
    if record.outcome is not None:
        outcome = record.outcome
        said = (
            f"outcome {outcome.decision} recorded by "
            f"{outcome.recorded_by or 'a supervisor'}"
            + (f" at {outcome.recorded_at}" if outcome.recorded_at else "")
            + (f" — {outcome.work_performed}" if outcome.work_performed else "")
        )
        if outcome.decision in PERFORMED_DECISIONS:
            row["state"] = "performed"
        elif outcome.decision == "rejected":
            row["state"] = "deferred"
        else:  # an outcome outside the vocabulary: name it, judge nothing
            row["detail"] = f"an outcome stands outside the vocabulary: {said}"
            return row
        row["detail"] = said
        return row
    if record.status == "closed_moot":
        stamp = record.closed_as_moot or {}
        row["detail"] = (
            "closed as moot — its machine was decommissioned"
            + (f" (by {stamp['by']}" if stamp.get("by") else "")
            + (f" at {stamp['at']}" if stamp.get("at") else "")
            + (")" if stamp.get("by") or stamp.get("at") else "")
            + "; the work will not be done"
        )
        return row
    if record.status == "dropped":
        stamp = record.dropped_from_board or {}
        row["detail"] = (
            "dropped from the board"
            + (f" at a shift acceptance (by {stamp['by']}" if stamp.get("by") else "")
            + (")" if stamp.get("by") else "")
            + " — the work was never done"
        )
        return row
    row["still_open"] = True
    row["detail"] = (
        "still open — decided at the gate, no outcome recorded; "
        "the work is still ahead of the floor"
    )
    return row


def _pm_review(store: Store, window: DowntimeWindow, item: WindowPlanItem) -> dict:
    """A packed obligation, judged by the cycle case that answered
    the due point the plan packed — the obligation's own record
    (its cycle list, its deferrals) read through that case."""
    row = {
        "kind": item.kind,
        "item_id": item.item_id,
        "machine_id": item.machine_id,
        "title": item.title,
        "estimate_min": item.estimate_min,
        "state": "not_worked",
        "still_open": False,
        "detail": "",
    }
    obligation: PmObligation | None = None
    for record in store.all_pm_records():
        if isinstance(record, PmObligation) and record.obligation_id == item.item_id:
            obligation = record
            break
    if obligation is None:
        row["detail"] = "the obligation is no longer on the record"
        return row
    due = _planned_due(window, item.item_id)
    cycles: list[CaseResult] = []
    for case_id in obligation.cycle_case_ids:
        case = store.get_case(case_id)
        if case is not None:
            cycles.append(case)
    answering: CaseResult | None = None
    for case in cycles:  # cycle ids are append-only, in due order
        if due and case.reported_at == due:
            answering = case
            break
    if answering is None:  # no exact due-point match: the latest cycle
        for case in cycles:  # the window's span had opened by its end
            if case.reported_at and case.reported_at <= window.ends_at:
                answering = case
    if answering is None:
        row["still_open"] = True
        row["detail"] = (
            f"no cycle case surfaced for the due point the window "
            f"packed ({due or 'the recorded due point'}) — the "
            "inspection never reached a board"
        )
        if obligation.status == "retired":
            row["detail"] += (
                f"; the obligation has since been retired "
                f"({obligation.retirement_kind or 'retired'}"
                + (f" — {obligation.retirement_reason}" if obligation.retirement_reason else "")
                + ")"
            )
        return row
    deferral = next(
        (d for d in obligation.deferrals if d.get("case_id") == answering.case_id),
        None,
    )
    if answering.outcome is not None:
        outcome = answering.outcome
        said = (
            f"cycle case {answering.case_id} — outcome {outcome.decision} "
            f"recorded by {outcome.recorded_by or 'a supervisor'}"
            + (f" at {outcome.recorded_at}" if outcome.recorded_at else "")
        )
        if outcome.decision in PERFORMED_DECISIONS:
            row["state"] = "performed"
            row["detail"] = said + "; the obligation's cycle closed performed"
        elif outcome.decision == "rejected":
            row["state"] = "deferred"
            row["detail"] = said + (
                f"; the obligation rolled its due point to "
                f"{deferral['rolled_to']}" if deferral and deferral.get("rolled_to") else ""
            )
        else:
            row["detail"] = f"an outcome stands outside the vocabulary: {said}"
        return row
    if answering.status == "dropped":
        # A dropped cycle case is a deferral: pm.py rolls the
        # obligation exactly as a rejected outcome rolls it.
        row["state"] = "deferred"
        row["detail"] = (
            f"cycle case {answering.case_id} was dropped at a board "
            "acceptance — the obligation rolled"
            + (
                f" its due point to {deferral['rolled_to']}"
                if deferral and deferral.get("rolled_to")
                else ""
            )
        )
        return row
    row["still_open"] = True
    row["detail"] = (
        f"cycle case {answering.case_id} surfaced for the due point "
        "and is still open — no outcome recorded"
    )
    return row


def build_review(
    store: Store, window: DowntimeWindow, *, now: datetime | None = None
) -> dict:
    """The window review: a confirmed window, answered for.

    Computed fresh from the records at the review's moment — an
    outcome that lands late still counts when it lands, and the
    review says the moment it was read at. Two refusals, by name:
    a proposed window (a proposal is not a promise — there is
    nothing to answer for) and a window whose span has not ended
    (the states only mean something once the span closes; naming
    work unworked while the window is still running would libel
    work in progress). The review writes nothing.
    """
    if window.status != STATUS_CONFIRMED:
        raise ValueError(
            f"window {window.window_id} is {window.status} — only a "
            "confirmed window has a review: a proposed plan is a "
            "proposal, not a promise the floor made"
        )
    moment = now or datetime.now(timezone.utc)
    end = parse_moment(window.ends_at)
    if end is not None and moment < end:
        raise ValueError(
            f"window {window.window_id} cannot be reviewed yet: its "
            f"span ends at {window.ends_at}, and the review opens "
            "when the span closes — naming work unworked while the "
            "window is still running would libel work in progress"
        )
    items = [
        _case_review(store, item)
        if item.kind == "case"
        else _pm_review(store, window, item)
        for item in window.packed
    ]
    counts = {
        "packed": len(items),
        "performed": sum(1 for i in items if i["state"] == "performed"),
        "deferred": sum(1 for i in items if i["state"] == "deferred"),
        "not_worked": sum(1 for i in items if i["state"] == "not_worked"),
    }
    minutes = {
        "planned": sum(i["estimate_min"] or 0 for i in items),
        "performed": sum(
            i["estimate_min"] or 0 for i in items if i["state"] == "performed"
        ),
        "deferred": sum(
            i["estimate_min"] or 0 for i in items if i["state"] == "deferred"
        ),
        "not_worked": sum(
            i["estimate_min"] or 0 for i in items if i["state"] == "not_worked"
        ),
    }
    return {
        "window_id": window.window_id,
        "reviewed_at": format_timestamp(moment),
        "starts_at": window.starts_at,
        "ends_at": window.ends_at,
        "basis": REVIEW_BASIS,
        "items": items,
        "counts": counts,
        "minutes": minutes,
        "performed_share": (
            round(minutes["performed"] / minutes["planned"], 4)
            if minutes["planned"]
            else None
        ),
    }


def window_ledger(store: Store, *, now: datetime | None = None) -> list[dict]:
    """The ledger of confirmed windows, in span order: what each
    promised, and — once its span has closed — what it got done.

    A row per confirmed window, never scored against another:
    different windows carry different work, and a league table
    would punish the shift that takes the hard window. The
    arithmetic on a row is that window's own — its packed count,
    its review's counts, its planned minutes performed as a share
    of its planned minutes. A window whose span has not ended has
    no review yet; its row says when the review opens instead of
    inventing a partial verdict.
    """
    moment = now or datetime.now(timezone.utc)
    rows = []
    for window in sorted(
        store.all_window_records(), key=lambda w: (w.starts_at, w.window_id)
    ):
        if window.status != STATUS_CONFIRMED:
            continue
        row = {
            "window_id": window.window_id,
            "starts_at": window.starts_at,
            "ends_at": window.ends_at,
            "length_min": window.length_min,
            "declared_by": window.declared_by,
            "confirmed_by": window.confirmed_by,
            "confirmed_at": window.confirmed_at,
            "packed": len(window.packed),
            "review_open": False,
            "review_note": (
                f"its span has not ended — the review opens at "
                f"{window.ends_at}"
            ),
            "counts": None,
            "minutes": None,
            "performed_share": None,
        }
        end = parse_moment(window.ends_at)
        if end is not None and moment >= end:
            review = build_review(store, window, now=moment)
            row["review_open"] = True
            row["review_note"] = ""
            row["counts"] = review["counts"]
            row["minutes"] = review["minutes"]
            row["performed_share"] = review["performed_share"]
        rows.append(row)
    return rows


def unfinished_windows(store: Store, *, now: datetime | None = None) -> list[dict]:
    """Ended confirmed windows whose work is still open, for the
    shift handoff: unfinished business is shift news.

    A window is named when its review shows a not-worked item
    whose work could still happen — a case still open, a cycle
    never surfaced or still open. An item closed as moot is in
    the review's counts, but it is settled history, not business
    the incoming shift can pick up, and it does not name its
    window here. Read-only, in span order.
    """
    moment = now or datetime.now(timezone.utc)
    rows = []
    for window in sorted(
        store.all_window_records(), key=lambda w: (w.starts_at, w.window_id)
    ):
        if window.status != STATUS_CONFIRMED:
            continue
        end = parse_moment(window.ends_at)
        if end is None or moment < end:
            continue
        review = build_review(store, window, now=moment)
        open_items = [
            {
                "item_id": i["item_id"],
                "kind": i["kind"],
                "machine_id": i["machine_id"],
                "detail": i["detail"],
            }
            for i in review["items"]
            if i["state"] == "not_worked" and i["still_open"]
        ]
        if not open_items:
            continue
        rows.append(
            {
                "window_id": window.window_id,
                "starts_at": window.starts_at,
                "ends_at": window.ends_at,
                "unfinished": open_items,
                "not_worked": review["counts"]["not_worked"],
            }
        )
    return rows


def render_review_text(review: dict) -> str:
    """The window review as plain text — the shift-log form."""
    counts = review["counts"]
    minutes = review["minutes"]
    share = review["performed_share"]
    lines = [
        f"WINDOW REVIEW {review['window_id']} — read at {review['reviewed_at']}",
        f"{review['starts_at']} → {review['ends_at']}",
        f"Review basis: {review['basis']}",
        (
            f"Packed {counts['packed']}: performed {counts['performed']} · "
            f"deferred {counts['deferred']} · not worked {counts['not_worked']}"
        ),
        (
            f"Planned minutes {minutes['planned']}: performed "
            f"{minutes['performed']} · deferred {minutes['deferred']} · "
            f"not worked {minutes['not_worked']}"
            + (f" · performed share {share}" if share is not None else "")
        ),
        "",
    ]
    for item in review["items"]:
        lines.append(
            f"  {item['state'].upper():<10} {item['item_id']} · "
            f"{item['machine_id']} · {item['title']} · "
            f"planned {item['estimate_min']} min — {item['detail']}"
        )
    return "\n".join(lines) + "\n"


def render_text(window: DowntimeWindow) -> str:
    """The window and its plan as plain text — the shift-log form."""
    scope = ", ".join(window.line_ids + window.machine_ids) or "—"
    lines = [
        f"DOWNTIME WINDOW {window.window_id} — {window.status.upper()}",
        (
            f"{window.starts_at} → {window.ends_at} "
            f"({window.length_min} min) · scope: {scope} · "
            f"declared by {window.declared_by} at {window.declared_at}"
        ),
        f"Plan basis: {window.plan_basis}",
        (
            f"Minutes: {window.minutes_used} used · "
            f"{window.minutes_left} left of {window.length_min}"
        ),
        "",
        "PACKED (in the order the window works them):",
    ]
    if window.packed:
        for item in window.packed:
            lines.append(
                f"  +{item.start_offset_min:>3}–{item.end_offset_min:<3} min · "
                f"{item.item_id} · {item.machine_id} · {item.title} · "
                f"est {item.estimate_min} min · priority {item.priority}"
            )
    else:
        lines.append("  (nothing packed)")
    for heading, items, empty in (
        ("DEFERRED (did not fit the minutes left):", window.deferred, "  (none)"),
        ("OVERSIZED (longer than the window itself):", window.oversized, "  (none)"),
        ("UNESTIMATED (no estimate on record — never given one):", window.unestimated, "  (none)"),
        ("BOOKED ELSEWHERE (confirmed into an overlapping window):", window.booked_elsewhere, "  (none)"),
    ):
        lines += ["", heading]
        if items:
            for item in items:
                lines.append(
                    f"  - {item.item_id} · {item.machine_id} · {item.title}"
                    + (
                        f" · est {item.estimate_min} min"
                        if item.estimate_min is not None
                        else ""
                    )
                    + (f" — {item.note}" if item.note else "")
                )
        else:
            lines.append(empty)
    for note in window.notes:
        lines.append(f"  note: {note}")
    if window.status == STATUS_CONFIRMED:
        lines.append(
            f"Confirmed by {window.confirmed_by} at {window.confirmed_at}."
        )
    return "\n".join(lines) + "\n"
