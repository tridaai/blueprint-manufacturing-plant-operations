"""The recurring-fault watch: same machine, same fault, again.

Individual cases get fixed at the gate. Patterns do not fix
themselves: three bearing failures on one spindle inside a fortnight
is not three maintenance jobs, it is one reliability problem. This
watch is the code that notices.

:meth:`ReliabilityWatch.check` runs whenever a case reaches the
gate or a supervisor records an outcome. It counts the cases for
the triggering case's machine + fault class inside the rolling
window (the supervisor's confirmed fault when an outcome exists,
the loop's top diagnosis otherwise) and, at the policy threshold,
raises one ``reliability_review`` board item carrying the
contributing cases as its evidence.

The item is the whole action. It lands on the board at the
supervisor gate like everything else; it triggers no work order, no
notification, no automatic anything. One open item per machine +
fault class at a time — a pattern already on the board does not
get raised twice.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from .events import format_timestamp
from .loops import ReliabilityPolicy
from .ports import Store
from .schemas import BoardItem, CaseResult

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_timestamp(value: str) -> datetime | None:
    """Parse a record timestamp ('' or garbage → None, never a crash)."""
    if not value:
        return None
    try:
        return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def effective_fault_class(record: CaseResult) -> str:
    """The fault a case counts as: the confirmed fault when a
    supervisor recorded one, else the loop's top diagnosis."""
    if record.outcome is not None and record.outcome.confirmed_fault:
        return record.outcome.confirmed_fault
    top = record.diagnosis.top()
    return top.fault_class if top else ""


def case_moment(record: CaseResult) -> datetime | None:
    """When the case entered the record (created, else reported)."""
    return parse_timestamp(record.created_at) or parse_timestamp(record.reported_at)


class ReliabilityWatch:
    """Counts recurrences and raises reliability_review board items."""

    def __init__(self, store: Store, policy: ReliabilityPolicy | None = None) -> None:
        self.store = store
        self.policy = policy or ReliabilityPolicy()

    def check(
        self, trigger: CaseResult, *, now: datetime | None = None
    ) -> BoardItem | None:
        """Raise (and return) a reliability item if the trigger's
        machine + fault class just crossed the threshold; else None."""
        machine_id = trigger.machine_id
        fault_class = effective_fault_class(trigger)
        if not machine_id or not fault_class:
            return None
        moment = now or datetime.now(timezone.utc)
        window_start = moment - timedelta(days=self.policy.window_days)
        contributing: list[CaseResult] = []
        for record in self.store.all_cases():
            if record.machine_id != machine_id:
                continue
            if effective_fault_class(record) != fault_class:
                continue
            at = case_moment(record)
            if at is None or not (window_start <= at <= moment):
                continue
            contributing.append(record)
        if len(contributing) < self.policy.threshold:
            return None
        for item in self.store.all_board_items():
            if (
                item.kind == "reliability_review"
                and item.status == "open"
                and item.machine_id == machine_id
                and item.fault_class == fault_class
            ):
                return None  # already on the board — never raised twice
        contributing.sort(key=lambda r: (case_moment(r) or moment, r.case_id))
        item = BoardItem(
            item_id=f"rel-{uuid.uuid4().hex[:8]}",
            kind="reliability_review",
            status="open",
            machine_id=machine_id,
            fault_class=fault_class,
            case_ids=[r.case_id for r in contributing],
            count=len(contributing),
            window_days=self.policy.window_days,
            threshold=self.policy.threshold,
            title=(
                f"Recurring fault: {fault_class} on {machine_id} — "
                f"{len(contributing)} cases in {self.policy.window_days} days"
            ),
            detail=(
                f"{len(contributing)} cases of {fault_class} on {machine_id} reached "
                f"the gate within {self.policy.window_days} days (threshold "
                f"{self.policy.threshold}). Repeatedly repairing the same fault is a "
                "reliability problem, not a run of bad luck — this item asks a "
                "supervisor to review the pattern. It triggers no automatic action."
            ),
            created_at=format_timestamp(moment),
        )
        self.store.save_board_item(item)
        return item
