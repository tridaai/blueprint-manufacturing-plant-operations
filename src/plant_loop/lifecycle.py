"""Machine lifecycle: machines end, and the record says so honestly.

The plant model (kb.json) is authored data — it names the
machines, and it never changes at runtime. But plants scrap
machines, and until this layer existed the system had no way to
say so: a standing PM obligation for a scrapped machine kept
surfacing due cases, and every board trained its supervisors to
ignore it a little more. This module is the machine's end, as a
supervisor's act on the record:

- **Decommission.** A supervisor marks a machine decommissioned —
  actor and reason required, effective from a stated moment in
  the record-timestamp idiom. The act is computed, not
  declarative: its standing PM obligations retire automatically
  (pm.py), each carrying the fixed reason "machine
  decommissioned" and this record's id — an automatic retirement,
  distinguishable from a supervisor's manual one forever. And its
  open board cases are *listed* on the record, nothing more:
  each waits for the supervisor's own disposition.
- **Disposition, case by case.** The supervisor chooses, per
  open case: **keep** (the case stays on the board and the loop
  keeps working it — a case about a scrapped machine can still
  carry a safety or quality lesson) or **close as moot** (the
  case leaves the board, stamped with the choice; its standing
  part claims release, like a gate rejection's). A disposition is
  recorded once per case. Nothing is ever disposed automatically
  — a safety case on the machine escalates exactly as before and
  can only be dispositioned by this explicit per-case choice.
- **Intake closes.** From the act onward, new cases naming the
  machine are refused where cases enter (the service's run
  paths), with the state named. Cases already on the board —
  kept dispositions, carried cases — are not new intake and are
  never refused.

Safety history is untouched by all of it: the machine's safety
records, its history entries, and every case record stay fully
readable forever. Decommissioning removes a machine from the
future; it rewrites nothing about its past.

Storage follows the house precedent (migration 0007): the
machine record lives in its own table, because later requests —
the intake guard, a board run, the handoff — must find the state
where it lives, not reconstruct it from other records' payloads.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from . import kb
from .events import format_timestamp
from .pm import PmManager
from .ports import Store
from .reliability import parse_timestamp
from .schemas import CaseResult, MachineRecord

# The fixed reason an automatic retirement carries. It is a
# statement of mechanism, not of judgement: the supervisor's own
# reason for ending the machine lives on the decommission record,
# and a manual retirement carries the supervisor's own words.
DECOMMISSION_RETIREMENT_REASON = "machine decommissioned"

# The dispositions a supervisor may choose for an open case.
DISPOSITION_KEEP = "keep"
DISPOSITION_CLOSE_MOOT = "close_moot"
DISPOSITIONS = (DISPOSITION_KEEP, DISPOSITION_CLOSE_MOOT)


def machine_state(store: Store, machine_id: str) -> str:
    """The machine's lifecycle state, as the records say it.

    "decommissioned" when a machine record stands; "in_service"
    when the plant model knows the machine and no record does;
    "unknown" when the plant model has never heard of it.
    """
    if store.get_machine_record(machine_id) is not None:
        return "decommissioned"
    return "in_service" if machine_id in kb.machines() else "unknown"


def open_cases(store: Store, machine_id: str) -> list[CaseResult]:
    """The machine's open board cases, in case-id order.

    Open is the board's own unfinished business: a case waiting
    at the gate (awaiting_supervisor — gate cases, blocked cases,
    and safety stops alike), or a case the supervisor has decided
    whose outcome is still outstanding. A case with its outcome
    recorded is settled history; a dropped or moot-closed case has
    already been disposed. Neither needs a disposition.
    """
    records = [
        r
        for r in store.all_cases()
        if r.machine_id == machine_id
        and (
            r.status == "awaiting_supervisor"
            or (r.status in ("approved", "rejected") and r.outcome is None)
        )
    ]
    return sorted(records, key=lambda r: r.case_id)


class MachineLifecycle:
    """Decommissions and dispositions — over the store + PM layer."""

    def __init__(self, store: Store, pm_manager: PmManager) -> None:
        self.store = store
        self.pm = pm_manager

    def record_for(self, machine_id: str) -> MachineRecord | None:
        return self.store.get_machine_record(machine_id)

    def decommission(
        self,
        machine_id: str,
        *,
        by: str = "",
        reason: str = "",
        effective_at: str = "",
        now: datetime | None = None,
    ) -> MachineRecord:
        """Record a machine's decommission and set its consequences.

        Validates the act before anything saves: the machine must
        be one the plant model knows, actor and reason are both
        required (the record must say whose act this was and
        why), a stated effective moment must parse in the
        record-timestamp idiom, and a machine is decommissioned
        once — a second act is refused, not merged.

        The consequences are computed in code, in this order: the
        machine's standing PM obligations retire automatically
        (each linked to this record), and the machine's open
        cases are snapshotted onto the record for disposition.
        No case is closed, dropped, or decided by this act — the
        list is the supervisor's to work, case by case.
        """
        if machine_id not in kb.machines():
            raise KeyError(f"unknown machine: {machine_id}")
        if self.store.get_machine_record(machine_id) is not None:
            raise ValueError(
                f"machine {machine_id} is already decommissioned — "
                "the record of its end already stands"
            )
        if not by.strip():
            raise ValueError(
                "decommissioning a machine needs the actor — the record "
                "must say whose act ended it"
            )
        if not reason.strip():
            raise ValueError(
                "decommissioning a machine needs the reason — the record "
                "must say why the plant is retiring it"
            )
        moment = now or datetime.now(timezone.utc)
        stamp = format_timestamp(moment)
        effective = stamp
        if effective_at:
            parsed = parse_timestamp(effective_at)
            if parsed is None:
                raise ValueError(
                    f"effective_at is not a record timestamp: {effective_at!r}"
                )
            effective = format_timestamp(parsed)
        record = MachineRecord(
            record_id=f"DEC-{uuid.uuid4().hex[:8]}",
            machine_id=machine_id,
            status="decommissioned",
            reason=reason,
            decided_by=by,
            decided_at=stamp,
            effective_at=effective,
            created_at=stamp,
        )
        # Automatic retirements first, so the record names exactly
        # the obligations the act ended. Already-retired
        # obligations are untouched: their own endings stand.
        for obligation in sorted(
            self.pm.obligations(), key=lambda o: o.obligation_id
        ):
            if obligation.machine_id != machine_id or obligation.status != "standing":
                continue
            self.pm.retire_obligation(
                obligation.obligation_id,
                by=by,
                reason=DECOMMISSION_RETIREMENT_REASON,
                kind="machine_decommission",
                decommission_id=record.record_id,
                now=moment,
            )
            record.retired_obligation_ids.append(obligation.obligation_id)
        record.open_case_ids = [r.case_id for r in open_cases(self.store, machine_id)]
        self.store.save_machine_record(record)
        return record

    def dispose(
        self,
        machine_id: str,
        case_id: str,
        disposition: str,
        *,
        by: str = "",
        now: datetime | None = None,
    ) -> MachineRecord:
        """Record the supervisor's disposition of one open case.

        The case must be on the decommission record's open list,
        not yet dispositioned, and still open — a case that has
        settled since the act (decided, outcome recorded, dropped
        at a turnover) no longer needs a disposition, and choosing
        one for it is refused with its current state named.

        "keep" changes nothing about the case: the disposition is
        the record of the choice, and the loop keeps working it.
        "close_moot" closes the case on its own record (status
        closed_moot + the stamp naming the decommission) — the
        service releases its standing part claims, the way a gate
        rejection's claims release. Either way the disposition is
        recorded once: this record is the audit trail.
        """
        record = self.store.get_machine_record(machine_id)
        if record is None:
            raise KeyError(f"machine {machine_id} is not decommissioned")
        if disposition not in DISPOSITIONS:
            raise ValueError(
                f"unknown disposition: {disposition!r} (keep | close_moot)"
            )
        if not by.strip():
            raise ValueError(
                "a disposition needs the actor — the record must say "
                "whose choice this was"
            )
        if case_id not in record.open_case_ids:
            raise ValueError(
                f"case {case_id} is not on {machine_id}'s decommission "
                "record — it was not open when the machine was "
                "decommissioned (or it belongs to another machine)"
            )
        if any(d["case_id"] == case_id for d in record.dispositions):
            raise ValueError(
                f"case {case_id} already has a recorded disposition — "
                "a disposition is recorded once"
            )
        case = self.store.get_case(case_id)
        if case is None:  # the snapshot named it; the store lost it
            raise KeyError(f"unknown case: {case_id}")
        still_open = case.status == "awaiting_supervisor" or (
            case.status in ("approved", "rejected") and case.outcome is None
        )
        if not still_open:
            raise ValueError(
                f"case {case_id} is no longer open (status: {case.status}"
                + (", outcome recorded" if case.outcome is not None else "")
                + ") — it settled on its own record and needs no disposition"
            )
        moment = now or datetime.now(timezone.utc)
        stamp = format_timestamp(moment)
        if disposition == DISPOSITION_CLOSE_MOOT:
            case.status = "closed_moot"
            case.closed_as_moot = {
                "decommission_id": record.record_id,
                "machine_id": machine_id,
                "by": by,
                "at": stamp,
            }
            self.store.save_case(case)
        record.dispositions.append(
            {"case_id": case_id, "disposition": disposition, "by": by, "at": stamp}
        )
        self.store.save_machine_record(record)
        return record
