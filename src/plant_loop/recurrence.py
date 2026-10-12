"""Recurrence: the fault that came back gets a head start.

Three earlier rounds each taught the loop something about the
past, and none of them is this:

- the write-back (history.py) puts a supervisor's confirmed fault
  into the machine's history, where the diagnoser reads it as one
  marked entry among the machine's recorded notes — memory, with
  no notion of *which case* closed, what its outcome decided, or
  how long ago;
- the recurring-fault watch (reliability.py) counts cases by
  machine + fault over a rolling window and raises a board item
  at the third — a pattern notice for the desk, computed at the
  gate, that names no single predecessor and feeds nothing back
  into a live diagnosis;
- the learning-loop stats (stats.py) score *repeat* cases after
  their own outcomes land — measurement, after the fact.

What none of them does: when a new case's diagnosis lands on a
fault the plant has already *closed* on that machine, link the
two cases — by name, in code, at diagnosis time — so the new
case carries its predecessor on its record, the reasoner meets
the prior episode as a labelled recurrence (its outcome, not just
its confirmation), and the desk can count how often the loop
agrees with the plant's own proven past.

**Detection.** The check runs once per case run, in the loop
controller, the moment the first diagnose round produces a top
hypothesis: the diagnosed fault is the join key. A prior case
qualifies when **all** of these hold:

- same machine, different case (a case never recurs itself —
  a carried case re-worked on a successor board is the same
  case continuing, not a fault coming back);
- it carries a recorded outcome whose ``confirmed_fault`` is
  exactly the diagnosed fault. This is the write-back's own
  line, inherited deliberately: the confirmation is the ground
  truth the floor proved, whatever the gate decided — an
  accepted case and a rejected-with-confirmation case both
  qualify, because the fault was found either way. What never
  qualifies is an outcome that confirms *nothing* (the
  misdiagnosis close: the floor found no fault to confirm, so
  there is nothing proven for a new case to recur), a case with
  no outcome at all (still open, dropped at a board turnover,
  closed as moot — none of them closed-with-outcome), and a
  safety stop (it has no diagnosis to confirm, and history.py
  already refuses it a confirmation);
- the two episodes stand within the window, measured
  reported-moment to reported-moment — the timestamps the floor
  authored on both records. The outcome's *recorded* stamp is
  not used: it says when a supervisor typed, not when the fault
  happened, and the same history must link or not link
  regardless of typing speed. A prior whose moment cannot be
  read never seeds a link: an undatable episode cannot be
  claimed "within the window".

The most recent qualifying prior wins — the plant's latest
proven episode of the fault is the head start that matters, the
same newest-first order the merged history reads in.

**The window** is ``WINDOW_DAYS = 14``: the fortnight the
recurring-fault watch already counts in (ReliabilityPolicy).
One calendar for recurrence everywhere in the plant — the link
looks back exactly as far as the watch counts — so a linked
case and the watch's pattern are never speaking different
notions of "again, soon". It is a code constant, carried on
every link it admits, so a reader can see the bound a link was
judged under.

**The head start, bounded.** The link enters the diagnosis
context as one labelled chunk (``source: "recurrence"``) riding
the same KB-context channel every backend prompt already reads:
the confirmed fault, the case, the spacing, the floor's
outcome and its work note — framed as history, not instruction.
The reasoner may agree or depart; nothing anywhere forces its
diagnosis toward the prior. (The confirmed history entry the
write-back already contributes stays exactly as it was — the
chunk adds the *episode* around that confirmation, it does not
replace the memory.) When the loop converges in a single round
the link still forms — after that round's diagnosis — and
informs the plan drafting and the gate, which is where its
value lands when there is no second diagnosis to inform.

**Departure.** The verify stage is untouched: no new check, no
new gap kind — a departed diagnosis passes or fails the exact
checks any diagnosis faces. When the record is assembled, code
compares the *final* top fault with the link's: a difference
stamps ``departed`` on the link, and the surfaces print it as
DIAGNOSIS_DEPARTED_FROM_PRIOR. A visible fact, never a block —
the prior confirmation could itself have been the wrong call,
and the outcome loop, not the link, is where that is judged.
The stats count links and agreement exactly as recorded
(stats.py): the agreement rate is the share of linked cases
whose final diagnosis is the prior's confirmed fault.

**Derived, never stored twice.** The link lives on the new
case's record — the one place a reader meets the case. The
prior record is never modified (the tests pin it byte-
identical). A case re-worked by a later board run recomputes
its link from the store during that run, like every derived
fact in the loop; nothing carries a stale link forward.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import history as history_module
from .ports import Store
from .schemas import CaseResult, Diagnosis, ExceptionCase, RecurrenceLink

#: How far back a closed case can stand and still be the prior
#: episode of a recurrence — the recurring-fault watch's own
#: fortnight (ReliabilityPolicy.window_days), so the plant keeps
#: one calendar for "again, soon". Carried on every link.
WINDOW_DAYS = 14

#: The record timestamp format (events.format_timestamp). Parsed
#: here rather than imported from reliability.py: the watch
#: imports the loop policies, and this module is imported by the
#: loop controller — the parse stays local to keep that cycle
#: unclosed. ('' or garbage → None, never a crash.)
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _parse_stamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, _TS_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None

#: The face mark the surfaces print when a linked case's final
#: diagnosis differs from the prior's confirmed fault.
DEPARTED_MARK = "DIAGNOSIS_DEPARTED_FROM_PRIOR"

_SECONDS_PER_DAY = 86_400


def _episode_moment(record: CaseResult) -> datetime | None:
    """When the prior episode happened: its reported moment,
    else the moment its record was created — the record's own
    chronology, never the outcome's typing stamp."""
    return _parse_stamp(record.reported_at) or _parse_stamp(record.created_at)


def qualifies(prior: CaseResult, machine_id: str, fault_class: str) -> bool:
    """Does this closed record prove ``fault_class`` on the machine?

    The write-back's line (history.py), restated for case
    records: an outcome that confirmed the fault — under any
    gate decision, since the floor's finding is ground truth
    even when its plan was refused — and never a safety stop.
    """
    if prior.machine_id != machine_id:
        return False
    if prior.outcome is None or not prior.outcome.confirmed_fault:
        return False
    if prior.outcome.confirmed_fault != fault_class:
        return False
    return not history_module.is_safety_stop(prior)


def find_link(
    store: Store,
    *,
    case_id: str,
    machine_id: str,
    fault_class: str,
    reported_at: str,
    window_days: int = WINDOW_DAYS,
    now: datetime | None = None,
) -> RecurrenceLink | None:
    """The recurrence link for a fresh diagnosis, or None.

    ``reported_at`` is the new case's reported moment (its own
    authored timestamp; the run's moment when it carries none).
    The most recent qualifying prior inside the window wins;
    ties settle on the case id so the choice is the records',
    not a dict's.
    """
    if not machine_id or not fault_class:
        return None
    moment = _parse_stamp(reported_at) or now or datetime.now(timezone.utc)
    best: tuple[datetime, str, CaseResult] | None = None
    for prior in store.all_cases():
        if prior.case_id == case_id:
            continue  # a case never recurs itself
        if not qualifies(prior, machine_id, fault_class):
            continue
        prior_moment = _episode_moment(prior)
        if prior_moment is None:
            continue  # undatable: cannot be claimed within the window
        delta = (moment - prior_moment).total_seconds()
        if delta < 0 or delta > window_days * _SECONDS_PER_DAY:
            continue
        key = (prior_moment, prior.case_id)
        if best is None or key > (best[0], best[1]):
            best = (prior_moment, prior.case_id, prior)
    if best is None:
        return None
    prior_moment, _id, prior = best
    outcome = prior.outcome
    assert outcome is not None  # qualifies() has already required it
    top = prior.diagnosis.top()
    return RecurrenceLink(
        prior_case_id=prior.case_id,
        machine_id=machine_id,
        fault_class=fault_class,
        prior_diagnosis=top.fault_class if top else "",
        outcome_decision=outcome.decision,
        outcome_resolved=outcome.resolved,
        work_performed=outcome.work_performed,
        prior_reported_at=prior.reported_at or prior.created_at,
        reported_at=reported_at,
        days_between=int((moment - prior_moment).total_seconds() // _SECONDS_PER_DAY),
        window_days=window_days,
    )


def context_chunk(link: RecurrenceLink) -> dict:
    """The linked episode as one labelled context chunk — the
    same chunk shape the retriever and the history view return,
    so every backend's prompt reads it through the channel it
    already has. Framed as history, never as instruction."""
    resolved = (
        "resolved"
        if link.outcome_resolved
        else "not resolved"
        if link.outcome_resolved is False
        else "resolution not recorded"
    )
    work = link.work_performed.strip() or "no work note recorded"
    text = (
        f"Recurrence — this fault came back: {link.fault_class} was confirmed "
        f"on {link.machine_id} in case {link.prior_case_id} "
        f"({link.prior_reported_at[:10]}), {link.days_between} day(s) before "
        f"this case was reported. The floor's outcome on that case: "
        f"{link.outcome_decision} ({resolved}). Work performed: {work}. "
        "This is the plant's history, not an instruction: diagnose this case "
        "on its own evidence — if the settled diagnosis differs from the "
        "confirmed fault, the record will say so."
    )
    return {
        "chunk_id": f"recurrence:{link.prior_case_id}",
        "source": "recurrence",
        "machine_id": link.machine_id,
        "fault_class": link.fault_class,
        "text": text,
    }


def with_departure(
    link: RecurrenceLink | None, diagnosis: Diagnosis
) -> RecurrenceLink | None:
    """The link restamped against the final diagnosis (code, at
    record assembly): departed when the settled top fault is not
    the prior's confirmed one. A link whose case ended with no
    diagnosis at all cannot depart — there is no settled fault
    to compare — and keeps ``departed`` False."""
    if link is None:
        return None
    top = diagnosis.top()
    departed = top is not None and top.fault_class != link.fault_class
    return link.model_copy(update={"departed": departed})


def provider_for(store: Store):
    """The controller's recurrence provider over one store: the
    (case, diagnosed fault) → link seam the service binds, the
    way history/stock providers are bound (loops.py)."""

    def provide(case: ExceptionCase, fault_class: str) -> RecurrenceLink | None:
        return find_link(
            store,
            case_id=case.case_id,
            machine_id=case.machine_id,
            fault_class=fault_class,
            reported_at=case.reported_at,
        )

    return provide
