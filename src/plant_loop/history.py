"""The plant's memory: seeded history + supervisor-confirmed write-backs.

Two kinds of maintenance history exist in this blueprint, and they
are kept apart on purpose:

- **Recorded history** — the KB's seeded entries (``data/kb.json``).
  Authored ground truth, packaged with the code. It is never
  appended to at runtime: a shipped artefact that quietly rewrites
  itself would make every diagnosis since the edit unauditable.
- **Supervisor-confirmed history** — what the floor has proven since.
  When a supervisor records an outcome with a ``confirmed_fault``
  (service.record_outcome), the confirmation is written back as a
  :class:`~plant_loop.schemas.HistoryEntry` through the record
  store — the same store that carries cases and board items
  (migration 0004), because a confirmation is runtime state, not
  documentation.

This module owns the seam between them:

- :func:`entry_for_record` builds the write-back entry for a case
  record (or refuses it — see below);
- :func:`merged_history` assembles the one history view the
  diagnoser reads: confirmed entries first (newest first), then the
  machine's recorded entries, every entry carrying its ``source``
  so the two kinds are never confused downstream;
- :func:`history_chunks` renders merged entries as retrieval-shaped
  context chunks, marked the same way, for the model backends'
  prompts.

The rules, all enforced here in code:

- **Only a confirmed fault writes back.** An accepted or
  accepted-with-changes outcome with a confirmed fault counts; an
  outcome that confirms nothing — whatever its decision — writes
  nothing. A rejection *without* a confirmed fault does not invent
  one; a rejection *with* one still writes, because the fault the
  floor found is ground truth even when the plan was refused.
- **One entry per case, ever.** The entry id derives from the case
  id and outcomes are recorded once per case; the store write is an
  upsert on that id, so a repeated write is the same fact restated,
  never a second memory.
- **Safety stops never write back.** A safety_observation case has
  no diagnosis to confirm; an outcome that tries to attach a
  confirmed fault to one is refused here with a ``ValueError``
  *before* anything is persisted, rather than the refusal being
  left to the accident that safety records carry no hypotheses.
"""

from __future__ import annotations

from . import kb
from .ports import Store
from .schemas import EXIT_SAFETY_STOP, CaseResult, HistoryEntry, SupervisorOutcome

SOURCE_RECORDED = "recorded"
SOURCE_CONFIRMED = "supervisor_confirmed"


def is_safety_stop(record: CaseResult) -> bool:
    return record.exit_reason == EXIT_SAFETY_STOP or record.type == "safety_observation"


def entry_for_record(
    record: CaseResult, outcome: SupervisorOutcome | None = None
) -> HistoryEntry | None:
    """The write-back entry a record's outcome earns (None = none).

    ``outcome`` defaults to the record's recorded outcome; the
    service passes the outcome it is *about* to record so the entry
    (and its refusal) is computed before anything mutates — a
    refused outcome then leaves no trace in either store, not just
    in the one that copies on read.

    Raises ValueError when the outcome attaches a confirmed fault
    to a safety stop — the one shape that must never enter the
    plant's memory as a diagnosis confirmation.
    """
    outcome = outcome if outcome is not None else record.outcome
    if outcome is None or not outcome.confirmed_fault:
        return None
    if is_safety_stop(record):
        raise ValueError(
            f"case {record.case_id} is a safety stop — it has no diagnosis to "
            "confirm, so no fault confirmation can be recorded for it; record "
            "the outcome without a confirmed fault"
        )
    title = record.title or f"Case {record.case_id}"
    return HistoryEntry(
        entry_id=f"hist-{record.case_id}",
        machine_id=record.machine_id,
        fault_class=outcome.confirmed_fault,
        case_id=record.case_id,
        source=SOURCE_CONFIRMED,
        date=(record.reported_at or outcome.recorded_at)[:10],
        recorded_at=outcome.recorded_at,
        summary=f"{title} — the floor confirmed {outcome.confirmed_fault}.",
        resolution=(
            outcome.work_performed.strip()
            or "Confirmed by the supervisor's recorded outcome; no work note was recorded."
        ),
    )


def record_confirmation(store: Store, record: CaseResult) -> HistoryEntry | None:
    """Write a record's confirmation into the store (idempotent).

    Returns the entry written, or None when the outcome confirms
    nothing. Raises ValueError for a safety stop carrying a
    confirmed fault — see :func:`entry_for_record`.
    """
    entry = entry_for_record(record)
    if entry is not None:
        store.save_history_entry(entry)
    return entry


def _as_context_entry(raw: dict, source: str, case_id: str = "") -> dict:
    """One history entry in the diagnoser's context shape: the KB
    entry fields, plus the source marking and the confirming case."""
    return {
        "hist_id": raw.get("hist_id") or raw.get("entry_id", ""),
        "machine_id": raw.get("machine_id", ""),
        "fault_class": raw.get("fault_class", ""),
        "date": raw.get("date", ""),
        "summary": raw.get("summary", ""),
        "resolution": raw.get("resolution", ""),
        "source": source,
        "case_id": case_id,
    }


def merged_history(
    store: Store | None, machine_id: str, fault_class: str = ""
) -> list[dict]:
    """The machine's history as the diagnoser reads it.

    Supervisor-confirmed entries first (newest confirmation first —
    the plant's latest proven truth outranks older notes), then the
    KB's recorded entries in KB order. ``store=None`` yields the
    recorded history alone (a context assembled without a store —
    direct backend use — still reads the KB's memory, never a
    silently empty one).
    """
    confirmed: list[dict] = []
    if store is not None:
        entries = [
            e
            for e in store.all_history_entries()
            if e.machine_id == machine_id and (not fault_class or e.fault_class == fault_class)
        ]
        entries.sort(key=lambda e: (e.date, e.entry_id), reverse=True)
        confirmed = [
            _as_context_entry(e.model_dump(), SOURCE_CONFIRMED, case_id=e.case_id)
            for e in entries
        ]
    recorded = [
        _as_context_entry(raw, SOURCE_RECORDED)
        for raw in kb.history_for(machine_id, fault_class)
    ]
    return confirmed + recorded


def history_chunks(entries: list[dict]) -> list[dict]:
    """Merged history entries as context chunks for the diagnose
    prompt — same chunk shape the retriever returns, with the
    history source marked in the dict and in the text itself, so a
    model (and a reader of the record) can tell a supervisor's
    confirmation from a recorded note."""
    chunks: list[dict] = []
    for entry in entries:
        if entry["source"] == SOURCE_CONFIRMED:
            text = (
                f"Supervisor-confirmed history (case {entry['case_id']}, {entry['date']}), "
                f"{entry['machine_id']}, {entry['fault_class']}: {entry['summary']} "
                f"Resolution: {entry['resolution']}"
            )
        else:
            text = (
                f"Maintenance history {entry['hist_id']} ({entry['date']}), "
                f"{entry['machine_id']}, {entry['fault_class']}: {entry['summary']} "
                f"Resolution: {entry['resolution']}"
            )
        chunks.append(
            {
                "chunk_id": f"history:{entry['hist_id']}",
                "source": "history",
                "history_source": entry["source"],
                "machine_id": entry["machine_id"],
                "fault_class": entry["fault_class"],
                "text": text,
            }
        )
    return chunks
