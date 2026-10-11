"""Run events: the structured trail a loop run leaves behind.

Every graph node and every loop round emits a :class:`RunEvent` to an
:class:`~plant_loop.ports.EventSink`. Sinks decide what an event is
for: the demo prints a narrated trace, tests collect, the API logs,
and a deployment ships them to its observability stack. Events are
how the loop's work is audited — rounds, gaps, exits, telemetry —
without scraping the final record.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TextIO


def format_timestamp(moment: datetime) -> str:
    """UTC timestamps in the record format used across the blueprint."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunEvent:
    type: str
    case_id: str = ""
    node: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    at: str = field(default_factory=lambda: format_timestamp(datetime.now(timezone.utc)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "case_id": self.case_id,
            "node": self.node,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "at": self.at,
        }


class NullSink:
    """Drops events (library default)."""

    def emit(self, event: RunEvent) -> None:
        pass


class CollectingSink:
    """Keeps events in memory (tests, evals, in-process inspection)."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)

    def of_type(self, type_: str) -> list[RunEvent]:
        return [e for e in self.events if e.type == type_]


class CallbackSink:
    """Forwards every event to a callable."""

    def __init__(self, callback: Callable[[RunEvent], None]) -> None:
        self._callback = callback

    def emit(self, event: RunEvent) -> None:
        self._callback(event)


class PrintingSink:
    """Human-readable trace lines (demo / CLI)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def emit(self, event: RunEvent) -> None:
        line = self._format(event)
        if line:
            print(line, file=self._stream)

    @staticmethod
    def _format(event: RunEvent) -> str:
        d = event.detail
        if event.type == "node_started":
            return ""
        if event.type == "node_finished":
            return f"    · {event.node} ({event.duration_ms} ms)"
        if event.type == "round_started":
            return f"    round {d.get('round')}: {d.get('phase', '')}"
        if event.type == "gaps_found":
            return f"    gaps: {d.get('count', 0)} open — {', '.join(d.get('kinds', [])) or 'none'}"
        if event.type == "loop_exit":
            return f"    loop exit: {d.get('exit_reason')} after {d.get('rounds')} round(s)"
        if event.type == "case_selected":
            return f"  ▸ {event.case_id} — priority {d.get('priority')} ({d.get('reason', '')})"
        if event.type == "board_finished":
            return (
                f"  board run finished: {d.get('processed', 0)} processed, "
                f"{d.get('remaining', 0)} remaining"
            )
        return ""
