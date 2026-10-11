"""The plant knowledge base: SOPs, maintenance history, parts inventory.

A synthetic, self-contained KB (``data/kb.json``) for the fictional
Meridian Fabrication, Plant 2. It is the ground the loops reason
over:

- **SOPs** per machine / fault class carry the *mandatory safety
  steps* the plan may never omit, the diagnostic checks that separate
  one fault class from another, the corrective steps, the required
  parts, and any approved substitutes.
- **Maintenance history** gives diagnosis its precedents ("this exact
  signature was a bearing in August").
- **Parts inventory** is checked in code: a plan that needs a part
  the shelf does not have (and no substitute) cannot converge — the
  case is blocked on the outside world, not on the model.

Retrieval over SOP/history chunks sits behind the house
:class:`~plant_loop.ports.Retriever` seam (embeddings + a vector
index); the loop also reads the KB directly for the code-side facts
(safety steps, stock counts) that must never depend on retrieval
luck.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def load_kb() -> dict:
    """Load the packaged knowledge base (cached per process)."""
    text = resources.files(__package__).joinpath("data/kb.json").read_text(encoding="utf-8")
    return json.loads(text)


def plant_name() -> str:
    return load_kb()["plant"]["name"]


def lines() -> dict[str, dict]:
    return {line["line_id"]: line for line in load_kb()["lines"]}


def line_criticality(line_id: str) -> int:
    """Line criticality 1–3 (1 when the line is unknown)."""
    line = lines().get(line_id or "")
    return int(line["criticality"]) if line else 1


def machines() -> dict[str, dict]:
    return {m["machine_id"]: m for m in load_kb()["machines"]}


def sops() -> list[dict]:
    return load_kb()["sops"]


def sop_by_id(sop_id: str) -> dict | None:
    for sop in sops():
        if sop["sop_id"] == sop_id:
            return sop
    return None


def sop_for(machine_id: str, fault_class: str) -> dict | None:
    """The SOP for one machine + fault class (the shortage SOP is plant-wide)."""
    for sop in sops():
        if sop["fault_class"] == fault_class and sop["machine_id"] in (machine_id, ""):
            return sop
    return None


def sops_for_machine(machine_id: str) -> list[dict]:
    return [s for s in sops() if s["machine_id"] in (machine_id, "")]


def mandatory_safety_steps(sop: dict | None) -> list[dict]:
    """The code-side safety list for an SOP — the verify round's law."""
    if not sop:
        return []
    return list(sop.get("mandatory_safety_steps", []))


def inventory() -> dict[str, dict]:
    return {p["part_id"]: p for p in load_kb()["parts"]}


def part_qty(part_id: str) -> int | None:
    """On-hand qty, or ``None`` when the part is not stocked at all."""
    part = inventory().get(part_id)
    return int(part["qty_on_hand"]) if part else None


def substitute_for(sop: dict | None, part_id: str) -> str:
    """The first KB-listed substitute for a part ('' when none listed)."""
    entries = substitute_entries(sop, part_id)
    return entries[0]["part_id"] if entries else ""


def substitute_entries(sop: dict | None, part_id: str) -> list[dict]:
    """The compatibility relation for one part slot, in search order.

    Each entry names a substitute the KB lists as compatible with the
    slot, the *basis* for that compatibility (fitting, service,
    approval), and the spec deltas that matter — the differences a
    supervisor should see before approving a plan that runs on the
    substitute. Listing is not approval to use: the verify stage
    still checks each candidate against the shelf and the SOP's
    constraints for the slot (see substitution.py).
    """
    if not sop:
        return []
    raw = sop.get("substitutes", {}).get(part_id, [])
    entries: list[dict] = []
    for entry in raw:
        if isinstance(entry, str):  # a bare id carries no basis/deltas
            entries.append({"part_id": entry, "basis": "", "spec_deltas": []})
        else:
            entries.append(
                {
                    "part_id": str(entry.get("part_id", "")),
                    "basis": str(entry.get("basis", "")),
                    "spec_deltas": list(entry.get("spec_deltas", [])),
                }
            )
    return [e for e in entries if e["part_id"]]


def part_specs(part_id: str) -> dict:
    """The part's spec sheet ({} when the register carries none)."""
    part = inventory().get(part_id)
    return dict(part.get("specs", {})) if part else {}


def part_constraints(sop: dict | None, part_id: str) -> list[dict]:
    """The SOP's constraints on whatever fills this part slot.

    Each constraint is ``{attribute, op, value, reason}`` over the
    candidate part's spec sheet — e.g. a fan slot that requires
    ``airflow_m3h >= 420``. A substitute that violates one is
    rejected at the verify stage, however compatible the KB calls
    it: the SOP's physics outranks the catalogue's optimism.
    """
    if not sop:
        return []
    return list(sop.get("part_constraints", {}).get(part_id, []))


def fault_classes() -> list[str]:
    """The fault vocabulary: every fault class the plant has an SOP
    or a history entry for. Diagnoses speak it, and a supervisor's
    confirmed fault must be one of these — a confirmation outside the
    vocabulary is a data error, not a new fact about the plant."""
    classes = {sop["fault_class"] for sop in sops()}
    classes |= {entry["fault_class"] for entry in load_kb()["history"]}
    return sorted(classes)


def history_for(machine_id: str, fault_class: str = "") -> list[dict]:
    entries = [h for h in load_kb()["history"] if h["machine_id"] == machine_id]
    if fault_class:
        entries = [h for h in entries if h["fault_class"] == fault_class]
    return entries


# ---------------------------------------------------------------------------
# Retrieval chunks
# ---------------------------------------------------------------------------


def kb_chunks() -> list[dict]:
    """Retrievable text chunks: one per SOP, one per history entry."""
    chunks: list[dict] = []
    for sop in sops():
        checks = " ".join(sop.get("diagnostic_checks", []))
        steps = " ".join(s["text"] for s in sop.get("corrective_steps", []))
        safety = " ".join(s["text"] for s in sop.get("mandatory_safety_steps", []))
        chunks.append(
            {
                "chunk_id": f"sop:{sop['sop_id']}",
                "source": "sop",
                "machine_id": sop["machine_id"],
                "fault_class": sop["fault_class"],
                "text": (
                    f"{sop['title']} ({sop['sop_id']}). Applies to {sop['machine_id'] or 'any machine'}. "
                    f"Diagnostic checks: {checks} Mandatory safety steps (must appear in every plan): {safety} "
                    f"Corrective outline: {steps}"
                ),
            }
        )
    for entry in load_kb()["history"]:
        chunks.append(
            {
                "chunk_id": f"history:{entry['hist_id']}",
                "source": "history",
                # The KB's own history is *recorded* history; the
                # supervisor-confirmed write-backs (history.py) carry
                # the other marking, so a retrieved chunk and an
                # appended one read the same way.
                "history_source": "recorded",
                "machine_id": entry["machine_id"],
                "fault_class": entry["fault_class"],
                "text": (
                    f"Maintenance history {entry['hist_id']} ({entry['date']}), "
                    f"{entry['machine_id']}, {entry['fault_class']}: {entry['summary']} "
                    f"Resolution: {entry['resolution']}"
                ),
            }
        )
    return chunks


class KBRetriever:
    """Retrieval over the KB chunks, behind the Retriever port.

    Chunks are embedded once at construction; a query returns the
    nearest chunks (SOPs and history mixed — the reasoner sees both
    the procedure and the precedent). The index is the house vector
    index: pgvector in production, in-memory as the test double.
    """

    def __init__(self, embedder, index) -> None:
        self._embedder = embedder
        self._index = index
        self._chunks = {c["chunk_id"]: c for c in kb_chunks()}
        for chunk in self._chunks.values():
            index.add(chunk["chunk_id"], embedder.embed(chunk["text"]), embedder.model)

    def retrieve(self, query: str, *, limit: int = 5) -> list[dict]:
        vector = self._embedder.embed(query)
        hits = self._index.nearest(vector, limit=limit)
        return [
            {**self._chunks[chunk_id], "similarity": round(float(score), 4)}
            for chunk_id, score in hits
            if chunk_id in self._chunks
        ]
