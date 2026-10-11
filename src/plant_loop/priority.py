"""Board priority, computed in code.

The board loop's first question — *what do we work next?* — is
answered here, deterministically, never by a model:

    score = type_base × line_criticality_factor + downtime_term

- **Type base** encodes the plant's triage doctrine: a safety
  observation outranks everything (and will stop at round 0 anyway);
  a stopped line outranks a quality escape; a shortage outranks a
  maintenance alert.
- **Line criticality** (1–3, from the KB) scales the base: the same
  fault on the press line matters more than on utilities.
- **Downtime** adds pressure to machine_down cases only — minutes
  already lost, capped so an old case cannot outrank a safety case.

Ties break by reported time, then case id, so a board run is
reproducible to the case.
"""

from __future__ import annotations

from . import kb
from .schemas import ExceptionCase

TYPE_BASE = {
    "safety_observation": 1000.0,
    "machine_down": 600.0,
    "quality_nonconformance": 400.0,
    "material_shortage": 250.0,
    "maintenance_alert": 150.0,
}

_DOWNTIME_CAP_MIN = 240


def priority_for(case: ExceptionCase) -> tuple[float, str]:
    """(score, human-readable reason) for one case."""
    base = TYPE_BASE.get(case.type, 100.0)
    criticality = kb.line_criticality(case.line_id)
    factor = 1 + 0.25 * (criticality - 1)
    score = base * factor
    reasons = [f"{case.type} base {base:.0f} × line criticality {criticality} (×{factor:.2f})"]
    if case.type == "machine_down" and case.downtime_minutes:
        added = min(case.downtime_minutes, _DOWNTIME_CAP_MIN) * 0.5
        score += added
        reasons.append(f"+ {added:.0f} for {case.downtime_minutes} min already down")
    return round(score, 2), "; ".join(reasons)


def board_order(cases: list[ExceptionCase]) -> list[ExceptionCase]:
    """Cases in the order the board loop works them."""
    return sorted(
        cases,
        key=lambda c: (-priority_for(c)[0], c.reported_at or "", c.case_id),
    )
