"""The parts outlook: the board, looking ahead at its own shelf.

Every round before this one measured parts in the present tense.
The allocation ledger (round 5) reads the shelf net of the claims
standing *now*; the verify stage blocks a case when the part it
needs is not there *now*. But the information to see a shortage
coming is already on the record — it is just scattered across
three rounds' worth of state:

- **standing claims** (round 5): quantities converged cases have
  committed to draw, standing until the gate decides;
- **open cases' verified plans**: requirements the verify stage
  measured and could not satisfy — a blocked case is a dated,
  quantified statement of future demand, and today nothing adds
  those statements up;
- **standing PM obligations** (round 8): each carries a next due
  point and a machine + fault class whose SOP names the parts
  the cycle will draw. A due date inside the horizon is demand
  with a date on it — the only genuinely *future* source, and
  the one that can tip a covered part into shortage while every
  present-tense check still passes.

This module is the projection that adds them up, in code, over a
stated horizon. Per part it walks the demand in the order it
lands — everything already on the books (claims, open-case
needs, overdue cycles) is effective at the projection moment;
future cycles bite on their due dates — and names the first
moment cumulative demand passes the shelf: the **bites-at**
date, and what tipped it over. A part whose projected demand
exactly equals the shelf is covered, not at risk: the boundary
is arithmetic, and it is pinned.

Discipline, in the house pattern:

- **Traced, never estimated.** Every projected unit is a line
  naming its source — the claiming case, the case whose plan
  needs it, the obligation whose cycle falls due. A line a
  supervisor cannot follow back to a record does not belong in
  the projection.
- **One count per unit.** A converged case's satisfied
  requirements *are* its claims, so plan demand counts only
  what no standing claim covers. A PM obligation with a cycle
  already on a board contributes through that case's own
  record, never twice. A retired obligation contributes
  nothing: its duty ended on the record (round 10), and a
  projection that kept spending its due cycles would be
  planning around a ghost.
- **Unquantified stays unquantified.** An obligation whose
  machine + fault class has no SOP on record names no parts to
  project; the outlook lists it as unquantified rather than
  inventing quantities. (An SOP that names no parts is a
  different fact: it was consulted, and it names none.)
- **A read model, purely.** Building the outlook writes
  nothing — no cases, no claims, no orders. It advises; humans
  procure. Procurement itself (suppliers, lead times, purchase
  orders) is outside this blueprint's plant, and the outlook
  does not pretend otherwise.

The horizon is a reporting window with an authored default
(:data:`DEFAULT_HORIZON_DAYS`), stated on every report — a
caller (the API, the CLI) may ask for a longer or shorter one,
and the report prints the window it used.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import kb
from .events import format_timestamp
from .ports import Store
from .reliability import parse_timestamp
from .schemas import PartDemandLine, PartOutlook, PartsOutlook, PmObligation

#: The authored reporting window: two weeks of foresight — long
#: enough to cover a PM interval's span and a stores order cycle,
#: short enough that every line in it is a record, not a hope.
DEFAULT_HORIZON_DAYS = 14

#: A case is *open* for demand purposes while it waits at the
#: gate or stands approved awaiting its outcome — its plan will
#: still draw parts. Rejected, dropped and closed-moot cases are
#: finished business: their claims are released and their plans
#: will never draw, so their requirements are not demand.
OPEN_STATUSES = ("awaiting_supervisor", "approved")

_KIND_ORDER = {"claim": 0, "case_plan": 1, "pm_cycle": 2}

BASIS = (
    "Demand is traced, never estimated: standing part claims, the "
    "unsatisfied requirements on open cases' verified plans, and the "
    "next due cycle of each standing PM obligation whose SOP names "
    "its parts, falling due inside the horizon. Supply is the plant "
    "model's authored on-hand quantity (data/kb.json). The outlook "
    "is a read model — it creates no cases, places no claims and "
    "orders nothing; procurement stays a human's job."
)


def _bites(lines: list[PartDemandLine], supply: int, moment: datetime, as_of: str) -> tuple[str, str]:
    """The first moment cumulative demand passes the shelf.

    Everything already on the books — claims, open-case needs and
    PM cycles already overdue — is effective at the projection
    moment; future cycles bite on their own due dates. Returns
    the (timestamp, note) of the tipping event, or ("", "") when
    the shelf holds — the caller only asks when it does not.
    """
    on_the_books = sum(
        line.qty for line in lines if line.source_kind in ("claim", "case_plan")
    )
    future: list[tuple[datetime, PartDemandLine]] = []
    for line in lines:
        if line.source_kind != "pm_cycle":
            continue
        due = parse_timestamp(line.date)
        if due is None or due <= moment:
            on_the_books += line.qty  # overdue: it bites now, with the rest
        else:
            future.append((due, line))
    if on_the_books > supply:
        return as_of, "already short on the current book"
    cumulative = on_the_books
    for _due, line in sorted(future, key=lambda e: (e[0], e[1].source_id)):
        cumulative += line.qty
        if cumulative > supply:
            return line.date, (
                f"tipped by the PM cycle of {line.source_id} (due {line.date})"
            )
    return "", ""


def build_outlook(
    store: Store,
    *,
    now: datetime | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> PartsOutlook:
    """Project part demand against the shelf over the horizon.

    Pure over the store and the plant model: the same records
    project the same outlook, byte for byte, and building it
    changes no record — the pin the tests hold it to.
    """
    if horizon_days < 1:
        raise ValueError("the outlook horizon must be at least 1 day")
    moment = now or datetime.now(timezone.utc)
    as_of = format_timestamp(moment)
    horizon_end_dt = moment + timedelta(days=horizon_days)

    lines_by_part: dict[str, list[PartDemandLine]] = {}

    def add(line: PartDemandLine) -> None:
        lines_by_part.setdefault(line.part_id, []).append(line)

    # -- committed demand: the standing claims (round 5) --------------
    claims = [c for c in store.all_part_claims() if c.status == "claimed"]
    claimed_by_case: dict[str, dict[str, int]] = {}
    for claim in claims:
        per_part = claimed_by_case.setdefault(claim.case_id, {})
        per_part[claim.part_id] = per_part.get(claim.part_id, 0) + claim.qty
        add(
            PartDemandLine(
                part_id=claim.part_id,
                qty=claim.qty,
                source_kind="claim",
                source_id=claim.case_id,
                date=claim.claimed_at,
                detail=(
                    f"claimed by case {claim.case_id} (board run {claim.run_id})"
                    + (
                        f" — drawn in place of {claim.original_part_id}"
                        if claim.original_part_id
                        else ""
                    )
                ),
            )
        )

    # -- open cases' verified plans: what no claim covers -------------
    for record in store.all_cases():
        if record.status not in OPEN_STATUSES:
            continue
        covered = claimed_by_case.get(record.case_id, {})
        for req in record.plan.required_parts:
            drawn = req.substitute_part_id or req.part_id
            uncovered = req.qty - covered.get(drawn, 0)
            if uncovered <= 0:
                continue  # a standing claim already carries this unit
            # An unsatisfied requirement is demand on the part it
            # names (the verify stage's substitute search already
            # failed against it); a satisfied one with no claim
            # behind it — a plan that verified but never
            # converged — will draw the part it runs on.
            part = drawn if req.satisfied else req.part_id
            add(
                PartDemandLine(
                    part_id=part,
                    qty=uncovered,
                    source_kind="case_plan",
                    source_id=record.case_id,
                    date=record.created_at or record.reported_at,
                    detail=(
                        f"case {record.case_id} ({record.machine_id} "
                        f"{_top_fault(record)}) plans to draw {uncovered} — "
                        + (
                            "verified against the shelf, not yet claimed"
                            if req.satisfied
                            else "the verify stage could not satisfy it"
                        )
                    ),
                )
            )

    # -- PM cycles falling due inside the horizon (round 8) -----------
    unquantified: list[dict] = []
    obligations = [r for r in store.all_pm_records() if isinstance(r, PmObligation)]
    for obligation in sorted(obligations, key=lambda o: o.obligation_id):
        if obligation.status != "standing":
            continue  # retired: the duty ended on the record (round 10)
        if obligation.cycle_case_id:
            # A cycle is already on a board: its demand rides on
            # that case's own record (its claim, or its plan's
            # unsatisfied requirements) — counting the obligation
            # too would spend the same unit twice.
            continue
        due = parse_timestamp(obligation.next_due_at)
        if due is None or due > horizon_end_dt:
            continue
        sop = kb.sop_for(obligation.machine_id, obligation.fault_class)
        if sop is None:
            unquantified.append(
                {
                    "obligation_id": obligation.obligation_id,
                    "machine_id": obligation.machine_id,
                    "fault_class": obligation.fault_class,
                    "next_due_at": obligation.next_due_at,
                    "reason": (
                        "no SOP on record for this machine + fault class — "
                        "the cycle's parts cannot be projected"
                    ),
                }
            )
            continue
        for required in sop.get("required_parts", []):
            part_id = str(required.get("part_id", ""))
            qty = int(required.get("qty", 0))
            if not part_id or qty <= 0:
                continue
            add(
                PartDemandLine(
                    part_id=part_id,
                    qty=qty,
                    source_kind="pm_cycle",
                    source_id=obligation.obligation_id,
                    date=obligation.next_due_at,
                    detail=(
                        f"PM cycle of {obligation.obligation_id} due "
                        f"{obligation.next_due_at} — {obligation.machine_id} "
                        f"{obligation.fault_class} inspection under "
                        f"{sop['sop_id']}"
                    ),
                )
            )

    # -- assemble -------------------------------------------------------
    parts: list[PartOutlook] = []
    for part_id in sorted(lines_by_part):
        lines = sorted(
            lines_by_part[part_id],
            key=lambda l: (_KIND_ORDER.get(l.source_kind, 9), l.date, l.source_id),
        )
        on_hand = kb.part_qty(part_id)
        supply = on_hand if on_hand is not None else 0
        claimed = sum(l.qty for l in lines if l.source_kind == "claim")
        case_demand = sum(l.qty for l in lines if l.source_kind == "case_plan")
        pm_demand = sum(l.qty for l in lines if l.source_kind == "pm_cycle")
        projected = claimed + case_demand + pm_demand
        shortfall = max(0, projected - supply)
        risk = shortfall > 0
        bites_at, bites_note = (
            _bites(lines, supply, moment, as_of) if risk else ("", "")
        )
        parts.append(
            PartOutlook(
                part_id=part_id,
                part_name=kb.inventory().get(part_id, {}).get("name", ""),
                stocked=on_hand is not None,
                on_hand=on_hand,
                claimed_qty=claimed,
                case_demand_qty=case_demand,
                pm_demand_qty=pm_demand,
                projected_demand=projected,
                shortfall=shortfall,
                shortage_risk=risk,
                bites_at=bites_at,
                bites_note=bites_note,
                lines=lines,
            )
        )
    shortages = sorted(
        (p for p in parts if p.shortage_risk), key=lambda p: (p.bites_at, p.part_id)
    )
    return PartsOutlook(
        as_of=as_of,
        horizon_days=horizon_days,
        horizon_end=format_timestamp(horizon_end_dt),
        basis=BASIS,
        parts=parts,
        shortages=shortages,
        unquantified_pm=unquantified,
    )


def _top_fault(record) -> str:
    top = record.diagnosis.top()
    return top.fault_class if top else "undiagnosed"


def _part_line(part: PartOutlook) -> str:
    shelf = (
        f"{part.on_hand} on hand"
        if part.stocked
        else "not stocked in the plant model"
    )
    return (
        f"  - {part.part_id} ({part.part_name or '—'}): projected "
        f"{part.projected_demand} against {shelf} · claimed "
        f"{part.claimed_qty} · open-case plans {part.case_demand_qty} · "
        f"PM cycles {part.pm_demand_qty}"
    )


def render_text(outlook: PartsOutlook, *, shortages_only: bool = False) -> str:
    """The outlook as plain text — the CLI's and the shift log's form."""
    lines = [
        f"PARTS OUTLOOK — as of {outlook.as_of}, horizon "
        f"{outlook.horizon_days} days (to {outlook.horizon_end})",
        outlook.basis,
        "",
        "SHORTAGE RISKS:",
    ]
    if outlook.shortages:
        for part in outlook.shortages:
            lines.append(
                f"  - {part.part_id} ({part.part_name or '—'}): SHORT "
                f"{part.shortfall} — projected {part.projected_demand} "
                f"against {part.on_hand if part.stocked else 0} on hand · "
                f"bites {part.bites_at} · {part.bites_note}"
            )
            for line in part.lines:
                lines.append(
                    f"      {line.source_kind} x{line.qty} — {line.detail}"
                )
    else:
        lines.append(
            "  (none — projected demand inside the horizon is covered "
            "by the shelf)"
        )
    if not shortages_only:
        covered = [p for p in outlook.parts if not p.shortage_risk]
        lines += ["", "COVERED DEMAND:"]
        lines += [_part_line(p) for p in covered] or ["  (no demand projected)"]
    if outlook.unquantified_pm:
        lines += ["", "PM CYCLES WHOSE PARTS CANNOT BE PROJECTED:"]
        for entry in outlook.unquantified_pm:
            lines.append(
                f"  - {entry['obligation_id']} · {entry['machine_id']} · "
                f"{entry['fault_class']} · due {entry['next_due_at']} — "
                f"{entry['reason']}"
            )
    return "\n".join(lines) + "\n"
