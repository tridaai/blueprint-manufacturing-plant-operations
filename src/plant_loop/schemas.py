"""Schemas: the vocabulary of the board, the loop, and the record.

Everything that crosses a boundary in this blueprint is one of these
pydantic models: a case as it arrives, a diagnosis / plan / critique
as a reasoning round produces it (validated in code before the loop
consumes it), and the :class:`CaseResult` record that waits at the
supervisor's gate. The models are deliberately plain data — all
behaviour lives in the loop controllers and guardrails.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# The five exception types the board carries. Priority between them is
# computed in code (priority.py), never asked of a model.
EXCEPTION_TYPES = (
    "machine_down",
    "quality_nonconformance",
    "material_shortage",
    "maintenance_alert",
    "safety_observation",
)

# The case loop's exits. Every case ends in exactly one of these, and
# the exit — not a model's mood — decides what gets drafted.
EXIT_CONVERGED = "converged"                # plan verified clean → work order
EXIT_EXHAUSTED = "exhausted"                # rounds ran out with gaps open
EXIT_BLOCKED_EXTERNAL = "blocked_external"  # part/vendor only an outsider can fix
EXIT_SAFETY_STOP = "safety_stop"            # safety case or safety bypass → now
EXIT_BUDGET_EXCEEDED = "budget_exceeded"    # case token budget spent mid-loop

# Printed on every record and every draft. It is a statement of the
# system's design, not a disclaimer the model chose to add.
GATE_NOTICE = (
    "This record was produced by an advisory agent. The agent never issues "
    "commands to machines, never marks work performed, and never closes a "
    "case. Only a supervisor's recorded decision moves a case past this gate."
)


class ExceptionCase(BaseModel):
    """One exception as it lands on the shift board."""

    case_id: str
    type: str  # one of EXCEPTION_TYPES
    title: str
    description: str = ""
    machine_id: str = ""
    line_id: str = ""
    product: str = ""
    batch: str = ""
    # Synthetic sensor / inspection readings, e.g.
    # {"spindle_vibration_mm_s": 8.4, "qc_porosity_pct": 6.2}.
    readings: dict[str, float | str] = Field(default_factory=dict)
    # Free text from the floor (operator note, maintenance log line).
    # UNTRUSTED: screened before any model reads it (screening.py).
    notes: str = ""
    reporter: str = ""
    reported_at: str = ""  # ISO-8601; ordering tiebreak on the board
    downtime_minutes: int = 0


class Hypothesis(BaseModel):
    """One ranked fault hypothesis. Evidence is not optional."""

    fault_class: str
    statement: str
    confidence: float = 0.0  # 0..1, the reasoner's own estimate
    evidence: list[str] = Field(default_factory=list)
    sop_id: str = ""


class Diagnosis(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    summary: str = ""

    def top(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None


class PlanStep(BaseModel):
    step_id: str
    text: str
    kind: str = "action"  # safety | action | check
    mandatory: bool = False  # a mandatory SOP safety step: non-removable
    part_id: str = ""
    qty: int = 0
    sop_ref: str = ""


class PartRequirement(BaseModel):
    part_id: str
    qty: int = 1
    available_qty: int = 0
    satisfied: bool = False
    substitute_part_id: str = ""  # set when the plan runs on a substitute


class SpecDelta(BaseModel):
    """One spec difference between an original part and a substitute."""

    attribute: str
    original: str | int | float | bool | None = None
    substitute: str | int | float | bool | None = None
    note: str = ""


class Substitution(BaseModel):
    """One substitute-search outcome, recorded on the plan.

    The verify stage runs the search in code (substitution.py): the
    KB lists candidates as *compatible* — with a basis and the spec
    deltas that matter — and each candidate is then checked against
    the shelf and the SOP's constraints for the slot. A candidate
    that clears both is ``applied``: the plan is revised onto it and
    the record carries the substitution openly, so the supervisor
    approves it knowingly. A candidate that fails is ``rejected``
    with the reason named (out of stock, or the constraint it
    violated). Rejections are paperwork, not decoration: they are
    how a blocked case documents that the search actually happened.
    """

    original_part_id: str
    substitute_part_id: str
    qty: int = 1
    basis: str = ""  # the KB's compatibility basis for the pairing
    spec_deltas: list[SpecDelta] = Field(default_factory=list)
    status: str = "applied"  # applied | rejected
    reason: str = ""  # applied: what it cleared · rejected: why it failed


class ActionPlan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    required_parts: list[PartRequirement] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    estimated_downtime_min: int = 0
    addresses: list[str] = Field(default_factory=list)  # fault classes the plan covers
    notes: str = ""


class Gap(BaseModel):
    """One defect the verify round found in the diagnosis or plan."""

    kind: str  # missing_safety_step | part_unavailable | unevidenced_hypothesis
    #          | banned_action | competing_hypothesis | sop_contradiction
    detail: str
    ref: str = ""  # step id / part id / fault class it points at
    blocking: bool = True
    safety: bool = False  # a safety-class gap stops the loop immediately


class Critique(BaseModel):
    gaps: list[Gap] = Field(default_factory=list)
    notes: str = ""


class RoundRecord(BaseModel):
    """One loop round, as telemetry and as console timeline material."""

    round: int
    top_fault_class: str = ""
    hypotheses: int = 0
    steps: int = 0
    gaps_opened: list[str] = Field(default_factory=list)  # gap kinds, this round
    gaps_closed: list[str] = Field(default_factory=list)  # gap kinds closed vs prior round
    substitutions: list[str] = Field(default_factory=list)  # verify-stage search, this round
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class GuardrailCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class GuardrailInfo(BaseModel):
    passed: bool
    checks: list[GuardrailCheck] = Field(default_factory=list)


class CaseTelemetry(BaseModel):
    backend: str = ""
    model: str = ""
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    latency_seconds: float = 0.0


class DraftInfo(BaseModel):
    kind: str = ""  # work_order | escalation
    text: str = ""


# The supervisor's verdict on the plan itself, recorded with the
# outcome: distinct from the gate decision (approved / rejected),
# which says whether the draft may proceed at all.
OUTCOME_DECISIONS = ("accepted", "accepted_with_changes", "rejected")


class SupervisorOutcome(BaseModel):
    """What actually happened after the gate, recorded by a supervisor.

    This is the ground truth the loop is measured against: the fault
    the floor actually found (``confirmed_fault``, from the same
    vocabulary the diagnoses use), whether the drafted plan was the
    plan that got executed, and the downtime the case really cost.
    Every field except ``decision`` may be left unrecorded — a partial
    outcome still counts, and the stats say what they were computed
    over rather than pretending the record is complete.
    """

    decision: str  # one of OUTCOME_DECISIONS
    confirmed_fault: str = ""  # fault class, "" when not confirmed/recorded
    work_performed: str = ""
    downtime_minutes_actual: int | None = None
    parts_used: list[str] = Field(default_factory=list)
    resolved: bool | None = None  # None = not recorded
    recorded_by: str = ""
    recorded_at: str = ""


class BoardItem(BaseModel):
    """A non-case item on the board, raised by a code-side watch.

    The board carries exceptions; it also carries the patterns the
    exceptions form. A ``reliability_review`` item is raised when the
    same machine keeps failing the same way (see reliability.py). It
    is not an exception and it triggers nothing automatic — it waits
    at the supervisor gate like everything else, carrying the cases
    that are its evidence.
    """

    item_id: str
    kind: str = "reliability_review"
    status: str = "open"  # open | acknowledged
    machine_id: str = ""
    fault_class: str = ""
    case_ids: list[str] = Field(default_factory=list)  # the evidence
    count: int = 0
    window_days: int = 0
    threshold: int = 0
    title: str = ""
    detail: str = ""
    created_at: str = ""


class HistoryEntry(BaseModel):
    """One supervisor-confirmed fault, written back to the plant's memory.

    The KB's seeded history is authored ground truth and never
    changes. This is the other kind of history: when a supervisor's
    outcome confirms the fault the floor actually found, the
    confirmation is appended to the machine's history as an entry of
    its own — same shape as a seeded entry (machine, fault class,
    date, what was seen, what was done) plus the provenance a seeded
    entry does not need: which case confirmed it, and the
    ``source`` marking (``supervisor_confirmed``) that keeps it
    distinguishable from the KB's recorded history forever.

    One entry per confirming case: the entry id derives from the
    case id, and outcomes are recorded once per case, so a
    confirmation can never be written twice. Only a confirmed fault
    writes one — an outcome that confirms nothing writes nothing,
    and a safety stop never writes one at all (history.py refuses
    it in code: a safety case has no diagnosis to confirm).
    """

    entry_id: str  # hist-<case_id>
    machine_id: str = ""
    fault_class: str = ""
    case_id: str = ""  # the case whose outcome confirmed the fault
    source: str = "supervisor_confirmed"
    date: str = ""  # the case's date (its reported_at, date part)
    recorded_at: str = ""  # when the outcome was recorded
    summary: str = ""
    resolution: str = ""


class TraceStep(BaseModel):
    name: str
    title: str
    details: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class CaseResult(BaseModel):
    """The record that waits at the supervisor's gate."""

    case_id: str
    status: str = "awaiting_supervisor"  # awaiting_supervisor | approved | rejected
    exit_reason: str = ""
    type: str = ""
    title: str = ""
    machine_id: str = ""
    line_id: str = ""
    priority: float = 0.0
    priority_reason: str = ""
    diagnosis: Diagnosis = Field(default_factory=Diagnosis)
    plan: ActionPlan = Field(default_factory=ActionPlan)
    open_gaps: list[Gap] = Field(default_factory=list)
    rounds: list[RoundRecord] = Field(default_factory=list)
    guardrails: GuardrailInfo = Field(default_factory=lambda: GuardrailInfo(passed=True))
    injection_flags: list[str] = Field(default_factory=list)
    draft: DraftInfo = Field(default_factory=DraftInfo)
    telemetry: CaseTelemetry = Field(default_factory=CaseTelemetry)
    gate_notice: str = GATE_NOTICE
    reported_at: str = ""
    created_at: str = ""
    # The supervisor's decision, when recorded.
    decision: str = ""
    decision_by: str = ""
    decision_reason: str = ""
    # What actually happened after the gate, when recorded.
    outcome: SupervisorOutcome | None = None
    trace: list[TraceStep] = Field(default_factory=list)


class BoardRunResult(BaseModel):
    """The outcome of one board-loop run (a shift's worth of work)."""

    run_id: str
    status: str = ""  # board_clear | shift_budget_exceeded
    processed: list[str] = Field(default_factory=list)  # case ids, in worked order
    exits: dict[str, str] = Field(default_factory=dict)  # case_id -> exit reason
    remaining: list[dict] = Field(default_factory=list)  # [{case_id, reason}]
    handoff_summary: str = ""
    telemetry: CaseTelemetry = Field(default_factory=CaseTelemetry)
    created_at: str = ""
