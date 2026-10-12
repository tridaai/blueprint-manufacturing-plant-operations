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
    #          | incompatible_part (a gate amendment's swapped part fails
    #            the SOP's constraints for its slot — amendments.py)
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


class PmProposal(BaseModel):
    """A preventive-maintenance proposal the reliability watch earned.

    When the watch's recurrence rule fires for a machine + fault
    class, the pattern stops being only a board item: this proposal
    is drafted in code (pm.py) from the watch's own contributing
    set — the cited cases are its evidence, and the proposed
    inspection interval is derived from their spacing by the
    plant model's authored rule, with the arithmetic carried on
    the record so the supervisor decides on the numbers, not on a
    summary of them. A proposal is PROPOSED until a supervisor
    approves it (the approved interval is what stands — it may
    differ from the proposal's) or rejects it with a reason; a
    rejection moves the evidence watermark, so the same evidence
    can never re-draft this proposal (pm.py).
    """

    proposal_id: str
    kind: str = "proposal"  # the pm_records kind column
    status: str = "proposed"  # proposed | approved | rejected
    machine_id: str = ""
    fault_class: str = ""
    line_id: str = ""
    # The watch item this proposal answers, when one stands open
    # for the pair (the usual case — the two are raised together).
    board_item_id: str = ""
    # The evidence: the watch's contributing cases, cited by id
    # and by the moment each entered the record. Never summarised.
    evidence: list[dict] = Field(default_factory=list)  # [{case_id, at}]
    fault_count: int = 0
    window_days: int = 0
    threshold: int = 0
    # The interval arithmetic, as printed for the supervisor.
    gaps_days: list[float] = Field(default_factory=list)
    mean_gap_days: float = 0.0
    interval_fraction: float = 0.0
    min_interval_days: int = 0
    max_interval_days: int = 0
    raw_interval_days: float = 0.0
    clamped: bool = False
    proposed_interval_days: int = 0
    rationale: str = ""
    created_at: str = ""
    # The supervisor's decision, when recorded.
    decided_by: str = ""
    decided_at: str = ""
    decision_reason: str = ""  # required for a rejection
    approved_interval_days: int | None = None


class PmObligation(BaseModel):
    """A standing preventive-maintenance obligation.

    Created by approving a PmProposal — and by nothing else: the
    agent drafts proposals and surfaces due cases, but an
    obligation exists only because a supervisor approved one, and
    a PM is marked performed only because a supervisor's outcome
    on the surfaced case said so (pm.py). The obligation carries
    the approved interval, the clock it runs on (from the last
    performed inspection — supervisor-declared at approval when
    the plant knows it, else from the approval itself), the cycle
    case currently on the board, and the deferral history: a
    deferred PM rolls its due point forward by the plant model's
    deferral step, with the supervisor's reason kept.

    An obligation ends only by retirement, and retirement is
    one-way: a supervisor retires it with a reason, or its
    machine's decommission (lifecycle.py) retires it automatically
    with the linkage on the record. The record is never deleted —
    why the plant stopped owing this inspection stays readable.
    """

    obligation_id: str
    kind: str = "obligation"  # the pm_records kind column
    status: str = "standing"  # standing | retired (retirement is one-way)
    proposal_id: str = ""
    machine_id: str = ""
    fault_class: str = ""
    line_id: str = ""
    interval_days: int = 0  # the APPROVED interval — the one that stands
    approved_by: str = ""
    approved_at: str = ""
    # The clock: due = basis + interval. The basis is the last
    # performed inspection when the supervisor declared it at
    # approval (last_performed_at set), else the approval itself.
    clock_basis: str = ""  # last_performed | approval
    last_performed_at: str = ""  # "" until a PM is performed
    next_due_at: str = ""
    # The open cycle: the case the board is carrying for this
    # obligation ("" between cycles). cycle_case_ids is every
    # cycle's case, append-only — the evidence exclusion list
    # (an inspection the system scheduled is not a recurrence).
    cycle_case_id: str = ""
    cycle_case_ids: list[str] = Field(default_factory=list)
    cycles_completed: int = 0
    last_cycle: dict = Field(default_factory=dict)  # {case_id, result, at, by}
    deferrals: list[dict] = Field(default_factory=list)
    created_at: str = ""
    # Retirement (lifecycle.py / pm.py): a supervisor ends the
    # obligation — by hand, with their own reason, or automatically
    # when its machine is decommissioned (kind
    # "machine_decommission", the reason fixed as "machine
    # decommissioned", and decommission_id linking the machine
    # record that ended it). Retirement is one-way: the record
    # keeps the reason forever, surfaces no further due cases, and
    # nothing un-retires it — a pattern that returns earns a fresh
    # proposal, and a fresh approval creates a NEW obligation.
    retired_at: str = ""
    retired_by: str = ""
    retirement_reason: str = ""
    retirement_kind: str = ""  # manual | machine_decommission
    decommission_id: str = ""  # the MachineRecord, when auto-retired


class MachineRecord(BaseModel):
    """A machine's recorded end: the decommission a supervisor declared.

    The plant model (kb.json) says which machines exist; it is
    authored data and never changes at runtime. Whether a machine
    still stands on the floor is a runtime fact, so it lives here,
    as a record (migration 0007): a supervisor decommissions a
    machine — actor and reason required, effective from a stated
    moment in the record-timestamp idiom — and the record carries
    everything the act set in motion: the standing PM obligations
    it retired automatically, and the machine's open board cases
    as they stood at that moment, each waiting for the
    supervisor's own disposition (kept, or closed as moot) —
    nothing about a case is decided by the decommission itself.

    One record per machine, ever: a machine is decommissioned
    once. The record is never deleted and never edited backwards;
    dispositions append to it as they are chosen.
    """

    record_id: str  # DEC-<random hex>
    machine_id: str
    status: str = "decommissioned"  # the only state a record marks
    reason: str = ""  # the supervisor's reason (required)
    decided_by: str = ""  # the supervisor (required)
    decided_at: str = ""  # when the act was recorded
    effective_at: str = ""  # the stated moment it takes effect
    # The standing obligations the act retired automatically,
    # by obligation id (each obligation carries the link back).
    retired_obligation_ids: list[str] = Field(default_factory=list)
    # The machine's open cases at the moment of the act — the
    # list the supervisor disposes from, one by one.
    open_case_ids: list[str] = Field(default_factory=list)
    # The dispositions, as chosen: [{case_id, disposition, by,
    # at}] with disposition keep | close_moot. A case appears
    # once — a disposition is a recorded decision, not an edit.
    dispositions: list[dict] = Field(default_factory=list)
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


class PartClaim(BaseModel):
    """One case's claim on a shelf part, made when its plan converged.

    The board shares one shelf. Before round 5, each case's verify
    stage measured that shelf as if the case were alone on it: two
    cases could both plan around the same last unit, both converge,
    and the floor discovered the conflict at the workbench. Now a
    converged plan *claims* the quantities it will draw (the
    substitute's quantity, when the plan runs on a substitute), in
    board-run order — which is priority order by construction — and
    later cases in the run (and every later run) measure the shelf
    net of outstanding claims.

    A claim is a planning reservation on the record, nothing more:
    it is not a warehouse transaction, it draws no stock, and it
    says nothing about work performed. While its case waits at the
    gate it stands — across board runs, too: a successor run that
    accepts the case as carry-over renews the claim into itself
    (``carried_from_run`` keeps the hop's provenance), and it
    stays releasable exactly as before. It is released in one of
    two ways, each stamped with who released it and why: the
    supervisor rejects the case at the gate
    (``released_by_gate_rejection``), or a board turnover finds
    the case's business finished — decided and done, dropped,
    superseded (``released_run_ended``, carryover.py). The one
    claim a turnover never releases is an approved case's while
    its outcome is still outstanding: the approved work order
    will draw the part. A release never rewrites the finished run
    it came from — runs are records; a case that lost a part to a
    claim keeps its blocked record until a fresh run re-plans it.
    """

    claim_id: str  # claim-<case_id>-<part_id drawn>
    run_id: str = ""  # the board run that produced the case, or "single"
    case_id: str = ""
    part_id: str = ""  # the part actually drawn (the substitute, when substituted)
    original_part_id: str = ""  # the requirement's part, when a substitute is drawn
    qty: int = 1
    status: str = "claimed"  # claimed | released
    claimed_at: str = ""
    released_at: str = ""
    released_by: str = ""
    release_reason: str = ""  # released_by_gate_rejection | released_run_ended
    # Set when a successor board run accepts this claim's case as
    # carry-over (carryover.py): the run the claim was carried
    # *from* at its latest renewal. ``run_id`` then names the run
    # that currently holds the claim; this field keeps the hop's
    # provenance, so a gap can say "carried over from board run X"
    # instead of pretending the claim was made on today's board.
    carried_from_run: str = ""


class CaseEconomics(BaseModel):
    """One case priced in downtime dollars (economics.py).

    The exposure is the plan's own estimated downtime priced at the
    line's rate in the plant model — an estimate of what leaving
    this case waiting costs per the plan, nothing more: it reorders
    nothing (priority stays the severity formula's), and it is never
    computed for a safety stop, whose record carries no economics
    at all. ``rate_usd_per_hour`` is ``None`` when the plant model
    carries no rate for the line; every dollar figure is then
    ``None`` too — minutes are shown, dollars are not invented.
    Actuals join the snapshot when the supervisor records the
    outcome, at the same rate.
    """

    line_id: str = ""
    rate_usd_per_hour: float | None = None
    rate_basis: str = ""  # where the rate comes from, cited on the figure
    estimated_minutes: int = 0
    estimated_usd: float | None = None
    actual_minutes: int | None = None
    actual_usd: float | None = None


class BoardEconomics(BaseModel):
    """A set of cases priced as one board (economics.py).

    Dollar totals cover only the cases on lines the plant model
    rates; the minutes on unrated lines are summed separately and
    never folded into a dollar figure. ``estimated_exposure_usd``
    is ``None`` when no case in the set is costed at all.
    """

    cases_costed: int = 0
    cases_minutes_only: int = 0
    estimated_minutes: int = 0
    unrated_minutes: int = 0
    estimated_exposure_usd: float | None = None
    largest_exposure_case_id: str = ""
    largest_exposure_usd: float | None = None


class PartDemandLine(BaseModel):
    """One traced line of projected part demand (outlook.py).

    Every projected unit names where it comes from — a standing
    claim, an open case's verified plan, or a standing PM
    obligation's next due cycle — because a projection a supervisor
    cannot trace back to a record is a guess wearing a table's
    clothes. ``date`` is the line's own record date: the claim's
    stamp, the case's creation, the cycle's due point.
    """

    part_id: str
    qty: int = 0
    source_kind: str = ""  # claim | case_plan | pm_cycle
    source_id: str = ""  # the case id, or the obligation id
    date: str = ""  # the line's own record timestamp
    detail: str = ""


class PartOutlook(BaseModel):
    """One part's projected position inside the outlook horizon.

    Projected demand is the sum of the traced lines: standing
    claims (already committed), open cases' unsatisfied plan
    requirements (still to draw), and PM cycles falling due inside
    the horizon (dated). ``shortfall`` is what the shelf cannot
    cover; a part at exactly zero shortfall is covered, not at
    risk — the boundary is the plant's own arithmetic, not a
    rounding mood. ``bites_at`` is the first moment cumulative
    demand passes the shelf: the projection moment itself when the
    book is already short, else the due date of the PM cycle that
    tips it over.
    """

    part_id: str
    part_name: str = ""
    stocked: bool = True  # False: the plant model stocks no such part
    on_hand: int | None = None  # the authored shelf qty (None: not stocked)
    claimed_qty: int = 0
    case_demand_qty: int = 0
    pm_demand_qty: int = 0
    projected_demand: int = 0
    shortfall: int = 0
    shortage_risk: bool = False
    bites_at: str = ""  # "" when there is no shortfall
    bites_note: str = ""  # what tips it over, in words
    lines: list[PartDemandLine] = Field(default_factory=list)


class PartsOutlook(BaseModel):
    """The board looking ahead: projected demand against the shelf.

    A pure read model over the store plus the plant model
    (outlook.py): it creates no cases, places no claims and orders
    nothing. ``unquantified_pm`` names the standing obligations
    due inside the horizon whose machine + fault class has no SOP
    on record — their cycles' parts cannot be projected, and the
    outlook says so rather than inventing quantities. (An SOP that
    names no parts is not unquantified: it was consulted, and it
    names none.)
    """

    as_of: str = ""
    horizon_days: int = 0
    horizon_end: str = ""
    basis: str = ""
    parts: list[PartOutlook] = Field(default_factory=list)
    shortages: list[PartOutlook] = Field(default_factory=list)
    unquantified_pm: list[dict] = Field(default_factory=list)


class KnowledgeGap(BaseModel):
    """One missing piece of plant knowledge, and what it has cost.

    A row exists because case records ended without a verified
    plan while their diagnosed fault had no SOP on record for
    their machine (gaps.py — the same ``sop_for`` test the loop
    and the parts outlook apply). The fields are the asset, the
    evidence and the cost, and nothing else: identifiers, counts,
    timestamps. There is deliberately no field that could carry
    SOP content — no steps, no text, no draft — because this
    report advises the knowledge base's maintainer; it must never
    become a machine-authored procedure wearing a report's
    clothes. ``fault_class`` may be ``unknown_fault``: the
    diagnosis itself had nothing to name, which is the same hole
    seen from the other side.
    """

    machine_id: str
    fault_class: str  # the diagnosed fault with no SOP (or unknown_fault)
    machine_on_register: bool = False  # False: the KB has no such machine
    blocked_cases: int = 0
    case_ids: list[str] = Field(default_factory=list)  # chronological
    first_case_id: str = ""
    latest_case_id: str = ""
    first_seen_at: str = ""  # the earliest contributing record's stamp
    latest_seen_at: str = ""
    rounds_burned: int = 0  # total loop rounds the gap's cases spent
    exits: dict[str, int] = Field(default_factory=dict)  # exit -> count
    open_now: bool = False  # a contributing case still waits at the gate
    open_case_ids: list[str] = Field(default_factory=list)
    # Standing PM obligations standing on the same hole — the
    # outlook's unquantified test (no SOP on record) applied to
    # the same machine + fault, without the outlook's horizon:
    # the maintainer who authors this SOP unblocks both.
    standing_obligation_ids: list[str] = Field(default_factory=list)


class KnowledgeGapReport(BaseModel):
    """What the plant doesn't know: the knowledge-gap report.

    A pure read model over the store plus the plant model
    (gaps.py): it creates no cases, drafts no SOPs, and unblocks
    nothing. A gap stops biting when new cases stop blocking on
    it — visible here as the row's case list no longer growing,
    and gone entirely when every contributing record has been
    superseded by a re-run that converged (records are upserts;
    finished runs are never rewritten).
    """

    as_of: str = ""
    basis: str = ""
    cases_examined: int = 0
    blocked_cases: int = 0  # summed over the rows
    gaps: list[KnowledgeGap] = Field(default_factory=list)


class RecurrenceLink(BaseModel):
    """This case's link to the closed case it recurs (recurrence.py).

    Stamped by code at the moment the loop's first diagnosis lands:
    the diagnosed top fault is the join key, and the link names the
    most recent earlier case on the same machine whose recorded
    outcome *confirmed that same fault* inside the recurrence
    window. It is an overlay on the new case's record only — the
    prior record is never touched by being linked from.

    The link carries the prior episode as the gate should meet it:
    the case, what the loop diagnosed there, what the floor's
    outcome decided and whether it resolved, and the spacing
    between the two episodes. ``departed`` is stamped when the
    record is assembled, from the *final* diagnosis: True when the
    loop's settled fault differs from the prior's confirmed one —
    the mark the surfaces print as DIAGNOSIS_DEPARTED_FROM_PRIOR.
    It blocks nothing; it is a fact for the supervisor, and the
    outcome stats are where it is judged.
    """

    prior_case_id: str = ""
    machine_id: str = ""
    fault_class: str = ""  # the prior's confirmed fault (the join key)
    prior_diagnosis: str = ""  # the prior case's own top diagnosis
    outcome_decision: str = ""  # accepted | accepted_with_changes | rejected
    outcome_resolved: bool | None = None  # None = not recorded
    work_performed: str = ""
    prior_reported_at: str = ""
    reported_at: str = ""  # this case's reported moment
    days_between: int = 0
    window_days: int = 0
    departed: bool = False


class AmendmentVerification(BaseModel):
    """The code re-verification of one gate amendment (amendments.py).

    Run at the moment the amendment is recorded, over the amended
    plan, with no model anywhere in it: the verify stage's own code
    checks (``guardrails.verify_plan``) plus, for a part swap, the
    SOP-constraint test the substitute search applies. ``gaps`` is
    the failing checks, verbatim; ``detail`` is the one-line
    verdict the gate faces print. A ``failed`` amendment enters
    nothing: the plan, the claims and the record's approved-able
    state all stay with the last verified plan, and approval is
    refused until a passing amendment (or a rejection) resolves it.
    """

    status: str = ""  # passed | failed
    checked_at: str = ""
    detail: str = ""
    gaps: list[Gap] = Field(default_factory=list)


class PlanAmendment(BaseModel):
    """One supervisor amendment to a gated case's plan (amendments.py).

    The gate's third path, beside approve and reject: the
    supervisor who can see the smaller fix — swap this part for
    the one on the shelf, strike the step that does not apply —
    records it here, by name, with a reason. The amendment is an
    overlay: the loop's original plan on the record is never
    edited, and the amended plan every surface prints is derived
    from it by replaying the *passing* amendments in order.

    Two kinds, deliberately no more. A ``part_swap`` replaces the
    part drawn for one requirement slot with another part from
    the plant register, named by id (free text is refused at
    recording — a part the register does not carry cannot be
    verified against anything). A ``step_strike`` removes one
    non-mandatory step; a mandatory safety step can never be
    struck (the refusal names the step and its SOP), because an
    amendment that thins a plan's safety steps is a rejection
    wearing a smaller hat. There is no step *addition*: an added
    step has no diagnosis behind it — the supervisor directs the
    work, the planner proposes it, and new steps come from the
    loop on a fresh run, not from the gate.

    A swap carries its compatibility paperwork the way a
    verify-stage substitution does: the KB compatibility basis
    when the pairing is listed for the slot (with the authored
    spec deltas), or the plain statement that the pairing is
    supervisor-specified and unlisted — with the spec deltas
    computed from the two parts' spec sheets either way.
    """

    amendment_id: str  # amd-<case_id>-<n>
    case_id: str = ""
    kind: str = ""  # part_swap | step_strike
    by: str = ""  # the supervisor (required)
    reason: str = ""  # the supervisor's reason (required)
    recorded_at: str = ""
    # part_swap: the requirement slot (its own part id), the part
    # drawn for it before this amendment, and the part drawn after.
    slot_part_id: str = ""
    from_part_id: str = ""
    to_part_id: str = ""
    qty: int = 0
    basis: str = ""
    spec_deltas: list[SpecDelta] = Field(default_factory=list)
    # The drawn part's shelf position (net of other cases' claims)
    # as the re-verification measured it — the number the verified
    # amended plan carries for the swapped requirement.
    to_available_qty: int = 0
    # step_strike: the struck step, quoted so the record shows
    # what was removed even though the amended plan omits it.
    step_id: str = ""
    step_text: str = ""
    verification: AmendmentVerification = Field(default_factory=AmendmentVerification)


class TraceStep(BaseModel):
    name: str
    title: str
    details: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class CaseResult(BaseModel):
    """The record that waits at the supervisor's gate."""

    case_id: str
    # awaiting_supervisor | approved | rejected | dropped (a
    # carried case the supervisor dropped when the next shift
    # accepted the board — carryover.py; the stamp is below) |
    # closed_moot (its machine was decommissioned and the
    # supervisor disposed it as moot — lifecycle.py; stamp below).
    status: str = "awaiting_supervisor"
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
    # The supervisor's amendments to the plan, recorded at the
    # gate in order (amendments.py). Append-only: an amendment is
    # never edited or removed, a failed one included — the record
    # shows what was tried, what the code checks said, and what
    # the plan became. The original ``plan`` above is never
    # touched by any of them.
    amendments: list[PlanAmendment] = Field(default_factory=list)
    # The recurrence link (recurrence.py): set when the loop's
    # first diagnosis matched a fault a supervisor had already
    # confirmed on this machine, on a closed case inside the
    # recurrence window. An overlay computed from the store at
    # diagnosis time; the prior case's record is never modified
    # by carrying it. None when this fault has no qualifying
    # prior — the ordinary cold start.
    recurrence: RecurrenceLink | None = None
    # When a successor board run accepted this case's predecessor
    # run as carry-over and the supervisor dropped the case at that
    # acceptance: {from_run, into_run, by, reason, at}. The case
    # leaves the board (status "dropped"); the stamp is the audit
    # trail. Empty for every case that was never dropped.
    dropped_from_board: dict = Field(default_factory=dict)
    # When the case's machine was decommissioned and the
    # supervisor chose this case's disposition as close-as-moot:
    # {decommission_id, machine_id, by, at}. The case leaves the
    # board (status "closed_moot") — closed by a supervisor's
    # recorded choice on the decommission record, never by the
    # decommission itself and never by the agent. Empty for every
    # case that was never closed this way.
    closed_as_moot: dict = Field(default_factory=dict)
    # What actually happened after the gate, when recorded.
    outcome: SupervisorOutcome | None = None
    # The case priced in downtime dollars (economics.py). Stamped
    # when the record is persisted and refreshed when the outcome
    # lands; None on a safety stop, by design.
    economics: CaseEconomics | None = None
    trace: list[TraceStep] = Field(default_factory=list)


class BoardRunResult(BaseModel):
    """The outcome of one board-loop run (a shift's worth of work)."""

    run_id: str
    status: str = ""  # board_clear | shift_budget_exceeded
    processed: list[str] = Field(default_factory=list)  # case ids, in worked order
    exits: dict[str, str] = Field(default_factory=dict)  # case_id -> exit reason
    remaining: list[dict] = Field(default_factory=list)  # [{case_id, reason}]
    # The allocation ledger for this run: the part claims its
    # converged cases made, with their standing (allocation.py).
    allocations: list[PartClaim] = Field(default_factory=list)
    # The run priced in downtime dollars: the estimated exposure
    # its processed cases carry (economics.py).
    economics: BoardEconomics | None = None
    # Carry-over (carryover.py): the run this one inherited its
    # board from ("" — it started clean), how the inheritance
    # settled (cases inherited / re-worked / dropped / closed, and
    # the claims renewed and released at the turnover), and this
    # run's own outgoing manifest — the unfinished business the
    # next shift is offered, computed in code when the run ends.
    carried_from: str = ""
    carry_over_settled: dict = Field(default_factory=dict)
    carry_over: list[dict] = Field(default_factory=list)
    # Preventive maintenance (pm.py): the due-obligation cases this
    # run surfaced onto its board — ordinary maintenance_alert
    # cases in `processed`, named here so a reader can tell which
    # of the board's cases the plant's standing obligations sent.
    pm_cases: list[str] = Field(default_factory=list)
    handoff_summary: str = ""
    telemetry: CaseTelemetry = Field(default_factory=CaseTelemetry)
    created_at: str = ""
