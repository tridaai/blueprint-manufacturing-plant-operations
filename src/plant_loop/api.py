"""HTTP API: cases in, loop records out, supervisor decisions back.

A thin transport over :class:`~plant_loop.service.PlantService` —
every route delegates, none re-implements. Notable behaviour:

- structured JSON logs, one line per request, correlated by a
  request id (``X-Request-ID`` header or a generated one);
- optional shared-key auth: set ``API_KEY`` and every route except
  the health/readiness probes and the console requires
  ``X-API-Key``;
- ``POST /cases/analyze`` runs one case through its loop to the
  supervisor gate and returns the full loop record; a dead provider
  call is a 502 with the translated error, never a half-record;
- ``POST /board/run`` works a list of cases (or the bundled board)
  through the board loop in priority order, to the shift budget;
  with ``carry_over_from`` naming a finished run it first settles
  that run's carry-over manifest (carryover.py) — blocked cases
  re-worked with their loop state, gate cases inherited standing,
  ``drop`` entries leaving the board audited;
- ``GET /board/runs/{run_id}/carry-over`` returns a stored run's
  carry-over contract: the manifest it offers the next shift, and
  how its own inheritance settled;
- ``POST /cases/{id}/decision`` records approve/reject — with the
  carry-over drop, the only ways a case leaves
  ``awaiting_supervisor``;
- ``POST /api/cases/{id}/outcome`` records what actually happened
  after the gate (the plan's disposition, the confirmed fault, the
  real downtime) — the ground truth behind ``GET
  /api/stats/diagnosis`` and the recurring-fault watch;
- ``GET /api/board/handoff`` renders the current board as a shift
  handoff (JSON, or ``?format=text`` for the shift-log text);
- ``GET /api/cases/{id}/packet`` renders one gated case as a
  work-order packet for the floor (JSON, or ``?format=text``) —
  converged cases only; anything else is a 409, because an
  unverified plan must not reach the floor as a work order;
- ``GET /api/board/items`` lists the non-case board items (the
  reliability reviews the watch raises);
- ``GET /api/parts/outlook`` projects part demand against the
  shelf over a horizon (outlook.py): standing claims, open
  cases' unsatisfied plan requirements, and PM cycles falling
  due, each traced to its source record, with the shortage rows
  and the moment each shortfall bites. ``?horizon_days=`` widens
  or narrows the window (the authored default is 14). A pure
  read — it creates nothing and orders nothing;
- ``GET /api/knowledge/gaps`` aggregates the case blocks the
  plant's own knowledge caused (gaps.py): records whose diagnosed
  fault has no SOP on record for their machine, grouped by the
  missing SOP, with the rounds the absence burned and whether a
  case is blocked on it right now. A pure read that drafts
  nothing — it advises the KB maintainer; authoring an SOP stays
  a human's act;
- ``GET /api/pm/proposals`` / ``GET /api/pm/obligations`` list the
  preventive-maintenance layer (pm.py): proposals the watch's
  pattern earned — evidence and interval arithmetic on the record —
  and the standing obligations approvals created;
  ``POST /api/pm/proposals/{id}/decision`` records the
  supervisor's approve (interval editable at the gate) or reject
  (reason required — it moves the evidence watermark);
  ``POST /api/pm/obligations/{id}/retire`` records a supervisor's
  manual retirement of an obligation (actor + reason required,
  one-way);
- ``GET /api/windows`` lists the downtime windows (windows.py)
  with their computed plans, the planned / confirmed counts, and
  the ledger of confirmed windows (each window's review counts
  once its span has closed); ``POST /api/windows`` declares one
  (supervisor, start, length, scope — the plan is computed in
  the same act); ``POST /api/windows/{id}/plan`` recomputes a
  proposed window's plan; ``POST /api/windows/{id}/confirm``
  confirms it — re-reading the fingerprinted state first, so a
  plan the board has drifted from refuses, naming the drift;
  ``GET /api/windows/{id}/review`` answers for a confirmed
  window once its span has closed — each packed item performed,
  deferred, or not worked (a proposed window, and one still
  running, refuse: 409);
- ``GET /api/machines`` lists the plant's machines with their
  lifecycle state (lifecycle.py); ``POST
  /api/machines/{id}/decommission`` records a machine's end
  (actor + reason required; its standing obligations retire
  automatically, linked, and its open cases are listed for
  disposition); ``POST /api/machines/{id}/dispositions`` records
  the supervisor's keep / close-as-moot choice for one of them.
  New cases for a decommissioned machine are refused at intake
  (``POST /cases/analyze`` and ``POST /board/run`` answer 409,
  state named);
- ``/readiness`` fails (503) when PostgreSQL is configured but
  unreachable, and says honestly when the in-memory test doubles are
  in use (records lost on restart — fine for a look, not for work).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, kb
from . import amendments as amendments_module
from . import outlook as outlook_module
from . import windows as windows_module
from .config import env_str, load_dotenv
from .errors import ProviderError
from .loops import BoardPolicy
from .samples import load_sample_cases
from .schemas import OUTCOME_DECISIONS, ExceptionCase, SupervisorOutcome
from .service import PlantService
from .wiring import build_service

SERVICE_SLUG = "manufacturing-plant-operations"

# ---------------------------------------------------------------------------
# structured logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger("plant_loop")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


log = _configure_logging()

# ---------------------------------------------------------------------------
# app + service (built lazily so imports/tests stay cheap)
# ---------------------------------------------------------------------------

_service: PlantService | None = None


def get_service() -> PlantService:
    global _service
    if _service is None:
        _service = build_service()
    return _service


def set_service(service: PlantService | None) -> None:
    """Test hook: inject a fully-double service (or reset with None)."""
    global _service
    _service = service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv()
    database = "PostgreSQL" if env_str("DATABASE_URL") else "in-memory (TEST DOUBLE)"
    bucket = env_str("S3_BUCKET")
    storage = f"S3 bucket {bucket!r}" if bucket else "in-memory (TEST DOUBLE)"
    log.info(
        "plant-loop %s starting · store: %s · objects: %s · reasoning backend: %s",
        __version__,
        database,
        storage,
        env_str("REASONING_BACKEND", "offline"),
    )
    yield
    log.info("plant-loop shutting down")


app = FastAPI(
    title="Trida AI Blueprint — Plant Operations Loop Agent",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "%s %s -> %s (%s ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={"request_id": request_id},
    )
    return response


def require_api_key(request: Request) -> None:
    configured = env_str("API_KEY")
    if configured and request.headers.get("X-API-Key") != configured:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


@app.middleware("http")
async def api_key_gate(request: Request, call_next):
    path = request.url.path
    public = (
        path in ("/", "/health", "/readiness", "/favicon.svg", "/favicon.ico")
        or path.startswith("/static/")
    )
    if not public and env_str("API_KEY"):
        try:
            require_api_key(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.exception_handler(ProviderError)
async def provider_error_handler(_request: Request, exc: ProviderError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# console + probes
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(str(resources.files(__package__).joinpath("static")))
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def console():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/favicon.svg")
def favicon_svg():
    return FileResponse(str(_STATIC_DIR / "brand" / "favicon.svg"))


@app.get("/favicon.ico")
def favicon_ico():
    return FileResponse(str(_STATIC_DIR / "brand" / "favicon.ico"))


@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_SLUG, "version": __version__}


@app.get("/readiness")
def readiness():
    service = get_service()
    database_url = env_str("DATABASE_URL", "")
    checks: dict = {}
    if database_url:
        ping = getattr(service.store, "ping", None)
        checks["postgres"] = bool(ping()) if ping else False
        store_kind = "postgres"
    else:
        checks["postgres"] = None
        store_kind = "in-memory (TEST DOUBLE — records lost on restart)"
    checks["objects"] = "s3" if env_str("S3_BUCKET") else "in-memory (TEST DOUBLE)"
    ready = checks["postgres"] is not False
    body = {
        "ready": ready,
        "store": store_kind,
        "checks": checks,
        "reasoning_backend": env_str("REASONING_BACKEND", "offline"),
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


# ---------------------------------------------------------------------------
# cases + board
# ---------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    decision: str  # approved | rejected
    by: str = ""
    reason: str = ""


class AmendmentRequest(BaseModel):
    kind: str  # part_swap | step_strike
    by: str = ""  # the supervisor (required — amendments.py refuses anonymous ones)
    reason: str = ""  # the supervisor's reason (required)
    from_part_id: str = ""  # part_swap: the part the plan currently draws
    to_part_id: str = ""  # part_swap: a part on the register, by id
    step_id: str = ""  # step_strike: a non-mandatory step in the plan


class PmDecisionRequest(BaseModel):
    decision: str  # approved | rejected
    by: str = ""
    reason: str = ""  # required for a rejection (pm.py)
    # The interval the supervisor approves (editable at the gate);
    # omit to stand on the proposal's own interval.
    interval_days: int | None = None
    # The last performed inspection, as a record timestamp, when
    # the supervisor declares it at approval — the obligation's
    # clock runs from it (pm.py).
    last_performed_at: str = ""


class PmRetireRequest(BaseModel):
    by: str = ""  # required (pm.py): whose act ends the obligation
    reason: str = ""  # required (pm.py): why the plant stops owing it


class WindowDeclareRequest(BaseModel):
    by: str = ""  # required (windows.py): the supervisor declaring it
    starts_at: str = ""  # a record timestamp (ISO-8601 accepted)
    length_min: int = 0  # the window's length in minutes (at least 1)
    line_ids: list[str] = []  # scope: whole lines, by plant-model id
    machine_ids: list[str] = []  # scope: named machines, by plant-model id


class WindowConfirmRequest(BaseModel):
    by: str = ""  # required (windows.py): who takes the plan as the schedule


class MachineDecommissionRequest(BaseModel):
    by: str = ""  # required (lifecycle.py): whose act ends the machine
    reason: str = ""  # required (lifecycle.py): why it is being retired
    # The stated moment the decommission takes effect, as a record
    # timestamp; omit for the moment of the act itself.
    effective_at: str = ""


class MachineDispositionRequest(BaseModel):
    case_id: str
    disposition: str  # keep | close_moot
    by: str = ""  # required (lifecycle.py): whose choice this is


class BoardRunRequest(BaseModel):
    cases: list[dict] | None = None  # omit for the bundled sample board
    max_cases: int | None = None
    max_tokens: int | None = None
    # Carry-over (carryover.py): name a finished board run to
    # inherit its board — blocked cases re-worked with their loop
    # state, gate cases inherited standing, unworked cases joined
    # fresh, claims turned over. Omit: the run starts clean.
    carry_over_from: str = ""
    # Cases to drop at that acceptance: [{case_id, by, reason}].
    drop: list[dict] = []


@app.get("/samples")
def samples():
    return [case.model_dump() for case in load_sample_cases()]


@app.post("/cases/analyze", status_code=201)
def analyze(case: ExceptionCase, request: Request):
    try:
        run = get_service().run_case(case)
    except ValueError as exc:
        # The intake guard (lifecycle.py): a decommissioned
        # machine's new cases are refused, with the state named.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return run.result.model_dump()


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    record = get_service().get_record(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")
    body = record.model_dump()
    # The plan as it stands at the gate (amendments.py): the
    # original plan above, with the passing amendments applied and
    # marked — plus the case's amendment face, so a reader never
    # has to guess which plan a decision would approve.
    body["effective_plan"] = (
        amendments_module.apply_passing_amendments(record).model_dump()
    )
    body["amendment_state"] = amendments_module.amendment_state(record)
    return body


@app.post("/cases/{case_id}/amendment")
def amend_case(case_id: str, body: AmendmentRequest, request: Request):
    """Record a supervisor's amendment to a waiting case's plan.

    The amendment is re-verified in code at recording
    (amendments.py) — passed or failed, the attempt and its
    verdict land on the record the response returns. Recording
    refusals (not at the gate, anonymous, reasonless, a
    free-text part, a mandatory safety step) are 409s naming
    the refusal; an unknown case is a 404.
    """
    service = get_service()
    try:
        record = service.amend_case(
            case_id,
            kind=body.kind,
            by=body.by,
            reason=body.reason,
            from_part_id=body.from_part_id,
            to_part_id=body.to_part_id,
            step_id=body.step_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return record.model_dump()


@app.post("/cases/{case_id}/decision")
def decide_case(case_id: str, body: DecisionRequest, request: Request):
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=422, detail="decision must be 'approved' or 'rejected'")
    service = get_service()
    try:
        record = service.record_decision(case_id, body.decision, by=body.by, reason=body.reason)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return record.model_dump()


@app.get("/board")
def board():
    """Cases waiting at the supervisor gate, in board order."""
    return get_service().board()


@app.post("/board/run", status_code=201)
def run_board(body: BoardRunRequest, request: Request):
    service = get_service()
    if body.cases is None:
        cases = load_sample_cases()
    else:
        try:
            cases = [ExceptionCase.model_validate(item) for item in body.cases]
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid case payload: {exc}") from None
    policy = BoardPolicy(
        max_cases=body.max_cases if body.max_cases is not None else service.board_policy.max_cases,
        board_token_budget=(
            body.max_tokens if body.max_tokens is not None else service.board_policy.board_token_budget
        ),
    )
    try:
        result = service.run_board(
            cases,
            board_policy=policy,
            carry_over_from=body.carry_over_from or "",
            drop=body.drop or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return result.model_dump()


@app.get("/board/runs/{run_id}")
def get_board_run(run_id: str):
    run = get_service().get_board_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown board run: {run_id}")
    return run


@app.get("/board/runs/{run_id}/carry-over")
def get_carry_over(run_id: str):
    """A board run's carry-over contract: the manifest it offers
    the next shift, and how its own inheritance settled."""
    try:
        return get_service().get_carry_over(run_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown board run: {run_id}"
        ) from None


@app.get("/metrics")
def metrics():
    return get_service().metrics()


# ---------------------------------------------------------------------------
# outcomes, live stats, handoff, board items (closing the loop)
# ---------------------------------------------------------------------------


class OutcomeRequest(BaseModel):
    decision: str  # accepted | accepted_with_changes | rejected
    confirmed_fault: str = ""  # a fault class from the vocabulary, or ""
    work_performed: str = ""
    downtime_minutes_actual: int | None = None
    parts_used: list[str] = []
    resolved: bool | None = None
    by: str = ""


@app.post("/api/cases/{case_id}/outcome")
def record_outcome(case_id: str, body: OutcomeRequest, request: Request):
    """Record what actually happened after the gate.

    Validation is strict on purpose: an outcome is ground truth, and
    ground truth with a typo'd fault class in it poisons every stat
    downstream. Unknown decisions, unknown fault classes and
    negative downtime are 422s — never silently dropped or coerced.
    """
    if body.decision not in OUTCOME_DECISIONS:
        raise HTTPException(
            status_code=422,
            detail=f"decision must be one of {', '.join(OUTCOME_DECISIONS)}",
        )
    confirmed = body.confirmed_fault.strip()
    if confirmed and confirmed not in kb.fault_classes():
        raise HTTPException(
            status_code=422,
            detail=f"unknown fault class: {confirmed!r} (see GET /api/fault-classes)",
        )
    if body.downtime_minutes_actual is not None and body.downtime_minutes_actual < 0:
        raise HTTPException(status_code=422, detail="downtime_minutes_actual cannot be negative")
    outcome = SupervisorOutcome(
        decision=body.decision,
        confirmed_fault=confirmed,
        work_performed=body.work_performed.strip(),
        downtime_minutes_actual=body.downtime_minutes_actual,
        parts_used=[p.strip() for p in body.parts_used if p.strip()],
        resolved=body.resolved,
        recorded_by=body.by.strip(),
    )
    service = get_service()
    try:
        record = service.record_outcome(case_id, outcome)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return record.model_dump()


@app.get("/api/cases/{case_id}/packet")
def case_packet(case_id: str, format: str = "json"):
    """The work-order packet for a gated case (JSON, or ?format=text).

    Only a converged case has a verified work order to packet: a
    non-converged exit (or a draft the supervisor rejected) is a
    409 — the escalation draft on the record is that case's
    document, and an unchecked plan must not reach the floor
    dressed as a work order.
    """
    service = get_service()
    try:
        payload = service.case_packet(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if format == "text":
        return PlainTextResponse(payload["text"])
    return payload


@app.get("/api/stats/diagnosis")
def stats_diagnosis():
    """Live diagnosis accuracy over outcome-carrying records.

    With no outcomes recorded yet the payload says so (``available:
    false`` with the honest note) instead of printing zeroes that
    look like a failing agent.
    """
    return get_service().diagnosis_stats()


@app.get("/api/board/handoff")
def board_handoff(format: str = "json"):
    """The shift handoff for the board as it stands.

    JSON carries every section plus the rendered ``text``; with
    ``?format=text`` the response IS the text, ready for a shift log.
    """
    payload = get_service().handoff()
    if format == "text":
        return PlainTextResponse(payload["text"])
    return payload


@app.get("/api/board/items")
def board_items():
    """Non-case board items (reliability reviews raised by the watch)."""
    return [item.model_dump() for item in get_service().board_items()]


@app.get("/api/allocations")
def allocations():
    """The allocation ledger: part claims, standing and released.

    A converged case claims the parts its plan will draw; later
    cases measure the shelf net of the claims standing. A gate
    rejection releases its case's claims (stamped on the entry);
    an approval leaves them standing.
    """
    return get_service().allocations()


@app.get("/api/parts/outlook")
def parts_outlook(horizon_days: int = outlook_module.DEFAULT_HORIZON_DAYS):
    """The parts outlook (outlook.py): projected demand against the
    shelf over the horizon — every unit traced to the claim, case
    plan or PM cycle that will draw it, the shortage rows named
    with the moment each shortfall bites."""
    if horizon_days < 1:
        raise HTTPException(
            status_code=422, detail="horizon_days must be at least 1"
        )
    return get_service().parts_outlook(horizon_days=horizon_days).model_dump()


@app.get("/api/knowledge/gaps")
def knowledge_gaps():
    """The knowledge-gap report (gaps.py): the blocks the plant's
    own missing knowledge caused, grouped by the missing SOP —
    the asset, the cases it blocked, the rounds it burned, and
    whether it is biting right now. Advisory only: the report
    carries no SOP content and drafts nothing."""
    return get_service().knowledge_gaps().model_dump()


@app.get("/api/pm/proposals")
def pm_proposals():
    """Preventive-maintenance proposals (pm.py).

    Drafted in code when the recurring-fault watch's pattern earns
    one: the cited evidence and the interval arithmetic ride on
    the record, so what the supervisor is asked to decide on is
    fully visible here.
    """
    return [p.model_dump() for p in get_service().pm_proposals()]


@app.get("/api/pm/obligations")
def pm_obligations():
    """Standing PM obligations (pm.py): the approved interval, the
    clock it runs on, the next due point, the open cycle case if
    one is on a board, and the deferral history."""
    return [o.model_dump() for o in get_service().pm_obligations()]


@app.post("/api/pm/proposals/{proposal_id}/decision")
def pm_decision(proposal_id: str, req: PmDecisionRequest):
    """Record the supervisor's decision on a PM proposal.

    Approval creates the standing obligation at the approved
    interval (the request's, else the proposal's); rejection
    requires a reason and spends the cited evidence (the
    watermark — pm.py).
    """
    if req.decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=422,
            detail=f"decision must be approved or rejected, got {req.decision!r}",
        )
    if req.interval_days is not None and req.interval_days < 1:
        raise HTTPException(status_code=422, detail="interval_days must be at least 1")
    try:
        proposal, obligation = get_service().decide_pm_proposal(
            proposal_id,
            req.decision,
            by=req.by,
            reason=req.reason,
            interval_days=req.interval_days,
            last_performed_at=req.last_performed_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "proposal": proposal.model_dump(),
        "obligation": obligation.model_dump() if obligation else None,
    }


@app.post("/api/pm/obligations/{obligation_id}/retire")
def pm_retire(obligation_id: str, req: PmRetireRequest):
    """Retire a standing PM obligation (pm.py).

    A supervisor's act: actor and reason required, retirement
    one-way, and the reason stays on the record — the obligation
    surfaces no further due cases. A machine's decommission
    retires its obligations through the machine endpoint below,
    never through here, so the two endings stay distinguishable.
    """
    try:
        obligation = get_service().retire_pm_obligation(
            obligation_id, by=req.by, reason=req.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return obligation.model_dump()


# ---------------------------------------------------------------------------
# machine lifecycle (lifecycle.py)
# ---------------------------------------------------------------------------


@app.get("/api/machines")
def machines():
    """The plant's machines with their lifecycle state.

    The plant model names the machines; the machine records say
    which of them have been decommissioned, and carry the open
    cases awaiting a disposition.
    """
    service = get_service()
    records = {r.machine_id: r for r in service.machine_records()}
    out = []
    for machine_id, machine in sorted(kb.machines().items()):
        record = records.get(machine_id)
        out.append(
            {
                "machine_id": machine_id,
                "name": machine.get("name", ""),
                "line_id": machine.get("line_id", ""),
                "state": "decommissioned" if record else "in_service",
                "decommission": record.model_dump() if record else None,
                "open_cases": (
                    [r.case_id for r in service.machine_open_cases(machine_id)]
                    if record
                    else []
                ),
            }
        )
    return out


@app.post("/api/machines/{machine_id}/decommission")
def machine_decommission(machine_id: str, req: MachineDecommissionRequest):
    """Decommission a machine (lifecycle.py).

    A supervisor's act: actor and reason required. The machine's
    standing PM obligations retire automatically — linked to this
    record, reason "machine decommissioned" — and its open cases
    are listed on the record for disposition. The act itself
    closes, drops, and decides nothing about any case.
    """
    try:
        record = get_service().decommission_machine(
            machine_id, by=req.by, reason=req.reason, effective_at=req.effective_at
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.model_dump()


@app.post("/api/machines/{machine_id}/dispositions")
def machine_disposition(machine_id: str, req: MachineDispositionRequest):
    """Record the supervisor's disposition of one open case of a
    decommissioned machine (lifecycle.py): keep it on the board,
    or close it as moot. Recorded once per case."""
    if req.disposition not in ("keep", "close_moot"):
        raise HTTPException(
            status_code=422,
            detail=f"disposition must be keep or close_moot, got {req.disposition!r}",
        )
    try:
        record = get_service().dispose_machine_case(
            machine_id, req.case_id, req.disposition, by=req.by
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.model_dump()


# ---------------------------------------------------------------------------
# downtime windows (windows.py)
# ---------------------------------------------------------------------------


@app.get("/api/windows")
def windows():
    """The downtime windows and their plans (windows.py), with
    the planned / proposed / confirmed counts. Each plan shows
    its sections — packed, deferred, oversized, unestimated,
    booked elsewhere — and the minutes arithmetic, as computed."""
    service = get_service()
    return {
        "windows": [w.model_dump() for w in service.windows()],
        "counts": service.window_counts(),
        "ledger": service.window_ledger(),
    }


@app.post("/api/windows")
def window_declare(req: WindowDeclareRequest):
    """Declare a downtime window and compute its first plan.

    A supervisor's act: ``by`` is required, the scope must name
    lines and/or machines from the plant model (an unknown id is
    refused), and the plan packs the work-ready cases and due PM
    obligations in board priority order in the same act. The
    plan is a proposal until confirmed.
    """
    try:
        window = get_service().declare_window(
            by=req.by,
            starts_at=req.starts_at,
            length_min=req.length_min,
            line_ids=req.line_ids,
            machine_ids=req.machine_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return window.model_dump()


@app.get("/api/windows/{window_id}")
def window_get(window_id: str, format: str = "json"):
    """One window and its plan (JSON, or ``?format=text`` for the
    shift-log rendering)."""
    window = get_service().get_window(window_id)
    if window is None:
        raise HTTPException(status_code=404, detail=f"unknown window: {window_id}")
    if format == "text":
        return PlainTextResponse(windows_module.render_text(window))
    return window.model_dump()


@app.get("/api/windows/{window_id}/review")
def window_review(window_id: str, format: str = "json"):
    """A confirmed window, answered for (windows.py): each packed
    item performed, deferred, or not worked, read off the records
    as they stand, with the plan's minutes restated by state.

    The review opens only when the span has closed: a proposed
    window has no promise to answer for, and a window still
    running refuses (409) rather than naming work unworked while
    it is in progress. ``?format=text`` renders the shift-log
    form."""
    try:
        review = get_service().window_review(window_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if format == "text":
        return PlainTextResponse(windows_module.render_review_text(review))
    return review


@app.post("/api/windows/{window_id}/plan")
def window_plan(window_id: str):
    """Recompute a proposed window's plan from the records as
    they stand — the remedy a confirmation drift refusal names.
    A confirmed window is never re-planned (409)."""
    try:
        window = get_service().plan_window(window_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return window.model_dump()


@app.post("/api/windows/{window_id}/confirm")
def window_confirm(window_id: str, req: WindowConfirmRequest):
    """Confirm a window's plan (windows.py).

    The state the plan was computed over is re-read first: drift
    — a member decided, outcomed, dropped, or stripped of its
    claims since the computation — refuses with the drift named
    (409). Confirmation itself moves no case; it books the
    plan's packed items against overlapping windows.
    """
    try:
        window = get_service().confirm_window(window_id, by=req.by)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return window.model_dump()


@app.get("/api/fault-classes")
def fault_classes():
    """The fault vocabulary diagnoses and confirmed faults speak."""
    return {"fault_classes": kb.fault_classes()}


@app.get("/evals/results")
def eval_results():
    packaged = Path(str(resources.files(__package__).joinpath("data/eval_results.json")))
    if packaged.is_file():
        return json.loads(packaged.read_text(encoding="utf-8"))
    return {"available": False, "note": "run `make evals` to generate results"}


@app.get("/evals/gate-baseline")
def gate_baseline():
    """The reasoner regression gate's committed baseline record
    (regression.py): the offline configuration's run over the fixed
    suite, with the gate criteria and the exit severity order a
    candidate configuration is judged against. Regenerated — in the
    open — by `uv run python evals/run_gate.py run --backend offline`.
    """
    from . import regression

    packaged = Path(str(resources.files(__package__).joinpath("data/gate_baseline.json")))
    if packaged.is_file():
        record = json.loads(packaged.read_text(encoding="utf-8"))
        return {
            "record": record,
            "criteria": regression.GATE_CRITERIA,
            "severity_order": sorted(regression.EXIT_SEVERITY, key=regression.EXIT_SEVERITY.get),
            "pin_violations": regression.pin_violations(record),
        }
    return {"available": False, "note": "run `uv run python evals/run_gate.py run` to generate the baseline"}
