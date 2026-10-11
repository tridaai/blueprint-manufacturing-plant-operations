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
- ``POST /cases/{id}/decision`` records approve/reject — the ONLY
  way a case leaves ``awaiting_supervisor``;
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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .config import env_str, load_dotenv
from .errors import ProviderError
from .loops import BoardPolicy
from .samples import load_sample_cases
from .schemas import ExceptionCase
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


class BoardRunRequest(BaseModel):
    cases: list[dict] | None = None  # omit for the bundled sample board
    max_cases: int | None = None
    max_tokens: int | None = None


@app.get("/samples")
def samples():
    return [case.model_dump() for case in load_sample_cases()]


@app.post("/cases/analyze", status_code=201)
def analyze(case: ExceptionCase, request: Request):
    run = get_service().run_case(case)
    return run.result.model_dump()


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    record = get_service().get_record(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")
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
    result = service.run_board(cases, board_policy=policy)
    return result.model_dump()


@app.get("/board/runs/{run_id}")
def get_board_run(run_id: str):
    run = get_service().get_board_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown board run: {run_id}")
    return run


@app.get("/metrics")
def metrics():
    return get_service().metrics()


@app.get("/evals/results")
def eval_results():
    packaged = Path(str(resources.files(__package__).joinpath("data/eval_results.json")))
    if packaged.is_file():
        return json.loads(packaged.read_text(encoding="utf-8"))
    return {"available": False, "note": "run `make evals` to generate results"}
