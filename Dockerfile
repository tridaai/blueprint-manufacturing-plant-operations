# syntax=docker/dockerfile:1
#
# Multi-stage build for the plant operations loop agent.
#   docker build -t plant-operations-loop .
#   docker compose up        (full production topology)
# Only the API service runs this image; PostgreSQL/pgvector and the S3
# gateway are their own compose services (docker-compose.yml).
# Same multi-stage pattern as the sibling blueprints; verify with
# `docker compose up --build`, then `curl localhost:8000/health`.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

# Frozen dependency set (see constraints.txt header for how it is made).
COPY constraints.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r constraints.txt

# The package itself, on top of the frozen set (--no-deps: the
# constraints file already resolved everything, including the [llm]
# extra's openai/anthropic SDKs).
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps ".[llm]"

FROM python:3.12-slim AS runtime

# Non-root runtime user; the app needs no privileges and writes nothing
# to disk (state lives in PostgreSQL and the S3 bucket).
RUN useradd --uid 10001 --create-home app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src ./src
COPY evals ./evals
COPY migrations ./migrations

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MIGRATIONS_DIR=/app/migrations

USER app
EXPOSE 8000

# /health is the cheap liveness probe (no store access); /readiness is
# for orchestrators that want the deeper check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

LABEL org.opencontainers.image.title="Trida AI Blueprint — Plant Operations Loop Agent" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.trida.blueprint="manufacturing-plant-operations"

CMD ["uvicorn", "plant_loop.api:app", "--host", "0.0.0.0", "--port", "8000"]
