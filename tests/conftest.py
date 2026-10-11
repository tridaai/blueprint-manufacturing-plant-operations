"""Pytest fixtures: hermetic tests, whatever the developer's shell holds.

The suite must behave identically on a bare machine and on one with
real keys in the environment (or a populated .env): a Muse key
leaking in from the shell would flip backend selection and turn unit
tests into live calls. So an autouse fixture scrubs every variable
the package reads (by exact name, by provider prefix, and by suffix)
before each test, and restores them afterwards.
"""

from __future__ import annotations

import os

import pytest

_EXACT = {
    "REASONING_BACKEND",
    "LLM_PROVIDER",
    "LLM_TIMEOUT_SECONDS",
    "MODEL_API_KEY",
    "MODEL_API_BASE_URL",
    "MODEL_SURFACE",
    "MUSE_MODEL",
    "EMBEDDER",
    "EMBEDDING_MODEL",
    "DATABASE_URL",
    "CHECKPOINTS",
    "MIGRATIONS_DIR",
    "API_KEY",
    "S3_BUCKET",
    "S3_ENDPOINT_URL",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "LOOP_MAX_ROUNDS",
    "LOOP_CASE_TOKEN_BUDGET",
    "BOARD_MAX_CASES",
    "BOARD_TOKEN_BUDGET",
    "EVAL_LIVE",
}

_PREFIXES = ("OPENAI_", "ANTHROPIC_", "OLLAMA_", "MODEL_", "MUSE_")
_SUFFIXES = ("_API_KEY", "_MODEL", "_BASE_URL")


def _is_agent_env(name: str) -> bool:
    return (
        name in _EXACT
        or name.startswith(_PREFIXES)
        or name.endswith(_SUFFIXES)
    )


@pytest.fixture(autouse=True)
def scrub_env(monkeypatch):
    saved = {k: v for k, v in os.environ.items() if _is_agent_env(k)}
    for key in saved:
        monkeypatch.delenv(key, raising=False)
    yield
    for key, value in saved.items():
        os.environ[key] = value
