"""Provider error translation.

Raw SDK/HTTP exceptions (``openai.APIConnectionError``, an httpx stack,
a Meta Model API 429 body, …) are what a library should raise and what a
product must never show. Every outbound call in the reasoning backends
is wrapped so a failure becomes a :class:`ProviderError` in the
product's ``error: …`` style: it names the backend, the endpoint it
tried, and the likely fix. A reasoning call cannot degrade silently —
there is no "the model shrugged" state in a case record — so the
service surfaces the translated message (CLI/demo exit 1, API 502)
instead of recording a guess.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """A provider call failed; the message says what to do about it."""


def _failure_kind(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (401, 403) or "unauthorized" in message or "authentication" in message:
        return "auth"
    if status == 404 or "not found" in message:
        return "not_found"
    if "connect" in name or "connection" in message or "refused" in message or "unreachable" in message:
        return "connection"
    if "timeout" in name or "timed out" in message or "timeout" in message:
        return "timeout"
    return "provider"


_FIX_HINTS = {
    ("muse", "auth"): (
        "check MODEL_API_KEY — keys are created in the Meta Model API "
        "dashboard (dev.meta.ai → API keys); a Spark 1.3 key on the "
        "standard tier is not interchangeable with a contributor key"
    ),
    ("muse", "not_found"): (
        "check MODEL_API_BASE_URL and MUSE_MODEL — the model id must be an "
        "exact id such as muse-spark-1.3; a 404 here is usually a typo'd model"
    ),
    ("ollama", "connection"): (
        "start Ollama (`ollama serve`), pull the model (`ollama pull llama3.1`), "
        "or point OLLAMA_BASE_URL at a running server"
    ),
    ("openai", "connection"): (
        "check OPENAI_BASE_URL and this machine's network/proxy settings"
    ),
    ("anthropic", "connection"): (
        "check ANTHROPIC_BASE_URL and this machine's network/proxy settings"
    ),
}


def _fix_hint(backend: str, kind: str) -> str:
    specific = _FIX_HINTS.get((backend, kind))
    if specific:
        return specific
    if kind == "auth":
        return "check that the API key in .env / the environment is valid for this endpoint"
    if kind == "timeout":
        return (
            "the endpoint did not answer within LLM_TIMEOUT_SECONDS — raise it, "
            "or check the endpoint's health"
        )
    if kind == "connection":
        return f"check the {backend} endpoint address and this machine's network connection"
    return "see the provider detail above; check the backend configuration in .env"


def _detail_of(exc: Exception) -> str:
    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return detail[:237] + "..." if len(detail) > 240 else detail


def translate_provider_error(
    exc: Exception, *, backend: str, base_url: str | None
) -> ProviderError:
    """Turn a raw provider/SDK exception into an actionable ProviderError."""
    if isinstance(exc, ProviderError):
        return exc
    kind = _failure_kind(exc)
    endpoint = base_url or "the provider default endpoint"
    return ProviderError(
        f"{backend} reasoning call failed ({kind}): {_detail_of(exc)} — endpoint: {endpoint}. "
        f"Likely fix: {_fix_hint(backend, kind)}."
    )


def not_configured_error(backend: str, variable: str, extra: str = "") -> ProviderError:
    """A backend was selected but its credential is missing.

    This is a configuration error, not a runtime degradation: the run
    fails loudly with the fix, and never substitutes another backend.
    """
    return ProviderError(
        f"{backend} reasoning backend is not configured: {variable} is not set. "
        f"Add {variable}=<your key> to the .env file in the repo root "
        "(copy .env.example to .env) or export it in your shell, then re-run."
        + (f" {extra}" if extra else "")
    )
