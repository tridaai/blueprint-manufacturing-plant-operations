"""Infrastructure modules: migrations, dotenv, object store, errors, KB."""

from __future__ import annotations

import io
import urllib.request

import pytest

from plant_loop import config, errors, kb
from plant_loop.migrations import migration_files
from plant_loop.objectstore import InMemoryObjectStore, S3ObjectStore


def test_migrations_are_numbered_and_cover_the_schema():
    files = migration_files()
    assert [f.name for f in files] == ["0001_case_records.sql", "0002_kb_embeddings.sql"]
    sql = files[0].read_text()
    for table in ("case_records", "board_runs", "run_errors"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "vector" in files[1].read_text()


def test_dotenv_loader(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nPLANT_TEST_A=from-file\nPLANT_TEST_B = spaced \n")
    monkeypatch.delenv("PLANT_TEST_A", raising=False)
    monkeypatch.setenv("PLANT_TEST_B", "from-shell")
    config.load_dotenv(env_file)
    assert config.env_str("PLANT_TEST_A") == "from-file"
    assert config.env_str("PLANT_TEST_B") == "from-shell"  # never overrides


def test_in_memory_object_store_roundtrip():
    store = InMemoryObjectStore()
    store.put("cases/A.json", b"{}")
    assert store.get("cases/A.json") == b"{}"
    assert store.get("cases/missing.json") is None
    assert store.exists("cases/A.json") is True
    assert store.exists("cases/missing.json") is False


def test_s3_object_store_signs_requests(monkeypatch):
    captured = {}

    class _Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=0):
        captured["request"] = request
        if request.get_method() == "PUT":
            return _Response(b"")
        return _Response(b"payload-bytes")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    store = S3ObjectStore(
        bucket="case-payloads", endpoint_url="http://minio:8333",
        access_key="AKIDEXAMPLE", secret_key="secret", region="us-east-1",
    )
    store.put("cases/A.json", b"{}")
    request = captured["request"]
    assert request.get_method() == "PUT"
    auth = request.headers["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "s3/aws4_request" in auth
    assert store.get("cases/A.json") == b"payload-bytes"


def test_provider_error_translation():
    class FakeStatusError(Exception):
        status_code = 401

    class APIConnectionError(Exception):
        pass

    translated = errors.translate_provider_error(
        FakeStatusError("bad key supplied"), backend="muse", base_url="https://api.meta.ai/v1"
    )
    message = str(translated)
    assert "muse" in message and "auth" in message
    assert "bad key supplied" in message  # the provider detail survives
    assert "https://api.meta.ai/v1" in message  # …and so does the endpoint tried
    translated = errors.translate_provider_error(
        APIConnectionError("refused"), backend="muse", base_url="https://api.meta.ai/v1"
    )
    assert "connection" in str(translated)
    original = errors.ProviderError("already translated")
    assert errors.translate_provider_error(original, backend="muse", base_url=None) is original


def test_kb_shape_and_integrity():
    assert kb.plant_name() == "Meridian Fabrication, Plant 2"
    assert len(kb.lines()) == 5 and len(kb.machines()) == 7
    machine_ids = set(kb.machines())
    for sop in kb.sops():
        assert sop["machine_id"] in machine_ids or sop["machine_id"] == ""
        if sop["sop_id"] != "SOP-MATL-EXP":
            assert sop["mandatory_safety_steps"], f"{sop['sop_id']} has no mandatory safety steps"
    history = kb.load_kb()["history"]
    for entry in history:
        assert entry["machine_id"] in machine_ids
    chunks = kb.kb_chunks()
    assert len(chunks) == len(kb.sops()) + len(history)
    assert all(chunk["text"] for chunk in chunks)


def test_not_configured_error_names_the_fix():
    err = errors.not_configured_error("muse", "MODEL_API_KEY", "Create a key at dev.meta.ai.")
    assert "MODEL_API_KEY" in str(err) and "dev.meta.ai" in str(err)
