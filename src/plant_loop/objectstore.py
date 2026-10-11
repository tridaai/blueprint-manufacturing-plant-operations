"""Object storage for raw case payloads (S3-compatible, stdlib SigV4).

The raw case JSON, exactly as submitted, is archived under
``cases/<case_id>.json`` so the record the loop reasoned over can
always be re-read byte-for-byte. The production implementation speaks
S3 with a hand-rolled AWS SigV4 signer (no SDK dependency — the same
pattern as the sibling blueprints): it works against AWS S3 and any
S3-compatible gateway (the compose stack runs SeaweedFS's S3 gateway
as the ``minio`` service). The in-memory double serves tests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .config import env_str, load_dotenv


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, date_stamp: str, region: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret).encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, "s3")
    return _hmac_sha256(k_service, "aws4_request")


class S3ObjectStore:
    """SigV4-signed S3 PUT/GET against a bucket + endpoint."""

    def __init__(self, bucket: str, endpoint_url: str, access_key: str,
                 secret_key: str, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.endpoint = endpoint_url.rstrip("/")
        self._access = access_key
        self._secret = secret_key
        self._region = region

    def _signed_request(self, method: str, key: str, data: bytes | None):
        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(data or b"").hexdigest()
        host = self.endpoint.split("://", 1)[-1]
        canonical_uri = f"/{self.bucket}/{key}"
        canonical_headers = f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join(
            [method, canonical_uri, "", canonical_headers, signed_headers, payload_hash]
        )
        scope = f"{date_stamp}/{self._region}/s3/aws4_request"
        string_to_sign = "\n".join(
            ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
        )
        signature = hmac.new(
            _signing_key(self._secret, date_stamp, self._region),
            string_to_sign.encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        headers = {
            "x-amz-date": amz_date,
            "x-amz-content-sha256": payload_hash,
            "Authorization": (
                f"AWS4-HMAC-SHA256 Credential={self._access}/{scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
        }
        request = urllib.request.Request(
            f"{self.endpoint}{canonical_uri}", data=data, method=method, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def ensure_bucket(self) -> None:
        status, _ = self._signed_request("PUT", "", None)
        # 200 = created; 409/301 = already exists under this account.
        if status not in (200, 301, 409):
            raise RuntimeError(f"S3 ensure-bucket failed (HTTP {status}) for bucket {self.bucket!r}")

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        status, _ = self._signed_request("PUT", key, data)
        if status not in (200, 201):
            raise RuntimeError(f"S3 put failed (HTTP {status}) for {key!r}")
        return f"s3://{self.bucket}/{key}"

    def get(self, key: str) -> bytes | None:
        status, body = self._signed_request("GET", key, None)
        return body if status == 200 else None

    def exists(self, key: str) -> bool:
        status, _ = self._signed_request("HEAD", key, None)
        return status == 200


class InMemoryObjectStore:
    """The test double: objects in a dict."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str = "text/plain") -> str:
        self._objects[key] = data
        return f"memory://{key}"

    def get(self, key: str) -> bytes | None:
        return self._objects.get(key)

    def exists(self, key: str) -> bool:
        return key in self._objects


def default_object_store():
    """S3 when a bucket is configured, else the in-memory double."""
    load_dotenv()
    bucket = env_str("S3_BUCKET", "")
    if not bucket:
        return InMemoryObjectStore()
    return S3ObjectStore(
        bucket=bucket,
        endpoint_url=env_str("S3_ENDPOINT_URL", "http://localhost:8333"),
        access_key=env_str("S3_ACCESS_KEY_ID", "minioadmin"),
        secret_key=env_str("S3_SECRET_ACCESS_KEY", "minioadmin"),
        region=env_str("S3_REGION", "us-east-1"),
    )


def payload_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=1, sort_keys=True).encode("utf-8")
