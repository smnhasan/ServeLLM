import os

# Must be set before the app is imported: tests never load real models,
# and the shared session client would otherwise trip the rate limiter.
os.environ["DUMMY_MODE"] = "true"
os.environ["DUMMY_LATENCY_MS"] = "0"
os.environ["STREAM_CHUNK_DELAY_MS"] = "0"
os.environ["RATE_LIMIT_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

# ─── Valid test keys (from default config) ───────────────────────────────────
VALID_KEY = "sk-llmserve-test-key-1234"
INVALID_KEY = "sk-invalid-key-0000"

AUTH_HEADERS = {"Authorization": f"Bearer {VALID_KEY}"}
INVALID_HEADERS = {"Authorization": f"Bearer {INVALID_KEY}"}
NO_HEADERS: dict = {}


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers():
    return AUTH_HEADERS


@pytest.fixture(scope="session")
def invalid_headers():
    return INVALID_HEADERS
