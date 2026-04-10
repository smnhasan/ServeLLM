import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

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
