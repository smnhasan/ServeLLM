import pytest
from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS, INVALID_HEADERS


class TestAuthentication:
    """Tests for API key authentication and rate limiting."""

    def test_no_auth_header_returns_401(self, client: TestClient):
        r = client.get("/v1/models")
        assert r.status_code == 401
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "invalid_api_key"

    def test_invalid_api_key_returns_401(self, client: TestClient):
        r = client.get("/v1/models", headers=INVALID_HEADERS)
        assert r.status_code == 401
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"

    def test_valid_api_key_succeeds(self, client: TestClient):
        r = client.get("/v1/models", headers=AUTH_HEADERS)
        assert r.status_code == 200

    def test_bearer_prefix_required_or_raw(self, client: TestClient):
        # Raw key without Bearer prefix should also work
        raw_headers = {"Authorization": "sk-llmserve-test-key-1234"}
        r = client.get("/v1/models", headers=raw_headers)
        assert r.status_code == 200

    def test_malformed_bearer_returns_401(self, client: TestClient):
        headers = {"Authorization": "Bearer"}
        r = client.get("/v1/models", headers=headers)
        # "Bearer" alone with no token — returns 401
        assert r.status_code == 401

    def test_wrong_scheme_returns_401(self, client: TestClient):
        headers = {"Authorization": "Basic dXNlcjpwYXNz"}
        r = client.get("/v1/models", headers=headers)
        assert r.status_code == 401


class TestSecurity:
    """Tests for security helpers."""

    def test_extract_bearer_token(self):
        from app.core.security import extract_bearer_token
        assert extract_bearer_token("Bearer abc123") == "abc123"
        assert extract_bearer_token("abc123") == "abc123"
        assert extract_bearer_token("") is None
        assert extract_bearer_token(None) is None
        assert extract_bearer_token("Bearer") is None

    def test_hash_api_key(self):
        from app.core.security import hash_api_key
        h1 = hash_api_key("mykey")
        h2 = hash_api_key("mykey")
        h3 = hash_api_key("otherkey")
        assert h1 == h2
        assert h1 != h3

    def test_verify_api_key(self):
        from app.core.security import hash_api_key, verify_api_key
        plain = "sk-test-key"
        hashed = hash_api_key(plain)
        assert verify_api_key(plain, hashed) is True
        assert verify_api_key("wrong-key", hashed) is False

    def test_generate_api_key_format(self):
        from app.core.security import generate_api_key
        key = generate_api_key()
        assert key.startswith("sk-llmserve-")
        assert len(key) > 20

    def test_get_valid_keys(self):
        from app.core.security import get_valid_keys
        keys = get_valid_keys()
        assert isinstance(keys, list)
        assert len(keys) >= 1
        assert "sk-llmserve-test-key-1234" in keys
