import hashlib
import hmac
import secrets
from typing import Optional
from app.core.config import settings


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256 for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Constant-time comparison to verify an API key against its hash."""
    return hmac.compare_digest(hash_api_key(plain_key), hashed_key)


def generate_api_key(prefix: str = "sk-llmserve") -> str:
    """Generate a new random API key."""
    token = secrets.token_urlsafe(32)
    return f"{prefix}-{token}"


def extract_bearer_token(authorization: str) -> Optional[str]:
    """Extract the token from a 'Bearer <token>' header value."""
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    # Allow raw key (no "Bearer" prefix), but reject bare "bearer" word
    if len(parts) == 1 and parts[0].lower() != "bearer":
        return parts[0]
    return None


def get_valid_keys() -> list[str]:
    """Return the list of valid API keys from config."""
    return [k.strip() for k in settings.VALID_API_KEYS.split(",") if k.strip()]
