import time
from collections import defaultdict, deque
from typing import Optional, Tuple

from app.core.config import settings
from app.core.security import extract_bearer_token, get_valid_keys, verify_api_key, hash_api_key


# ─── Simple in-memory rate limiter (sliding window) ──────────────────────────

class RateLimiter:
    """Per-key sliding-window rate limiter (in-memory, single-process)."""

    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).
        """
        now = time.monotonic()
        window = self._windows[key]

        # Drop timestamps older than 60 seconds
        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self.rpm:
            retry_after = int(60 - (now - window[0])) + 1
            return False, retry_after

        window.append(now)
        return True, 0

    def get_usage(self, key: str) -> dict:
        now = time.monotonic()
        window = self._windows[key]
        while window and now - window[0] > 60:
            window.popleft()
        count = len(window)
        return {
            "requests_in_last_minute": count,
            "limit": self.rpm,
            "remaining": max(0, self.rpm - count),
        }


rate_limiter = RateLimiter(rpm=settings.RATE_LIMIT_RPM)


# ─── Auth Service ─────────────────────────────────────────────────────────────

class AuthService:
    def __init__(self):
        self._valid_keys = get_valid_keys()

    def reload_keys(self):
        """Reload keys from config (useful for hot-reload in production)."""
        self._valid_keys = get_valid_keys()

    def verify_key(self, authorization: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify the API key from the Authorization header.

        Returns:
            (is_valid, api_key, error_message)
        """
        if not authorization:
            return False, None, "Missing Authorization header. Provide: Authorization: Bearer <api-key>"

        token = extract_bearer_token(authorization)
        if not token:
            return False, None, "Invalid Authorization header format. Use: Bearer <api-key>"

        if settings.AUTH_USE_HASHED:
            # Compare against hashed keys stored in config
            token_hash = hash_api_key(token)
            for stored_key in self._valid_keys:
                if hmac.compare_digest(token_hash, stored_key):
                    return True, token, None
        else:
            # Plain comparison (dev/demo mode)
            if token in self._valid_keys:
                return True, token, None

        return False, None, "Invalid API key. Check your credentials."

    def check_rate_limit(self, api_key: str) -> Tuple[bool, int]:
        """
        Check rate limit for the given key.

        Returns:
            (allowed, retry_after_seconds)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, 0
        return rate_limiter.is_allowed(api_key)

    def get_rate_limit_info(self, api_key: str) -> dict:
        return rate_limiter.get_usage(api_key)


auth_service = AuthService()
