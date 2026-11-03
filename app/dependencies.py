
from fastapi import Header, Request, status

from app.services.auth import auth_service
from app.utils.errors import APIError

__all__ = ["APIError", "get_current_api_key"]


async def get_current_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    is_valid, api_key, error_msg = auth_service.verify_key(authorization)

    if not is_valid:
        raise APIError(
            message=error_msg,
            error_type="invalid_request_error",
            code="invalid_api_key",
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    allowed, retry_after = auth_service.check_rate_limit(api_key)
    if not allowed:
        raise APIError(
            message=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
        )

    return api_key
