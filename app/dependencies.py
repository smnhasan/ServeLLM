from typing import Optional
from fastapi import Header, Request, status
from fastapi.responses import JSONResponse

from app.services.auth import auth_service


class APIError(Exception):
    """OpenAI-style JSON error raised inside route handlers or dependencies."""
    def __init__(self, message: str, error_type: str, code: str, status_code: int,
                 param: str = None, headers: dict = None):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.status_code = status_code
        self.param = param
        self.extra_headers = headers or {}

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content={"error": {"message": self.message, "type": self.error_type,
                               "param": self.param, "code": self.code}},
            headers=self.extra_headers,
        )


async def get_current_api_key(
    request: Request,
    authorization: Optional[str] = Header(default=None),
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
