"""Helpers for producing OpenAI-compatible JSON error responses."""

from fastapi.responses import JSONResponse


def error_response(
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "invalid_request",
    param: str | None = None,
    status_code: int = 400,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": param, "code": code}},
    )


class APIError(Exception):
    """OpenAI-style JSON error raised inside route handlers or dependencies."""
    def __init__(self, message: str, error_type: str, code: str, status_code: int,
                 param: str | None = None, headers: dict | None = None):
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
