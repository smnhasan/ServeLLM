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
