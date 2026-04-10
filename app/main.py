import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.dependencies import APIError
from app.routers.v1 import chat, completions, embeddings, models


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[LLM Serve] Starting up — version {settings.APP_VERSION}")
    print(f"[LLM Serve] Dummy mode: {settings.DUMMY_MODE}")
    print(f"[LLM Serve] Rate limiting: {settings.RATE_LIMIT_ENABLED} ({settings.RATE_LIMIT_RPM} RPM)")
    yield
    print("[LLM Serve] Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "OpenAI-compatible LLM serving API. "
        "Supports chat completions, text completions, embeddings, and model listing. "
        "Includes API key authentication and rate limiting."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request timing middleware ────────────────────────────────────────────────

@app.middleware("http")
async def api_error_middleware(request: Request, call_next):
    """Catch APIError raised inside Depends() before Starlette wraps it."""
    try:
        return await call_next(request)
    except APIError as exc:
        return exc.to_response()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
    return response

# ─── Exception handlers ───────────────────────────────────────────────────────

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return exc.to_response()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    param = ".".join(str(x) for x in first.get("loc", [])[1:]) or None
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": first.get("msg", "Validation error"),
                "type": "invalid_request_error",
                "param": param,
                "code": "invalid_request",
            }
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"No handler found for {request.method} {request.url.path}",
                "type": "invalid_request_error",
                "code": "not_found",
            }
        },
    )


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(chat.router, prefix=settings.API_V1_PREFIX, tags=["Chat"])
app.include_router(completions.router, prefix=settings.API_V1_PREFIX, tags=["Completions"])
app.include_router(models.router, prefix=settings.API_V1_PREFIX, tags=["Models"])
app.include_router(embeddings.router, prefix=settings.API_V1_PREFIX, tags=["Embeddings"])


# ─── Health / Info endpoints ──────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "dummy_mode": settings.DUMMY_MODE,
    }


@app.get("/", tags=["System"], summary="Root / API info")
async def root() -> Dict[str, Any]:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "OpenAI-compatible LLM serving API",
        "docs": "/docs",
        "endpoints": {
            "chat_completions": f"{settings.API_V1_PREFIX}/chat/completions",
            "completions": f"{settings.API_V1_PREFIX}/completions",
            "models": f"{settings.API_V1_PREFIX}/models",
            "embeddings": f"{settings.API_V1_PREFIX}/embeddings",
        },
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
