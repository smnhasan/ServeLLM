"""
llm_serve.routes.completions — /v1/completions route factory.
"""

import json
import time
import uuid
import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm_serve.models import CompletionRequest
from llm_serve.backends import LLMBackend

logger = logging.getLogger(__name__)


def make_completions_router(llm: LLMBackend, semaphore: asyncio.Semaphore) -> APIRouter:
    router = APIRouter()

    # ── non-streaming ─────────────────────────────────────────────────────

    async def _complete(request: CompletionRequest) -> dict:
        async with semaphore:
            try:
                resp = llm.complete(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
                text = resp["choices"][0]["text"]
                return {
                    "id":      f"cmpl-{uuid.uuid4().hex[:8]}",
                    "object":  "text_completion",
                    "created": int(time.time()),
                    "model":   request.model,
                    "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens":     len(request.prompt.split()),
                        "completion_tokens": len(text.split()),
                        "total_tokens":      len(request.prompt.split()) + len(text.split()),
                    },
                }
            except Exception as exc:
                logger.error("Completion error: %s", exc)
                raise HTTPException(status_code=500, detail=str(exc))

    # ── streaming ─────────────────────────────────────────────────────────

    async def _stream(request: CompletionRequest):
        async with semaphore:
            try:
                cid     = f"cmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())

                for output in llm.stream(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                ):
                    token = output["choices"][0]["text"]
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": cid, "object": "text_completion",
                                "created": created, "model": request.model,
                                "choices": [{"text": token, "index": 0, "finish_reason": None}],
                            }
                        )
                        + "\n\n"
                    )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid, "object": "text_completion",
                            "created": created, "model": request.model,
                            "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            except Exception as exc:
                logger.error("Completion stream error: %s", exc)
                yield (
                    "data: "
                    + json.dumps({"error": {"message": str(exc), "type": "internal_error"}})
                    + "\n\n"
                )

    # ── endpoint ──────────────────────────────────────────────────────────

    @router.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        if request.stream:
            return StreamingResponse(_stream(request), media_type="text/event-stream")
        return await _complete(request)

    return router
