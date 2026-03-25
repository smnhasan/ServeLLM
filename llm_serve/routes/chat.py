"""
llm_serve.routes.chat — /v1/chat/completions route factory.
"""

import json
import time
import uuid
import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm_serve.models import ChatCompletionRequest
from llm_serve.backends import LLMBackend

logger = logging.getLogger(__name__)


def make_chat_router(llm: LLMBackend, semaphore: asyncio.Semaphore) -> APIRouter:
    router = APIRouter()

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_response(request: ChatCompletionRequest, text: str, prompt: str) -> dict:
        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   request.model,
            "choices": [
                {
                    "index":   0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens":     len(prompt.split()),
                "completion_tokens": len(text.split()),
                "total_tokens":      len(prompt.split()) + len(text.split()),
            },
        }

    # ── non-streaming ─────────────────────────────────────────────────────

    async def _complete(request: ChatCompletionRequest) -> dict:
        async with semaphore:
            try:
                prompt = llm.messages_to_prompt(request.messages)
                resp   = llm.complete(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
                text = resp["choices"][0]["text"].strip()
                return _build_response(request, text, prompt)
            except Exception as exc:
                logger.error("Chat completion error: %s", exc)
                raise HTTPException(status_code=500, detail=str(exc))

    # ── streaming ─────────────────────────────────────────────────────────

    async def _stream(request: ChatCompletionRequest):
        async with semaphore:
            try:
                prompt  = llm.messages_to_prompt(request.messages)
                cid     = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())

                # opening chunk with role
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": request.model,
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant", "content": ""},
                                 "finish_reason": None}
                            ],
                        }
                    )
                    + "\n\n"
                )

                for output in llm.stream(
                    prompt,
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
                                "id": cid, "object": "chat.completion.chunk",
                                "created": created, "model": request.model,
                                "choices": [
                                    {"index": 0, "delta": {"content": token},
                                     "finish_reason": None}
                                ],
                            }
                        )
                        + "\n\n"
                    )

                # stop chunk
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid, "object": "chat.completion.chunk",
                            "created": created, "model": request.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                    )
                    + "\n\n"
                )
                yield "data: [DONE]\n\n"

            except Exception as exc:
                logger.error("Chat stream error: %s", exc)
                yield (
                    "data: "
                    + json.dumps({"error": {"message": str(exc), "type": "internal_error"}})
                    + "\n\n"
                )

    # ── endpoint ──────────────────────────────────────────────────────────

    @router.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        if request.stream:
            return StreamingResponse(_stream(request), media_type="text/event-stream")
        return await _complete(request)

    return router
