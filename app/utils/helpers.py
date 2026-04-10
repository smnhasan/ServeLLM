import json
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi.responses import StreamingResponse

from app.schemas.openai import (
    ChatCompletionStreamResponse,
    ChatStreamChoice,
    DeltaMessage,
    ErrorDetail,
    ErrorResponse,
)


def openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 400,
) -> tuple[ErrorResponse, int]:
    """Create an OpenAI-compatible error response."""
    return (
        ErrorResponse(error=ErrorDetail(message=message, type=error_type, param=param, code=code)),
        status_code,
    )


async def stream_chat_response(
    token_generator: AsyncGenerator[str, None],
    model: str,
    completion_id: Optional[str] = None,
) -> StreamingResponse:
    """Wrap a token generator in an SSE StreamingResponse."""
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def generate():
        # First chunk: role delta
        first_chunk = ChatCompletionStreamResponse(
            id=cid,
            created=created,
            model=model,
            choices=[
                ChatStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Content chunks
        async for token in token_generator:
            chunk = ChatCompletionStreamResponse(
                id=cid,
                created=created,
                model=model,
                choices=[
                    ChatStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=token),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=cid,
            created=created,
            model=model,
            choices=[
                ChatStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def estimate_tokens(text: str) -> int:
    """Quick token estimation (~4 chars per token)."""
    return max(1, len(text) // 4)


def normalize_stop(stop) -> Optional[list]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop
