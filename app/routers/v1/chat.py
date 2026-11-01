import time
import uuid

from fastapi import APIRouter, Depends

from app.dependencies import get_current_api_key
from app.schemas.openai import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    MessageRole,
    UsageInfo,
)
from app.services.inference import get_engine
from app.services.model_manager import model_manager
from app.utils.errors import error_response
from app.utils.helpers import estimate_tokens, normalize_stop, stream_chat_response

router = APIRouter()


@router.post(
    "/chat/completions",
    summary="Create chat completion",
    description="Creates a model response for the given chat conversation. Compatible with OpenAI API.",
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_current_api_key),
):
    if not model_manager.model_exists(request.model):
        return error_response(f"The model '{request.model}' does not exist.", param="model", code="model_not_found", status_code=404)

    if not model_manager.has_capability(request.model, "chat"):
        return error_response(f"Model '{request.model}' does not support chat completions.", param="model", code="model_not_supported")

    stop = normalize_stop(request.stop)
    model_max = model_manager.get_max_tokens(request.model)
    max_tokens = request.max_tokens or min(1024, model_max)
    n = request.n or 1

    if max_tokens > model_max:
        return error_response(f"max_tokens ({max_tokens}) exceeds model maximum ({model_max}).", param="max_tokens", code="max_tokens_exceeded")

    if request.stream:
        if n > 1:
            return error_response("n > 1 is not supported with stream=true.", param="n", code="invalid_request")
        engine = get_engine()
        token_gen = engine.stream_chat(
            messages=request.messages, model=request.model,
            temperature=request.temperature, top_p=request.top_p,
            max_tokens=max_tokens, stop=stop,
        )
        return await stream_chat_response(token_gen, model=request.model)

    choices: list[ChatChoice] = []
    total_completion_tokens = 0

    for i in range(n):
        engine = get_engine()
        content = await engine.chat(
            messages=request.messages, model=request.model,
            temperature=request.temperature, top_p=request.top_p,
            max_tokens=max_tokens, stop=stop,
        )
        total_completion_tokens += estimate_tokens(content)
        choices.append(ChatChoice(index=i, message=ChatMessage(role=MessageRole.assistant, content=content), finish_reason="stop"))

    prompt_tokens = estimate_tokens(" ".join(m.content or "" for m in request.messages))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=total_completion_tokens, total_tokens=prompt_tokens + total_completion_tokens),
        system_fingerprint=f"fp_{uuid.uuid4().hex[:10]}",
    )
