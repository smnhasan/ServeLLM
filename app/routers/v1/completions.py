import time
import uuid

from fastapi import APIRouter, Depends

from app.dependencies import get_current_api_key
from app.schemas.openai import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    UsageInfo,
)
from app.services.inference import get_engine
from app.services.model_manager import model_manager
from app.utils.errors import error_response
from app.utils.helpers import estimate_tokens, normalize_stop

router = APIRouter()


@router.post(
    "/completions",
    summary="Create text completion",
    description="Creates a completion for the provided prompt and parameters. Legacy OpenAI endpoint.",
)
async def create_completion(
    request: CompletionRequest,
    api_key: str = Depends(get_current_api_key),
):
    if not model_manager.model_exists(request.model):
        return error_response(f"The model '{request.model}' does not exist.", param="model", code="model_not_found", status_code=404)

    if not model_manager.has_capability(request.model, "completions"):
        return error_response(f"Model '{request.model}' does not support text completions.", param="model", code="model_not_supported")

    if request.stream:
        return error_response("Streaming is not supported for /v1/completions.", param="stream", code="invalid_request")

    stop = normalize_stop(request.stop)
    model_max = model_manager.get_max_tokens(request.model)
    max_tokens = min(request.max_tokens or 16, model_max)
    n = request.n or 1

    if isinstance(request.prompt, str):
        prompts = [request.prompt]
    elif len(request.prompt) == 0:
        return error_response("prompt cannot be empty", param="prompt", code="invalid_request")
    elif all(isinstance(p, str) for p in request.prompt):
        prompts = request.prompt
    else:
        return error_response("Token-ID prompts are not supported; send text.", param="prompt", code="invalid_request")

    choices = []
    total_completion_tokens = 0
    prompt_tokens = sum(estimate_tokens(p) for p in prompts)

    for i, prompt in enumerate(prompts):
        for j in range(n):
            engine = get_engine()
            text = await engine.complete(
                prompt=prompt, model=request.model,
                temperature=request.temperature, top_p=request.top_p,
                max_tokens=max_tokens, stop=stop,
            )
            if request.echo:
                text = prompt + text
            total_completion_tokens += estimate_tokens(text)
            choices.append(CompletionChoice(text=text, index=i * n + j, finish_reason="stop"))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=total_completion_tokens, total_tokens=prompt_tokens + total_completion_tokens),
    )
