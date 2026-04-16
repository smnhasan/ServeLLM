import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_current_api_key
from app.schemas.openai import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    UsageInfo,
)
from app.services.inference import get_engine
from app.services.model_manager import model_manager
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
        return JSONResponse(status_code=404, content={"error": {"message": f"The model '{request.model}' does not exist.", "type": "invalid_request_error", "param": "model", "code": "model_not_found"}})

    stop = normalize_stop(request.stop)
    model_max = model_manager.get_max_tokens(request.model)
    max_tokens = min(request.max_tokens, model_max)

    if isinstance(request.prompt, str):
        prompts = [request.prompt]
    elif isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return JSONResponse(status_code=400, content={"error": {"message": "prompt cannot be empty", "type": "invalid_request_error", "param": "prompt", "code": "invalid_request"}})
        prompts = request.prompt if isinstance(request.prompt[0], str) else ["<token_ids>"]
    else:
        prompts = [str(request.prompt)]

    choices = []
    total_completion_tokens = 0
    prompt_tokens = sum(estimate_tokens(p) for p in prompts)

    for i, prompt in enumerate(prompts):
        for j in range(request.n):
            engine = get_engine()
            text = await engine.complete(
                prompt=prompt, model=request.model,
                temperature=request.temperature, top_p=request.top_p,
                max_tokens=max_tokens, stop=stop,
            )
            if request.echo:
                text = prompt + text
            total_completion_tokens += estimate_tokens(text)
            choices.append(CompletionChoice(text=text, index=i * request.n + j, finish_reason="stop"))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=total_completion_tokens, total_tokens=prompt_tokens + total_completion_tokens),
    )
