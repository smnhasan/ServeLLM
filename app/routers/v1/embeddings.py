from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_current_api_key
from app.schemas.openai import EmbeddingData, EmbeddingRequest, EmbeddingResponse, UsageInfo
from app.services.inference import get_engine
from app.services.model_manager import model_manager
from app.utils.helpers import estimate_tokens

router = APIRouter()


@router.post(
    "/embeddings",
    summary="Create embeddings",
    description="Creates an embedding vector representing the input text.",
)
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: str = Depends(get_current_api_key),
):
    if not model_manager.model_exists(request.model):
        return JSONResponse(status_code=404, content={"error": {"message": f"The model '{request.model}' does not exist.", "type": "invalid_request_error", "param": "model", "code": "model_not_found"}})

    if not model_manager.has_capability(request.model, "embeddings"):
        return JSONResponse(status_code=400, content={"error": {"message": f"Model '{request.model}' does not support embeddings.", "type": "invalid_request_error", "param": "model", "code": "model_not_supported"}})

    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list):
        if len(request.input) == 0:
            return JSONResponse(status_code=400, content={"error": {"message": "input cannot be empty", "type": "invalid_request_error", "param": "input", "code": "invalid_request"}})
        texts = request.input if isinstance(request.input[0], str) else [f"<tokens:{i}>" for i in range(len(request.input))]
    else:
        texts = [str(request.input)]

    engine = get_engine()
    embeddings = await engine.embed(texts=texts, model=request.model)
    total_tokens = sum(estimate_tokens(t) for t in texts)

    return EmbeddingResponse(
        data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)],
        model=request.model,
        usage=UsageInfo(prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens),
    )
