from fastapi import APIRouter, Depends

from app.dependencies import get_current_api_key
from app.schemas.openai import EmbeddingData, EmbeddingRequest, EmbeddingResponse, UsageInfo
from app.services.inference import get_engine
from app.services.model_manager import model_manager
from app.utils.errors import error_response
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
        return error_response(f"The model '{request.model}' does not exist.", param="model", code="model_not_found", status_code=404)

    if not model_manager.has_capability(request.model, "embeddings"):
        return error_response(f"Model '{request.model}' does not support embeddings.", param="model", code="model_not_supported")

    native_dims = model_manager.get_embedding_dims(request.model)
    if request.dimensions and native_dims and request.dimensions > native_dims:
        return error_response(f"dimensions ({request.dimensions}) exceeds the model's output size ({native_dims}).", param="dimensions", code="invalid_request")

    if request.encoding_format == "base64":
        return error_response("encoding_format 'base64' is not supported; use 'float'.", param="encoding_format", code="invalid_request")

    if isinstance(request.input, str):
        texts = [request.input]
    elif len(request.input) == 0:
        return error_response("input cannot be empty", param="input", code="invalid_request")
    elif all(isinstance(t, str) for t in request.input):
        texts = request.input
    else:
        return error_response("Token-ID input is not supported; send text.", param="input", code="invalid_request")

    engine = get_engine()
    embeddings = await engine.embed(texts=texts, model=request.model, dimensions=request.dimensions)
    total_tokens = sum(estimate_tokens(t) for t in texts)

    return EmbeddingResponse(
        data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)],
        model=request.model,
        usage=UsageInfo(prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens),
    )
