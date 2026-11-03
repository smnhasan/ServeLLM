from fastapi import APIRouter, Depends

from app.dependencies import get_current_api_key
from app.schemas.openai import ModelCard, ModelList
from app.services.model_manager import model_manager
from app.utils.errors import error_response

router = APIRouter()


@router.get("/models", response_model=ModelList, summary="List models")
async def list_models(api_key: str = Depends(get_current_api_key)):
    return ModelList(data=model_manager.list_models())


@router.get("/models/{model_id:path}", response_model=ModelCard, summary="Retrieve model")
async def retrieve_model(model_id: str, api_key: str = Depends(get_current_api_key)):
    model = model_manager.get_model(model_id)
    if model is None:
        return error_response(f"The model '{model_id}' does not exist.", code="model_not_found", status_code=404)
    return model
