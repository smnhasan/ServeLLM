from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_current_api_key
from app.schemas.openai import ModelCard, ModelList
from app.services.model_manager import model_manager

router = APIRouter()


@router.get("/models", response_model=ModelList, summary="List models")
async def list_models(api_key: str = Depends(get_current_api_key)):
    return ModelList(data=model_manager.list_models())


@router.get("/models/{model_id:path}", response_model=ModelCard, summary="Retrieve model")
async def retrieve_model(model_id: str, api_key: str = Depends(get_current_api_key)):
    model = model_manager.get_model(model_id)
    if model is None:
        return JSONResponse(status_code=404, content={"error": {"message": f"The model '{model_id}' does not exist.", "type": "invalid_request_error", "param": None, "code": "model_not_found"}})
    return model
