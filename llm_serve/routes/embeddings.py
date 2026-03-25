"""
llm_serve.routes.embeddings — /v1/embeddings route factory.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from llm_serve.models import EmbeddingRequest
from llm_serve.backends import E5Backend, InstructorBackend
from llm_serve.config import EMBEDDING_MODEL_E5_ID, EMBEDDING_MODEL_INSTRUCTOR_ID

logger = logging.getLogger(__name__)


def make_embeddings_router(
    e5:         E5Backend,
    instructor: InstructorBackend,
    semaphore:  asyncio.Semaphore,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/embeddings")
    async def create_embeddings(request: EmbeddingRequest) -> dict:
        """
        OpenAI-compatible embeddings endpoint.

        Routes to the correct backend based on ``request.model``:

        ======================================== ====== ============
        Model                                    Dim    Normalised
        ======================================== ====== ============
        ``intfloat/multilingual-e5-large``       1024   L2 ✓
        ``hkunlp/instructor-large``               768   L2 ✓
        ======================================== ====== ============
        """
        async with semaphore:
            texts = (
                [request.input]
                if isinstance(request.input, str)
                else list(request.input)
            )

            if not texts:
                raise HTTPException(status_code=400, detail="'input' must not be empty.")

            model_id = request.model.strip()

            try:
                if model_id == EMBEDDING_MODEL_INSTRUCTOR_ID:
                    vectors   = instructor.encode(texts, instruction=request.instruction)
                    used_model = EMBEDDING_MODEL_INSTRUCTOR_ID
                elif model_id in (EMBEDDING_MODEL_E5_ID, ""):
                    vectors   = e5.encode(texts)
                    used_model = EMBEDDING_MODEL_E5_ID
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unknown embedding model '{model_id}'. "
                            f"Supported: '{EMBEDDING_MODEL_E5_ID}', '{EMBEDDING_MODEL_INSTRUCTOR_ID}'."
                        ),
                    )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error("Embedding error: %s", exc)
                raise HTTPException(status_code=500, detail=str(exc))

            data_list, total_tokens = [], 0
            for idx, (text, vec) in enumerate(zip(texts, vectors)):
                tokens        = len(text.split())
                total_tokens += tokens
                data_list.append(
                    {"object": "embedding", "embedding": vec.tolist(), "index": idx}
                )

            return {
                "object": "list",
                "data":   data_list,
                "model":  used_model,
                "usage":  {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            }

    return router
