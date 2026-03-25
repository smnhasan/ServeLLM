"""
llm_serve.server — CombinedServer: wires backends + routes into a FastAPI app.
"""

import asyncio
import logging
import time
from typing import Optional

from fastapi import FastAPI

from llm_serve.config import (
    LLM_MODEL_REPO,
    LLM_MODEL_FILE,
    LLM_MODEL_ID,
    EMBEDDING_MODEL_E5_ID,
    EMBEDDING_MODEL_E5_DIM,
    EMBEDDING_MODEL_INSTRUCTOR_ID,
    EMBEDDING_MODEL_INSTRUCTOR_DIM,
    DEFAULT_N_CTX,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_MAX_REQUESTS,
)
from llm_serve.backends import LLMBackend, E5Backend, InstructorBackend
from llm_serve.routes import (
    make_chat_router,
    make_completions_router,
    make_embeddings_router,
)

logger = logging.getLogger(__name__)


class CombinedServer:
    """
    Loads all three models and exposes a FastAPI ``app`` with the full
    OpenAI-compatible endpoint set.

    Parameters
    ----------
    llm_repo      : HuggingFace repo id for the GGUF model
    llm_file      : filename inside the repo
    n_ctx         : LLM context window size (tokens)
    n_gpu_layers  : layers to offload (-1 = all)
    max_requests  : max concurrent in-flight requests (semaphore)
    """

    def __init__(
        self,
        llm_repo:     str = LLM_MODEL_REPO,
        llm_file:     str = LLM_MODEL_FILE,
        n_ctx:        int = DEFAULT_N_CTX,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
    ) -> None:
        # ── backends ──────────────────────────────────────────────────────
        self.llm        = LLMBackend(llm_repo, llm_file, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        self.e5         = E5Backend()
        self.instructor = InstructorBackend()

        # ── concurrency ───────────────────────────────────────────────────
        self.semaphore  = asyncio.Semaphore(max_requests)

        # ── FastAPI app ───────────────────────────────────────────────────
        self.app = FastAPI(
            title=(
                "GPT-OSS-20B + Multilingual-E5-Large + "
                "Instructor-Large OpenAI-Compatible API"
            ),
            description=(
                "Drop-in OpenAI-compatible API serving GPT-OSS-20B (chat/completions), "
                "multilingual-e5-large (embeddings, dim=1024), and "
                "instructor-large (embeddings, dim=768)."
            ),
            version="2.0.0",
        )
        self._register_routes()
        logger.info("CombinedServer initialised.")

    # ── route registration ────────────────────────────────────────────────

    def _register_routes(self) -> None:
        # root + health + models
        self.app.include_router(self._make_meta_router())

        # model-specific routes
        self.app.include_router(
            make_chat_router(self.llm, self.semaphore)
        )
        self.app.include_router(
            make_completions_router(self.llm, self.semaphore)
        )
        self.app.include_router(
            make_embeddings_router(self.e5, self.instructor, self.semaphore)
        )

    def _make_meta_router(self):
        from fastapi import APIRouter
        router = APIRouter()

        @router.get("/")
        async def root():
            return {
                "message": (
                    "GPT-OSS-20B + Multilingual-E5-Large + "
                    "Instructor-Large OpenAI-Compatible API"
                ),
                "endpoints": {
                    "chat":        "/v1/chat/completions",
                    "completions": "/v1/completions",
                    "embeddings":  "/v1/embeddings",
                    "models":      "/v1/models",
                    "health":      "/health",
                },
            }

        @router.get("/health")
        async def health():
            return {
                "status":               "healthy",
                "llm_model":            LLM_MODEL_ID,
                "embedding_model_e5":   EMBEDDING_MODEL_E5_ID,
                "embedding_dim_e5":     EMBEDDING_MODEL_E5_DIM,
                "embedding_model_ins":  EMBEDDING_MODEL_INSTRUCTOR_ID,
                "embedding_dim_ins":    EMBEDDING_MODEL_INSTRUCTOR_DIM,
                "backend":              (
                    "llama-cpp-python + sentence-transformers + InstructorEmbedding"
                ),
            }

        @router.get("/v1/models")
        async def list_models():
            ts = int(time.time())
            return {
                "object": "list",
                "data": [
                    _model_entry(LLM_MODEL_ID, ts),
                    _model_entry(EMBEDDING_MODEL_E5_ID, ts),
                    _model_entry(EMBEDDING_MODEL_INSTRUCTOR_ID, ts),
                ],
            }

        return router


# ── helpers ───────────────────────────────────────────────────────────────────

def _model_entry(model_id: str, created: int) -> dict:
    return {
        "id":         model_id,
        "object":     "model",
        "created":    created,
        "owned_by":   "organization",
        "permission": [],
        "root":       model_id,
        "parent":     None,
    }
