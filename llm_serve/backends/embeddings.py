"""
llm_serve.backends.embeddings — SentenceTransformer and Instructor embedding backends.
"""

import threading
import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

from llm_serve.config import (
    EMBEDDING_MODEL_E5_ID,
    EMBEDDING_MODEL_E5_DIM,
    EMBEDDING_MODEL_INSTRUCTOR_ID,
    EMBEDDING_MODEL_INSTRUCTOR_DIM,
)

logger = logging.getLogger(__name__)


class E5Backend:
    """
    Wraps ``intfloat/multilingual-e5-large`` via SentenceTransformer.

    Prefix convention (caller's responsibility):
      - Retrieval queries  → ``"query: <text>"``
      - Documents/passages → ``"passage: <text>"``
    """

    MODEL_ID = EMBEDDING_MODEL_E5_ID
    DIM      = EMBEDDING_MODEL_E5_DIM

    def __init__(self, model_id: str = EMBEDDING_MODEL_E5_ID) -> None:
        logger.info("Loading E5 embedding model: %s", model_id)
        self.model = SentenceTransformer(model_id)
        self.lock  = threading.Lock()
        logger.info("E5 embedding model loaded — dim=%d", self.DIM)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, 1024)."""
        with self.lock:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )


class InstructorBackend:
    """
    Wraps ``hkunlp/instructor-large`` via InstructorEmbedding.

    Pass an optional ``instruction`` string; defaults to
    ``"Represent the sentence: "`` when omitted.
    """

    MODEL_ID = EMBEDDING_MODEL_INSTRUCTOR_ID
    DIM      = EMBEDDING_MODEL_INSTRUCTOR_DIM
    DEFAULT_INSTRUCTION = "Represent the sentence: "

    def __init__(self, model_id: str = EMBEDDING_MODEL_INSTRUCTOR_ID) -> None:
        logger.info("Loading Instructor embedding model: %s", model_id)
        self.model = INSTRUCTOR(model_id)
        self.lock  = threading.Lock()
        logger.info("Instructor embedding model loaded — dim=%d", self.DIM)

    def encode(
        self,
        texts:       List[str],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return L2-normalised embeddings, shape (N, 768).

        Parameters
        ----------
        texts:       list of raw text strings
        instruction: task-specific instruction prefix (applied to every text)
        """
        instr = instruction or self.DEFAULT_INSTRUCTION
        pairs = [[instr, t] for t in texts]
        with self.lock:
            vecs = self.model.encode(pairs)

        # L2-normalise (matches OpenAI convention)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms
