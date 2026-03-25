"""
llm_serve.models — OpenAI-compatible Pydantic request / response schemas.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field

from llm_serve.config import LLM_MODEL_ID, DEFAULT_EMBEDDING_MODEL_ID


# ─────────────────────────────────────────────────────────────────────────────
#  Shared
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role:    str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


# ─────────────────────────────────────────────────────────────────────────────
#  Chat completions  /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    model:             str                           = Field(default=LLM_MODEL_ID)
    messages:          List[Message]                 = Field(...)
    temperature:       Optional[float]               = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens:        Optional[int]                 = Field(default=500, ge=1)
    stream:            Optional[bool]                = Field(default=False)
    top_p:             Optional[float]               = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float]               = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty:  Optional[float]               = Field(default=0.0, ge=-2.0, le=2.0)
    stop:              Optional[Union[str, List[str]]] = Field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
#  Text completions  /v1/completions
# ─────────────────────────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model:       str                           = Field(default=LLM_MODEL_ID)
    prompt:      str                           = Field(...)
    temperature: Optional[float]               = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens:  Optional[int]                 = Field(default=500, ge=1)
    stream:      Optional[bool]                = Field(default=False)
    top_p:       Optional[float]               = Field(default=1.0, ge=0.0, le=1.0)
    stop:        Optional[Union[str, List[str]]] = Field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
#  Embeddings  /v1/embeddings
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    """
    OpenAI-compatible /v1/embeddings request.
    https://platform.openai.com/docs/api-reference/embeddings/create

    Supported models
    ----------------
    - ``intfloat/multilingual-e5-large``  dim=1024, L2-normalised
    - ``hkunlp/instructor-large``         dim=768,  L2-normalised

    multilingual-e5-large prefix convention (caller-controlled)
    -----------------------------------------------------------
    - Retrieval queries  → ``"query: <text>"``
    - Documents/passages → ``"passage: <text>"``

    instructor-large instruction convention
    ----------------------------------------
    - Pass an optional ``instruction`` field (applied uniformly to all texts).
    - Defaults to ``"Represent the sentence: "`` if omitted.
    """
    model:           str                    = Field(default=DEFAULT_EMBEDDING_MODEL_ID)
    input:           Union[str, List[str]]  = Field(..., description="Text or list of texts to embed")
    encoding_format: Optional[str]          = Field(default="float", description="'float' or 'base64'")
    dimensions:      Optional[int]          = Field(default=None)
    user:            Optional[str]          = Field(default=None)
    instruction:     Optional[str]          = Field(
        default=None,
        description=(
            "Instruction prefix for hkunlp/instructor-large only. "
            "E.g. \"Represent the sentence for retrieval: \". "
            "Ignored when using multilingual-e5-large."
        ),
    )
