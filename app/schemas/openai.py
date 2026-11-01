from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ─── Shared ──────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ─── Models ──────────────────────────────────────────────────────────────────

class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex[:12]}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: list[ModelPermission] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ─── Chat Completions ─────────────────────────────────────────────────────────

class MessageRole(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    function = "function"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1, le=10)
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = "stop"
    logprobs: Any | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
    system_fingerprint: str | None = None


# ─── Streaming ────────────────────────────────────────────────────────────────

class DeltaMessage(BaseModel):
    role: MessageRole | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: Any | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatStreamChoice]
    system_fingerprint: str | None = None


# ─── Completions (legacy) ─────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    suffix: str | None = None
    max_tokens: int | None = Field(default=16, ge=1)
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1, le=10)
    stream: bool | None = False
    logprobs: int | None = Field(default=None, ge=0, le=5)
    echo: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: int | None = Field(default=1, ge=1)
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Any | None = None
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


# ─── Embeddings ───────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] | None = "float"
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo
