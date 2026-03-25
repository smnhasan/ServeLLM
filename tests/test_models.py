"""
tests/test_models.py — unit tests for Pydantic request schemas.
No heavy dependencies (llama-cpp / sentence-transformers) are imported.
"""

import pytest
from llm_serve.models import (
    Message,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
from llm_serve.config import (
    LLM_MODEL_ID,
    EMBEDDING_MODEL_E5_ID,
    EMBEDDING_MODEL_INSTRUCTOR_ID,
)


class TestMessage:
    def test_basic(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"


class TestChatCompletionRequest:
    def test_defaults(self):
        req = ChatCompletionRequest(
            messages=[Message(role="user", content="Hi")]
        )
        assert req.model == LLM_MODEL_ID
        assert req.temperature == 0.7
        assert req.max_tokens == 500
        assert req.stream is False

    def test_custom(self):
        req = ChatCompletionRequest(
            model="my-model",
            messages=[Message(role="user", content="Hi")],
            temperature=0.3,
            max_tokens=200,
            stream=True,
        )
        assert req.temperature == 0.3
        assert req.max_tokens == 200
        assert req.stream is True

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                messages=[Message(role="user", content="Hi")],
                temperature=3.0,   # > 2.0 — should fail
            )


class TestCompletionRequest:
    def test_defaults(self):
        req = CompletionRequest(prompt="Once upon a time")
        assert req.model == LLM_MODEL_ID
        assert req.prompt == "Once upon a time"

    def test_stop_string(self):
        req = CompletionRequest(prompt="test", stop="<END>")
        assert req.stop == "<END>"

    def test_stop_list(self):
        req = CompletionRequest(prompt="test", stop=["<END>", "\n"])
        assert req.stop == ["<END>", "\n"]


class TestEmbeddingRequest:
    def test_defaults_single(self):
        req = EmbeddingRequest(input="hello world")
        assert req.model == EMBEDDING_MODEL_E5_ID
        assert req.input == "hello world"
        assert req.instruction is None

    def test_batch(self):
        req = EmbeddingRequest(
            model=EMBEDDING_MODEL_E5_ID,
            input=["text one", "text two"],
        )
        assert len(req.input) == 2

    def test_instructor_with_instruction(self):
        req = EmbeddingRequest(
            model=EMBEDDING_MODEL_INSTRUCTOR_ID,
            input="Represent this document",
            instruction="Represent the document for retrieval: ",
        )
        assert req.instruction == "Represent the document for retrieval: "
