"""
tests/test_routes.py — integration tests for FastAPI endpoints.

Uses unittest.mock to avoid loading real models during CI.
"""

import time
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from llm_serve.routes import (
    make_chat_router,
    make_completions_router,
    make_embeddings_router,
)
from llm_serve.config import (
    LLM_MODEL_ID,
    EMBEDDING_MODEL_E5_ID,
    EMBEDDING_MODEL_INSTRUCTOR_ID,
)
import asyncio


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.messages_to_prompt.return_value = "<|user|>\nHello<|end|>\n<|start|>assistant…\n"
    llm.build_stops.return_value = ["<|end|>", "<|user|>"]
    llm.complete.return_value = {"choices": [{"text": "I am an AI assistant."}]}
    # stream yields a single token then stops
    llm.stream.return_value = iter([{"choices": [{"text": "Hello"}]}])
    return llm


@pytest.fixture()
def mock_e5():
    e5 = MagicMock()
    e5.encode.return_value = np.random.rand(1, 1024).astype(np.float32)
    return e5


@pytest.fixture()
def mock_instructor():
    inst = MagicMock()
    inst.encode.return_value = np.random.rand(1, 768).astype(np.float32)
    return inst


@pytest.fixture()
def app(mock_llm, mock_e5, mock_instructor):
    sem = asyncio.Semaphore(5)
    a   = FastAPI()
    a.include_router(make_chat_router(mock_llm, sem))
    a.include_router(make_completions_router(mock_llm, sem))
    a.include_router(make_embeddings_router(mock_e5, mock_instructor, sem))
    return a


@pytest.fixture()
def client(app):
    return TestClient(app)


# ── chat completions ──────────────────────────────────────────────────────────

class TestChatCompletions:
    def test_basic(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": LLM_MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)

    def test_usage_fields(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        usage = r.json()["usage"]
        assert "prompt_tokens"     in usage
        assert "completion_tokens" in usage
        assert "total_tokens"      in usage


# ── text completions ──────────────────────────────────────────────────────────

class TestCompletions:
    def test_basic(self, client, mock_llm):
        mock_llm.complete.return_value = {"choices": [{"text": "Once upon a time…"}]}
        r = client.post(
            "/v1/completions",
            json={"prompt": "Once upon"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert "Once upon" in body["choices"][0]["text"] or body["choices"][0]["text"]


# ── embeddings ────────────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_e5_single(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_E5_ID, "input": "hello"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == EMBEDDING_MODEL_E5_ID
        assert len(body["data"]) == 1
        assert len(body["data"][0]["embedding"]) == 1024

    def test_instructor_single(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_INSTRUCTOR_ID, "input": "hello"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == EMBEDDING_MODEL_INSTRUCTOR_ID
        assert len(body["data"][0]["embedding"]) == 768

    def test_unknown_model_returns_400(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"model": "some/unknown-model", "input": "test"},
        )
        assert r.status_code == 400

    def test_empty_input_returns_400(self, client):
        r = client.post(
            "/v1/embeddings",
            json={"model": EMBEDDING_MODEL_E5_ID, "input": []},
        )
        assert r.status_code == 400

    def test_batch(self, client, mock_e5):
        mock_e5.encode.return_value = np.random.rand(3, 1024).astype(np.float32)
        r = client.post(
            "/v1/embeddings",
            json={
                "model": EMBEDDING_MODEL_E5_ID,
                "input": ["text one", "text two", "text three"],
            },
        )
        assert r.status_code == 200
        assert len(r.json()["data"]) == 3
