from fastapi.testclient import TestClient

from tests.conftest import AUTH_HEADERS

# ─── System / Health ──────────────────────────────────────────────────────────

class TestSystem:
    def test_health_check(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_root_returns_api_info(self, client: TestClient):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "endpoints" in body
        assert "/v1/chat/completions" in body["endpoints"]["chat_completions"]

    def test_unknown_path_returns_404(self, client: TestClient):
        r = client.get("/v1/nonexistent", headers=AUTH_HEADERS)
        assert r.status_code == 404

    def test_process_time_header_present(self, client: TestClient):
        r = client.get("/health")
        assert "x-process-time" in r.headers


# ─── Models ───────────────────────────────────────────────────────────────────

class TestModels:
    def test_list_models_unauthenticated(self, client: TestClient):
        r = client.get("/v1/models")
        assert r.status_code == 401

    def test_list_models_returns_list(self, client: TestClient):
        r = client.get("/v1/models", headers=AUTH_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0

    def test_list_models_schema(self, client: TestClient):
        r = client.get("/v1/models", headers=AUTH_HEADERS)
        model = r.json()["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "owned_by" in model
        assert "created" in model

    def test_retrieve_existing_model(self, client: TestClient):
        r = client.get("/v1/models/gpt-oss-20b", headers=AUTH_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "gpt-oss-20b"
        assert body["object"] == "model"

    def test_retrieve_nonexistent_model(self, client: TestClient):
        r = client.get("/v1/models/fake-model-xyz", headers=AUTH_HEADERS)
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "model_not_found"

    def test_default_models_listed(self, client: TestClient):
        r = client.get("/v1/models", headers=AUTH_HEADERS)
        ids = [m["id"] for m in r.json()["data"]]
        assert "gpt-oss-20b" in ids
        assert "intfloat/multilingual-e5-large" in ids
        assert "hkunlp/instructor-large" in ids


# ─── Chat Completions ─────────────────────────────────────────────────────────

class TestChatCompletions:
    BASE_PAYLOAD = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    def test_unauthenticated_returns_401(self, client: TestClient):
        r = client.post("/v1/chat/completions", json=self.BASE_PAYLOAD)
        assert r.status_code == 401

    def test_basic_chat_completion(self, client: TestClient):
        r = client.post("/v1/chat/completions", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert "id" in body
        assert body["id"].startswith("chatcmpl-")
        assert "choices" in body
        assert len(body["choices"]) == 1
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)
        assert len(body["choices"][0]["message"]["content"]) > 0

    def test_usage_tokens_in_response(self, client: TestClient):
        r = client.post("/v1/chat/completions", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        usage = r.json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_multiple_n_completions(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "n": 3}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        choices = r.json()["choices"]
        assert len(choices) == 3
        for i, choice in enumerate(choices):
            assert choice["index"] == i

    def test_system_message_supported(self, client: TestClient):
        payload = {
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200

    def test_nonexistent_model_returns_404(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "model": "non-existent-model"}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

    def test_embedding_model_not_supported_for_chat(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "model": "intfloat/multilingual-e5-large"}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_not_supported"

    def test_max_tokens_respected(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "max_tokens": 5}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        # With max_tokens=5, content should be short
        assert len(content.split()) <= 20  # generous upper bound

    def test_exceeding_max_tokens_returns_400(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "max_tokens": 999999}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "max_tokens_exceeded"

    def test_missing_messages_returns_422(self, client: TestClient):
        payload = {"model": "gpt-oss-20b"}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 422

    def test_invalid_temperature_returns_422(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "temperature": 5.0}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 422

    def test_finish_reason_present(self, client: TestClient):
        r = client.post("/v1/chat/completions", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r.json()["choices"][0]["finish_reason"] == "stop"

    def test_model_echoed_in_response(self, client: TestClient):
        r = client.post("/v1/chat/completions", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r.json()["model"] == "gpt-oss-20b"

    def test_streaming_returns_event_stream(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "stream": True}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_streaming_content_structure(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "stream": True}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        lines = [line for line in r.text.split("\n") if line.startswith("data:")]
        assert len(lines) > 1
        # Last data line should be [DONE]
        assert lines[-1].strip() == "data: [DONE]"

    def test_null_n_treated_as_one(self, client: TestClient):
        for stream in (False, True):
            payload = {**self.BASE_PAYLOAD, "n": None, "stream": stream}
            r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
            assert r.status_code == 200

    def test_stream_engine_error_sends_error_event(self, client: TestClient, monkeypatch):
        from app.services.inference import dummy_engine

        async def failing_stream(*args, **kwargs):
            yield "partial "
            raise RuntimeError("engine crashed")

        monkeypatch.setattr(dummy_engine, "stream_chat", failing_stream)
        payload = {**self.BASE_PAYLOAD, "stream": True}
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        lines = [line for line in r.text.split("\n") if line.startswith("data:")]
        assert '"engine_error"' in lines[-2]
        assert lines[-1].strip() == "data: [DONE]"

    def test_multi_turn_conversation(self, client: TestClient):
        payload = {
            "model": "gpt-oss-20b",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 4+4?"},
            ],
        }
        r = client.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200


# ─── Completions (legacy) ─────────────────────────────────────────────────────

class TestCompletions:
    BASE_PAYLOAD = {
        "model": "gpt-oss-20b",
        "prompt": "Once upon a time",
    }

    def test_basic_completion(self, client: TestClient):
        r = client.post("/v1/completions", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["id"].startswith("cmpl-")
        assert len(body["choices"]) == 1
        assert isinstance(body["choices"][0]["text"], str)

    def test_echo_option(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "echo": True}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        text = r.json()["choices"][0]["text"]
        assert text.startswith("Once upon a time")

    def test_multiple_prompts(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "prompt": ["Hello", "World"]}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert len(r.json()["choices"]) == 2

    def test_n_completions(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "n": 2}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert len(r.json()["choices"]) == 2

    def test_invalid_model(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "model": "fake-model"}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 404

    def test_usage_present(self, client: TestClient):
        r = client.post("/v1/completions", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        usage = r.json()["usage"]
        assert usage["total_tokens"] > 0

    def test_unauthenticated_returns_401(self, client: TestClient):
        r = client.post("/v1/completions", json=self.BASE_PAYLOAD)
        assert r.status_code == 401

    def test_embedding_model_not_supported(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "model": "intfloat/multilingual-e5-large"}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_not_supported"

    def test_null_n_and_max_tokens(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "n": None, "max_tokens": None}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert len(r.json()["choices"]) == 1

    def test_stream_rejected(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "stream": True}
        r = client.post("/v1/completions", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400


# ─── Embeddings ───────────────────────────────────────────────────────────────

class TestEmbeddings:
    BASE_PAYLOAD = {
        "model": "intfloat/multilingual-e5-large",
        "input": "The food was delicious and the waiter was very helpful.",
    }

    def test_basic_embedding(self, client: TestClient):
        r = client.post("/v1/embeddings", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["object"] == "embedding"
        embedding = body["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # multilingual-e5-large dimensions

    def test_multiple_inputs(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "input": ["Hello", "World", "Test"]}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert len(r.json()["data"]) == 3

    def test_embedding_indices(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "input": ["a", "b", "c"]}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        for i, d in enumerate(r.json()["data"]):
            assert d["index"] == i

    def test_instructor_model_dimensions(self, client: TestClient):
        payload = {"model": "hkunlp/instructor-large", "input": "test"}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert len(r.json()["data"][0]["embedding"]) == 768

    def test_dimensions_param_truncates(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "dimensions": 256}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 200
        assert len(r.json()["data"][0]["embedding"]) == 256

    def test_token_id_input_rejected(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "input": [1, 2, 3]}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400

    def test_deterministic_embeddings(self, client: TestClient):
        """Same input should produce same embedding vector."""
        r1 = client.post("/v1/embeddings", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        r2 = client.post("/v1/embeddings", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r1.json()["data"][0]["embedding"] == r2.json()["data"][0]["embedding"]

    def test_different_inputs_different_embeddings(self, client: TestClient):
        payload1 = {**self.BASE_PAYLOAD, "input": "apple"}
        payload2 = {**self.BASE_PAYLOAD, "input": "banana"}
        r1 = client.post("/v1/embeddings", json=payload1, headers=AUTH_HEADERS)
        r2 = client.post("/v1/embeddings", json=payload2, headers=AUTH_HEADERS)
        assert r1.json()["data"][0]["embedding"] != r2.json()["data"][0]["embedding"]

    def test_chat_model_not_supported_for_embeddings(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "model": "gpt-oss-20b"}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_not_supported"

    def test_nonexistent_model_returns_404(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "model": "fake-embed-model"}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 404

    def test_empty_input_returns_400(self, client: TestClient):
        payload = {**self.BASE_PAYLOAD, "input": []}
        r = client.post("/v1/embeddings", json=payload, headers=AUTH_HEADERS)
        assert r.status_code == 400

    def test_usage_has_zero_completion_tokens(self, client: TestClient):
        r = client.post("/v1/embeddings", json=self.BASE_PAYLOAD, headers=AUTH_HEADERS)
        assert r.json()["usage"]["completion_tokens"] == 0

    def test_unauthenticated_returns_401(self, client: TestClient):
        r = client.post("/v1/embeddings", json=self.BASE_PAYLOAD)
        assert r.status_code == 401
