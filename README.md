# LLM Serve

An **OpenAI-compatible LLM serving API** built with FastAPI. Drop-in replacement for the OpenAI API — works with any client library that supports a custom `base_url`.

---

## Features

| Feature | Details |
|---|---|
| **Auth** | Bearer token API key verification |
| **Rate limiting** | Per-key sliding-window (configurable RPM) |
| **Chat completions** | `POST /v1/chat/completions` — streaming + non-streaming |
| **Text completions** | `POST /v1/completions` — legacy endpoint |
| **Embeddings** | `POST /v1/embeddings` — deterministic dummy vectors |
| **Model registry** | `GET /v1/models`, `GET /v1/models/{id}` — YAML-configurable |
| **OpenAI schema** | Pydantic models matching the OpenAI API spec exactly |
| **Dummy mode** | Fully functional without a real model — swap in vLLM/HF/Ollama later |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/llm-serve.git
cd llm-serve
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set your VALID_API_KEYS etc.

# 3. Run
uvicorn app.main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

---

## Usage

### With curl

```bash
export API_KEY="sk-llmserve-test-key-1234"

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $API_KEY"

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role":"user","content":"Hi"}], "stream": true}'

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-ada-002", "input": "Hello world"}'
```

### With the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-llmserve-test-key-1234",
)

# Chat
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is FastAPI?"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# Embeddings
emb = client.embeddings.create(
    model="text-embedding-ada-002",
    input="The quick brown fox",
)
print(len(emb.data[0].embedding))  # 1536
```

---

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `VALID_API_KEYS` | `sk-llmserve-test-key-1234,...` | Comma-separated valid API keys |
| `RATE_LIMIT_ENABLED` | `true` | Enable per-key rate limiting |
| `RATE_LIMIT_RPM` | `60` | Requests per minute per key |
| `DUMMY_MODE` | `true` | Use dummy responses (no real model) |
| `DUMMY_LATENCY_MS` | `200` | Simulated inference latency |
| `STREAM_CHUNK_DELAY_MS` | `50` | Delay between streamed tokens |
| `MODELS_CONFIG_PATH` | `configs/models.yaml` | Path to custom model registry |
| `DEBUG` | `false` | Enable FastAPI debug / auto-reload |

### Adding models

Edit `configs/models.yaml`:

```yaml
models:
  - id: "my-custom-model"
    owned_by: "my-org"
    capabilities: ["chat"]
    context_window: 32768
    max_tokens: 32768
```

---

## Running Tests

```bash
pytest -v
```

---

## Connecting a Real Model Backend

1. Open `app/services/inference.py`
2. Create a new class implementing `LLMEngine` (defined in `app/services/llm_engine.py`)
3. Replace `dummy_engine` with your implementation in the routers

Example backends to implement:
- **vLLM**: `AsyncLLMEngine` via `vllm.engine.async_llm_engine`
- **HuggingFace Transformers**: `AutoModelForCausalLM` + `pipeline`
- **Ollama**: HTTP calls to `http://localhost:11434`
- **LiteLLM**: Proxy to any supported provider

---

## Project Structure

```
llm-serve/
├── app/
│   ├── main.py              # FastAPI app, middleware, lifespan
│   ├── dependencies.py      # Auth + rate limit FastAPI dependencies
│   ├── routers/v1/
│   │   ├── chat.py          # POST /v1/chat/completions
│   │   ├── completions.py   # POST /v1/completions
│   │   ├── models.py        # GET  /v1/models[/{id}]
│   │   └── embeddings.py    # POST /v1/embeddings
│   ├── schemas/openai.py    # Pydantic models (OpenAI-spec)
│   ├── services/
│   │   ├── llm_engine.py    # Abstract LLM interface
│   │   ├── inference.py     # DummyInferenceEngine (swap for real)
│   │   ├── model_manager.py # Model registry
│   │   └── auth.py          # Auth + rate limiter
│   ├── core/
│   │   ├── config.py        # pydantic-settings
│   │   └── security.py      # Key hashing helpers
│   └── utils/helpers.py     # Streaming, token counting, errors
├── configs/models.yaml      # Custom model registry
├── tests/                   # pytest test suite
└── .env.example
```
