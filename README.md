# llm-serve

> **OpenAI-compatible API server** for **GPT-OSS-20B** (chat & text completions) and two embedding models — all in one installable Python package.

---

## Models

| Role | Model | Dim | Notes |
|------|-------|-----|-------|
| LLM | `gpt-oss-20b` | — | GGUF via llama-cpp-python |
| Embeddings | `intfloat/multilingual-e5-large` | 1024 | L2-normalised |
| Embeddings | `hkunlp/instructor-large` | 768 | instruction-tuned, L2-normalised |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI chat (streaming supported) |
| `POST` | `/v1/completions` | OpenAI text completion (streaming supported) |
| `POST` | `/v1/embeddings` | Embeddings, model-routed |
| `GET`  | `/v1/models` | List available models |
| `GET`  | `/health` | Health check |

---

## Installation

### 1 — Clone & install (editable)

```bash
git clone https://github.com/your-org/llm-serve.git
cd llm-serve
pip install -e ".[dev]"
```

### 2 — GPU-accelerated llama-cpp-python (CUDA 12.2 — Kaggle / Colab default)

```bash
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
    --no-cache-dir
```

For other CUDA versions or CPU-only, see the
[llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python).

### 3 — CPU-only (slower)

```bash
pip install llama-cpp-python
```

---

## Quick Start

### Python API

```python
from llm_serve import start_server_and_keep_alive

# Blocking — keeps Kaggle / Colab session alive
start_server_and_keep_alive(
    authtoken="YOUR_NGROK_TOKEN",   # or set NGROK_AUTHTOKEN env var
    port=8001,
    max_hours=12,
)
```

For interactive / testing use:

```python
from llm_serve import start_server_only

manager, server = start_server_only(authtoken="YOUR_NGROK_TOKEN")
print(manager.get_public_url())

# … do testing …

manager.stop()
```

### CLI

```bash
# Blocking (recommended for Kaggle)
llm-serve start --authtoken YOUR_TOKEN --port 8001 --max-hours 12

# Non-blocking (returns immediately — for quick tests)
llm-serve start --authtoken YOUR_TOKEN --no-keep-alive

# CPU-only, custom context
llm-serve start --authtoken YOUR_TOKEN --n-gpu-layers 0 --n-ctx 4096
```

Full option list:

```
llm-serve start --help
```

---

## API Reference

### Chat completions

```bash
curl -s https://YOUR_NGROK_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "system",  "content": "You are a helpful assistant."},
      {"role": "user",    "content": "What is machine learning?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

Streaming (`"stream": true`) is supported — responses are delivered as
Server-Sent Events in the standard OpenAI chunk format.

---

### Text completions

```bash
curl -s https://YOUR_NGROK_URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "prompt": "Once upon a time", "max_tokens": 100}'
```

---

### Embeddings

#### multilingual-e5-large (dim = 1024)

Prefix convention (caller's responsibility):

| Use case | Prefix |
|----------|--------|
| Retrieval query | `"query: <text>"` |
| Document / passage | `"passage: <text>"` |

```bash
# Single
curl -s https://YOUR_NGROK_URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intfloat/multilingual-e5-large",
    "input": "query: What is artificial intelligence?"
  }'

# Batch (multilingual — English, Bengali, French)
curl -s https://YOUR_NGROK_URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intfloat/multilingual-e5-large",
    "input": [
      "passage: Artificial intelligence simulates human intelligence.",
      "passage: মেশিন লার্নিং একটি কৃত্রিম বুদ্ধিমত্তার শাখা।",
      "passage: L'\''apprentissage automatique est une branche de l'\''IA."
    ]
  }'
```

#### instructor-large (dim = 768)

Pass an optional `instruction` field (applied uniformly to all texts).
Defaults to `"Represent the sentence: "` when omitted.

```bash
curl -s https://YOUR_NGROK_URL/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hkunlp/instructor-large",
    "input": [
      "Artificial intelligence simulates human intelligence.",
      "Machine learning is a branch of AI."
    ],
    "instruction": "Represent the document for retrieval: "
  }'
```

Response shape (both models):

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "embedding": [0.023, -0.045, ...], "index": 0 }
  ],
  "model": "<model-id>",
  "usage": { "prompt_tokens": 9, "total_tokens": 9 }
}
```

---

## Configuration

All defaults live in `llm_serve/config.py` and can be overridden at
call time via keyword arguments or CLI flags.

| Constant | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_REPO` | `ggml-org/gpt-oss-20b-GGUF` | HF repo for the GGUF |
| `LLM_MODEL_FILE` | `gpt-oss-20b-mxfp4.gguf` | GGUF filename |
| `DEFAULT_N_CTX` | `10048` | LLM context window |
| `DEFAULT_N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `DEFAULT_MAX_REQUESTS` | `3` | Semaphore concurrency |
| `DEFAULT_PORT` | `8000` | uvicorn port |
| `DEFAULT_MAX_HOURS` | `12` | Keep-alive duration |

---

## Project Layout

```
llm_serve/
├── llm_serve/
│   ├── __init__.py          # public API + version
│   ├── config.py            # all constants in one place
│   ├── models/
│   │   └── __init__.py      # Pydantic request schemas
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── llm.py           # LLMBackend (llama-cpp-python)
│   │   └── embeddings.py    # E5Backend + InstructorBackend
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py          # /v1/chat/completions
│   │   ├── completions.py   # /v1/completions
│   │   └── embeddings.py    # /v1/embeddings
│   ├── server.py            # CombinedServer (wires everything together)
│   ├── tunnel.py            # NgrokTunnelManager
│   ├── manager.py           # ServerManager (uvicorn thread + ngrok)
│   ├── launch.py            # start_server_and_keep_alive / start_server_only
│   └── cli.py               # llm-serve CLI entry point
├── tests/
│   ├── test_models.py       # Pydantic schema unit tests
│   └── test_routes.py       # FastAPI route integration tests (mocked)
├── examples/
│   └── kaggle_notebook.py   # Drop-in replacement notebook
├── pyproject.toml
├── MANIFEST.in
├── LICENSE
└── README.md
```

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run tests
pytest

# Lint + format
ruff check llm_serve tests
ruff format llm_serve tests

# Type check
mypy llm_serve
```

---

## License

MIT — see [LICENSE](LICENSE).
