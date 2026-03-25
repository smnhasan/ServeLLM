# %% [markdown]
# # llm-serve — Kaggle Notebook (Package Edition)
#
# Install the package, then launch the server with a single function call.
#
# **Models served:**
# - 🤖 **LLM** — `gpt-oss-20b` via `llama-cpp-python`
# - 🔢 **Embeddings** — `intfloat/multilingual-e5-large` (dim=1024)
# - 🔢 **Embeddings** — `hkunlp/instructor-large` (dim=768)

# %% [markdown]
# ## Cell 1 — Install the package

# %% [code]
# Option A — install from PyPI (once published)
# !pip install llm-serve

# Option B — install from local wheel / source directory
# !pip install /path/to/llm_serve  --quiet

# GPU-accelerated llama-cpp-python for CUDA 12.2 (Kaggle default)
# !pip install llama-cpp-python \
#     --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
#     --no-cache-dir --quiet

# %% [markdown]
# ## Cell 2 — Load ngrok secret

# %% [code]
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["NGROK_AUTHTOKEN"] = user_secrets.get_secret("NGROK_AUTHTOKEN")

# %% [markdown]
# ## Cell 3 — Launch (blocking keep-alive)

# %% [code]
# %%scrolled true
from llm_serve import start_server_and_keep_alive

start_server_and_keep_alive(
    # authtoken=os.environ["NGROK_AUTHTOKEN"],  # picked up automatically from env
    port=8001,
    max_hours=12,
)

# %% [markdown]
# ## Cell 4 — Quick API test (run in a separate cell)

# %% [code]
import requests, json

PUBLIC_URL = "PASTE_YOUR_NGROK_URL_HERE"

# ── Health ────────────────────────────────────────────────────────────────────
print("── Health ──")
print(requests.get(f"{PUBLIC_URL}/health").json())

# ── Models ────────────────────────────────────────────────────────────────────
print("\n── Models ──")
print(requests.get(f"{PUBLIC_URL}/v1/models").json())

# ── Chat completion ───────────────────────────────────────────────────────────
print("\n── Chat Completion ──")
resp = requests.post(
    f"{PUBLIC_URL}/v1/chat/completions",
    json={
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "system",  "content": "You are a helpful assistant. Be concise."},
            {"role": "user",    "content": "What is machine learning in one sentence?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    },
    timeout=60,
)
print(resp.json()["choices"][0]["message"]["content"])

# ── Embeddings — multilingual-e5-large ───────────────────────────────────────
print("\n── Embeddings: multilingual-e5-large ──")
r = requests.post(
    f"{PUBLIC_URL}/v1/embeddings",
    json={
        "model": "intfloat/multilingual-e5-large",
        "input": "query: What is artificial intelligence?",
    },
    timeout=30,
)
result = r.json()
print(f"Dim   : {len(result['data'][0]['embedding'])}")
print(f"Usage : {result['usage']}")

# ── Embeddings — instructor-large (custom instruction) ───────────────────────
print("\n── Embeddings: hkunlp/instructor-large ──")
r = requests.post(
    f"{PUBLIC_URL}/v1/embeddings",
    json={
        "model": "hkunlp/instructor-large",
        "input": ["Artificial intelligence simulates human intelligence.",
                  "Machine learning is a branch of AI."],
        "instruction": "Represent the document for retrieval: ",
    },
    timeout=30,
)
for item in r.json()["data"]:
    print(f"  index {item['index']}  ->  dim={len(item['embedding'])}")
