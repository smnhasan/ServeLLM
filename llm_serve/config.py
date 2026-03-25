"""
llm_serve.config — Model identifiers and server defaults.

All tuneable constants live here so they are importable from one place
and can be overridden without touching server logic.
"""

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_MODEL_REPO: str = "ggml-org/gpt-oss-20b-GGUF"
LLM_MODEL_FILE: str = "gpt-oss-20b-mxfp4.gguf"
LLM_MODEL_ID:   str = "gpt-oss-20b"

# ── Embedding: multilingual-e5-large ──────────────────────────────────────
EMBEDDING_MODEL_E5_ID:  str = "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_E5_DIM: int = 1024

# ── Embedding: instructor-large ───────────────────────────────────────────
EMBEDDING_MODEL_INSTRUCTOR_ID:  str = "hkunlp/instructor-large"
EMBEDDING_MODEL_INSTRUCTOR_DIM: int = 768

# ── Fallback / default embedding model ────────────────────────────────────
DEFAULT_EMBEDDING_MODEL_ID: str = EMBEDDING_MODEL_E5_ID

# ── Server defaults ───────────────────────────────────────────────────────
DEFAULT_PORT:         int = 8000
DEFAULT_N_CTX:        int = 10_048
DEFAULT_N_GPU_LAYERS: int = -1          # -1 = offload all layers to GPU
DEFAULT_MAX_REQUESTS: int = 3
DEFAULT_MAX_HOURS:    int = 12
