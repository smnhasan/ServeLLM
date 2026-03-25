"""
llm_serve.launch — One-call convenience functions (mirrors the original notebook API).
"""

import logging
import time
import traceback
from typing import Optional, Tuple

from llm_serve.config import (
    LLM_MODEL_REPO,
    LLM_MODEL_FILE,
    DEFAULT_N_CTX,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_PORT,
    DEFAULT_MAX_HOURS,
)
from llm_serve.server import CombinedServer
from llm_serve.manager import ServerManager

logger = logging.getLogger(__name__)


def start_server_and_keep_alive(
    llm_repo:     str           = LLM_MODEL_REPO,
    llm_file:     str           = LLM_MODEL_FILE,
    n_ctx:        int           = DEFAULT_N_CTX,
    n_gpu_layers: int           = DEFAULT_N_GPU_LAYERS,
    max_requests: int           = DEFAULT_MAX_REQUESTS,
    port:         int           = DEFAULT_PORT,
    authtoken:    Optional[str] = None,
    max_hours:    int           = DEFAULT_MAX_HOURS,
) -> None:
    """
    Download models, start the API server, open an ngrok tunnel, and block
    the calling thread to keep the session alive (useful in Kaggle notebooks).

    Press Ctrl+C / Interrupt to stop.

    Parameters
    ----------
    llm_repo     : HuggingFace repo id (GGUF)
    llm_file     : GGUF filename
    n_ctx        : LLM context length
    n_gpu_layers : GPU layers (-1 = all)
    max_requests : max concurrent requests
    port         : local port for uvicorn
    authtoken    : ngrok auth token (or set NGROK_AUTHTOKEN env var)
    max_hours    : max keep-alive duration
    """
    print("=" * 65)
    print("  Initialising LLM + Embedding Server")
    print("=" * 65)

    try:
        print("\n[1/2] Downloading & loading models (may take a few minutes) …")
        server = CombinedServer(
            llm_repo=llm_repo,
            llm_file=llm_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_requests=max_requests,
        )

        print("\n[2/2] Starting FastAPI server + ngrok tunnel …")
        manager = ServerManager(server)
        success = manager.start_with_ngrok(port=port, authtoken=authtoken)

        if success:
            print("\n⚠  Cell will now BLOCK — [*] stays active to keep session alive.")
            print("   Press Interrupt / Ctrl+C to stop.\n")
            time.sleep(2)
            manager.keep_alive_blocking(max_hours=max_hours)
        else:
            print("✖  Server failed to start.")

    except Exception as exc:
        print(f"✖  Fatal error: {exc}")
        traceback.print_exc()


def start_server_only(
    llm_repo:     str           = LLM_MODEL_REPO,
    llm_file:     str           = LLM_MODEL_FILE,
    n_ctx:        int           = DEFAULT_N_CTX,
    n_gpu_layers: int           = DEFAULT_N_GPU_LAYERS,
    max_requests: int           = DEFAULT_MAX_REQUESTS,
    port:         int           = DEFAULT_PORT,
    authtoken:    Optional[str] = None,
) -> Tuple[Optional[ServerManager], Optional[CombinedServer]]:
    """
    Start the server WITHOUT a blocking keep-alive loop.
    Returns ``(manager, server)`` for interactive use.

    Call ``manager.keep_alive_blocking()`` later if needed.
    """
    print("Initialising server (no keep-alive) …")
    try:
        server  = CombinedServer(
            llm_repo=llm_repo,
            llm_file=llm_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_requests=max_requests,
        )
        manager = ServerManager(server)
        success = manager.start_with_ngrok(port=port, authtoken=authtoken)

        if success:
            print("\n⚠  Keep-alive NOT active. Session may time out.")
            print("   Run manager.keep_alive_blocking() to enable it.")
            return manager, server

        return None, None

    except Exception as exc:
        print(f"✖  Fatal error: {exc}")
        traceback.print_exc()
        return None, None
