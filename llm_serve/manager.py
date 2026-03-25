"""
llm_serve.manager — ServerManager: runs uvicorn in a thread + manages the ngrok tunnel.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import uvicorn

from llm_serve.config import DEFAULT_PORT
from llm_serve.server import CombinedServer
from llm_serve.tunnel import NgrokTunnelManager

logger = logging.getLogger(__name__)


class ServerManager:
    """
    Spins up the FastAPI server in a background thread and opens a
    public ngrok tunnel.

    Usage
    -----
    >>> manager = ServerManager(server)
    >>> manager.start_with_ngrok(port=8000, authtoken="…")
    >>> manager.keep_alive_blocking(max_hours=12)
    """

    def __init__(self, server: CombinedServer) -> None:
        self.server         = server
        self._thread:  Optional[threading.Thread] = None
        self.tunnel_manager = NgrokTunnelManager()
        self.is_running     = False

    # ── start ─────────────────────────────────────────────────────────────

    def start_with_ngrok(
        self,
        port:      int           = DEFAULT_PORT,
        authtoken: Optional[str] = None,
    ) -> bool:
        if self.is_running:
            logger.warning("Server is already running.")
            return True

        print(f"Starting server on port {port} …")
        try:
            self.tunnel_manager.setup_auth(authtoken)

            # start uvicorn in daemon thread
            self._thread = threading.Thread(
                target=lambda: uvicorn.run(
                    self.server.app, host="0.0.0.0", port=port, log_level="info"
                ),
                daemon=True,
            )
            self._thread.start()
            print("Waiting for server to initialise …")
            time.sleep(5)

            public_url = self.tunnel_manager.create_tunnel(port=port)
            if not public_url:
                print("✖  Failed to create ngrok tunnel.")
                return False

            self.is_running = True
            time.sleep(3)

            if self.tunnel_manager.test_tunnel():
                self._print_banner(public_url)
                return True

            print("✖  Tunnel health check failed.")
            return False

        except Exception as exc:
            print(f"✖  Server start failed: {exc}")
            return False

    # ── keep-alive ────────────────────────────────────────────────────────

    def keep_alive_blocking(self, max_hours: int = 12) -> None:
        """
        Block the calling thread, printing periodic status lines.
        Useful in Jupyter / Kaggle to prevent session timeout.
        Press Ctrl+C to stop.
        """
        start_time = datetime.now()
        end_time   = start_time + timedelta(hours=max_hours)
        iteration  = 0

        print(f"\n{'=' * 65}")
        print("🟢  Keep-Alive active  (cell is running)")
        print(f"  Started : {start_time:%Y-%m-%d %H:%M:%S}")
        print(f"  Ends at : {end_time:%Y-%m-%d %H:%M:%S}")
        print(f"  URL     : {self.get_public_url()}")
        print(f"  Stop    : Interrupt / Ctrl+C")
        print(f"{'=' * 65}\n")

        try:
            while datetime.now() < end_time:
                iteration += 1
                now       = datetime.now()
                elapsed   = now - start_time
                remaining = end_time - now

                if iteration % 10 == 0:
                    def _fmt(secs: float):
                        h, r = divmod(int(secs), 3600)
                        m, s = divmod(r, 60)
                        return h, m, s

                    eh, em, es = _fmt(elapsed.total_seconds())
                    rh, rm, _  = _fmt(remaining.total_seconds())
                    print(
                        f"[{now:%H:%M:%S}] 🟢 Status | "
                        f"Elapsed: {eh}h {em}m {es}s | "
                        f"Remaining: {rh}h {rm}m | Iter: {iteration}"
                    )

                    if self.get_public_url():
                        try:
                            r = requests.get(
                                f"{self.get_public_url()}/health", timeout=10
                            )
                            status = "✓ OK" if r.status_code == 200 else f"✗ {r.status_code}"
                        except Exception as exc:
                            status = f"✗ {str(exc)[:40]}"
                        print(f"                 Health : {status}")
                    print()

                time.sleep(30)

        except KeyboardInterrupt:
            print("\n⚠  Keep-alive interrupted.")
        finally:
            elapsed       = datetime.now() - start_time
            h, r          = divmod(int(elapsed.total_seconds()), 3600)
            m, s          = divmod(r, 60)
            print(f"\n{'=' * 65}")
            print(f"🔴 Keep-Alive ended | Total runtime: {h}h {m}m {s}s")
            print(f"{'=' * 65}\n")

    # ── stop ──────────────────────────────────────────────────────────────

    def stop(self) -> None:
        print("Stopping ngrok tunnel …")
        self.tunnel_manager.cleanup()
        self.is_running = False
        print("Tunnel stopped.")

    # ── helpers ───────────────────────────────────────────────────────────

    def get_public_url(self) -> Optional[str]:
        return self.tunnel_manager.public_url

    def _print_banner(self, public_url: str) -> None:
        from llm_serve.config import (
            LLM_MODEL_ID,
            EMBEDDING_MODEL_E5_ID, EMBEDDING_MODEL_E5_DIM,
            EMBEDDING_MODEL_INSTRUCTOR_ID, EMBEDDING_MODEL_INSTRUCTOR_DIM,
        )
        print(f"\n{'=' * 65}")
        print("  Server Active!")
        print(f"{'=' * 65}")
        print(f"\n  Public URL : {public_url}")
        print(f"\n  Endpoints:")
        print(f"    POST  {public_url}/v1/chat/completions")
        print(f"    POST  {public_url}/v1/completions")
        print(f"    POST  {public_url}/v1/embeddings")
        print(f"    GET   {public_url}/v1/models")
        print(f"    GET   {public_url}/health")
        print(f"\n  Models:")
        print(f"    LLM            : {LLM_MODEL_ID}")
        print(f"    Embeddings (1) : {EMBEDDING_MODEL_E5_ID} (dim={EMBEDDING_MODEL_E5_DIM})")
        print(f"    Embeddings (2) : {EMBEDDING_MODEL_INSTRUCTOR_ID} (dim={EMBEDDING_MODEL_INSTRUCTOR_DIM})")
        print(f"{'=' * 65}\n")
