"""
llm_serve.tunnel — NgrokTunnelManager: thin wrapper around pyngrok.
"""

import atexit
import logging
import os
import time
from typing import Optional

import requests
from pyngrok import ngrok

logger = logging.getLogger(__name__)


class NgrokTunnelManager:
    """Create and manage a single ngrok HTTP tunnel."""

    def __init__(self) -> None:
        self.tunnel: Optional[object] = None
        self.public_url: Optional[str] = None
        self.is_active: bool = False
        atexit.register(self.cleanup)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def setup_auth(self, authtoken: Optional[str] = None) -> None:
        token = authtoken or os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
            logger.info("Ngrok auth token configured.")
        else:
            logger.warning(
                "No ngrok auth token found. "
                "Set NGROK_AUTHTOKEN env var or pass authtoken= to setup_auth()."
            )

    def create_tunnel(self, port: int = 8000) -> Optional[str]:
        self.cleanup()
        logger.info("Creating ngrok tunnel on port %d …", port)
        try:
            self.tunnel     = ngrok.connect(port, "http")
            self.public_url = str(self.tunnel.public_url)
            self.is_active  = True
            logger.info("Ngrok tunnel active: %s", self.public_url)
            return self.public_url
        except Exception as exc:
            logger.error("Failed to create ngrok tunnel: %s", exc)
            return None

    def cleanup(self) -> None:
        try:
            if self.tunnel:
                ngrok.disconnect(self.tunnel.public_url)
            ngrok.kill()
        except Exception as exc:
            logger.debug("Ngrok cleanup warning: %s", exc)
        finally:
            self.tunnel     = None
            self.public_url = None
            self.is_active  = False

    # ── health probe ──────────────────────────────────────────────────────

    def test_tunnel(self, max_retries: int = 5, retry_delay: float = 3.0) -> bool:
        if not self.public_url:
            return False
        for attempt in range(max_retries):
            try:
                r = requests.get(f"{self.public_url}/health", timeout=15)
                if r.status_code == 200:
                    logger.info("Tunnel health check passed.")
                    return True
            except Exception as exc:
                if attempt == max_retries - 1:
                    logger.error("Tunnel health check failed: %s", exc)
                else:
                    time.sleep(retry_delay)
        return False
