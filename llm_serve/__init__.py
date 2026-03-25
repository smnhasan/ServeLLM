"""
llm_serve — OpenAI-compatible API server for GPT-OSS-20B + embedding models.

Quick start:
    from llm_serve import start_server_and_keep_alive
    start_server_and_keep_alive(authtoken="YOUR_NGROK_TOKEN")

CLI:
    llm-serve start --authtoken YOUR_TOKEN
"""

from llm_serve.server import CombinedServer
from llm_serve.manager import ServerManager
from llm_serve.launch import start_server_and_keep_alive, start_server_only

__all__ = [
    "CombinedServer",
    "ServerManager",
    "start_server_and_keep_alive",
    "start_server_only",
]

__version__ = "2.0.0"
__author__ = "llm_serve contributors"
