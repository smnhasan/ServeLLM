"""
llm_serve.backends.llm — llama-cpp-python LLM backend.
"""

import threading
import logging
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from llm_serve.models import Message

logger = logging.getLogger(__name__)


class LLMBackend:
    """Downloads and wraps a GGUF model via llama-cpp-python."""

    def __init__(
        self,
        repo_id:    str,
        filename:   str,
        n_ctx:      int = 10_048,
        n_gpu_layers: int = -1,
    ) -> None:
        logger.info("Downloading LLM: %s / %s", repo_id, filename)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info("LLM downloaded to: %s", model_path)

        logger.info("Loading LLM into memory (n_gpu_layers=%d) …", n_gpu_layers)
        self.llm  = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.lock = threading.Lock()
        logger.info("LLM loaded successfully.")

    # ── prompt formatting ─────────────────────────────────────────────────

    @staticmethod
    def messages_to_prompt(messages: List[Message]) -> str:
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}<|end|>")
            elif msg.role == "assistant":
                parts.append(
                    f"<|start|>assistant<|channel|>final<|message|>\n{msg.content}<|end|>"
                )
        parts.append("<|start|>assistant<|channel|>final<|message|>\n")
        return "\n".join(parts)

    @staticmethod
    def build_stops(stop: Optional[Union[str, List[str]]] = None) -> List[str]:
        defaults = ["<|end|>", "<|user|>"]
        if stop is None:
            return defaults
        if isinstance(stop, str):
            return defaults + [stop]
        return defaults + list(stop)

    # ── inference ─────────────────────────────────────────────────────────

    def complete(
        self,
        prompt:      str,
        max_tokens:  int   = 500,
        temperature: float = 0.7,
        top_p:       float = 1.0,
        stop:        Optional[Union[str, List[str]]] = None,
    ) -> dict:
        stops = self.build_stops(stop)
        with self.lock:
            return self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stops,
            )

    def stream(
        self,
        prompt:      str,
        max_tokens:  int   = 500,
        temperature: float = 0.7,
        top_p:       float = 1.0,
        stop:        Optional[Union[str, List[str]]] = None,
    ):
        stops = self.build_stops(stop)
        with self.lock:
            yield from self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stops,
                stream=True,
            )
