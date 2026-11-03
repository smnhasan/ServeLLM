import asyncio
import logging
import threading
from collections.abc import AsyncGenerator

import numpy as np
from huggingface_hub import hf_hub_download
from InstructorEmbedding import INSTRUCTOR
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.schemas.openai import ChatMessage
from app.services.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

class LocalInferenceEngine(LLMEngine):
    """Real LLM inference engine using llama-cpp-python and sentence-transformers."""

    def __init__(self):
        self.llm_model_name = settings.LLM_MODEL_REPO
        self.llm_model_file = settings.LLM_MODEL_FILE
        self.n_ctx = settings.LLM_N_CTX
        self.n_gpu_layers = settings.LLM_N_GPU_LAYERS
        max_concurrent = settings.MAX_CONCURRENT_REQUESTS

        logger.info(f"Downloading LLM: {self.llm_model_name}/{self.llm_model_file}")
        model_path = hf_hub_download(repo_id=self.llm_model_name, filename=self.llm_model_file)
        logger.info(f"LLM downloaded to: {model_path}")

        logger.info("Loading LLM into memory (GPU offload)...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False
        )
        logger.info("LLM loaded successfully.")

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_E5_ID}")
        self.embedder_e5 = SentenceTransformer(settings.EMBEDDING_MODEL_E5_ID)
        logger.info("E5 embedding model loaded")

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_INSTRUCTOR_ID}")
        self.embedder_instructor = INSTRUCTOR(settings.EMBEDDING_MODEL_INSTRUCTOR_ID)
        logger.info("Instructor embedding model loaded")

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.llm_lock = threading.Lock()
        self.embed_e5_lock = threading.Lock()
        self.embed_ins_lock = threading.Lock()

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Render messages in the gpt-oss "harmony" chat format."""
        parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            content = msg.content or ''
            if role == 'system':
                # Harmony reserves "system" for model metadata; user-supplied
                # system prompts go in the developer message.
                parts.append(f'<|start|>developer<|message|># Instructions\n\n{content}<|end|>')
            elif role == 'user':
                parts.append(f'<|start|>user<|message|>{content}<|end|>')
            elif role == 'assistant':
                parts.append(f'<|start|>assistant<|channel|>final<|message|>{content}<|end|>')
        parts.append('<|start|>assistant<|channel|>final<|message|>')
        return ''.join(parts)

    def _stop_sequences(self, stop: str | list[str] | None, chat: bool = True) -> list[str]:
        # Harmony end-of-turn markers only make sense for chat-formatted prompts;
        # raw /v1/completions prompts get just the caller's stop sequences.
        defaults = ['<|end|>', '<|return|>', '<|start|>'] if chat else []
        if stop is None:
            return defaults
        if isinstance(stop, str):
            return defaults + [stop]
        return defaults + stop

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        async with self.semaphore:
            prompt = self._messages_to_prompt(messages)
            stops = self._stop_sequences(stop)
            
            def _run_llm():
                with self.llm_lock:
                    return self.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stops)
            
            resp = await asyncio.to_thread(_run_llm)
            text = resp['choices'][0]['text'].strip()
            return text

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        async with self.semaphore:
            prompt = self._messages_to_prompt(messages)
            stops = self._stop_sequences(stop)

            # Generation runs entirely in one worker thread so llm_lock is acquired
            # and released by the same thread. Tokens are handed to the event loop
            # through a queue; `cancelled` stops the worker when the consumer goes
            # away (e.g. client disconnect), which releases the lock.
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            cancelled = threading.Event()
            done = object()

            def _put(item):
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:  # event loop already closed
                    cancelled.set()

            def _produce():
                item = done
                try:
                    with self.llm_lock:
                        stream = self.llm(prompt, max_tokens=max_tokens, temperature=temperature,
                                          top_p=top_p, stop=stops, stream=True)
                        try:
                            for output in stream:
                                if cancelled.is_set():
                                    break
                                _put(output['choices'][0]['text'])
                        finally:
                            stream.close()
                except Exception as exc:
                    item = exc
                finally:
                    _put(item)

            threading.Thread(target=_produce, name="llm-stream", daemon=True).start()
            try:
                while True:
                    item = await queue.get()
                    if item is done:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                cancelled.set()

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16,
        stop: list[str] | None = None,
        **kwargs,
    ) -> str:
        async with self.semaphore:
            stops = self._stop_sequences(stop, chat=False) or None
            def _run_llm():
                with self.llm_lock:
                    return self.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stops)
            
            resp = await asyncio.to_thread(_run_llm)
            text = resp['choices'][0]['text']
            return text

    async def embed(
        self,
        texts: list[str],
        model: str,
        **kwargs,
    ) -> list[list[float]]:
        async with self.semaphore:
            used_model = model
            if settings.EMBEDDING_MODEL_INSTRUCTOR_ID in used_model or "instructor" in used_model.lower():
                raw_vecs = await asyncio.to_thread(self._embed_instructor, texts, kwargs.get("instruction"))
            else:
                raw_vecs = await asyncio.to_thread(self._embed_e5, texts)

            dims = kwargs.get("dimensions")
            
            data_list = []
            for vec in raw_vecs:
                if dims is not None:
                    vec = vec[:dims]
                data_list.append(vec.tolist())
            return data_list

    def _embed_e5(self, texts: list[str]) -> np.ndarray:
        with self.embed_e5_lock:
            return self.embedder_e5.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )

    def _embed_instructor(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        instr = instruction or 'Represent the sentence: '
        pairs = [[instr, t] for t in texts]
        with self.embed_ins_lock:
            vecs = self.embedder_instructor.encode(pairs, batch_size=32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    def count_tokens(self, text: str, model: str) -> int:
        try:
            return len(self.llm.tokenize(text.encode('utf-8'), add_bos=False))
        except Exception:
            return len(text.split())
