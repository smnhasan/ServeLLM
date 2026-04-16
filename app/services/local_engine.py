import asyncio
import base64
import threading
import uuid
import time
import logging
from typing import AsyncGenerator, List, Optional, Union, Any, Dict
import numpy as np

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

from app.services.llm_engine import LLMEngine
from app.schemas.openai import ChatMessage
from app.core.config import settings

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

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            if role == 'system':
                parts.append(f'<|system|>\\n{msg.content}')
            elif role == 'user':
                parts.append(f'<|user|>\\n{msg.content}<|end|>')
            elif role == 'assistant':
                parts.append(f'<|start|>assistant<|channel|>final<|message|>\\n{msg.content}<|end|>')
        parts.append('<|start|>assistant<|channel|>final<|message|>\\n')
        return '\\n'.join(parts)

    def _stop_sequences(self, stop: Optional[Union[str, List[str]]]) -> List[str]:
        defaults = ['<|end|>', '<|user|>']
        if stop is None:
            return defaults
        if isinstance(stop, str):
            return defaults + [stop]
        return defaults + stop

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
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
        messages: List[ChatMessage],
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        async with self.semaphore:
            prompt = self._messages_to_prompt(messages)
            stops = self._stop_sequences(stop)

            def _gen_llm():
                with self.llm_lock:
                    for output in self.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stops, stream=True):
                        yield output['choices'][0]['text']
                        
            loop = asyncio.get_event_loop()
            iterator = _gen_llm()
            while True:
                try:
                    token = await loop.run_in_executor(None, next, iterator)
                    yield token
                except StopIteration:
                    break

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        async with self.semaphore:
            stops = self._stop_sequences(stop)
            def _run_llm():
                with self.llm_lock:
                    return self.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stops)
            
            resp = await asyncio.to_thread(_run_llm)
            text = resp['choices'][0]['text']
            return text

    async def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs,
    ) -> List[List[float]]:
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

    def _embed_e5(self, texts: List[str]) -> np.ndarray:
        with self.embed_e5_lock:
            return self.embedder_e5.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )

    def _embed_instructor(self, texts: List[str], instruction: Optional[str] = None) -> np.ndarray:
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
