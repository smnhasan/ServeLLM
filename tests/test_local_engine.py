"""LocalInferenceEngine tests with the heavy ML backends stubbed out."""

import asyncio
import sys
import threading
import time
import types

import numpy as np
import pytest

# Stub the model libraries so the engine module imports without them installed.
for _name, _attr in [
    ("llama_cpp", "Llama"),
    ("huggingface_hub", "hf_hub_download"),
    ("sentence_transformers", "SentenceTransformer"),
    ("InstructorEmbedding", "INSTRUCTOR"),
]:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, object)
        sys.modules[_name] = _mod

from app.schemas.openai import ChatMessage, MessageRole  # noqa: E402
from app.services.local_engine import LocalInferenceEngine  # noqa: E402

HARMONY_STOPS = {"<|end|>", "<|return|>", "<|start|>"}
MESSAGES = [ChatMessage(role=MessageRole.user, content="hi")]


class FakeLlama:
    """Mimics Llama.__call__: a dict, or a generator of chunks when stream=True."""

    def __init__(self, tokens=("a", "b", "c"), delay=0.0, fail_after=None):
        self.tokens = tokens
        self.delay = delay
        self.fail_after = fail_after
        self.calls: list[dict] = []
        self.stream_closed = threading.Event()

    def __call__(self, prompt, stream=False, **kwargs):
        self.calls.append(kwargs)
        if not stream:
            return {"choices": [{"text": "".join(self.tokens)}]}
        return self._stream()

    def _stream(self):
        try:
            for i, tok in enumerate(self.tokens):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("boom")
                time.sleep(self.delay)
                yield {"choices": [{"text": tok}]}
        finally:
            self.stream_closed.set()


class FakeEmbedder:
    def __init__(self, dims):
        self.dims = dims

    def encode(self, texts, **kwargs):
        vecs = np.random.default_rng(0).normal(size=(len(texts), self.dims))
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def make_engine(llm: FakeLlama) -> LocalInferenceEngine:
    engine = object.__new__(LocalInferenceEngine)
    engine.llm = llm
    engine.embedder_e5 = FakeEmbedder(1024)
    engine.embedder_instructor = FakeEmbedder(768)
    engine.semaphore = asyncio.Semaphore(3)
    engine.llm_lock = threading.Lock()
    engine.embed_e5_lock = threading.Lock()
    engine.embed_ins_lock = threading.Lock()
    return engine


async def test_stream_yields_all_tokens():
    engine = make_engine(FakeLlama())
    tokens = [t async for t in engine.stream_chat(MESSAGES, model="gpt-oss-20b")]
    assert tokens == ["a", "b", "c"]
    assert not engine.llm_lock.locked()


async def test_stream_cancel_releases_lock():
    llm = FakeLlama(tokens=tuple("x" * 20), delay=0.2)
    engine = make_engine(llm)
    gen = engine.stream_chat(MESSAGES, model="gpt-oss-20b")
    assert await gen.__anext__() == "x"

    # Client disconnects while the next token is still being generated:
    # Starlette cancels the task that is awaiting the stream.
    pending = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await gen.aclose()

    # The worker must notice the cancel, close the llama stream and free the lock.
    acquired = await asyncio.to_thread(engine.llm_lock.acquire, True, 2)
    assert acquired
    engine.llm_lock.release()
    assert llm.stream_closed.is_set()


async def test_stream_error_propagates_and_releases_lock():
    engine = make_engine(FakeLlama(fail_after=1))
    gen = engine.stream_chat(MESSAGES, model="gpt-oss-20b")
    assert await gen.__anext__() == "a"
    with pytest.raises(RuntimeError, match="boom"):
        await gen.__anext__()
    assert not engine.llm_lock.locked()


async def test_completions_do_not_use_chat_stop_tokens():
    llm = FakeLlama()
    engine = make_engine(llm)
    await engine.complete("Once upon a time", model="gpt-oss-20b", stop=["\n"])
    assert llm.calls[-1]["stop"] == ["\n"]
    await engine.complete("Once upon a time", model="gpt-oss-20b")
    assert llm.calls[-1]["stop"] is None


async def test_chat_uses_harmony_stop_tokens():
    llm = FakeLlama()
    engine = make_engine(llm)
    await engine.chat(MESSAGES, model="gpt-oss-20b")
    assert HARMONY_STOPS <= set(llm.calls[-1]["stop"])
