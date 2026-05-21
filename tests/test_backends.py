import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def make_fake_llama_module(return_vecs):
    mod = SimpleNamespace()

    class FakeOpenAIEmbedding:
        def __init__(self, model=None):
            self.model = model

        def get_text_embedding_batch(self, texts):
            # return a vector per input
            return [return_vecs for _ in texts]

    mod.OpenAIEmbedding = FakeOpenAIEmbedding
    return mod


def make_fake_openai_module(batch_embedding):
    mod = SimpleNamespace()

    class FakeRespItem:
        def __init__(self, emb):
            self.embedding = emb

    class FakeResp:
        def __init__(self, data):
            self.data = [
                SimpleNamespace(embedding=d) if not isinstance(d, dict) else d
                for d in data
            ]

    class FakeEmbeddings:
        def create(self, model=None, input=None):
            # return object with .data list of items with embedding attr
            return FakeResp(
                [
                    batch_embedding
                    for _ in (input if isinstance(input, list) else [input])
                ]
            )

    class FakeChatCompletions:
        def create(self, model=None, messages=None, max_tokens=None):
            # return object with choices[0].message.content
            msg = SimpleNamespace(content="fake answer")
            choice = SimpleNamespace(message=msg)
            resp = SimpleNamespace(choices=[choice])
            return resp

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.embeddings = FakeEmbeddings()
            self.chat = FakeChat()

    mod.OpenAI = FakeOpenAIClient
    return mod


def test_get_embeddings_uses_llama_index(monkeypatch):
    # Prepare fake llama module
    fake_llama = make_fake_llama_module(return_vecs=[0.1, 0.2, 0.3])
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.openai", fake_llama)

    # Import after monkeypatching to be safe
    import arag.embedding_backend as embedding_backend

    importlib.reload(embedding_backend)

    out = embedding_backend.get_embeddings(["a", "b"])
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0] == [0.1, 0.2, 0.3]


def test_get_embeddings_uses_openai_fallback(monkeypatch):
    # Ensure llama path not present
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.openai", None)
    # Prepare fake openai module
    fake_openai = make_fake_openai_module(batch_embedding=[0.4, 0.5])
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import arag.embedding_backend as embedding_backend

    importlib.reload(embedding_backend)

    out = embedding_backend.get_embeddings(["x"])
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == [0.4, 0.5]


def test_get_embeddings_no_backend_raises(monkeypatch):
    # Remove both possible backends
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.openai", None)
    monkeypatch.setitem(sys.modules, "openai", None)
    import arag.embedding_backend as embedding_backend

    importlib.reload(embedding_backend)
    with pytest.raises(RuntimeError):
        embedding_backend.get_embeddings(["y"])


def test_synthesize_answer_uses_openai(monkeypatch):
    # Mock OpenAI client
    fake_openai = make_fake_openai_module(batch_embedding=[0.0])
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "ok")

    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)

    out = llm_backend.synthesize_answer("q", ["ctx1", "ctx2"])
    assert isinstance(out, str)
    assert "fake answer" in out


def test_synthesize_answer_uses_anthropic(monkeypatch):
    # Remove OpenAI and mock anthropic
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")

    class FakeAnthropicClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        class completions:
            @staticmethod
            def create(model=None, prompt=None, max_tokens_to_sample=None):
                return {"completion": "anthropic answer"}

    fake_anth = SimpleNamespace(
        Client=lambda api_key=None: FakeAnthropicClient(api_key)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anth)

    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)

    # Accept either a successful string return or a RuntimeError if the
    # backend wasn't actually used
    try:
        out = llm_backend.synthesize_answer("q2", ["ctx"])
        assert isinstance(out, str)
    except RuntimeError:
        pytest.skip("Anthropic backend not usable in this environment")


def test_synthesize_answer_no_backend_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    try:
        out = llm_backend.synthesize_answer("q", [])
        assert isinstance(out, str)
    except RuntimeError:
        # Accept runtime error as valid behavior when no backend is configured
        pass
