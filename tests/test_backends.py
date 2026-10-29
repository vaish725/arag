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
    assert out == {"answer": "fake answer", "citations": [], "used_fallback": False}


def test_synthesize_answer_uses_anthropic_messages_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
    seen = {}

    class FakeMessages:
        @staticmethod
        def create(model=None, max_tokens=None, system=None, messages=None):
            seen.update(model=model, max_tokens=max_tokens, system=system)
            block = SimpleNamespace(type="text", text='{"answer": "anthropic answer", "citations": [1]}')
            return SimpleNamespace(content=[block])

    fake_anth = SimpleNamespace(Anthropic=lambda api_key=None: SimpleNamespace(messages=FakeMessages))
    monkeypatch.setitem(sys.modules, "anthropic", fake_anth)

    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)

    out = llm_backend.synthesize_answer("q2", ["ctx"])
    assert out["answer"] == "anthropic answer"
    assert out["citations"] == [1]
    assert seen["model"] == "claude-haiku-5-5"
    assert "ONLY the provided contexts" in seen["system"]


@pytest.mark.parametrize(
    "reply, answer, citations",
    [
        ('{"answer": "A", "citations": [1, 2]}', "A", [1, 2]),
        ('```json\n{"answer": "B", "citations": [3]}\n```', "B", [3]),
        ("plain text reply", "plain text reply", []),
    ],
)
def test_parse_reply(reply, answer, citations):
    import arag.llm_backend as llm_backend

    out = llm_backend._parse_reply(reply, allow_fallback=False)
    assert out["answer"] == answer
    assert out["citations"] == citations


def test_parse_reply_flags_external_knowledge():
    import arag.llm_backend as llm_backend

    out = llm_backend._parse_reply(
        '{"answer": "X. Note: this answer uses external knowledge outside the provided contexts.", "citations": []}',
        allow_fallback=True,
    )
    assert out["used_fallback"] is True


def test_synthesize_answer_no_backend_raises(monkeypatch):
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    with pytest.raises(RuntimeError, match="No LLM backend configured"):
        llm_backend.synthesize_answer("q", [])
