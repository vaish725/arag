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
    monkeypatch.setenv("ARAG_EMBEDDING_PROVIDER", "openai")
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
    monkeypatch.setenv("ARAG_EMBEDDING_PROVIDER", "openai")
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
    monkeypatch.setenv("ARAG_EMBEDDING_PROVIDER", "openai")
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
    monkeypatch.setenv("ARAG_LLM_PROVIDER", "openai")  # OpenAI answers are opt-in

    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)

    out = llm_backend.synthesize_answer("q", ["ctx1", "ctx2"])
    assert (out["answer"], out["citations"], out["used_fallback"]) == ("fake answer", [], False)
    assert out["model"] == "gpt-4o-mini" and out["input_tokens"] is None and out["output_tokens"] is None  # the fake reports no usage


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


# ---- OpenAI answers are opt-in: OPENAI_API_KEY alone activates nothing ---------------------------------
def _fake_openai_that_must_not_be_used(monkeypatch):
    calls = []

    class Boom:
        def __init__(self, *a, **k):
            calls.append("OpenAI client constructed")
            raise AssertionError("the OpenAI client must not be used")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Boom))
    return calls


def test_openai_key_alone_activates_nothing(monkeypatch):
    calls = _fake_openai_that_must_not_be_used(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ARAG_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    assert llm_backend.get_llm_provider() == "anthropic"
    assert llm_backend.llm_available() is False
    with pytest.raises(RuntimeError, match="No LLM backend configured"):
        llm_backend.synthesize_answer("q", ["ctx"])
    assert calls == []


def test_openai_key_does_not_override_an_anthropic_key(monkeypatch):
    calls = _fake_openai_that_must_not_be_used(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
    monkeypatch.delenv("ARAG_LLM_PROVIDER", raising=False)

    class FakeMessages:
        @staticmethod
        def create(**kw):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"answer": "from anthropic", "citations": []}')])

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda api_key=None: SimpleNamespace(messages=FakeMessages)))
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    assert llm_backend.synthesize_answer("q", ["ctx"])["answer"] == "from anthropic"
    assert calls == []


def test_openai_needs_the_explicit_provider_and_its_key(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", make_fake_openai_module(batch_embedding=[0.0]))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ARAG_LLM_PROVIDER", "openai")
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    assert llm_backend.llm_available() is False  # the provider alone is not enough
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    assert llm_backend.llm_available() is True
    assert llm_backend.synthesize_answer("q", ["ctx"])["answer"] == "fake answer"


def test_anthropic_key_is_ignored_when_openai_is_selected(monkeypatch):
    monkeypatch.setenv("ARAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    assert llm_backend.llm_available() is False


def test_unknown_llm_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("ARAG_LLM_PROVIDER", "carrier-pigeon")
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    with pytest.raises(ValueError, match="ARAG_LLM_PROVIDER"):
        llm_backend.llm_available()


# ---- ARAG_LLM_PROVIDER=openai without the openai package: a clear error, not an ImportError traceback -------
def test_openai_provider_without_the_package_fails_with_a_clear_message(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "openai", None)  # `import openai` now raises ImportError
    monkeypatch.setenv("ARAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import arag.llm_backend as llm_backend

    importlib.reload(llm_backend)
    with caplog.at_level("WARNING"):
        assert llm_backend.llm_available() is False
    assert "pip install openai" in caplog.text  # the explicit choice is not silently ignored
    with pytest.raises(RuntimeError) as err:
        llm_backend.synthesize_answer("q", ["ctx"])
    assert err.type is RuntimeError and not isinstance(err.value, ImportError)
    assert "ARAG_LLM_PROVIDER=openai" in str(err.value) and "pip install openai" in str(err.value)
    assert "No module named" not in str(err.value)
