import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for test imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def test_agent_with_mocked_llm(monkeypatch):
    """Agent should call semantic_search -> chunk_read and then synthesize
    via LLM when available.
    """
    # Import package modules fresh
    import arag.agent_loop as agent_loop
    import arag.llm_backend as llm_backend
    import arag.retrieval_tools as retrieval_tools

    importlib.reload(agent_loop)
    importlib.reload(retrieval_tools)
    importlib.reload(llm_backend)

    # Mock retrieval behavior
    def fake_semantic_search(query, k=3):
        return [{"chunk_id": "c1", "text": "fake snippet", "score": 1.0}]

    def fake_chunk_read(chunk_id):
        return {
            "chunk_id": chunk_id,
            "source": "test",
            "full_context": "Full context about director.",
        }

    monkeypatch.setattr(retrieval_tools, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(retrieval_tools, "keyword_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieval_tools, "chunk_read", fake_chunk_read)
    # Ensure agent_loop's local bindings point to the mocked functions
    monkeypatch.setattr(agent_loop, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(agent_loop, "keyword_search", lambda *a, **k: [])
    monkeypatch.setattr(agent_loop, "chunk_read", fake_chunk_read)

    # Mock LLM backend to assert it is called and to return a predictable answer
    monkeypatch.setattr(llm_backend, "llm_available", lambda: True)
    monkeypatch.setattr(
        llm_backend, "synthesize_answer", lambda q, ctxs, **kw: {"answer": "SYNTHESIZED BY LLM", "citations": [], "used_fallback": False}
    )

    res = agent_loop.run_agent(
        "Who directed X?", max_steps=4, max_tokens=1000, allow_fallback=False
    )

    assert isinstance(res, dict)
    assert res["answer"] == "SYNTHESIZED BY LLM"
    # Ensure the trace contains a semantic_search call and a chunk_read call
    tools = [t.get("tool") for t in res["tool_trace"]]
    assert "semantic_search" in tools
    assert "chunk_read" in tools


def test_agent_without_llm_falls_back(monkeypatch):
    """When no LLM is available, agent should return concatenated retrieved contexts."""
    import arag.agent_loop as agent_loop
    import arag.llm_backend as llm_backend
    import arag.retrieval_tools as retrieval_tools

    importlib.reload(agent_loop)
    importlib.reload(retrieval_tools)
    importlib.reload(llm_backend)

    def fake_semantic_search(query, k=3):
        return [{"chunk_id": "c2", "text": "snippet2", "score": 1.0}]

    def fake_chunk_read(chunk_id):
        return {
            "chunk_id": chunk_id,
            "source": "test",
            "full_context": "Some full context here.",
        }

    monkeypatch.setattr(retrieval_tools, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(retrieval_tools, "keyword_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieval_tools, "chunk_read", fake_chunk_read)
    monkeypatch.setattr(agent_loop, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(agent_loop, "keyword_search", lambda *a, **k: [])
    monkeypatch.setattr(agent_loop, "chunk_read", fake_chunk_read)

    # Ensure agent uses heuristic fallback (no LLM)
    monkeypatch.setattr(llm_backend, "llm_available", lambda: False)

    res = agent_loop.run_agent(
        "Who directed Y?", max_steps=4, max_tokens=1000, allow_fallback=False
    )
    assert isinstance(res, dict)
    assert "Some full context here." in res["answer"]


from arag import agent_loop  # noqa: E402


def _stub_llm(monkeypatch):
    import arag.llm_backend as llm_backend

    calls = []

    def fake_synth(q, ctxs, **kw):
        calls.append(q)
        return {"answer": "LLM ANSWER", "citations": [], "used_fallback": True}

    monkeypatch.setattr(llm_backend, "llm_available", lambda: True)
    monkeypatch.setattr(llm_backend, "synthesize_answer", fake_synth)
    return calls


def test_director_flow_with_fallback_uses_template_not_blank(monkeypatch):
    """C-02: fallback mode used to return answer == '' for director questions."""
    calls = _stub_llm(monkeypatch)
    res = agent_loop.run_agent(
        "Who directed Inception and did they win any Academy Awards?",
        allow_fallback=True,
        enable_director_flow=True,
    )
    assert "Christopher Nolan" in res["answer"]
    assert calls == [], "LLM must not be called when the template found the director"


def test_director_flow_with_fallback_calls_llm_when_director_unknown(monkeypatch):
    calls = _stub_llm(monkeypatch)
    ctx = {"chunk_id": "c1", "text": "A passage that names nobody.", "score": 1.0}
    monkeypatch.setattr(agent_loop, "semantic_search", lambda q, k=3: [ctx])
    monkeypatch.setattr(agent_loop, "keyword_search", lambda *a, **k: [ctx])
    monkeypatch.setattr(
        agent_loop, "chunk_read", lambda cid: {"chunk_id": cid, "source": "s", "full_context": ctx["text"]}
    )
    res = agent_loop.run_agent(
        "Who directed Zzyzx Qwerty and did they win any Academy Awards?",
        allow_fallback=True,
        enable_director_flow=True,
    )
    assert calls, "LLM should be asked when the director is Unknown"
    assert res["answer"] == "LLM ANSWER"
