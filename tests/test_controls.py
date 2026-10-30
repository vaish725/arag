"""Every UI control and CLI flag changes behaviour (none is accepted and ignored).

Streamlit controls: Mode, Question, Max steps, Max tokens, the fallback checkbox, Run, and the
per-session question limit. CLI flags: --question, --max_steps, --max_tokens, --allow-fallback,
--no-allow-fallback, --enable-director-flow, --no-enable-director-flow.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from arag import agent_loop

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
MODE_LLM = "semantic + BM25 + optional LLM fallback"
Q = "Who directed Inception and did they win any Academy Awards?"


def fake_result(**over):
    res = {
        "answer": "Director: Test.", "tool_trace": [], "retrieved_texts": [], "citations": [], "step_count": 1,
        "token_count": 2, "budget_exhausted": False, "tool_calls": 3, "vector_search": {"enabled": True, "reason": ""},
    }
    res.update(over)
    return res


@pytest.fixture
def calls(monkeypatch):
    """Replace run_agent with a recorder (streamlit_app does `from arag.agent_loop import run_agent` on every run)."""
    recorded = []

    def fake(question, **kwargs):
        recorded.append({"question": question, **kwargs})
        return fake_result()

    monkeypatch.setattr(agent_loop, "run_agent", fake)
    return recorded


def run_app(**actions):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


# ---------------------------------------------------------------- Streamlit controls
def test_question_max_steps_max_tokens_and_run_reach_run_agent(calls):
    at = run_app()
    at.text_input[0].set_value(Q)
    at.slider[0].set_value(7)  # Max steps (tool calls)
    at.slider[1].set_value(321)  # Max tokens
    assert not calls, "nothing runs until Run is clicked"
    at.button[0].click().run()
    assert len(calls) == 1
    assert calls[0]["question"] == Q and calls[0]["max_steps"] == 7 and calls[0]["max_tokens"] == 321
    assert calls[0]["enable_director_flow"] is True


def test_mode_and_fallback_checkbox_set_allow_fallback(calls):
    at = run_app()
    at.text_input[0].set_value(Q)
    at.button[0].click().run()
    assert calls[-1]["allow_fallback"] is False  # evidence-only mode

    at.selectbox[0].select(MODE_LLM).run()
    at.button[0].click().run()
    assert calls[-1]["allow_fallback"] is True  # LLM mode, checkbox ticked by default

    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert calls[-1]["allow_fallback"] is False  # LLM mode, checkbox unticked


def test_session_limit_stops_further_questions(calls, monkeypatch):
    monkeypatch.setenv("ARAG_MAX_QUESTIONS_PER_SESSION", "1")
    at = run_app()
    at.text_input[0].set_value(Q)
    at.button[0].click().run()
    assert len(calls) == 1
    at.button[0].click().run()
    assert len(calls) == 1, "the second question must not run"
    assert any("Session limit reached" in e.value for e in at.error)


def test_budget_exhausted_is_shown_to_the_user(monkeypatch):
    monkeypatch.setattr(agent_loop, "run_agent", lambda q, **k: fake_result(budget_exhausted=True, tool_calls=5))
    at = run_app()
    at.text_input[0].set_value(Q)
    at.button[0].click().run()
    assert any("Step budget exhausted after 5 tool calls" in w.value for w in at.warning)


# ---------------------------------------------------------------- CLI flags
def cli(monkeypatch, argv):
    seen = {}

    def fake(question, **kwargs):
        seen.update(question=question, **kwargs)
        return fake_result()

    monkeypatch.setattr(agent_loop, "run_agent", fake)
    assert agent_loop.main(argv) == 0
    return seen


def test_cli_question_max_steps_and_max_tokens(monkeypatch):
    seen = cli(monkeypatch, ["--question", "Who directed X?", "--max_steps", "9", "--max_tokens", "77"])
    assert (seen["question"], seen["max_steps"], seen["max_tokens"]) == ("Who directed X?", 9, 77)


def test_cli_defaults(monkeypatch):
    seen = cli(monkeypatch, [])
    assert seen["max_steps"] == agent_loop.DEFAULT_MAX_STEPS and seen["max_tokens"] == 2000
    assert seen["allow_fallback"] is True and seen["enable_director_flow"] is True


@pytest.mark.parametrize(
    "flag, key, value",
    [
        ("--allow-fallback", "allow_fallback", True),
        ("--no-allow-fallback", "allow_fallback", False),
        ("--enable-director-flow", "enable_director_flow", True),
        ("--no-enable-director-flow", "enable_director_flow", False),
    ],
)
def test_cli_boolean_flags_reach_run_agent(monkeypatch, flag, key, value):
    assert cli(monkeypatch, [flag])[key] is value


# ---------------------------------------------------------------- behaviour behind the controls
def test_max_tokens_caps_the_llm_call(monkeypatch):
    import arag.llm_backend as llm_backend

    seen = {}

    def fake_synth(q, ctxs, **kw):
        seen.update(kw)
        return {"answer": "x", "citations": [], "used_fallback": False}

    monkeypatch.setattr(llm_backend, "llm_available", lambda: True)
    monkeypatch.setattr(llm_backend, "synthesize_answer", fake_synth)
    agent_loop.run_agent("What is the capital of France?", max_tokens=123, allow_fallback=True)
    assert seen["max_tokens"] == 123 and seen["allow_fallback"] is True


def test_director_flow_flag_changes_the_answer():
    on = agent_loop.run_agent(Q, allow_fallback=False, enable_director_flow=True)
    off = agent_loop.run_agent(Q, allow_fallback=False, enable_director_flow=False)
    assert on["answer"].startswith("Director:")  # the two-hop director->award template
    assert not off["answer"].startswith("Director:") and off["answer"] != on["answer"]  # generic retrieval dump


def test_allow_fallback_gates_the_llm(monkeypatch):
    import arag.llm_backend as llm_backend

    called = []
    monkeypatch.setattr(llm_backend, "llm_available", lambda: True)
    monkeypatch.setattr(llm_backend, "synthesize_answer", lambda q, c, **kw: called.append(kw) or {"answer": "LLM", "citations": [], "used_fallback": True})
    # an unanswerable director question: only a run that allows fallback may ask the LLM
    unknown_q = "Who directed Zzyzx Qwerty and did they win any Academy Awards?"
    ctx = {"chunk_id": "c1", "text": "A passage that names nobody.", "score": 1.0}
    monkeypatch.setattr(agent_loop, "semantic_search", lambda q, k=3: [ctx])
    monkeypatch.setattr(agent_loop, "keyword_search", lambda *a, **k: [ctx])
    monkeypatch.setattr(agent_loop, "chunk_read", lambda cid: {"chunk_id": cid, "source": "s", "full_context": ctx["text"]})
    agent_loop.run_agent(unknown_q, allow_fallback=False, enable_director_flow=True)
    assert not called
    agent_loop.run_agent(unknown_q, allow_fallback=True, enable_director_flow=True)
    assert called
