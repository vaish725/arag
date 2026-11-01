"""Every LLM decision is recorded in tool_trace (called with model and tokens, skipped with a reason, failed
with the error), and an Unknown verdict with fallback on really calls the client. Fake client only."""
import sys

import pytest

from arag import agent_loop

INCEPTION = "Who directed Inception and did they win any Academy Awards?"  # director found, verdict Unknown
LA_LA_LAND = "Who directed La La Land and did they win any Academy Awards?"  # verdict Won
UNKNOWN_FILM = "Who directed Zzyzx Qwerty and did they win any Academy Awards?"  # director not found


def llm_entries(res):
    return [e for e in res["tool_trace"] if e.get("tool") == "llm"]


def run(question, allow_fallback=True):
    return agent_loop.run_agent(question, allow_fallback=allow_fallback, enable_director_flow=True)


@pytest.mark.needs_model
def test_unknown_verdict_with_fallback_on_calls_the_client_once(fake_anthropic):
    res = run(INCEPTION)
    assert fake_anthropic.calls == 1
    (entry,) = llm_entries(res)
    assert entry["decision"] == "called"
    assert entry["model"] == "claude-haiku-5-5" and (entry["input_tokens"], entry["output_tokens"]) == (812, 64)
    assert res["answer"] == "LLM ANSWER"
    # the corpus-evidence answer is kept next to it
    assert res["template_answer"].startswith("Director: Christopher Nolan.") and "Unknown" in res["template_answer"]
    assert res["llm"]["decision"] == "called" and res["llm"]["input_tokens"] == 812


def test_director_not_found_still_asks_the_llm(fake_anthropic):
    res = run(UNKNOWN_FILM)
    assert fake_anthropic.calls == 1 and res["answer"] == "LLM ANSWER" and llm_entries(res)[0]["decision"] == "called"


def test_skipped_because_the_corpus_already_gave_a_verdict(fake_anthropic):
    res = run(LA_LA_LAND)
    assert fake_anthropic.calls == 0
    (entry,) = llm_entries(res)
    assert entry["decision"] == "skipped" and "already gave a verdict" in entry["reason"] and "Won" in entry["reason"]


@pytest.mark.needs_model
def test_skipped_because_fallback_is_not_allowed(fake_anthropic):
    res = run(INCEPTION, allow_fallback=False)
    assert fake_anthropic.calls == 0
    (entry,) = llm_entries(res)
    assert entry["decision"] == "skipped" and "fallback not allowed" in entry["reason"]


@pytest.mark.needs_model
def test_skipped_because_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = run(INCEPTION)
    (entry,) = llm_entries(res)
    assert entry["decision"] == "skipped" and "ANTHROPIC_API_KEY is not set" in entry["reason"]
    assert res["answer"].startswith("Director: Christopher Nolan.")  # the template answer stands


@pytest.mark.needs_model
def test_skipped_because_the_package_is_not_installed(fake_anthropic, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    res = run(INCEPTION)
    (entry,) = llm_entries(res)
    assert entry["decision"] == "skipped" and "'anthropic' package is not installed" in entry["reason"]


@pytest.mark.needs_model
def test_skipped_because_openai_is_selected_without_its_key(fake_anthropic, monkeypatch):
    monkeypatch.setenv("ARAG_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = run(INCEPTION)
    (entry,) = llm_entries(res)
    assert entry["decision"] == "skipped" and "OPENAI_API_KEY is not set" in entry["reason"] and fake_anthropic.calls == 0


@pytest.mark.needs_model
def test_a_failing_client_is_recorded_and_the_template_answer_is_kept(fake_anthropic):
    fake_anthropic.fail = "service overloaded"
    res = run(INCEPTION)
    assert fake_anthropic.calls == 1
    (entry,) = llm_entries(res)
    assert entry["decision"] == "failed" and "service overloaded" in entry["error"]
    assert res["llm"]["decision"] == "failed"
    assert res["answer"].startswith("Director: Christopher Nolan.") and "Unknown" in res["answer"]  # not blank, not the LLM's


def test_generic_route_records_its_decision(fake_anthropic, monkeypatch):
    res = run("What is the capital of France?")
    (entry,) = llm_entries(res)
    assert entry["decision"] == "called" and fake_anthropic.calls == 1
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    (entry,) = llm_entries(run("What is the capital of France?"))
    assert entry["decision"] == "skipped" and "ANTHROPIC_API_KEY is not set" in entry["reason"]


def test_every_decision_shows_up_as_the_top_level_llm_field(fake_anthropic):
    assert run(LA_LA_LAND)["llm"]["decision"] == "skipped"
    assert run("Who is the president of nowhere?", allow_fallback=False)["llm"]["decision"] in ("called", "skipped", "failed")
