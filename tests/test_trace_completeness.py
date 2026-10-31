"""tool_trace records every tool call (including the chunk_read probes) and each one counts toward max_steps:
the number of recorded calls equals the number of actual calls."""
import threading
from collections import Counter

import pytest

from arag import agent_loop, health
from arag import retrieval_tools as rt

TOOLS = ("semantic_search", "keyword_search", "chunk_read")
QUESTIONS = [
    "Who directed Inception and did they win any Academy Awards?",
    "Who directed Titanic (1997 film) and did they win any Academy Awards?",
    "Who directed Last Action Hero and did they win any Academy Awards?",
    "Who directed The Godfather Part II and did they win any Academy Awards?",
]


@pytest.fixture
def actual_calls(monkeypatch):
    """Count the calls the retrieval layer really receives (the real functions still run)."""
    counts = Counter()
    for name in TOOLS:
        real = getattr(rt, name)

        def counting(*a, _real=real, _name=name, **k):
            counts[_name] += 1
            return _real(*a, **k)

        monkeypatch.setattr(rt, name, counting)
    return counts


def recorded(res):
    return Counter(e["tool"] for e in res["tool_trace"] if e["tool"] in TOOLS and not e.get("blocked"))


@pytest.mark.parametrize("question", QUESTIONS)
def test_recorded_calls_equal_actual_calls(actual_calls, question):
    res = agent_loop.run_agent(question, allow_fallback=False, enable_director_flow=True)
    assert sum(actual_calls.values()) > 0
    assert recorded(res) == actual_calls, "every actual call (per tool) must appear in tool_trace"
    assert res["tool_calls"] == sum(actual_calls.values()), "and each one counts toward max_steps"
    assert res["budget_exhausted"] is False


def test_chunk_read_probes_are_in_the_trace(actual_calls):
    res = agent_loop.run_agent(QUESTIONS[1], allow_fallback=False, enable_director_flow=True)
    assert actual_calls["chunk_read"] > 5, "the recipe probes many chunk ids"
    assert recorded(res)["chunk_read"] == actual_calls["chunk_read"]
    probes = [e for e in res["tool_trace"] if e["tool"] == "chunk_read"]
    assert all("chunk_id" in e["args"] and "result_summary" in e for e in probes)


def test_entries_are_in_call_order_with_step_numbers(actual_calls):
    res = agent_loop.run_agent(QUESTIONS[0], allow_fallback=False, enable_director_flow=True)
    steps = [e["step"] for e in res["tool_trace"] if e["tool"] in TOOLS]
    assert steps == list(range(1, len(steps) + 1))


def test_blocked_attempts_are_flagged_and_do_not_count_as_calls(actual_calls):
    res = agent_loop.run_agent(QUESTIONS[0], max_steps=3, allow_fallback=False, enable_director_flow=True)
    assert res["budget_exhausted"] is True and res["tool_calls"] == 3
    assert sum(actual_calls.values()) == 3, "once the budget is spent the tools are not called"
    assert recorded(res) == actual_calls
    blocked = [e for e in res["tool_trace"] if e.get("blocked")]
    assert blocked and all(e["error"] == "step budget exhausted" for e in blocked)


def test_a_failing_call_is_recorded_with_its_error(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(rt, "semantic_search", broken)
    res = agent_loop.run_agent(QUESTIONS[0], allow_fallback=False, enable_director_flow=True)
    sem = [e for e in res["tool_trace"] if e["tool"] == "semantic_search"]
    assert sem and all("index exploded" in e["error"] for e in sem)
    assert res["tool_calls"] == len([e for e in res["tool_trace"] if e["tool"] in TOOLS and not e.get("blocked")])


def test_a_tool_substituted_at_module_level_still_shows_the_flows_own_entries(monkeypatch):
    monkeypatch.setattr(agent_loop, "semantic_search", lambda q, k=5: [])  # bypasses the recording wrapper
    res = agent_loop.run_agent(QUESTIONS[0], allow_fallback=False, enable_director_flow=True)
    assert any(e["tool"] == "semantic_search" for e in res["tool_trace"])  # the flow's own entries are kept
    assert any(e["tool"] == "chunk_read" and "step" in e for e in res["tool_trace"])  # the others are still recorded


def test_concurrent_runs_keep_separate_records():
    out = {}

    def run(name, limit):
        out[name] = agent_loop.run_agent(QUESTIONS[0], max_steps=limit, allow_fallback=False, enable_director_flow=True)

    threads = [threading.Thread(target=run, args=("tight", 4)), threading.Thread(target=run, args=("roomy", None))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for res in out.values():
        assert recorded(res) and sum(recorded(res).values()) == res["tool_calls"]
    assert out["tight"]["tool_calls"] == 4 and out["roomy"]["tool_calls"] > 4


def test_a_blocked_semantic_search_is_not_reported_as_degraded_vector_search():
    trace = [{"tool": "semantic_search", "error": "step budget exhausted", "blocked": True}]
    assert health.runtime_status(trace, (True, ""))["enabled"] is True
    trace = [{"tool": "semantic_search", "error": "model exploded"}]
    assert health.runtime_status(trace, (True, ""))["enabled"] is False
