"""Degraded mode is visible (Streamlit banner, CLI stderr line, run_agent field) and the startup
health line says what loaded."""
import json
import logging
import threading
from pathlib import Path

import pytest

pytest.importorskip("faiss")
from streamlit.testing.v1 import AppTest  # noqa: E402
from test_model_fingerprint import FakeModel, REVISION, local_world  # noqa: E402,F401  (fixture)

from arag import agent_loop, health  # noqa: E402
from arag import embedding_backend as eb  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
QUESTION = "Who directed Inception and did they win any Academy Awards?"
BANNER = "Meaning search unavailable: keyword-only mode. Reason: "


def force_mismatch(monkeypatch):
    """Configure a different embedding model than the committed indexes were built with."""
    monkeypatch.setattr(eb, "get_embedding_info", lambda: {"provider": "local", "model": "some/other-model", "dim": 384})


def banners(at):
    return [w.value for w in at.warning if w.value.startswith(BANNER)]


def element_order(at):
    return [(e.type, e.value if isinstance(getattr(e, "value", None), str) else "") for e in at.main]


# ---------------------------------------------------------------- Streamlit banner
def test_banner_is_shown_before_and_above_the_answer_when_degraded(monkeypatch):
    force_mismatch(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(banners(at)) == 1 and "Embedding mismatch" in banners(at)[0]  # before any question

    at.text_input[0].set_value(QUESTION).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(banners(at)) == 1, "exactly one banner once an answer is shown"
    order = element_order(at)
    banner_at = next(i for i, (t, v) in enumerate(order) if t == "warning" and v.startswith(BANNER))
    answer_at = next(i for i, (t, v) in enumerate(order) if t == "subheader" and v == "Answer")
    assert banner_at < answer_at, order


@pytest.mark.needs_model
def test_banner_is_hidden_when_vector_search_is_on():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not banners(at)
    at.text_input[0].set_value(QUESTION).run()
    at.button[0].click().run()
    assert not at.exception and not banners(at)
    assert "Christopher Nolan" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------- CLI
def test_cli_prints_the_degraded_line_to_stderr_before_the_answer(monkeypatch, capsys):
    force_mismatch(monkeypatch)
    seen = {}
    real_run_agent = agent_loop.run_agent

    def spy(*a, **k):
        seen["stderr_before_answer"] = capsys.readouterr().err  # what was already on stderr when the answer started
        return real_run_agent(*a, **k)

    monkeypatch.setattr(agent_loop, "run_agent", spy)
    assert agent_loop.main(["--question", QUESTION]) == 0
    out, err = capsys.readouterr()
    assert BANNER in seen["stderr_before_answer"], "the line must be printed before the answer is produced"
    result = json.loads(out)
    assert result["vector_search"]["enabled"] is False and "Embedding mismatch" in result["vector_search"]["reason"]
    assert "Christopher Nolan" in result["answer"]


@pytest.mark.needs_model
def test_cli_prints_no_degraded_line_when_vector_search_is_on(capsys):
    assert agent_loop.main(["--question", QUESTION]) == 0
    out, err = capsys.readouterr()
    assert BANNER not in err and json.loads(out)["vector_search"] == {"enabled": True, "reason": ""}


# ---------------------------------------------------------------- run_agent field
def test_run_agent_reports_a_semantic_search_failure_during_the_run(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("embedding service exploded")

    monkeypatch.setattr(agent_loop, "semantic_search", broken)
    res = agent_loop.run_agent(QUESTION, allow_fallback=False, enable_director_flow=True)
    vs = res["vector_search"]
    # whether or not the static check passes, a failing semantic_search is never silent
    assert vs["enabled"] is False and vs["reason"]
    assert "Christopher Nolan" in res["answer"]


# ---------------------------------------------------------------- startup health line
def health_records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "arag.health"]


def test_health_line_has_provider_model_revision_dims_counts_and_state(local_world, caplog):
    rt, paths, tmp_path = local_world
    caplog.set_level(logging.INFO, logger="arag.health")
    health._last_logged = None
    rt.load_indexes(**paths)
    lines = health_records(caplog)
    assert len(lines) == 1, lines
    line = lines[0]
    assert "embedding=local/BAAI/bge-small-en-v1.5" in line
    assert f"revision={REVISION[:8]}" in line
    assert "dims(chunk/sentence)=64/64" in line
    assert f"chunks={len(rt._chunks)}" in line and f"sentences={len(rt._sentences)}" in line
    assert line.endswith("vector_search=ON")


def test_health_line_says_off_with_the_reason(local_world, caplog, monkeypatch):
    rt, paths, tmp_path = local_world
    monkeypatch.setattr(eb, "get_embedding_info", lambda: {"provider": "local", "model": "some/other-model", "dim": 64})
    caplog.set_level(logging.INFO, logger="arag.health")
    health._last_logged = None
    rt.load_indexes(**paths)
    (line,) = health_records(caplog)
    assert "vector_search=OFF (reason: Embedding mismatch" in line and "some/other-model" in line


def test_health_line_is_logged_once_not_on_every_reload(local_world, caplog):
    rt, paths, tmp_path = local_world
    caplog.set_level(logging.INFO, logger="arag.health")
    health._last_logged = None
    for _ in range(3):
        rt.load_indexes(**paths)  # the agent re-reads the indexes on every question
    assert len(health_records(caplog)) == 1


def test_health_line_when_the_model_cannot_load(local_world, caplog):
    rt, paths, tmp_path = local_world
    FakeModel.fail = True
    eb._local_models.clear()
    eb._fingerprints.clear()
    caplog.set_level(logging.INFO, logger="arag.health")
    health._last_logged = None
    rt.load_indexes(**paths)
    (line,) = health_records(caplog)
    assert "vector_search=OFF (reason: The local embedding model" in line and "could not be loaded" in line


# ---------------------------------------------------------------- step budget (max_steps)
def test_max_steps_is_enforced_and_reported():
    res = agent_loop.run_agent(QUESTION, max_steps=2, allow_fallback=False, enable_director_flow=True)
    assert res["budget_exhausted"] is True
    assert res["tool_calls"] == 2
    assert isinstance(res["answer"], str)  # the run still finishes with what it has


def test_default_budget_is_not_hit_by_any_regression_film():
    baseline = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "regression_baseline.json").read_text())
    for film in baseline:
        res = agent_loop.run_agent(f"Who directed {film} and did they win any Academy Awards?", allow_fallback=False, enable_director_flow=True)
        assert res["budget_exhausted"] is False, film
        assert res["tool_calls"] < agent_loop.DEFAULT_MAX_STEPS, (film, res["tool_calls"])


def test_max_steps_none_means_unlimited():
    res = agent_loop.run_agent(QUESTION, max_steps=None, allow_fallback=False, enable_director_flow=True)
    assert res["budget_exhausted"] is False and res["tool_calls"] > 0


def test_step_budgets_are_per_run_even_with_concurrent_runs():
    out = {}

    def run(name, limit):
        out[name] = agent_loop.run_agent(QUESTION, max_steps=limit, allow_fallback=False, enable_director_flow=True)

    threads = [threading.Thread(target=run, args=("tight", 1)), threading.Thread(target=run, args=("roomy", agent_loop.DEFAULT_MAX_STEPS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert out["tight"]["budget_exhausted"] is True and out["tight"]["tool_calls"] == 1
    assert out["roomy"]["budget_exhausted"] is False
