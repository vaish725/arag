import pytest

from arag import runtime_ops
from arag.runtime_ops import BudgetExceededError, BudgetMonitor, RateLimiter


@pytest.fixture(autouse=True)
def _fresh_budget(monkeypatch):
    monkeypatch.setattr(runtime_ops, "_GLOBAL_BUDGET", None)
    yield
    runtime_ops.reset_global_budget()


def test_api_call_cap_raises():
    b = BudgetMonitor(max_api_calls=2)
    b.consume_api_call()
    b.consume_api_call()
    with pytest.raises(BudgetExceededError):
        b.consume_api_call()


def test_token_cap_raises():
    b = BudgetMonitor(max_tokens=10)
    b.consume_tokens(10)
    with pytest.raises(BudgetExceededError):
        b.consume_tokens(1)


def test_defaults_are_finite(monkeypatch):
    for name in runtime_ops.DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    b = runtime_ops.get_global_budget()
    b.start_run()
    assert runtime_ops._RUN_BUDGET.get().max_api_calls == 50
    assert b.daily.max_api_calls == 500


def test_zero_is_literal_and_none_removes_cap(monkeypatch):
    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "0")
    assert runtime_ops._limit_from_env("ARAG_MAX_API_CALLS_PER_DAY") == 0
    with pytest.raises(BudgetExceededError):
        runtime_ops.get_global_budget().consume_api_call()
    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "none")
    assert runtime_ops._limit_from_env("ARAG_MAX_API_CALLS_PER_DAY") is None


def test_run_budget_resets_each_question(monkeypatch):
    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_RUN", "2")
    b = runtime_ops.get_global_budget()
    runtime_ops.start_run()
    b.consume_api_call()
    b.consume_api_call()
    with pytest.raises(BudgetExceededError):
        b.consume_api_call()
    runtime_ops.start_run()  # next question starts fresh
    b.consume_api_call()


def test_daily_cap_spans_runs(monkeypatch):
    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "3")
    b = runtime_ops.get_global_budget()
    for _ in range(3):
        runtime_ops.start_run()
        b.consume_api_call()
    runtime_ops.start_run()
    with pytest.raises(BudgetExceededError):
        b.consume_api_call()


def test_daily_budget_rolls_over_at_midnight(monkeypatch):
    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "1")
    b = runtime_ops.get_global_budget()
    b.consume_api_call()
    b.day = b.day.replace(year=b.day.year - 1)  # pretend it is a new day
    b.consume_api_call()


def test_rate_limiter_runs_dry():
    rl = RateLimiter(capacity=2, refill_rate=0.0)
    assert rl.acquire(timeout=0.05)
    assert rl.acquire(timeout=0.05)
    assert not rl.acquire(timeout=0.05)


def test_embedding_backend_stops_at_cap(monkeypatch):
    import arag.embedding_backend as eb

    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "1")
    runtime_ops.reset_global_budget()
    # First call spends the single allowed call (fails later: no backend/key).
    with pytest.raises(RuntimeError):
        eb.get_embeddings(["a"])
    with pytest.raises(BudgetExceededError):
        eb.get_embeddings(["b"])


def test_concurrent_runs_do_not_reset_each_other(monkeypatch):
    """Two threads, per-run cap of 1: each gets exactly one call, then stops."""
    import threading

    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_RUN", "1")
    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "none")
    b = runtime_ops.get_global_budget()
    started = threading.Barrier(2)
    first_done = threading.Barrier(2)
    results = {}

    def worker(name):
        runtime_ops.start_run()
        started.wait(timeout=5)  # both runs have started before either spends
        b.consume_api_call()
        first_done.wait(timeout=5)  # both have spent their one call
        try:
            b.consume_api_call()
            results[name] = "no-error"
        except BudgetExceededError:
            results[name] = "stopped"

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert results == {"a": "stopped", "b": "stopped"}


def test_daily_counter_is_thread_safe(monkeypatch):
    import threading

    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "none")
    b = runtime_ops.get_global_budget()

    def spend():
        for _ in range(1000):
            b.consume_api_call()

    threads = [threading.Thread(target=spend) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert b.daily.api_calls == 8000


def test_old_env_vars_log_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("ARAG_MAX_API_CALLS", "5")
    with caplog.at_level("WARNING"):
        runtime_ops.get_global_budget()
    assert "ARAG_MAX_API_CALLS is set but no longer used" in caplog.text
