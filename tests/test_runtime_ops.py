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
    assert b.run.max_api_calls == 50
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
