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


def test_global_budget_has_safe_default(monkeypatch):
    monkeypatch.delenv("ARAG_MAX_API_CALLS", raising=False)
    assert runtime_ops.get_global_budget().max_api_calls == runtime_ops.DEFAULT_MAX_API_CALLS


def test_env_overrides_and_zero_means_unlimited(monkeypatch):
    monkeypatch.setenv("ARAG_MAX_API_CALLS", "7")
    assert runtime_ops.get_global_budget().max_api_calls == 7
    runtime_ops.reset_global_budget()
    monkeypatch.setenv("ARAG_MAX_API_CALLS", "0")
    assert runtime_ops.get_global_budget().max_api_calls is None


def test_rate_limiter_runs_dry():
    rl = RateLimiter(capacity=2, refill_rate=0.0)
    assert rl.acquire(timeout=0.05)
    assert rl.acquire(timeout=0.05)
    assert not rl.acquire(timeout=0.05)


def test_embedding_backend_stops_at_cap(monkeypatch):
    import arag.embedding_backend as eb

    monkeypatch.setenv("ARAG_MAX_API_CALLS", "1")
    runtime_ops.reset_global_budget()
    # First call spends the single allowed call (fails later: no backend/key).
    with pytest.raises(RuntimeError):
        eb.get_embeddings(["a"])
    with pytest.raises(BudgetExceededError):
        eb.get_embeddings(["b"])
