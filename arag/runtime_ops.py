"""Minimal runtime operations shim used by the demo.

This provides tiny local implementations for rate limiting and budget
metrics so embedding/LLM backends can import them during local tests.
This is intentionally lightweight and safe for development; hook your
real telemetry/budgeting system here in production.
"""
from __future__ import annotations

import threading
from typing import Any


class BudgetExceededError(Exception):
    pass


class _Budget:
    def __init__(self):
        self._lock = threading.Lock()
        self.api_calls = 0
        self.tokens = 0

    def consume_api_call(self, n: int = 1) -> None:
        with self._lock:
            self.api_calls += n

    def consume_api_call_if_allowed(self, n: int = 1) -> None:
        # no real limits for demo
        self.consume_api_call(n)

    def consume_tokens(self, n: int) -> None:
        with self._lock:
            self.tokens += n


class _RateLimiter:
    def acquire(self, timeout: float = 5.0) -> bool:
        return True


_global_budget = _Budget()
_global_rl = _RateLimiter()


def get_global_budget() -> _Budget:
    return _global_budget


def get_global_rate_limiter() -> _RateLimiter:
    return _global_rl


def inc_api_call_metric(n: int = 1) -> None:
    try:
        _global_budget.consume_api_call(n)
    except Exception:
        pass


def inc_token_metric(n: int = 1) -> None:
    try:
        _global_budget.consume_tokens(n)
    except Exception:
        pass


def start_metrics_server(port: int = 8000) -> None:
    # simple no-op server starter for tests; keep process alive elsewhere
    print(f"metrics-server-listening:{port}")
