"""Runtime helpers moved into arag package."""

from __future__ import annotations

import contextvars
import datetime
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, start_http_server

    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False

_PROM_API_CALLS = None
_PROM_TOKENS = None
_PROM_AGENT_REQUESTS = None


class BudgetExceededError(RuntimeError):
    pass


def setup_logging(log_file: str = "logs/arag.log", level: int = logging.INFO) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)


def start_metrics_server(port: int = 8000):
    if not _PROM_AVAILABLE:
        return False
    global _PROM_API_CALLS, _PROM_TOKENS, _PROM_AGENT_REQUESTS
    _PROM_API_CALLS = Counter("arag_api_calls_total", "Total external API calls")
    _PROM_TOKENS = Counter("arag_tokens_total", "Total tokens consumed")
    _PROM_AGENT_REQUESTS = Counter("arag_agent_requests_total", "Total agent requests")
    start_http_server(port)
    return True


def inc_api_call_metric(n: int = 1):
    if _PROM_AVAILABLE and _PROM_API_CALLS is not None:
        _PROM_API_CALLS.inc(n)


def inc_token_metric(n: int = 1):
    if _PROM_AVAILABLE and _PROM_TOKENS is not None:
        _PROM_TOKENS.inc(n)


class RateLimiter:
    def __init__(self, capacity: float = 10.0, refill_rate: float = 1.0):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last = now

    def acquire(self, n: float = 1.0, timeout: Optional[float] = 10.0) -> bool:
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return True
            if deadline and time.monotonic() > deadline:
                return False
            time.sleep(0.01)


class BudgetMonitor:
    def __init__(self, max_api_calls: Optional[int] = None, max_tokens: Optional[int] = None):
        self.max_api_calls = max_api_calls
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.api_calls = 0
        self.tokens = 0

    def consume_api_call(self, n: int = 1) -> None:
        with self.lock:
            self.api_calls += n
            if self.max_api_calls is not None and self.api_calls > self.max_api_calls:
                raise BudgetExceededError(f"API call budget exceeded: {self.api_calls} > {self.max_api_calls}")

    def consume_tokens(self, n: int) -> None:
        with self.lock:
            self.tokens += int(n)
            if self.max_tokens is not None and self.tokens > self.max_tokens:
                raise BudgetExceededError(f"Token budget exceeded: {self.tokens} > {self.max_tokens}")

    def remaining_api_calls(self) -> Optional[int]:
        if self.max_api_calls is None:
            return None
        with self.lock:
            return max(0, self.max_api_calls - self.api_calls)

    def remaining_tokens(self) -> Optional[int]:
        if self.max_tokens is None:
            return None
        with self.lock:
            return max(0, self.max_tokens - self.tokens)


_GLOBAL_RATE_LIMITER: Optional[RateLimiter] = None


def get_global_rate_limiter() -> RateLimiter:
    global _GLOBAL_RATE_LIMITER
    if _GLOBAL_RATE_LIMITER is None:
        _GLOBAL_RATE_LIMITER = RateLimiter(capacity=10.0, refill_rate=5.0)
    return _GLOBAL_RATE_LIMITER


# Spend caps. API calls are counted per request (one embeddings or chat
# request = 1), not per text. Two scopes:
#   per run : reset at the start of every run_agent() call (one question)
#   per day : whole process, resets when the calendar day changes
# Override with the env vars below; use the word "none" to remove a cap.
# A number is taken literally, so 0 means "no calls allowed".
DEFAULTS = {
    "ARAG_MAX_API_CALLS_PER_RUN": 50,
    "ARAG_MAX_TOKENS_PER_RUN": 100_000,
    "ARAG_MAX_API_CALLS_PER_DAY": 500,
    "ARAG_MAX_TOKENS_PER_DAY": 1_000_000,
}


_OLD_VARS = {
    "ARAG_MAX_API_CALLS": "ARAG_MAX_API_CALLS_PER_RUN / ARAG_MAX_API_CALLS_PER_DAY",
    "ARAG_MAX_TOKENS": "ARAG_MAX_TOKENS_PER_RUN / ARAG_MAX_TOKENS_PER_DAY",
}


def _warn_old_vars() -> None:
    for old, new in _OLD_VARS.items():
        if os.environ.get(old) is not None:
            logging.getLogger(__name__).warning(
                "%s is set but no longer used; set %s instead.", old, new
            )


def _limit_from_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return DEFAULTS[name]
    if raw.strip().lower() == "none":
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULTS[name]


def _make_monitor(scope: str) -> BudgetMonitor:
    return BudgetMonitor(
        max_api_calls=_limit_from_env(f"ARAG_MAX_API_CALLS_PER_{scope}"),
        max_tokens=_limit_from_env(f"ARAG_MAX_TOKENS_PER_{scope}"),
    )


# The per-run budget lives in a ContextVar so concurrent requests (Streamlit
# runs each session in its own thread) each get their own and never reset one
# another. The daily budget is shared and guarded by a lock.
_RUN_BUDGET: contextvars.ContextVar[Optional[BudgetMonitor]] = contextvars.ContextVar(
    "arag_run_budget", default=None
)


class _Budget:
    """Charges the current run's budget (if any) and the shared daily budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.day = datetime.date.today()
        self.daily = _make_monitor("DAY")

    def _daily(self) -> BudgetMonitor:
        with self._lock:
            today = datetime.date.today()
            if today != self.day:
                self.day = today
                self.daily = _make_monitor("DAY")
            return self.daily

    def start_run(self) -> None:
        _RUN_BUDGET.set(_make_monitor("RUN"))

    def consume_api_call(self, n: int = 1) -> None:
        run = _RUN_BUDGET.get()
        if run is not None:
            run.consume_api_call(n)
        self._daily().consume_api_call(n)

    def consume_tokens(self, n: int) -> None:
        run = _RUN_BUDGET.get()
        if run is not None:
            run.consume_tokens(n)
        self._daily().consume_tokens(n)


_GLOBAL_BUDGET: Optional[_Budget] = None
_BUDGET_INIT_LOCK = threading.Lock()


def get_global_budget() -> _Budget:
    global _GLOBAL_BUDGET
    if _GLOBAL_BUDGET is None:
        with _BUDGET_INIT_LOCK:
            if _GLOBAL_BUDGET is None:
                _warn_old_vars()
                _GLOBAL_BUDGET = _Budget()
    return _GLOBAL_BUDGET


def start_run() -> None:
    """Begin a new per-question budget. Called at the start of run_agent()."""
    get_global_budget().start_run()


def reset_global_budget() -> None:
    """Drop the cached budget so the environment is re-read (used by tests)."""
    global _GLOBAL_BUDGET
    _GLOBAL_BUDGET = None
