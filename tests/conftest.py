"""Test-suite guard: tests must never reach a real model API.

* Provider keys are removed from the environment for every test.
* ``load_dotenv`` is neutralised so a local ``.env`` cannot put them back.
* Any attempt to open a network connection fails the test.
"""

import os
import socket

import pytest

# Never let the Hugging Face hub (used by fastembed) reach the network. Tests that
# need the real local model use it from the cache; CI fills that cache in a step
# before pytest runs (see .github/workflows/ci.yml).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    try:
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    except ImportError:
        pass

    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    # Each test starts with a fresh spend budget (the cap is process-wide).
    from arag import runtime_ops

    runtime_ops.reset_global_budget()
    yield
    runtime_ops.reset_global_budget()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_model: needs the real local embedding model from the fastembed cache"
    )


_MODEL_AVAILABLE = None


def _local_model_available() -> bool:
    global _MODEL_AVAILABLE
    if _MODEL_AVAILABLE is None:
        try:
            from arag import embedding_backend

            embedding_backend.warm_up()
            _MODEL_AVAILABLE = True
        except Exception:
            _MODEL_AVAILABLE = False
    return _MODEL_AVAILABLE


def pytest_runtest_setup(item):
    """Skip needs_model tests when the model isn't cached, unless
    ARAG_REQUIRE_MODEL=1 (set in CI), in which case a missing model is a failure."""
    if item.get_closest_marker("needs_model") and not _local_model_available():
        msg = "local embedding model not in the fastembed cache (run: python scripts/download_model.py)"
        if os.environ.get("ARAG_REQUIRE_MODEL") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
