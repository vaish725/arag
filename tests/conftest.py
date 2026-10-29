"""Test-suite guard: tests must never reach a real model API.

* Provider keys are removed from the environment for every test.
* ``load_dotenv`` is neutralised so a local ``.env`` cannot put them back.
* Any attempt to open a network connection fails the test.
"""

import socket

import pytest

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
    yield
