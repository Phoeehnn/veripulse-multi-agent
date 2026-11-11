"""Offline test boundary and dedicated temporary directories."""
import os
from pathlib import Path
import socket
from uuid import uuid4
import pytest

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY", "YOUTUBE_API_KEY"):
    os.environ.pop(name, None)
# Avoid loading any developer .env even if one happens to exist locally.
os.environ["PROJECTA_ENV_FILE"] = str(Path(__file__).parent / "no-credentials.env")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Choose a new project-local temporary folder unless the caller supplies one."""
    if config.getoption("basetemp") is None:
        config.option.basetemp = str(Path(__file__).resolve().parents[1] / "traces/tmp" / ("pytest-" + uuid4().hex))


@pytest.fixture(autouse=True)
def local_network_only(monkeypatch):
    """Allow Windows asyncio loopback while refusing external socket connections."""
    connect = socket.socket.connect
    def guarded(sock, address):
        if address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network disabled during offline tests")
        return connect(sock, address)
    monkeypatch.setattr(socket.socket, "connect", guarded)
