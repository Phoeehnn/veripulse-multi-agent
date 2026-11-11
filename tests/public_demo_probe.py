"""Isolated UI dependency and network probe used by the public demo test."""
import os
from pathlib import Path
import socket
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
original = socket.socket.connect


def loopback_only(sock, address):
    """Permit the local event loop and reject external connections."""
    if address[0] not in {"127.0.0.1", "::1", "localhost"}:
        raise AssertionError("external network forbidden")
    return original(sock, address)


socket.socket.connect = loopback_only
from scripts.run_public_demo import build_app
app = build_app()
names = {row.get("api_name") for row in app.config["dependencies"]}
assert {"open_history", "refresh_history"} <= names
assert not names & {"investigate", "execute_live_sample", "sample_increment", "youtube_window"}
assert "src.yuqing.llm" not in sys.modules, "UI must not import the model gateway"
app.close()
print("readonly_ui_verified")
