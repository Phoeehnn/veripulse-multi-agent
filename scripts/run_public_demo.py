"""Open bundled synthetic histories with no investigation backend or credentials."""
import os
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
cache = Path(os.environ.get("GRADIO_TEMP_DIR", str(ROOT / "traces" / "demo-temp")))
cache.mkdir(parents=True, exist_ok=True)
for name in ("TEMP", "TMP", "GRADIO_TEMP_DIR"):
    os.environ[name] = str(cache)


def build_app():
    """Use the production view with synthetic data and no live execution bindings."""
    from src.yuqing.serve.research_ui import create_research_ui
    return create_research_ui({}, history_root=ROOT / "examples/synthetic/history")


if __name__ == "__main__":
    from src.yuqing.serve.research_ui import _css
    parser = argparse.ArgumentParser(description="Read-only synthetic news investigation demo")
    parser.add_argument("--port", type=int, default=7883)
    args = parser.parse_args()
    build_app().launch(server_name="127.0.0.1", server_port=args.port,
                       share=False, inbrowser=False, css=_css())
