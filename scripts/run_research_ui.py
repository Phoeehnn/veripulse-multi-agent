"""Local launcher for live investigation and explicitly whitelisted historical replay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import sys
from uuid import uuid4
import argparse
import gradio as gr
from src.yuqing.serve.research_ui import create_research_ui, _css


def launch_research_ui() -> None:
    """Launch the local investigation workspace with a separate core Python worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--resume-case")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--youtube-replay", action="store_true")
    parser.add_argument("--youtube-lookup", action="store_true", help="Enable only explicit on-click YouTube reads")
    parser.add_argument("--youtube-case-replay", action="store_true")
    parser.add_argument("--youtube-investigation", action="store_true")
    parser.add_argument("--youtube-approvals", type=Path)
    arguments = parser.parse_args()
    if arguments.youtube_investigation and any((arguments.read_only, arguments.youtube_lookup, arguments.youtube_replay, arguments.youtube_case_replay, arguments.resume_case)):
        parser.error("--youtube-investigation 必须独立使用")
    if arguments.read_only and arguments.youtube_lookup:
        parser.error("--read-only 与 --youtube-lookup 互斥")
    root = Path(__file__).resolve().parents[1] / "traces/m6b-fixed-ten-live-40d8e85f1381488280c62ba0c7e072bb/cases/01/traces/2026-10-29"
    catalog = {"m6b-propagation-01": root / "3ca12486b33c489c8e36c04655c56e61.jsonl",
        "m6b-arbiter-01": root / "7fef3cefff114c66b43ab7b969faea5b.jsonl"}
    worktree = Path(__file__).resolve().parents[1]
    source = (worktree / "traces/research-runs" / arguments.resume_case).resolve() if arguments.resume_case else None
    if source is not None and source.parent != (worktree / "traces/research-runs").resolve():
        raise ValueError("resume_source_outside_run_root")
    investigation = None
    if arguments.youtube_investigation:
        from src.yuqing.serve.research_youtube_ui import YoutubeInvestigationUI
        run_root = worktree / "traces/youtube-investigations"
        investigation = YoutubeInvestigationUI(worktree, run_root, arguments.youtube_approvals or run_root / "authorizations.json")
    app = create_research_ui(catalog, youtube_investigation=investigation, history_root=investigation.root if investigation else None, featured_history_root=(worktree / "examples/synthetic/history") if investigation else None, resume_source=source, youtube_case_root=(worktree / "traces/tmp" / ("youtube-case-ui-" + uuid4().hex)) if arguments.youtube_case_replay else None, youtube_replay=arguments.youtube_replay, youtube_lookup=arguments.youtube_lookup, core_python=None if arguments.read_only or arguments.youtube_lookup or arguments.youtube_case_replay or arguments.youtube_investigation else Path(sys.executable))
    app.launch(server_name="127.0.0.1", server_port=arguments.port, share=False,
        inbrowser=False, show_error=True, css=_css(), theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_research_ui()
