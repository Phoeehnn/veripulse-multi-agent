"""Render one fixed offline YouTube receipt independently of the live investigation."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sys

from src.yuqing.retrieval.youtube_comments import youtube_comment_batch


WINDOWS = {
    "报道前的讨论（7 月 29 日至 8 月 4 日）": ("2024-07-29", "2024-08-05"),
    "报道发布后一周": ("2024-09-13", "2024-09-20"),
}


def render_youtube_window(window: str) -> str:
    """Project a verified sample signal only when its publication falls in the chosen window."""
    from urllib.parse import parse_qs, urlsplit
    fixture = Path(__file__).resolve().parents[3] / "docs/cards/LR-youtube-comment-replay.json"
    row = json.loads(fixture.read_bytes())
    receipt, signal = row["receipt"], row["signal"]
    parts = parse_qs(urlsplit(receipt["source_url"]).query)
    start, end = WINDOWS[window]
    def midnight(date: str) -> datetime:
        """Interpret calendar boundaries in Beijing time."""
        return datetime.fromisoformat(date + "T00:00:00+08:00")
    batch = youtube_comment_batch([dict(id=parts["lc"][0], snippet=dict(videoId=parts["v"][0],
        publishedAt=row["published_at"], textDisplay=receipt["text"]))], video_id=parts["v"][0],
        start=midnight(start), end=midnight(end), fetched_at=midnight("2026-11-06"), case_id=receipt["case_id"],
        claim=row["topic"], provenance="synthetic_fixture")
    header = f"北京时间 {start} 至 {end}（不含结束日）。回放样本 1 条，窗内 {len(batch.observations)} 条。"
    if not batch.observations:
        return header + "\n\n本次回放样本中，没有符合时间窗的评论。"
    item = batch.observations[0]
    if (item.body_sha256 != receipt["body_sha256"] or item.material_id != signal["material_id"]
            or str(item.url) != receipt["source_url"] or signal["quote"] not in item.text
            or receipt["claim_sha256"] != sha256(row["topic"].encode()).hexdigest()):
        raise ValueError("youtube_replay_receipt_mismatch")
    return (header + f"\n\n**{item.material_id}** · {signal['interpretation']}\n\n"
            + f"引文：{signal['quote']} · [查看原评论]({item.url})")

def bind_youtube_lookup(gr: object, news: object, worktree: Path) -> None:
    """Run the quota-consuming lookup only on its own explicit user button."""
    import subprocess
    from datetime import datetime, timedelta, timezone
    import os
    today = datetime.now(timezone(timedelta(hours=8))).date()
    with gr.Accordion("YouTube 实时评论样本", open=True):
        start = gr.Textbox(label="开始日（北京时间）", value=str(today-timedelta(days=6)))
        end = gr.Textbox(label="结束日（含当天）", value=str(today))
        button = gr.Button("用上方新闻文本查 YouTube（最多 2 次请求）")
        output = gr.JSON(label="候选视频与时间窗内的顶层评论")
    def submit(query: str, start: str, end: str) -> dict:
        """Pass user text as JSON, never as a shell command."""
        python = Path(sys.executable)
        try:
            process = subprocess.run([str(python), "-B", "-m", "scripts.run_youtube_lookup"],
                input=json.dumps(dict(query=query, start=start, end=end)), capture_output=True, text=True,
                encoding="utf8", cwd=worktree, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                timeout=90, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return json.loads(process.stdout) if process.returncode == 0 else {"status":"查询进程失败"}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {"status":"查询未完成"}
    button.click(submit, inputs=[news, start, end], outputs=output, concurrency_limit=1, api_visibility="private")

def bind_youtube_case_ui(gr: object, project: Path, root: Path) -> None:
    """Expose a fixed-news offline sequence with server-owned selection state."""
    from src.yuqing.serve.research_youtube_case import YoutubeCaseReplay
    service = YoutubeCaseReplay(root, project)
    claim = json.loads((project / "docs/cards/LR-news-retirement-fixture.json").read_bytes())["claim"]
    with gr.Accordion("新闻与 YouTube 评论联调（离线回放）", open=True):
        statement = gr.Textbox(label="本轮固定声明", value=claim)
        create = gr.Button("1. 建立离线案件")
        case_id = gr.Textbox(label="案件编号", interactive=False)
        keyword = gr.Textbox(label="评论查询词", value="延迟退休")
        start = gr.Textbox(label="开始日（北京时间）", value="2024-07-29")
        end = gr.Textbox(label="结束日（含当天）", value="2024-08-04")
        query = gr.Button("2. 回放本案评论查询")
        preview = gr.JSON(label="候选视频、时间窗与样本数")
        query_id = gr.Textbox(label="查询编号", interactive=False)
        use = gr.Checkbox(label="核对候选后用于本案（取消勾选则跳过）", value=True)
        select = gr.Button("3. 确认选择")
        selection = gr.Textbox(label="选择回执", interactive=False)
        generate = gr.Button("4. 生成离线调查报告")
        report = gr.Markdown()
        detail = gr.JSON(label="案件报告与引文")
    create_page, query_page, select_page, generate_page = _case_page_actions(service)
    create.click(create_page, inputs=statement, outputs=[case_id, preview, query_id, selection, report, detail], api_name="yt_case_create")
    query.click(query_page, inputs=[case_id, keyword, start, end], outputs=[preview, query_id, selection, report, detail], api_name="yt_case_query")
    select.click(select_page, inputs=[case_id, query_id, use], outputs=[selection, report, detail], api_name="yt_case_select")
    generate.click(generate_page, inputs=case_id, outputs=[report, detail], concurrency_limit=1, api_name="yt_case_generate")


def _case_page_actions(service: object) -> tuple:
    """Build page transitions that clear stale downstream outputs on rejected actions."""
    def create_page(text: str) -> tuple:
        """Clear downstream display even when creating the replacement case fails."""
        try:
            return service.create(text), None, "", "", "", None
        except (ValueError, OSError) as error:
            return "", {"error": str(error)}, "", "", "", None
    def query_page(identity: str, word: str, first: str, last: str) -> tuple:
        """Replace the old preview and selection even when the new query fails."""
        try:
            value = service.query(identity, word, first, last)
            return value, value["query_id"], "", "", None
        except (ValueError, KeyError, OSError) as error:
            return {"error": str(error)}, "", "", "", None
    def select_page(identity: str, query: str, use: bool) -> tuple:
        """Clear previous report and show only the latest selection outcome."""
        try:
            return service.select(identity, query, use), "", None
        except (ValueError, KeyError, OSError) as error:
            return "选择失败：" + str(error), "", None
    def generate_page(identity: str) -> tuple:
        """Display the actual worker snapshot and its report."""
        value = service.generate(identity)
        return value["report"], value
    return create_page, query_page, select_page, generate_page
