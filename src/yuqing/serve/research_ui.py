"""Gradio view for explicitly supplied archived traces."""
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote
from src.yuqing.serve.research_live_ui import LiveUIBinding
from src.yuqing.serve.research_events import load_trace
from src.yuqing.serve.research_frontend import run_investigation
from src.yuqing.serve.research_progress_ui import bind_offline_demo
from src.yuqing.serve.research_history import ResearchHistory, friendly_report


def create_research_ui(trace_catalog: Mapping[str, Path], *, core_python: Path | None = None, resume_source: Path | None = None, history_root: Path | None = None, featured_history_root: Path | None = None, sample_increment: Any = None, live_sample: LiveUIBinding | None = None, synthetic_acceptance: bool = False, youtube_replay: bool = False, youtube_lookup: bool = False, youtube_case_root: Path | None = None, youtube_investigation: Any = None, offline_demo: bool = False) -> Any:
    """Open the latest archived result by default; history controls are strictly read-only."""
    import gradio as gr
    banner = Path(__file__).with_name("assets") / "olive-editorial-reference.png"
    gr.set_static_paths(paths=[banner])
    worktree = Path(__file__).resolve().parents[3]
    history = ResearchHistory(history_root if history_root is not None else worktree / "traces/research-runs", featured_root=featured_history_root)
    choices, selected = history.choices()
    initial = history.load(selected)
    initial_text = json.loads((resume_source / "request.json").read_text(encoding="utf-8"))["input_value"] if resume_source else ""
    with gr.Blocks(title="VeriPulse · 新闻研判", fill_width=True) as app:
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
                gr.Markdown("# VeriPulse\n新闻核查与研判平台", elem_id="brand")
                gr.HTML('<a href="#news-input" class="new-investigation">＋ 新建调查</a>', elem_id="new-investigation")
                gr.Markdown("### 调查历史\n选择一条归档，查看报告和证据。", elem_id="history-heading", elem_classes=["section-intro"])
                selector = gr.Radio(choices=choices, value=selected, label="历史调查", show_label=False, elem_id="history-list")
                refresh = gr.Button("↻  刷新记录", elem_id="history-refresh", variant="secondary")
                trace_button = gr.Button("早期 Agent Trace", elem_id="trace-open", visible=bool(trace_catalog))
            with gr.Column(scale=3, min_width=480, elem_id="main-panel"):
                with gr.Group(elem_id="masthead"):
                    gr.HTML('<div class="editorial-photo" aria-hidden="true"></div>', elem_id="masthead-art")
                    gr.Markdown("## 新闻调查\n从新闻声明开始，核对证据并形成研判报告。", elem_id="welcome")
                with gr.Row(elem_id="composer", visible=not offline_demo):
                    news = gr.Textbox(value=initial_text, label="新闻声明", show_label=False, placeholder="粘贴或输入需要核查的新闻声明、报道内容或网络说法。", lines=3, elem_id=None if offline_demo else "news-input")
                    start = gr.Button("继续上次调查" if resume_source else "开始调查", variant="primary", interactive=core_python is not None, visible=youtube_investigation is None)
                    create = gr.Button("建立调查", variant="primary", visible=youtube_investigation is not None)
                with gr.Column(elem_id="result-area", render=False) as result_area:
                    outputs = _result_panel(gr, initial)
                if youtube_investigation is not None:
                    from src.yuqing.serve.research_youtube_ui import bind_youtube_investigation
                    with gr.Column(elem_id="investigation-controls"):
                        bind_youtube_investigation(gr, youtube_investigation, news, outputs, create_button=create)
                if offline_demo:
                    bind_offline_demo(gr, worktree, outputs, embedded=not offline_demo)
                result_area.render()
        if youtube_case_root is not None:
            from src.yuqing.serve.research_youtube_replay import bind_youtube_case_ui
            bind_youtube_case_ui(gr, worktree, youtube_case_root)
        if youtube_lookup:
            from src.yuqing.serve.research_youtube_replay import bind_youtube_lookup
            bind_youtube_lookup(gr, news, worktree)
        _bind_youtube_window(gr, youtube_replay)
        _bind_history(gr, selector, refresh, outputs, history)
        _trace_panel(gr, trace_catalog, trace_button)
        _bind_sample_increment(gr, selector, outputs, history, sample_increment)
        _bind_live_sample(gr, selector, outputs, history, live_sample)
        if core_python is not None:
            _bind_investigation(gr, start, news, outputs, selector, history, worktree, core_python, resume_source, synthetic_acceptance)
    return app


def _bind_history(gr: Any, selector: Any, refresh: Any, outputs: list[Any], history: ResearchHistory) -> None:
    """Bind read-only history selection and refresh to the same archived results."""
    def refresh_history() -> tuple:
        """Reload the archive list and latest completed result."""
        choices, selected = history.choices()
        return (gr.update(choices=choices, value=selected), *history.load(selected))

    selector.change(history.load, inputs=selector, outputs=outputs, api_name="open_history")
    refresh.click(refresh_history, outputs=[selector, *outputs], api_name="refresh_history")


def _bind_investigation(gr: Any, start: Any, news: Any, outputs: list[Any], selector: Any,
                        history: ResearchHistory, worktree: Path, core_python: Path,
                        resume_source: Path | None, synthetic_acceptance: bool = False) -> None:
    """Keep new work and archive browsing on one surface, refreshing history only after results exist."""
    def investigate(text: str) -> Any:
        """Publish actual worker observations and add the completed or failed run to history."""
        for result in run_investigation(text, worktree=worktree, core_python=core_python,
                run_root=history.root, resume_source=resume_source, synthetic_acceptance=synthetic_acceptance):
            values = list(result)
            values[4] = friendly_report(values[4])
            identity = values[5].get("research_id")
            update = gr.skip()
            if identity:
                choices, _ = history.choices()
                update = gr.update(choices=choices, value=identity)
            yield (*values, update)
    start.click(investigate, inputs=news, outputs=[*outputs, selector], concurrency_limit=1, api_name="investigate")


def _result_panel(gr: Any, initial: tuple) -> list[Any]:
    """Build the shared archived/live result surface with developer details collapsed."""
    with gr.Row(elem_id="result-grid"):
        with gr.Column(scale=7, min_width=420, elem_id="report-column"):
            gr.Markdown("## 研判报告\n阅读结论与依据；在报告内滚动查看完整正文。", elem_classes=["section-intro"])
            gr.HTML('<div class="report-guide" aria-label="报告阅读顺序"><span><b>01 · 核对结论</b>判断与调查范围</span><i>→</i><span><b>02 · 回看证据</b>E 原文 · W 样本</span><i>→</i><span><b>03 · 查看限制</b>缺席与待解决问题</span></div>')
            with gr.Tabs():
                with gr.Tab("报告"):
                    report = gr.Markdown(initial[4], elem_id="report-body")
                with gr.Tab("证据来源"):
                    gr.Markdown("查看报告引用的原文、出处和链接。", elem_classes=["section-intro"])
                    sources = gr.HTML(initial[3])
                with gr.Tab("时间线"):
                    gr.Markdown("按时间查看取证和研判记录。", elem_classes=["section-intro"])
                    timeline = gr.Markdown(initial[2])
        with gr.Column(scale=3, min_width=280, elem_id="progress-column"):
            gr.Markdown("## 调查进展\n查看每个角色的分工和状态。", elem_classes=["section-intro"])
            agents = gr.Markdown(initial[1], elem_id="agent-results")
            with gr.Group(elem_id="usage-card"):
                gr.Markdown("### 调用与费用\n查看本次请求和预算记录。", elem_classes=["section-intro"])
                status = gr.Markdown(initial[0], elem_id="case-status")
            with gr.Accordion("归档详情", open=False):
                gr.Markdown("查看报告所对应的结构化归档。", elem_classes=["section-intro"])
                details = gr.JSON(initial[5], label="归档数据")
    return [status, agents, timeline, sources, report, details]


def _trace_panel(gr: Any, trace_catalog: Mapping[str, Path], trigger: Any) -> None:
    """Keep earlier individual Agent traces separate from complete investigation results."""
    if not trace_catalog:
        return
    catalog = {key: path for key, path in trace_catalog.items() if key and path.is_file() and path.suffix == ".jsonl"}
    choices = [(f"Agent记录 {number}", key) for number, key in enumerate(catalog, start=1)]

    def inspect(key: str | None) -> tuple:
        """Serialize only the projected Agent state for the read-only JSON viewer."""
        status, agents, summary, rows = _render_trace(key, catalog)
        return status, json.dumps(agents, ensure_ascii=False, indent=2), summary, rows

    selected = choices[0][1] if choices else None
    initial = inspect(selected)
    with gr.Column(visible=False, elem_id="trace-drawer") as drawer:
        with gr.Row(elem_id="trace-heading"):
            gr.Markdown("## 早期 Agent Trace\n查看历史 Agent 执行记录。")
            close = gr.Button("关闭", elem_id="trace-close", size="sm")
        selector = gr.Dropdown(choices=choices, label="历史记录", value=selected, interactive=bool(choices))
        gr.Markdown("### 扫描状态", elem_classes=["trace-section"])
        status = gr.Markdown(initial[0], elem_id="trace-status")
        summary = gr.Markdown(initial[2], elem_id="trace-note")
        gr.Markdown("### 事件时间线", elem_classes=["trace-section"])
        table = gr.Dataframe(initial[3], headers=["序号", "时间", "角色", "事件", "状态", "摘要", "引用"],
                             interactive=False, max_height=240, elem_id="trace-events")
        with gr.Accordion("JSON 详情", open=False, elem_id="trace-json-section"):
            agents = gr.Code(initial[1], language="json", label="归档状态", interactive=False,
                             lines=6, max_lines=10, buttons=["copy"], elem_id="trace-json")
    trigger.click(lambda: gr.update(visible=True), outputs=drawer, queue=False, api_name=False)
    close.click(lambda: gr.update(visible=False), outputs=drawer, queue=False, api_name=False)
    selector.change(inspect, inputs=selector, outputs=[status, agents, summary, table], api_name=False)


def _render_trace(trace_id: str | None, catalog: Mapping[str, Path]) -> tuple[str, dict[str, object], str, list[list[object]]]:
    """Render only fields projected by the trace adapter."""
    path = catalog.get(trace_id or "")
    if path is None:
        return "未选择记录", {}, "本地回放", []
    snapshot, rows = load_trace(path, research_id=trace_id or "history")
    table = [[row.seq, row.occurred_at.isoformat(), row.actor, row.kind, row.status, row.summary, row.source_event_ref] for row in rows]
    state = {item.role: {"status": item.status, "reason": item.reason} for item in snapshot.agents}
    amount = snapshot.known_cost_cny or ("部分已知" if snapshot.cost_status == "partial" else "未知")
    if snapshot.cost_status == "partial":
        amount = f"已知小计 ¥{snapshot.known_cost_cny}（总额未知）"
    labels = {"completed": "✓ 已完成", "failed": "! 失败", "interrupted": "– 已中断"}
    label = labels.get(snapshot.status, "· 状态待确认")
    mode = f'<span class="trace-status-badge {snapshot.status}">{label}</span>　历史回放 · {len(rows)} 条事件 · 归档费用：{amount}'
    report = "Trace 不含完整案件简报；本页不重建模型报告。"
    return mode, state, report, table


def _css() -> str:
    """Return a small theme that styles Gradio's documented container surface."""
    assets = Path(__file__).with_name("assets")
    banner = quote((assets / "olive-editorial-reference.png").resolve().as_posix(), safe="/:")
    css = (assets / "olive-workspace.css").read_text(encoding="utf-8")
    return css + f"\n.editorial-photo {{ background-image: url('/gradio_api/file={banner}'); }}"


def _bind_sample_increment(gr: Any, selector: Any, outputs: list[Any], history: ResearchHistory, runner: Any) -> None:
    """Keep the explicit offline increment action separate from read-only history selection."""
    if runner is None:
        return
    button=gr.Button("仅样本离线增量（复用所选主报告）",interactive=runner is not None)
    gr.Markdown("仅样本增量独立归档，事实与仲裁不重跑；真实调用未启用。")
    if runner is None:
        return
    def increment(identity: str) -> tuple:
        """Call the installed offline runner once and open its new archived result."""
        new_id=runner(identity)
        choices,_=history.choices()
        return (*history.load(new_id),gr.update(choices=choices,value=new_id))
    button.click(increment,inputs=selector,outputs=[*outputs,selector],concurrency_limit=1,api_name="sample_increment")


def _bind_live_sample(gr: Any, selector: Any, outputs: list[Any], history: ResearchHistory,
                      binding: LiveUIBinding | None) -> None:
    """Build a distinct live region; no execution API exists without explicit backend injection."""
    if binding is None:
        return
    with gr.Accordion("真实样本增量（复用主报告）",open=False):
        gr.Markdown("真实调用未启用。" if binding is None else "已安装执行入口；预览与历史浏览不执行。")
        preview=gr.Button("只读预览本次计划",interactive=binding is not None)
        text=gr.Markdown("尚未预览计划，未执行任何请求。")
        confirmed=gr.Checkbox(value=False,label="我确认仅执行本次样本增量，按所示累计钱包计费",interactive=binding is not None)
        run=gr.Button("开始真实样本增量",interactive=binding is not None)
        token=gr.State(None)
    if binding is None:
        return
    def inspect(identity: str) -> tuple:
        """Preview only and clear any previous confirmation."""
        description,bound=binding.preview(identity)
        return description,bound,False
    def execute(identity: str, checked: bool, bound: tuple | None) -> tuple:
        """Execute only after explicit confirmation; always clear confirmation and preview state."""
        try:
            new_id=binding.execute(identity,checked,bound)
            choices,_=history.choices()
            return (*history.load(new_id),gr.update(choices=choices,value=new_id),False,None,"本次已归档；再次执行须重新预览。")
        except (ValueError,OSError):
            return (*[gr.skip() for _ in outputs],gr.skip(),False,None,"计划未确认、已变化或执行未完成；请重新预览并查看归档状态。")
    preview.click(inspect,inputs=selector,outputs=[text,token,confirmed],api_name="preview_live_sample")
    run.click(execute,inputs=[selector,confirmed,token],outputs=[*outputs,selector,confirmed,token,text],
              concurrency_limit=1,api_name="execute_live_sample")
    selector.change(lambda:("所选主报告已变化，请重新预览。",None,False),outputs=[text,token,confirmed],api_name=False)


def _bind_youtube_window(gr: Any, enabled: bool) -> None:
    """Keep the existing offline window preview separate from investigation controls."""
    if not enabled:
        return
    from src.yuqing.serve.research_youtube_replay import WINDOWS, render_youtube_window
    with gr.Accordion("YouTube 评论调查 · 延迟退休（离线回放）", open=True):
        window = gr.Radio(list(WINDOWS), label="观察时间窗", value=list(WINDOWS)[0])
        display = gr.Markdown(render_youtube_window(list(WINDOWS)[0]))
        window.change(render_youtube_window, inputs=window, outputs=display, api_name="youtube_window")
