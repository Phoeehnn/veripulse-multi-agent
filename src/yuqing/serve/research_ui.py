"""Gradio view for explicitly supplied archived traces."""
import json
import os
from pathlib import Path
from typing import Any, Mapping
from src.yuqing.serve.research_live_ui import LiveUIBinding
from src.yuqing.serve.research_events import load_trace
from src.yuqing.serve.research_frontend import run_investigation, REQUIRED_ENV
from src.yuqing.serve.research_history import ResearchHistory, friendly_report


def create_research_ui(trace_catalog: Mapping[str, Path], *, core_python: Path | None = None, resume_source: Path | None = None, history_root: Path | None = None, sample_increment: Any = None, live_sample: LiveUIBinding | None = None, synthetic_acceptance: bool = False, youtube_replay: bool = False, youtube_lookup: bool = False, youtube_case_root: Path | None = None, youtube_investigation: Any = None) -> Any:
    """Open the latest archived result by default; history controls are strictly read-only."""
    import gradio as gr
    worktree = Path(__file__).resolve().parents[3]
    history = ResearchHistory(history_root if history_root is not None else worktree / "traces/research-runs")
    choices, selected = history.choices()
    initial = history.load(selected)
    initial_text = json.loads((resume_source / "request.json").read_text(encoding="utf-8"))["input_value"] if resume_source else ""
    with gr.Blocks(title="AI Investigation Dashboard") as app:
        gr.Markdown("# AI Investigation Dashboard\n新闻调查工作台 · 事实核查 / 来源证据 / 风险研判")
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
        gr.Markdown("实时调查配置：" + ("缺少 " + "、".join(missing) if missing else "已就绪；新声明会实际调用搜索和模型") if core_python else "仅样本增量入口已安装；历史浏览只读" if live_sample is not None or sample_increment is not None else "YouTube 样本调查 · 点击查询才使用配额" if youtube_lookup else "YouTube 案件调查 · 外发需本案许可" if youtube_investigation is not None else "只读归档预览")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 调查输入")
                news = gr.Textbox(value=initial_text, label="新闻或事实声明", placeholder="输入待核查的具体新闻声明…", lines=6)
                start = gr.Button("继续上次调查（复用取证）" if resume_source else "开始调查", variant="primary", interactive=core_python is not None, visible=youtube_investigation is None)
                gr.Markdown("选择或刷新历史结果仅查看归档，不调用搜索或模型。" if core_python or live_sample is not None or sample_increment is not None or youtube_lookup else "通过下方 YouTube 案件流程申请并执行获准调查。" if youtube_investigation is not None else "**只读结果预览** · 本页面不能发起API调用。")
                selector = gr.Dropdown(choices=choices, value=selected, label="调查历史（含失败记录）")
                refresh = gr.Button("刷新历史（只读）")
                _trace_panel(gr, trace_catalog)
            with gr.Column(scale=3):
                outputs = _result_panel(gr, initial)
        if youtube_investigation is not None:
            from src.yuqing.serve.research_youtube_ui import bind_youtube_investigation
            bind_youtube_investigation(gr, youtube_investigation, news, outputs)
        if youtube_case_root is not None:
            from src.yuqing.serve.research_youtube_replay import bind_youtube_case_ui
            bind_youtube_case_ui(gr, worktree, youtube_case_root)
        if youtube_lookup:
            from src.yuqing.serve.research_youtube_replay import bind_youtube_lookup
            bind_youtube_lookup(gr, news, worktree)
        _bind_youtube_window(gr, youtube_replay)
        selector.change(history.load, inputs=selector, outputs=outputs, api_name="open_history")
        def refresh_history() -> tuple:
            """Reload the archive list and latest completed result with no investigation side effects."""
            choices, selected = history.choices()
            return (gr.update(choices=choices, value=selected), *history.load(selected))
        refresh.click(refresh_history, outputs=[selector, *outputs], api_name="refresh_history")
        _bind_sample_increment(gr, selector, outputs, history, sample_increment)
        _bind_live_sample(gr, selector, outputs, history, live_sample)
        if core_python is not None:
            _bind_investigation(gr, start, news, outputs, selector, history, worktree, core_python, resume_source, synthetic_acceptance)
    return app


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
    status = gr.Markdown(initial[0])
    agents = gr.Markdown(initial[1])
    with gr.Tabs():
        with gr.Tab("研判报告与样本附录"):
            report = gr.Markdown(initial[4])
        with gr.Tab("来源与原文引用"):
            sources = gr.HTML(initial[3])
        with gr.Tab("执行时间线"):
            timeline = gr.Markdown(initial[2])
    with gr.Accordion("开发者信息", open=False):
        details = gr.JSON(initial[5], label="归档来源与公共合同")
    return [status, agents, timeline, sources, report, details]


def _trace_panel(gr: Any, trace_catalog: Mapping[str, Path]) -> None:
    """Keep earlier individual Agent traces separate from complete investigation results."""
    catalog = {key: path for key, path in trace_catalog.items() if key and path.is_file() and path.suffix == ".jsonl"}
    choices = [(f"Agent记录 {number}", key) for number, key in enumerate(catalog, start=1)]
    initial = _render_trace(choices[0][1], catalog) if choices else ("暂无记录", {}, "暂无完整案件报告", [])
    with gr.Accordion("早期 Agent Trace（辅助）", open=False):
        selector = gr.Dropdown(choices=choices, label="单Agent归档", value=choices[0][1] if choices else None)
        status = gr.Markdown(initial[0])
        summary = gr.Markdown(initial[2])
        with gr.Accordion("Trace详细数据", open=False):
            agents = gr.JSON(initial[1], label="归档状态")
            table = gr.Dataframe(initial[3], headers=["序号", "时间", "角色", "事件", "状态", "摘要", "引用"], interactive=False)
    selector.change(lambda key: _render_trace(key, catalog), inputs=selector, outputs=[status, agents, summary, table])


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
    mode = f"历史回放 · 终态：{snapshot.status} · 归档费用：{amount}"
    report = "Trace 不含完整案件简报；本页不重建模型报告。"
    return mode, state, report, table


def _css() -> str:
    """Return a small theme that styles Gradio's documented container surface."""
    return ".gradio-container{max-width:1440px!important;background:#f3f6fb;color:#152238}.gradio-container h1{color:#10233f}.gradio-container button{border-radius:12px}.source-card{background:#fff;border:1px solid #dce4ef;border-radius:12px;padding:16px;margin:12px 0}.source-card pre{white-space:pre-wrap;line-height:1.7}"


def _bind_sample_increment(gr: Any, selector: Any, outputs: list[Any], history: ResearchHistory, runner: Any) -> None:
    """Keep the explicit offline increment action separate from read-only history selection."""
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
