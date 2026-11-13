"""Lightweight two-stage screen; only the core subprocess imports model dependencies."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
import subprocess
from time import monotonic
from uuid import uuid4
from src.yuqing.retrieval.web_contracts import SearchCandidate
from src.yuqing.serve.research_contracts import ResearchRequest
from src.yuqing.serve.research_frontend import render_case


@dataclass
class YoutubeInvestigationUI:
    """Keep browser identifiers separate from trusted paths and approval files."""
    project: Path
    root: Path
    approvals: Path
    cases: dict = field(default_factory=dict)
    selected: dict = field(default_factory=dict)

    def create(self, text: str) -> str:
        """Create a stable text case without performing any external operation."""
        request = ResearchRequest(input_kind='text',input_value=text.strip())
        if not text.strip() or self.root.resolve().drive.lower() != 'd:':
            raise ValueError('invalid_case_input')
        identity = uuid4().hex
        case = self.root/identity
        case.mkdir(parents=True)
        payload = request.model_dump_json().encode()
        (case/'request.json').write_bytes(payload)
        self.cases[identity] = sha256(payload).hexdigest()
        return identity

    def case(self, identity: str) -> Path:
        """Reject browser case substitutions and changed requests."""
        if identity not in self.cases:
            raise ValueError('unknown_case')
        case = self.root/identity
        if sha256((case/'request.json').read_bytes()).hexdigest() != self.cases[identity]:
            raise ValueError('case_request_drift')
        return case

    def worker(self, identity: str, data: dict) -> dict:
        """Run a hidden core process, keeping secrets out of command arguments and browser output."""
        case = self.case(identity)
        temp = case/'tmp'
        temp.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(self.project), PYTHONIOENCODING='utf-8',
            PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TEMP=str(temp), TMP=str(temp))
        result = subprocess.run([sys.executable,'-B','-m',
            'scripts.run_youtube_investigation','--case',str(case),'--root',str(self.root),
            '--approvals',str(self.approvals)], input=json.dumps(data), text=True, encoding='utf8',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.project, env=env,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        # Raw stderr may contain provider details; keep it local, never return it to the browser.
        (case/'frontend-worker.log').write_text(result.stderr,encoding='utf8')
        return json.loads(result.stdout) if result.returncode == 0 else dict(status='failed',message='调查进程未完成。')

    def query(self, identity: str, word: str, first: str, last: str) -> dict:
        """Invalidate prior selection before attempting or requesting a new query."""
        self.selected.pop(identity,None)
        return self.worker(identity,dict(action='query',keyword=word,start=first,end=last))

    def select(self, identity: str, use: bool) -> str:
        """Pin the server-owned collected receipt and explicit user selection."""
        self.selected.pop(identity,None)
        case = self.case(identity)
        token = json.loads((case/'collection-token.json').read_bytes())
        raw = (case/'youtube-collection.json').read_bytes()
        if sha256(raw).hexdigest() != token['receipt_sha']:
            raise ValueError('collection_drift')
        selected = dict(receipt_sha=token['receipt_sha'],use=use)
        (case/'youtube-selection.json').write_text(json.dumps(selected),encoding='utf8')
        self.selected[identity] = selected
        return '已选用本案评论，等待模型许可。' if use else '已跳过评论，等待主报告运行许可。'

    def generate(self, identity: str) -> tuple:
        """Display actual archived role results, citations and whole-case costs after explicit execution."""
        case = self.case(identity)
        selected = self.selected[identity]
        if json.loads((case/'youtube-selection.json').read_bytes()) != selected:
            raise ValueError('selection_changed')
        started = monotonic()
        result = self.worker(identity,dict(action='run',use=selected['use']))
        elapsed = monotonic() - started
        if result['status'] != 'finished' and not (case/'snapshot.json').exists():
            return (result.get('message','本次操作未完成。'), '尚未取得新的角色结果。', '', '', '等待获准调查完成。', {})
        return render_case(case,elapsed,finished=True)


def bind_youtube_investigation(gr: object, service: YoutubeInvestigationUI, news: object, outputs: list,
                               *, create_button: object | None = None) -> None:
    """Bind one news input to query, selection and the existing report display."""
    with gr.Accordion('调查设置与材料审核',open=False):
        gr.Markdown('先建立案件，再确认事实核查用的网页和传播分析用的评论。查询与模型分析各自需要本案许可。')
        create = create_button if create_button is not None else gr.Button('建立调查', variant='primary')
        identity = gr.Textbox(label='案件编号',interactive=False)
        reset_web = bind_web_review(gr, service, identity, outputs)
        gr.Markdown('### YouTube 评论\n核对候选视频和时间窗内的评论，再选择是否用于传播分析。')
        word = gr.Textbox(label='评论查询词（与原新闻分开保存）')
        with gr.Row():
            first = gr.Textbox(label='开始日（北京时间 YYYY-MM-DD）')
            last = gr.Textbox(label='结束日（含当天）')
        gr.Markdown('填写关键词和时间范围，查询与这条新闻有关的公开评论。')
        query = gr.Button('查询公开评论')
        preview = gr.JSON(label='候选视频、原评论链接与窗内样本')
        use = gr.Checkbox(label='选用这些评论；取消勾选表示跳过',value=True)
        select = gr.Button('确认样本选择')
        status = gr.Markdown()
        gr.Markdown('确认材料后，生成包含结论、依据和原文引用的报告。')
        run = gr.Button('生成研判报告',variant='primary')
    create_page, query_page, select_page, run_page = page_actions(service)
    created = create.click(create_page,inputs=news,outputs=[identity,preview,status,*outputs],api_name='youtube_create')
    created.then(*reset_web, api_name=False)
    query.click(query_page,inputs=[identity,word,first,last],outputs=[preview,status,*outputs],api_name='youtube_query')
    select.click(select_page,inputs=[identity,use],outputs=[status,*outputs],api_name='youtube_select')
    run.click(run_page,inputs=identity,outputs=outputs,api_name='youtube_run',concurrency_limit=1)


def bind_web_review(gr: object, service: YoutubeInvestigationUI, identity: object, outputs: list) -> tuple:
    """Expose search and review through the case-bound worker actions."""
    gr.Markdown('### Tavily 网页候选\n核对标题、摘要与网址，选出与新闻相关的网页。正文在获准生成报告后提取。')
    with gr.Row():
        keyword = gr.Textbox(label='网页查询词')
        limit = gr.Textbox(label='本场共享预算上限（元）', placeholder='填写申请或许可中的金额')
    search = gr.Button('查询网页候选')
    candidates = gr.JSON(label='网页标题、摘要与网址')
    urls = gr.CheckboxGroup(label='选用的网页', choices=[])
    reason = gr.Textbox(label='相关性审核说明', placeholder='说明选中的网页与本案有何关联')
    confirm = gr.Button('确认网页选择')
    status = gr.Markdown()

    def search_page(case: str, query: str, limit_cny: str) -> tuple:
        """Present candidates from the current search receipt and clear prior choices."""
        try:
            result = service.worker(case, dict(action='web_search', query=query, limit_cny=limit_cny))
            rows = [SearchCandidate(url=row['url'], title=row['title'], snippet=row['content'])
                    for row in result.get('candidates', [])]
        except (ValueError, OSError, KeyError):
            result, rows = dict(message='网页查询未完成，请核对案件、查询词、预算和许可。'), []
        data = [row.model_dump(mode='json') for row in rows]
        choices = [(f'{row.title} · {row.url}', str(row.url)) for row in rows]
        message = '请核对候选网页并填写审核说明。' if rows else '本次没有可审核的网页候选。'
        return data, gr.update(choices=choices, value=[]), '', result.get('message', message), '', '', '', '', '', {}

    def confirm_page(case: str, chosen: list[str], note: str) -> tuple:
        """Persist the human decision via the worker and require a separately pinned model grant."""
        try:
            result = service.worker(case, dict(action='web_select', urls=chosen, reason=note))
            message = ('网页审核已归档，等待绑定该选择的模型许可。' if result['status'] == 'waiting_model_approval'
                       else result.get('message', '网页选择未保存。'))
        except (ValueError, OSError, KeyError):
            message = '网页选择未保存，请核对本案候选和审核说明。'
        return message, '', '', '', '', '', {}

    def reset_page() -> tuple:
        """Clear the previous case's web review when a new case is created."""
        return '', '', None, gr.update(choices=[], value=[]), '', ''

    search.click(search_page, inputs=[identity, keyword, limit],
                 outputs=[candidates, urls, reason, status, *outputs], api_name='web_search')
    confirm.click(confirm_page, inputs=[identity, urls, reason], outputs=[status, *outputs], api_name='web_select')
    return reset_page, None, [keyword, limit, candidates, urls, reason, status]


def page_actions(service: YoutubeInvestigationUI) -> tuple:
    """Return controlled callbacks that clear stale views on refusal."""
    def create_page(text: str) -> tuple:
        """Clear all stale results when switching to a new case."""
        try:
            return service.create(text),None,'案件已建立。','','','','','',{}
        except (ValueError, OSError):
            return '',None,'建立失败，请填写新闻声明。','','','','','',{}
    def query_page(case: str, query: str, start: str, end: str) -> tuple:
        """Clear old selection/report after a query attempt, including a denial."""
        try:
            value=service.query(case,query,start,end)
        except (ValueError, OSError):
            value=dict(status='failed',message='查询未完成，请检查案件和时间窗。')
        return value,value.get('message','请核对候选视频及窗内评论。'),'','','','','',{}
    def select_page(case: str, use: bool) -> tuple:
        """Present missing or changed collection without exposing local paths."""
        try:
            return service.select(case,use),'','','','','',{}
        except (ValueError, OSError, KeyError):
            return '尚无可选用的完整查询归档，请先完成查询。','','','','','',{}
    def run_page(case: str) -> object:
        """Clear stale reports when selection or execution is refused."""
        try:
            from src.yuqing.serve.research_progress_ui import stream_operation
            yield from stream_operation(service.case(case), lambda: service.generate(case))
        except (ValueError, OSError, KeyError):
            yield '调查未启动或未完成，请核对选择和许可。','','','','',{}
    return create_page, query_page, select_page, run_page
