"""Lightweight two-stage screen; only the core subprocess imports model dependencies."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import subprocess
from time import monotonic
from uuid import uuid4
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


def bind_youtube_investigation(gr: object, service: YoutubeInvestigationUI, news: object, outputs: list) -> None:
    """Bind one news input to query, selection and the existing report display."""
    with gr.Accordion('YouTube 评论与新闻调查',open=True):
        gr.Markdown('先建立案件并查询候选，核对后选用或跳过。外部查询与模型分析分别需要本案授权。')
        create = gr.Button('1. 用上方新闻建立案件')
        identity = gr.Textbox(label='当前案件',interactive=False)
        word = gr.Textbox(label='评论查询词（与原新闻分开保存）')
        with gr.Row():
            first = gr.Textbox(label='开始日（北京时间 YYYY-MM-DD）')
            last = gr.Textbox(label='结束日（含当天）')
        query = gr.Button('2. 申请或执行获准查询')
        preview = gr.JSON(label='候选视频、原评论链接与窗内样本')
        use = gr.Checkbox(label='选用这些评论；取消勾选表示跳过',value=True)
        select = gr.Button('3. 确认选用或跳过')
        status = gr.Markdown()
        run = gr.Button('4. 执行获准调查并显示报告',variant='primary')
    create_page, query_page, select_page, run_page = page_actions(service)
    create.click(create_page,inputs=news,outputs=[identity,preview,status,*outputs],api_name='youtube_create')
    query.click(query_page,inputs=[identity,word,first,last],outputs=[preview,status,*outputs],api_name='youtube_query')
    select.click(select_page,inputs=[identity,use],outputs=[status,*outputs],api_name='youtube_select')
    run.click(run_page,inputs=identity,outputs=outputs,api_name='youtube_run',concurrency_limit=1)


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
    def run_page(case: str) -> tuple:
        """Clear stale reports when selection or execution is refused."""
        try:
            return service.generate(case)
        except (ValueError, OSError, KeyError):
            return '调查未启动或未完成，请核对选择和许可。','','','','',{}
    return create_page, query_page, select_page, run_page
