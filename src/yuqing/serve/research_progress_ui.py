"""Stream observed worker progress and an explicitly offline fixed-news demonstration."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Iterator
import json
from uuid import uuid4

from src.yuqing.serve.research_frontend import render_case


def stream_operation(case: Path, operation: Callable, *, offline: bool = False) -> Iterator[tuple]:
    """Keep one worker running and render only its observed events until it terminates."""
    started = monotonic()
    label = '演示回放' if offline else '本案调查'
    latest = ('等待执行回执。', '尚无角色结果。', '', '', '等待调查报告。', {})
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(operation)
        while not task.done():
            try:
                values = list(render_case(case, monotonic() - started))
                latest = tuple(values)
            except Exception:
                yield _stopped(latest, offline)
                return
            values[0] = '<span class="activity-dot"></span> ' + label + ' · ' + values[0]
            values[0] += '\n\n工作进程处理中；没有新事件时正在等待处理或模型响应。'
            yield _present(values, offline)
            sleep(0.5)
        try:
            result = task.result()
            values = list(render_case(case, monotonic() - started, finished=True) if offline else result)
        except Exception:
            yield _stopped(latest, offline)
            return
        values[0] = label + ' · ' + values[0]
        yield _present(values, offline)


def offline_demo(project: Path, root: Path, text: str, confirmed: bool) -> Iterator[tuple]:
    """Run only the existing fixed-news, network-blocked three-role subprocess."""
    from src.yuqing.serve.research_youtube_case import YoutubeCaseReplay
    service = YoutubeCaseReplay(root, project)
    try:
        if not confirmed:
            raise ValueError('materials_not_confirmed')
        identity = service.create(text)
        query = service.query(identity, '延迟退休', '2024-07-29', '2024-08-04')
        service.select(identity, query['query_id'], True)
    except (ValueError, OSError, KeyError):
        yield ('未启动：请确认材料。若已修改新闻，请切换到实时调查入口。', '三个 Agent 均未启动。',
               '未启动子进程，未发起外部请求。', '', '未生成报告。', {})
        return
    case = service.cases[identity][0]
    yield from stream_operation(case, lambda: service.generate(identity), offline=True)


def bind_offline_demo(gr: object, project: Path, outputs: list, *, embedded: bool = False) -> None:
    """Keep the fixed demonstration visibly separate from real investigation grants."""
    spec = json.loads((project/'docs/cards/LR-news-retirement-fixture.json').read_bytes())
    with gr.Accordion('离线调查演示' if embedded else '调查演示', open=not embedded):
        text = gr.Textbox(label='新闻声明', value=spec['claim'], elem_id='demo-news-input' if embedded else 'news-input')
        gr.Markdown('材料来源：演示归档。')
        gr.Markdown('事实材料：退休改革报道摘录。传播材料：2024-07-29 至 08-04 的固定 YouTube 评论样本。')
        confirmed = gr.Checkbox(label='确认使用上述网页和评论材料', value=False)
        button = gr.Button('开始调查', variant='primary')
    root = project/'traces/tmp'/('offline-demo-' + uuid4().hex)

    def execute(text: str, confirmed: bool) -> Iterator[tuple]:
        """Stream a fresh case without reusing historical results or live authorizations."""
        yield from offline_demo(project, root, text, confirmed)

    button.click(execute, inputs=[text, confirmed], outputs=outputs,
                 concurrency_limit=1, api_name='offline_demo')


def _present(values: list, offline: bool) -> tuple:
    """Label fixture accounting as simulated without changing archived receipts."""
    if offline:
        values[0] = values[0].replace('**实时调查**', '**模拟执行**').replace('本次账本估算', '模拟账本估算').replace('累计已知', '模拟累计')
    return tuple(values)


def _stopped(latest: tuple, offline: bool) -> tuple:
    """Retain the last valid display while stopping UI animation, not the underlying worker."""
    values = list(latest)
    values[0] = '进度显示已停止：读取或执行异常；以下为最近有效记录，后台是否结束请查本案归档。'
    values[1] = values[1].replace('<span class="activity-dot"></span> ', '')
    values[1] = values[1].replace('正在处理 · running', '最近记录为运行中，当前状态未知')
    return _present(values, offline)
