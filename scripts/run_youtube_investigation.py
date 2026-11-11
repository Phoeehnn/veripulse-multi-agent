"""Core subprocess for the permission-gated YouTube investigation screen."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys


def handle(case: Path, root: Path, approvals: Path, data: dict) -> dict:
    """Resolve only a server-owned case and explicit server-side approvals before actions."""
    if case.resolve().parent != root.resolve() or root.resolve().drive.lower() != 'd:':
        raise ValueError('case_outside_run_root')
    permission = json.loads(approvals.read_bytes()).get(case.name, {}) if approvals.is_file() else {}
    if data['action'] == 'web_search':
        from src.yuqing.serve.research_web_review import search_candidates
        return search_candidates(case, root, data['query'], Decimal(data['limit_cny']), permission.get('web_pin', ''))
    if data['action'] == 'web_select':
        from src.yuqing.serve.research_web_review import selection_record
        row = selection_record(case, data['urls'], data['reason'])
        raw = json.dumps(row, ensure_ascii=False, indent=2).encode('utf8')
        with (case/'web-selection.json').open('xb') as stream:
            stream.write(raw)
        return dict(status='waiting_model_approval', web_selection_sha=sha256(raw).hexdigest())
    if data['action'] == 'query':
        return query(case, permission, data)
    if data['action'] == 'run':
        return investigate(case, root, permission, data)
    raise ValueError('invalid_action')


def query(case: Path, permission: dict, data: dict) -> dict:
    """Publish a query application or execute its separately approved exact scope once."""
    from src.yuqing.serve.research_youtube_activation import activation_fingerprint, request_identity, collect_youtube
    from scripts.run_youtube_lookup import window_dates
    window_dates(data['start'], data['end'])
    if not data['keyword'].strip() or len(data['keyword']) > 300:
        raise ValueError('invalid_keyword')
    scope = dict(stage='youtube_query', mode='live', identity=request_identity(case), keyword=data['keyword'],
        start=data['start'], end=data['end'], runtime=activation_fingerprint(), max_requests=2,
        endpoint='https://youtube.googleapis.com/youtube/v3', timeout_s=15)
    (case/'query-application.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding='utf8')
    if not permission.get('query_pin'):
        return dict(status='waiting_query_approval', message='查询申请已保存，等待本案授权。')
    receipt = collect_youtube(case, keyword=data['keyword'], start=data['start'], end=data['end'],
        grant=case/'query-grant.json', pin=permission['query_pin'], external_calls_authorized=True)
    collected = receipt.load()
    if any(attempt['status'] != 'received' for attempt in collected['attempts']):
        return dict(status='failed', message='YouTube 请求回执不明，本案停止。原尝试记录已保留。')
    token = dict(receipt_sha=receipt.receipt_sha, mode=receipt.mode, query_grant_sha=receipt.query_grant_sha)
    with (case/'collection-token.json').open('x', encoding='utf8') as stream:
        json.dump(token, stream)
    response = collected['response']
    return dict(status='queried', receipt_sha=receipt.receipt_sha, preview=response)


def investigate(case: Path, root: Path, permission: dict, data: dict) -> dict:
    """Run the existing live investigator only with the selected receipt and a model grant."""
    from src.yuqing import llm, settings
    from src.yuqing.serve.research_youtube_activation import CollectedYoutube, YoutubeActivation
    from src.yuqing.serve.research_runner import run_live_research
    if not permission.get('model_pin'):
        return dict(status='waiting_model_approval', message='选择已保存，等待本案模型发送授权。')
    raw = (case/'model-grant.json').read_bytes()
    if sha256(raw).hexdigest() != permission['model_pin']:
        raise ValueError('model_permission_drift')
    scope = json.loads(raw)
    token = json.loads((case/'collection-token.json').read_bytes())
    selected = json.loads((case/'youtube-selection.json').read_bytes())
    if selected != dict(receipt_sha=token['receipt_sha'], use=data['use']):
        raise ValueError('selection_changed')
    receipt = CollectedYoutube(case, **token)
    act = YoutubeActivation(receipt, case/'model-grant.json', permission['model_pin'], Path(scope['wallet']),
        Decimal(scope['limit_cny']), scope['model_limit'], scope['web_limit'], data['use'],
        tuple((Path(p),h) for p,h in scope['historical_files'].items()), external_calls_authorized=True,
        web_selection_sha=scope.get('web_selection_sha', ''))
    act.validate(case, live=True)
    cfg = settings.get_settings()
    client = llm._get_client(cfg, 0, cfg.llm_timeout_s)
    client._client.follow_redirects = False
    try:
        run_live_research(case, root, limit_cny=act.limit_cny, activation=act)
    finally:
        client.close()
    return dict(status='finished')


def archive_failure(case: Path) -> None:
    """Expose an incurred-cost failure without replacing any existing result or ledger."""
    from src.yuqing.serve.research_contracts import ResearchSnapshot
    if not (case/'cost.json').exists() or (case/'snapshot.json').exists():
        return
    cost = json.loads((case/'cost.json').read_bytes())
    snapshot = ResearchSnapshot(research_id=case.name,execution_mode='live',status='failed',
        report='调查异常停止，未形成完整结论。原回执及费用保留。',
        known_cost_cny=cost.get('known_cost_cny'),cost_status=cost.get('cost_status','unknown'))
    with (case/'snapshot.json').open('x',encoding='utf8') as stream:
        stream.write(snapshot.model_dump_json())


def main() -> None:
    """Keep secrets in the core process and return controlled statuses to the UI."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--approvals',type=Path,required=True)
    args = parser.parse_args()
    # Reuse the explicitly designated local key store; never return secrets to the UI.
    import os
    from dotenv import dotenv_values
    values = dotenv_values(Path(os.environ.get("PROJECTA_ENV_FILE", str(Path(__file__).resolve().parents[1] / ".env"))), interpolate=False)
    for name in ('YOUTUBE_API_KEY', 'DEEPSEEK_API_KEY', 'TAVILY_API_KEY'):
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]
    try:
        value = handle(args.case,args.root,args.approvals,json.load(sys.stdin))
    except Exception as exc:
        archive_failure(args.case)
        value = dict(status='failed', message='本次操作未完成，请核对许可或查看本案日志。', error_type=type(exc).__name__)
    print(json.dumps(value,ensure_ascii=True))


if __name__ == '__main__':
    main()
