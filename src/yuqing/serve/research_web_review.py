"""One-shot candidate search and explicit, grant-bound relevance selection."""
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from src.yuqing import settings
from src.yuqing.retrieval.web_contracts import SearchCandidate
from src.yuqing.retrieval.web_provider_contracts import WebRetrievalRequest, WebRetrievalReceipt
from src.yuqing.serve.research_youtube_activation import activation_fingerprint, check_grant, request_identity


def search_scope(case: Path, root: Path, query: str, limit: Decimal) -> dict:
    """Freeze exact query, case, wallet and basic-search scope before one external request."""
    request = WebRetrievalRequest(operation='search', payload=dict(query=query,
        max_results=settings.get_harness_config().search_k, search_depth='basic', include_usage=True))
    if case.resolve().parent != root.resolve() or root.resolve().drive.lower() != 'd:':
        raise ValueError('web_review_case_outside_root')
    if not limit.is_finite() or limit <= 0:
        raise ValueError('web_review_invalid_limit')
    return dict(stage='web_candidate_search', identity=request_identity(case), request=request.model_dump(),
        wallet=str((root/'wallet.sqlite').resolve()), limit_cny=str(limit), runtime=activation_fingerprint(),
        max_requests=1, endpoint='https://api.tavily.com/search')


def search_candidates(case: Path, root: Path, query: str, limit: Decimal, pin: str) -> dict:
    """Search once on the investigation wallet, preserving receipts and stopping on unknown."""
    from src.yuqing.serve.research_runner import _live_research_rate, open_research_wallet, budgeted_web_call, save_case_cost
    scope = search_scope(case, root, query, limit)
    (case/'web-search-application.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2), encoding='utf8')
    if not pin:
        return dict(status='waiting_web_approval', message='网页搜索申请已保存，尚未外发。')
    check_grant(case/'web-search-grant.json', pin, scope)
    rate = _live_research_rate()
    wallet = open_research_wallet(root, limit_cny=limit, credit_cny=rate)
    try:
        if wallet.storage_path.resolve() != Path(scope['wallet']) or wallet.snapshot().halted:
            raise ValueError('web_review_wallet_stopped_or_changed')
        with (case/'web-search.claim').open('x', encoding='utf8') as stream:
            stream.write(pin)
        folder = case/'web-search'
        folder.mkdir(exist_ok=False)
        request = WebRetrievalRequest.model_validate(scope['request'])
        (folder/'request.json').write_text(request.model_dump_json(indent=2), encoding='utf8')
        receipt = budgeted_web_call(request, wallet, credit_cny=rate, sample_id=case.name)
        (folder/'receipt.json').write_text(receipt.model_dump_json(indent=2), encoding='utf8')
        if receipt.status != 'success' or receipt.billing_status != 'usage_reported' or wallet.snapshot().halted:
            raise ValueError('web_review_search_failed_or_unknown')
        return dict(status='waiting_web_selection', candidates=receipt.response['results'])
    finally:
        save_case_cost(case, case.name, wallet)
        wallet.close()


def selection_record(case: Path, urls: list[str], reason: str) -> dict:
    """Bind an explicit relevance judgment to exact original search bytes and selected candidates."""
    folder = case/'web-search'
    request = WebRetrievalRequest.model_validate_json((folder/'request.json').read_bytes())
    receipt = WebRetrievalReceipt.model_validate_json((folder/'receipt.json').read_bytes())
    scope = json.loads((case/'web-search-grant.json').read_bytes())
    check_grant(case/'web-search-grant.json', (case/'web-search.claim').read_text(encoding='utf8'),
        search_scope(case, case.parent, request.payload['query'], Decimal(scope['limit_cny'])))
    if scope['request'] != request.model_dump() or receipt.status != 'success' or receipt.billing_status != 'usage_reported':
        raise ValueError('web_review_invalid_search')
    candidates = [SearchCandidate(url=row['url'], title=row['title'], snippet=row['content'])
                  for row in receipt.response['results']]
    allowed = {str(row.url) for row in candidates}
    if not urls or len(urls) != len(set(urls)) or not set(urls) <= allowed or not reason.strip():
        raise ValueError('web_review_selection_required')
    return dict(identity=request_identity(case), search_sha=sha256((folder/'receipt.json').read_bytes()).hexdigest(),
        request_sha=sha256((folder/'request.json').read_bytes()).hexdigest(),
        search_grant_sha=sha256((case/'web-search-grant.json').read_bytes()).hexdigest(),
        urls=urls, reason=reason, wallet=scope['wallet'], limit_cny=scope['limit_cny'])


def approved_search(case: Path, pin: str, wallet: Path, limit: Decimal) -> tuple:
    """Reject unapproved or drifted selections before extraction and return only approved candidates."""
    raw = (case/'web-selection.json').read_bytes()
    if not pin or sha256(raw).hexdigest() != pin:
        raise ValueError('web_review_selection_permission_mismatch')
    selected = json.loads(raw)
    if (selected != selection_record(case, selected['urls'], selected['reason'])
            or selected['wallet'] != str(wallet.resolve()) or selected['limit_cny'] != str(limit)):
        raise ValueError('web_review_selection_drift')
    request = WebRetrievalRequest.model_validate_json((case/'web-search/request.json').read_bytes())
    receipt = WebRetrievalReceipt.model_validate_json((case/'web-search/receipt.json').read_bytes())
    rows = {str(SearchCandidate(url=r['url'], title=r['title'], snippet=r['content']).url): r
            for r in receipt.response['results']}
    # The original receipt remains immutable; the derived extraction view contains selected URLs only.
    filtered = dict(receipt.response, results=[rows[url] for url in selected['urls']])
    return request, receipt.model_copy(update={'response': filtered})


def validate_activation_web(case: Path, activation: object, limit: Decimal) -> tuple | None:
    """Validate the activation and its selected search before claiming or opening model execution."""
    if activation is None:
        return None
    activation.validate(case, live=True)
    if limit != activation.limit_cny:
        raise ValueError('youtube_activation_limit_mismatch')
    if activation.collection.mode == 'live':
        return approved_search(case, activation.web_selection_sha, activation.wallet_path, limit)
    return None
