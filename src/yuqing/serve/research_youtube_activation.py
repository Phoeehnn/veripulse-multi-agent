"""Explicit YouTube activation: pinned grants, collector-owned provenance and one capped wallet."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from typing import Callable

from src.yuqing import llm, settings
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.harness.budget import BudgetDecision
from src.yuqing.retrieval.youtube_comments import youtube_comment_batch
from src.yuqing.retrieval.public_materials import archive_public_material, load_public_material
from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask
from src.yuqing.serve.research_public import APPENDIX_TITLE, PublicSampleSession


def activation_fingerprint() -> str:
    """Freeze runtime, all prompt bytes, effective configuration and model endpoint without credentials."""
    cfg = settings.get_settings()
    root = settings.PROJECT_ROOT
    files = [cfg.weights_path, cfg.fact_v2_eval_path, root/'src/yuqing/llm.py',
             root/'scripts/run_youtube_lookup.py', root/'scripts/run_youtube_investigation.py',
             root/'src/yuqing/serve/research_web_review.py', root/'src/yuqing/serve/research_runner.py',
             root/'src/yuqing/serve/research_youtube_activation.py', *sorted((root/'prompts').rglob('*.md'))]
    row = dict(files={str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in files},
        python=platform.python_version(), sdk=version('openai'), pydantic=version('pydantic'), httpx=version('httpx'),
        web_credit_cny=os.environ.get('LIVE_RESEARCH_CNY_PER_CREDIT'),
        model=cfg.deepseek_model, endpoint=cfg.deepseek_base_url, timeout=cfg.llm_timeout_s,
        retries=cfg.llm_max_retries, validation_retries=cfg.llm_validation_retries, thinking=cfg.llm_thinking)
    return sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def check_grant(path: Path, pin: str, expected: dict) -> None:
    """Require separately approved bytes and exact scope; a file alone grants no authority."""
    raw = path.read_bytes()
    if not pin or sha256(raw).hexdigest() != pin or json.loads(raw) != expected:
        raise ValueError('youtube_activation_permission_mismatch')


def request_identity(case: Path) -> dict:
    """Read the original text request identity before collection or model activation."""
    from src.yuqing.serve.research_contracts import ResearchRequest
    raw = (case/'request.json').read_bytes()
    request = ResearchRequest.model_validate_json(raw)
    if request.input_kind != 'text':
        raise ValueError('youtube_requires_text_claim')
    return dict(case_id=case.name, case_path=str(case.resolve()),
                request_sha=sha256(raw).hexdigest(), claim=request.input_value)


@dataclass(frozen=True)
class CollectedYoutube:
    """Carry collector provenance separately from untrusted provider response fields."""
    case: Path
    receipt_sha: str
    mode: str
    query_grant_sha: str

    def load(self) -> dict:
        """Reject receipt drift and bind collection to the unchanged original request."""
        raw = (self.case/'youtube-collection.json').read_bytes()
        row = json.loads(raw)
        if (sha256(raw).hexdigest() != self.receipt_sha or row['identity'] != request_identity(self.case)
                or row['mode'] != self.mode or row['grant_sha'] != self.query_grant_sha):
            raise ValueError('youtube_collection_drift')
        return row


def collect_youtube(case: Path, *, keyword: str, start: str, end: str, grant: Path, pin: str,
                    external_calls_authorized: bool = False, replay_get: Callable | None = None) -> CollectedYoutube:
    """Collect once through the sole gateway, or mark explicitly injected transport as mock forever."""
    from scripts.run_youtube_lookup import lookup, window_dates
    mode = 'mock' if replay_get is not None else 'live'
    expected = dict(stage='youtube_query', mode=mode, identity=request_identity(case), keyword=keyword,
        start=start, end=end, runtime=activation_fingerprint(), max_requests=2,
        endpoint='https://youtube.googleapis.com/youtube/v3', timeout_s=15)
    check_grant(grant, pin, expected)
    window_dates(start, end)
    if mode == 'live' and not external_calls_authorized:
        raise ValueError('youtube_external_calls_not_authorized')
    with (case/'youtube-query.claim').open('x', encoding='utf8') as stream:
        stream.write(pin)
    calls = []
    def get(operation: str, params: dict) -> dict:
        """Persist each attempt before dispatch and retain unknown on failed requests."""
        calls.append(dict(operation=operation, status='unknown'))
        if len(calls) > 2:
            raise ValueError('youtube_query_request_limit')
        (case/'youtube-query-attempts.json').write_text(json.dumps(calls), encoding='utf8')
        if replay_get is not None:
            result = replay_get(operation, params)
        else:
            import os
            from pydantic import SecretStr
            result = llm.youtube_public_get(operation, params, SecretStr(os.environ['YOUTUBE_API_KEY']))
        calls[-1]['status'] = 'received'
        (case/'youtube-query-attempts.json').write_text(json.dumps(calls), encoding='utf8')
        return result
    response = lookup(keyword, start, end, get)
    row = dict(identity=expected['identity'], mode=mode, grant_sha=pin, start=start, end=end,
               response=response, attempts=calls)
    raw = json.dumps(row, ensure_ascii=False).encode()
    with (case/'youtube-collection.json').open('xb') as stream:
        stream.write(raw)
    return CollectedYoutube(case, sha256(raw).hexdigest(), mode, pin)


class ActivationWallet:
    """Delegate to the original wallet while serializing all model/web dispatch count checks."""

    def __init__(self, wallet: object, model_limit: int, web_limit: int) -> None:
        """Wrap one existing ledger; never allocate a replacement wallet or reset its state."""
        self.wallet, self.model_limit, self.web_limit = wallet, model_limit, web_limit

    def __getattr__(self, name: str) -> object:
        """Keep every existing coordinator operation on the same persistent wallet."""
        return getattr(self.wallet, name)

    def reserve(self, request_id: str, amount: Decimal, *, identity: dict | None = None) -> BudgetDecision:
        """Stop the whole case on the first monetary refusal."""
        with self.wallet.coordination():
            result = self.wallet.reserve(request_id, amount, identity=identity)
            if not result.allowed and not self.wallet.snapshot().halted:
                self.wallet.halt(result.reason)
            return result

    def mark_dispatched(self, request_id: str) -> None:
        """Count persisted attempts under the wallet lock, including failed and unknown requests."""
        with self.wallet.coordination():
            history = self.wallet.load_history()
            state = replay_history(history, self.wallet.manifest_snapshot())
            web = state.requests[request_id].identity.get('group_id') == 'live-research-web'
            ids = {e['request_id'] for e in history.events if e['event_type'] == 'dispatched'}
            count = sum((state.requests[r].identity.get('group_id') == 'live-research-web') == web for r in ids)
            if self.wallet.snapshot().halted or count >= (self.web_limit if web else self.model_limit):
                self.wallet.cancel_unsent(request_id)
                if not self.wallet.snapshot().halted:
                    self.wallet.halt('youtube_activation_request_limit')
                raise ValueError('youtube_activation_stopped')
            self.wallet.mark_dispatched(request_id)


@dataclass(frozen=True)
class YoutubeActivation:
    """Bind a second explicit model grant to a collected page, selection and existing wallet."""
    collection: CollectedYoutube
    grant: Path
    pin: str
    wallet_path: Path
    limit_cny: Decimal
    model_limit: int
    web_limit: int
    use: bool
    historical_files: tuple[tuple[Path, str], ...] = ()
    external_calls_authorized: bool = False
    web_selection_sha: str = ""

    def scope(self) -> dict:
        """Build the exact reviewable scope; historical ledger bytes must be pinned separately."""
        scope = dict(stage='youtube_models', mode=self.collection.mode, identity=request_identity(self.collection.case),
            receipt_sha=self.collection.receipt_sha, query_grant_sha=self.collection.query_grant_sha, use=self.use,
            wallet=str(self.wallet_path.resolve()), limit_cny=str(self.limit_cny), model_limit=self.model_limit,
            web_limit=self.web_limit, runtime=activation_fingerprint(),
            historical_files={str(p.resolve()): h for p,h in self.historical_files})
        if self.web_selection_sha:
            scope['web_selection_sha'] = self.web_selection_sha
        return scope

    def validate(self, case: Path, *, live: bool) -> dict:
        """Reject absent permission, source promotion, drift and historical changes before any dispatch."""
        check_grant(self.grant, self.pin, self.scope())
        if case.resolve() != self.collection.case.resolve() or (live and self.collection.mode != 'live'):
            raise ValueError('youtube_activation_source_mismatch')
        if live and not self.external_calls_authorized:
            raise ValueError('youtube_external_calls_not_authorized')
        if self.model_limit < 1 or self.web_limit < 1 or not self.limit_cny.is_finite() or self.limit_cny <= 0:
            raise ValueError('youtube_activation_invalid_limits')
        for path, digest in self.historical_files:
            if sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('youtube_historical_ledger_drift')
        cfg = settings.get_settings()
        if live and (cfg.llm_max_retries != 0 or cfg.deepseek_base_url.rstrip('/') != 'https://api.deepseek.com'
                     or cfg.deepseek_model != 'deepseek-flash'):
            raise ValueError('youtube_live_runtime_not_approved')
        row = self.collection.load()
        if any(a['status'] != 'received' for a in row['attempts']):
            raise ValueError('youtube_collection_unknown')
        return row

    def bind_wallet(self, wallet: object) -> ActivationWallet:
        """Check the originally opened wallet path and cap before wrapping its dispatch boundary."""
        if wallet.storage_path.resolve() != self.wallet_path.resolve() or wallet.snapshot().max_cost_cny != self.limit_cny:
            raise ValueError('youtube_wallet_binding_mismatch')
        return ActivationWallet(wallet, self.model_limit, self.web_limit)

    def task(self, *, live: bool) -> PublicPropagationTask | None:
        """Archive validated selected materials with provenance derived from the collector mode."""
        from scripts.run_youtube_lookup import window_dates
        row = self.validate(self.collection.case, live=live)
        response = row['response']
        start, end = window_dates(row['start'], row['end'])
        batch = youtube_comment_batch(response['comments'], video_id=response['video_id'], start=start, end=end,
            fetched_at=datetime.fromisoformat(response['fetched_at']), case_id=self.collection.case.name,
            claim=row['identity']['claim'], provenance='recorded_response' if live else 'synthetic_fixture')
        directory = self.collection.case/'public-material'
        archive_public_material(batch, directory)
        bound = load_public_material(directory, case_id=batch.case_id, claim=row['identity']['claim'])
        return PublicPropagationTask(sample_id=batch.case_id, claim=row['identity']['claim'], batch=bound) if self.use and bound.observations else None

    def claim(self, case: Path) -> None:
        """Acquire a one-shot activation claim; any attempted run requires a new explicit permission to retry."""
        self.validate(case, live=self.collection.mode == 'live')
        with (case.resolve()/'youtube-models.claim').open('x', encoding='utf8') as stream:
            stream.write(self.pin)

    def guard_fact(self, handler: Callable, comp: object) -> Callable:
        """Stop the shared wallet before graph recovery records an unexpected fact failure."""
        def run(*args: object, **kwargs: object) -> dict:
            """Preserve the original exception and allow in-flight receipts to settle after halt."""
            try:
                return handler(*args, **kwargs)
            except Exception:
                if not comp.inputs.wallet.snapshot().halted:
                    comp.inputs.wallet.halt('youtube_fact_audit_error')
                raise
        return run

    def bind(self, comp: object, task: PublicPropagationTask | None) -> Callable | None:
        """Attach the existing public sample loop to this exact composition and wallet."""
        if task is None:
            return None
        session = PublicSampleSession(task, lambda: run_youtube_sample(task, comp))
        comp.public_sample = session
        def run(state: dict) -> dict:
            """Stop subsequent requests on sample archive/validation failure."""
            result = session.round1(state)
            if session.reason == 'sample_task_failed' and not comp.inputs.wallet.snapshot().halted:
                comp.inputs.wallet.halt('youtube_sample_audit_error')
            return result
        return run

    def project(self, snapshot: object, task: object, cost: dict, halted: bool) -> object:
        """Display the selected page scope and the shared whole-case cost without altering citations."""
        row = self.collection.load()
        response = row['response']
        adopted = len(task.batch.observations) if task else 0
        main, marker, appendix = (snapshot.report or '').partition(APPENDIX_TITLE)
        main = main.replace('\n## 用量记录\n', '\n## 主图模型用量（事实与仲裁）\n', 1)
        if adopted and self.collection.mode == 'live':
            appendix = appendix.replace('\n\n材料来源：已归档网页材料\n',
                                        '\n\n材料来源：已归档 YouTube 评论\n', 1)
        report = main + marker + appendix + (f"\n\nYouTube 窗口：{row['start']} 至 {row['end']}（北京时间，含结束日）。"
            f"返回 {len(response['comments'])} 条，窗内 {len(response['samples'])} 条，采用 {adopted} 条。"
            f"状态：{response['outcome']}；选用：{self.use}。")
        links = dict.fromkeys((citation.evidence_id, citation.source_url)
            for citation in snapshot.citations
            if citation.citation_id.startswith('public:') and citation.validation == 'passed' and citation.source_url)
        for evidence_id, url in links:
            report += f"\n\n{evidence_id}：[原评论]({url})"
        if 'model_request_count' in cost:
            report += f"\n\n整案模型请求数（含传播）：{cost['model_request_count']}"
        report += f"\n\n整案已知费用：{cost['known_cost_cny']} 元；费用状态：{cost['cost_status']}。"
        return snapshot.model_copy(update=dict(report=report, known_cost_cny=cost['known_cost_cny'],
            cost_status=cost['cost_status'], status='failed' if halted else snapshot.status))


def run_youtube_sample(task: PublicPropagationTask, comp: object) -> object:
    """Reuse the public propagation loop with the same coordinator and per-run identity."""
    from src.yuqing.agents.propagation import run_propagation_agent
    from src.yuqing.schemas import AgentSpec, TaskInput
    from src.yuqing.serve.public_propagation import PUBLIC_SLOTS
    from src.yuqing.serve.research_public_activation import APPROVED_PROMPT_SHA
    raw = (settings.PROJECT_ROOT/'prompts/propagation/public_materials.live-v1.md').read_bytes()
    if sha256(raw).hexdigest() != APPROVED_PROMPT_SHA:
        raise ValueError('youtube_prompt_drift')
    call = comp.propagation_call({})
    names = [*PUBLIC_SLOTS, 'submit_propagation_report']
    policy = settings.PolicyConfig(allow={'propagation_analyst':names}, risk_tier={n:'R0' for n in names}, r2_mode='deny')
    return run_propagation_agent(AgentSpec(agent='propagation_analyst', system_prompt=raw.decode()),
        TaskInput(sample_id=task.sample_id, claim=task.claim), call.context, public_task=task,
        node_config=call.node_config, worker_scope=comp.inputs.propagation_worker_scope, policy_config=policy,
        coordinator_factory=comp.coordinator, experiment_identity=comp.identity(call.context))
