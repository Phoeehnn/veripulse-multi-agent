"""Explicit offline news investigation using synthetic search archives and per-role SDK replays."""
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from functools import wraps
from hashlib import sha256
import os
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from src.yuqing import llm, settings
from src.yuqing.agents.propagation import run_propagation_agent
from src.yuqing.eval.dual_budget import price_fingerprint
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.graph.runtime import CaseServiceBindings
from src.yuqing.harness.budget import BudgetDecision
from src.yuqing.harness.initial_evidence import read_initial_evidence_manifest
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.retrieval.public_material_contracts import PublicSourceDescriptor
from src.yuqing.retrieval.public_materials import adapt_public_material, archive_public_material, load_public_material
from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask, PublicSignalResult
from src.yuqing.retrieval.web_contracts import WebMaterialBatch
from src.yuqing.retrieval.web_evidence import _digest
from src.yuqing.retrieval.youtube_comments import youtube_comment_batch
from src.yuqing.schemas import AgentSpec, TaskInput
from src.yuqing.serve.public_propagation import PUBLIC_SLOTS
from src.yuqing.serve.research_contracts import ResearchRequest
from src.yuqing.serve.research_public import PublicSampleSession
from src.yuqing.serve.research_public_activation import APPROVED_PROMPT_SHA
from src.yuqing.serve.research_runner import _case_inputs, save_case_cost
from src.yuqing.serve.research_service import (
    ResearchComposition, ResearchSample, assemble_research_case, research_snapshot, save_research_snapshot,
)


@dataclass(frozen=True)
class OfflineYoutubePage:
    """Bind an explicitly selected offline page and observation window to one investigation."""
    case_id: str
    claim: str
    video_id: str
    start: datetime
    end: datetime
    fetched_at: datetime
    comments: tuple[dict, ...]
    selection_reason: str = "selected"


@dataclass(frozen=True)
class OfflineNews:
    """Bind a previously archived synthetic search receipt, explicit classifications and finite responses."""
    manifest_sha256: str
    sources: tuple[PublicSourceDescriptor, ...]
    fact: llm.OfflineCompletionReplay
    propagation: llm.OfflineCompletionReplay
    arbiter: llm.OfflineCompletionReplay
    youtube: OfflineYoutubePage | None = None


class _OfflineCaseWallet(PersistentExperimentBudget):
    """Treat the first shared-cap refusal as a case stop, including for concurrent branches."""

    def reserve(self, request_id: str, amount: Decimal, *, identity: Mapping[str, str] | None = None) -> BudgetDecision:
        """Record the refusal before another role can acquire the same ledger lock."""
        with self._lock:
            decision = super().reserve(request_id, amount, identity=identity)
            if not decision.allowed and not self.snapshot().halted:
                self.halt(decision.reason)
            return decision

    def mark_dispatched(self, request_id: str) -> None:
        """Cancel a still-unsent reservation if another branch has already stopped the wallet."""
        with self._lock:
            if self.snapshot().halted:
                self.cancel_unsent(request_id)
                raise ValueError("offline_news_halted")
            super().mark_dispatched(request_id)


def _prepare(case: Path, root: Path, fixture: OfflineNews) -> tuple[ResearchRequest, PublicPropagationTask | None, int]:
    """Validate original case and retrieval provenance before creating an output or a wallet."""
    sandbox = (settings.PROJECT_ROOT / "traces/tmp").resolve()
    if root.resolve().drive.lower() != "d:" or not root.resolve().is_relative_to(sandbox) or case.resolve().parent != root.resolve():
        raise ValueError("offline_news_requires_isolated_case")
    if type(fixture) is not OfflineNews or any(type(x) is not llm.OfflineCompletionReplay
            for x in (fixture.fact, fixture.propagation, fixture.arbiter)):
        raise ValueError("offline_news_requires_all_replays")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")) or (root / "wallet.sqlite").exists():
        raise ValueError("offline_news_requires_fresh_offline_environment")
    request = ResearchRequest.model_validate_json((case / "request.json").read_bytes())
    directory = case / "retrieval/evidence"
    manifest_path, batch_path = directory / "initial-manifest.json", directory / "materials.json"
    if any(p.resolve() != p.absolute() for p in (case / "request.json", manifest_path, batch_path)):
        raise ValueError("offline_news_archive_escape")
    manifest, original_sha = read_initial_evidence_manifest(manifest_path, case.name, request.input_value)
    if request.input_kind != "text" or original_sha != fixture.manifest_sha256:
        raise ValueError("offline_news_manifest_mismatch")
    batch = WebMaterialBatch.model_validate_json(batch_path.read_bytes())
    if (batch.provenance != "synthetic_fixture" or batch.query != request.input_value
            or _digest(batch.model_dump(mode="json")) != manifest.retrieval_sha256):
        raise ValueError("offline_news_retrieval_mismatch")
    page = fixture.youtube
    if page is None:
        public = adapt_public_material(batch, fixture.sources, case_id=case.name, claim=request.input_value)
    else:
        if page.case_id != case.name or page.claim != request.input_value or fixture.sources:
            raise ValueError("offline_youtube_case_or_sources_mismatch")
        public = youtube_comment_batch(list(page.comments),
            video_id=page.video_id, start=page.start,
            end=page.end, fetched_at=page.fetched_at, case_id=page.case_id, claim=page.claim,
            provenance="synthetic_fixture")
    destination = case / "public-material"
    archive_public_material(public, destination)
    bound = load_public_material(destination, case_id=case.name, claim=request.input_value)
    adopted = bool(bound.observations) and (page is None or not page.selection_reason.startswith("user_skipped"))
    task = PublicPropagationTask(sample_id=case.name, claim=request.input_value, batch=bound) if adopted else None
    return request, task, len(bound.observations)


def _sample(task: PublicPropagationTask, comp: ResearchComposition) -> PublicSignalResult | None:
    """Call the original public propagation loop on this composition's own run and coordinator."""
    from src.yuqing.serve.research_youtube_activation import run_youtube_sample
    return run_youtube_sample(task, comp)


def _bind(bindings: CaseServiceBindings, comp: ResearchComposition, task: PublicPropagationTask | None,
          fixture: OfflineNews) -> CaseServiceBindings:
    """Install replay inside each graph worker, including followup, before reaching a model gateway."""
    wallet = comp.inputs.wallet
    def guarded(handler: Callable, replay: llm.OfflineCompletionReplay) -> Callable:
        """Gate a single graph worker and keep existing in-flight settlement semantics."""
        @wraps(handler)
        def invoke(*args: object, **kwargs: object) -> object:
            """Run this graph node under its own finite transport."""
            if wallet.snapshot().halted:
                raise ValueError("offline_news_halted")
            with llm.use_offline_completion_replay(replay):
                try:
                    return handler(*args, **kwargs)
                except Exception:
                    if not wallet.snapshot().halted:
                        wallet.halt("offline_node_exception")
                    raise
        return invoke
    prop = bindings.propagation_round1
    if task is not None:
        session = PublicSampleSession(task, lambda: _sample(task, comp))
        comp.public_sample = session
        def round1(state: dict) -> dict:
            """Escalate unexpected sample/validation exceptions beyond the side task."""
            result = session.round1(state)
            if session.reason == "sample_task_failed" and not wallet.snapshot().halted:
                wallet.halt("offline_sample_audit_error")
            return result
        prop = round1
    return replace(bindings, fact_round1=guarded(bindings.fact_round1, fixture.fact),
        fact_followup=guarded(bindings.fact_followup, fixture.fact), arbitrate=guarded(bindings.arbitrate, fixture.arbiter),
        propagation_round1=guarded(prop, fixture.propagation))


def run_offline_news(case: Path, root: Path, limit: Decimal, fixture: OfflineNews) -> None:
    """Execute the news worker on a pinned search fixture with one fresh mock ledger and snapshot."""
    request, task, window_count = _prepare(case, root, fixture)
    manifest = dict(experiment_id=uuid4().hex, mode="full", max_cost_cny=str(limit),
        price_fingerprint=price_fingerprint(settings.get_cost_config()), conditions_fingerprint=fixture.manifest_sha256,
        format_version=1)
    wallet = _OfflineCaseWallet.create(root / "wallet.sqlite", manifest, settings.get_evaluation_config())
    fact, prop = EvidenceWorkerScope(), EvidenceWorkerScope()
    try:
        sample = ResearchSample(sample_id=case.name, claim=request.input_value)
        inputs = _case_inputs(case, sample, wallet, case / "index", fact, prop)
        assembly, comp = assemble_research_case(request, inputs, case / "retrieval/evidence",
            external_calls_authorized=True, offline_bind=lambda b, c: _bind(b, c, task, fixture))
        output = assembly.invoke()
        snapshot = research_snapshot(output, comp, execution_mode="mock")
        if wallet.snapshot().halted:
            snapshot = snapshot.model_copy(update={"status": "failed"})
        cost = save_case_cost(case, case.name, wallet)
        sent = sum(e["event_type"] == "dispatched" for e in wallet.load_history().events)
        report = (snapshot.report or "").replace("## 用量记录", "## 主图用量（事实与仲裁）", 1)
        if fixture.youtube is not None:
            page = fixture.youtube
            count = len(task.batch.observations) if task else 0
            report += (f"\n\nYouTube 评论观察窗：{page.start.isoformat()} 至 {page.end.isoformat()}（不含结束时刻）。"
                       f"本页回放 {len(page.comments)} 条，窗内去重 {window_count} 条，采用 {count} 条。选择状态：{page.selection_reason}。")
            for citation in snapshot.citations:
                if citation.citation_id.startswith("public:") and citation.validation == "passed":
                    report += f"\n\n{citation.evidence_id}：[原评论]({citation.source_url})"
        report += (f"\n\n## 整案离线用量（含传播样本）\n请求数：{sent}；"
                   f"模拟已知费用小计：¥{cost['known_cost_cny']}；费用状态：{cost['cost_status']}。")
        save_research_snapshot(case, snapshot.model_copy(update={"report": report,
            "known_cost_cny": cost["known_cost_cny"], "cost_status": cost["cost_status"]}))
    finally:
        fact.close()
        prop.close()
        wallet.close()
