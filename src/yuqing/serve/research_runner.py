"""Run one real investigation in the core Python process with an isolated cumulative wallet."""
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.yuqing.serve.research_three_agent import OfflineNews

from pydantic import SecretStr
from src.yuqing import settings
from src.yuqing.eval.dual_budget import price_fingerprint
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.graph.composition import CaseInputs
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.llm import call_web_retrieval
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.retrieval.tavily_adapter import run_tavily_materials, WebResearchMaterialResult
from src.yuqing.retrieval.web_provider_contracts import WebRetrievalRequest, WebRetrievalReceipt
from src.yuqing.schemas import AgentSpec
from src.yuqing.serve.research_contracts import ResearchRequest
from src.yuqing.serve.research_index import build_research_index, verify_research_index
from src.yuqing.serve.research_service import ResearchSample, assemble_research_case, research_snapshot, save_research_snapshot
from src.yuqing.serve.research_recovery import continuation_root, prepare_continuation


def missing_live_configuration() -> list[str]:
    """List absent configuration names without reading or returning credential values."""
    return [name for name in ("TAVILY_API_KEY", "DEEPSEEK_API_KEY", "MODEL_CACHE_DIR",
                              "LIVE_RESEARCH_CNY_PER_CREDIT") if not os.environ.get(name, "").strip()]


def publish_stage(case: Path, stage: str, status: str) -> None:
    """Publish an observed stage atomically for the UI process."""
    row = {"stage": stage, "status": status, "time": datetime.now(timezone.utc).isoformat()}
    with (case / "stages.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def open_research_wallet(root: Path, *, limit_cny: Decimal, credit_cny: Decimal) -> PersistentExperimentBudget:
    """Keep one cumulative search-plus-model ledger across cases and UI restarts."""
    if (root / "continuation-plan.json").exists():
        target, allowance = continuation_root(root, total_limit=limit_cny)
        return open_research_wallet(target, limit_cny=allowance, credit_cny=credit_cny)
    manifest_path = root / "session-manifest.json"
    conditions = sha256(str(credit_cny).encode()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["max_cost_cny"] != str(limit_cny) or manifest["conditions_fingerprint"] != conditions:
            raise ValueError("research_wallet_conditions_changed")
        return PersistentExperimentBudget.resume(root / "wallet.sqlite", manifest, settings.get_evaluation_config())
    manifest = {"experiment_id": uuid4().hex, "mode": "full", "max_cost_cny": str(limit_cny),
        "price_fingerprint": price_fingerprint(settings.get_cost_config()),
        "conditions_fingerprint": conditions, "format_version": 1}
    wallet = PersistentExperimentBudget.create(root / "wallet.sqlite", manifest, settings.get_evaluation_config())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return wallet


def budgeted_web_call(request: WebRetrievalRequest, wallet: PersistentExperimentBudget, *,
                       credit_cny: Decimal, sample_id: str) -> WebRetrievalReceipt:
    """Reserve known basic-search/extract credit ceilings before the sole HTTP gateway."""
    credits = 1 if request.operation == "search" else (len(request.payload["urls"]) + 4) // 5
    request_id = uuid4().hex
    identity = {"sample_id": sample_id, "run_id": request_id, "group_id": "live-research-web", "variant": "web"}
    bound = Decimal(credits) * credit_cny
    wallet.prepare(request_id, bound, identity)
    decision = wallet.reserve(request_id, bound, identity=identity)
    if not decision.allowed:
        raise ValueError("research_cumulative_budget_exhausted")
    wallet.mark_dispatched(request_id)
    try:
        receipt = call_web_retrieval(request, api_key=SecretStr(os.environ["TAVILY_API_KEY"]),
            timeout_s=settings.get_settings().llm_timeout_s, external_calls_authorized=True)
    except Exception:
        wallet.settle(request_id, None)
        raise
    usage = receipt.usage or {}
    actual = Decimal(str(usage["credits"])) * credit_cny if receipt.billing_status == "usage_reported" else None
    wallet.settle(request_id, actual)
    return receipt


def run_live_research(case: Path, root: Path, *, limit_cny: Decimal, offline: "OfflineNews | None" = None, activation: object = None) -> None:
    """Execute actual search, indexing and graph services; never fabricate a live result."""
    if offline is not None:
        from src.yuqing.serve.research_three_agent import run_offline_news
        run_offline_news(case, root, limit_cny, offline)
        return
    from src.yuqing.serve.research_web_review import validate_activation_web
    approved_web = validate_activation_web(case, activation, limit_cny)
    rate = _live_research_rate()
    request = ResearchRequest.model_validate_json((case / "request.json").read_bytes())
    if request.input_kind != "text":
        raise ValueError("url_input_requires_claim_text")
    sample = ResearchSample(sample_id=case.name, claim=request.input_value)
    wallet = open_research_wallet(root, limit_cny=limit_cny, credit_cny=rate)
    fact_scope, propagation_scope = EvidenceWorkerScope(), EvidenceWorkerScope()
    try:
        if activation is not None:
            wallet = activation.bind_wallet(wallet)
            activation.claim(case)
        task = activation.task(live=True) if activation else None
        if activation is not None and activation.use and task is None:
            wallet.halt("youtube_selected_material_empty")
            raise ValueError("youtube_selected_material_empty")
        publish_stage(case, "网页搜索与正文提取", "running")
        material = _fetch_material(case, sample, wallet, rate, approved_web)
        _require_web_material(material, wallet, activation)
        publish_stage(case, "网页搜索与正文提取", "completed")
        publish_stage(case, "本案检索索引", "running")
        store = build_research_index(case / "retrieval/evidence", case / "index",
                                     sample_id=sample.sample_id, claim=sample.claim)
        verify_research_index(case / "retrieval/evidence", store, sample_id=sample.sample_id, claim=sample.claim)
        publish_stage(case, "本案检索索引", "completed")
        inputs = _case_inputs(case, sample, wallet, store, fact_scope, propagation_scope)
        assembly, composition = assemble_research_case(request, inputs, case / "retrieval/evidence",
            external_calls_authorized=True, sample_bind=(lambda c: activation.bind(c, task)) if activation else None,
            fact_guard=activation.guard_fact if activation else None)
        publish_stage(case, "Agent调查与仲裁", "running")
        output = assembly.invoke()
        snapshot = research_snapshot(output, composition, execution_mode="live")
        cost = save_case_cost(case, sample.sample_id, wallet)
        if activation is not None:
            snapshot = activation.project(snapshot, task, cost, wallet.snapshot().halted)
        save_research_snapshot(case, snapshot)
        publish_stage(case, "Agent调查与仲裁", snapshot.status)
    finally:
        save_case_cost(case, sample.sample_id, wallet)
        fact_scope.close()
        propagation_scope.close()
        wallet.close()


def _require_web_material(material: WebResearchMaterialResult, wallet: PersistentExperimentBudget, activation: object) -> None:
    """Halt on unusable retrieval before indexing or model dispatch."""
    if material.status not in {"prepared", "empty"} or wallet.snapshot().halted:
        if activation is not None and not wallet.snapshot().halted:
            wallet.halt("web_retrieval_audit_failed")
        raise ValueError("web_retrieval_failed_or_usage_unknown")
    if activation is not None and activation.use and material.status == "empty":
        wallet.halt("web_selected_material_empty")
        raise ValueError("web_selected_material_empty")


def _live_research_rate() -> Decimal:
    """Validate live environment before opening a wallet or dispatching retrieval."""
    missing = missing_live_configuration()
    if missing:
        raise ValueError("missing_live_configuration:" + ",".join(missing))
    if Path(os.environ["MODEL_CACHE_DIR"]).resolve().drive.lower() != "d:":
        raise ValueError("model_cache_must_be_on_d_drive")
    rate = Decimal(os.environ["LIVE_RESEARCH_CNY_PER_CREDIT"])
    if not rate.is_finite() or rate <= 0:
        raise ValueError("invalid_credit_conversion")
    return rate


def _fetch_material(case: Path, sample: ResearchSample, wallet: PersistentExperimentBudget,
                    rate: Decimal, approved_web: tuple | None = None) -> WebResearchMaterialResult:
    """Use explicit basic requests with usage receipts and the existing candidate bound."""
    config = settings.get_harness_config()
    def invoke(request: WebRetrievalRequest) -> WebRetrievalReceipt:
        """Charge this invocation's web requests to the same cumulative ledger."""
        if approved_web is not None and request.operation == "search":
            if request != approved_web[0]:
                raise ValueError("web_review_request_changed")
            return approved_web[1]
        return budgeted_web_call(request, wallet, credit_cny=rate, sample_id=sample.sample_id)
    search = approved_web[0] if approved_web is not None else WebRetrievalRequest(operation="search", payload={"query": sample.claim,
        "max_results": config.search_k, "search_depth": "basic", "include_usage": True})
    return run_tavily_materials(search,
        extract_options={"extract_depth": "basic", "format": "text", "include_usage": True},
        max_extract_urls=config.search_k, invoke=invoke, destination=case / "retrieval", sample_id=sample.sample_id,
        claim=sample.claim, chunk_chars=config.max_tool_result_chars,
        fetched_at=datetime.now(timezone.utc).isoformat(), provenance="recorded_response")


def _case_inputs(case: Path, sample: ResearchSample, wallet: PersistentExperimentBudget,
                  store: Path, fact: EvidenceWorkerScope, propagation: EvidenceWorkerScope) -> CaseInputs:
    """Use existing frozen specs and per-case writable stores; no original workspace state."""
    root = settings.PROJECT_ROOT
    return CaseInputs(case.name, sample, "full", case / "traces", wallet, fact, propagation,
        BM25Index([], None), store, case / "retrieval/evidence/initial-manifest.json",
        AgentSpec(agent="propagation_analyst", system_prompt=(root / "prompts/propagation/v1.md").read_text(encoding="utf-8")),
        AgentSpec(agent="arbiter", system_prompt=(root / "prompts/arbiter/v2.md").read_text(encoding="utf-8")),
        group_id="live-research", variant="web")


def save_case_cost(case: Path, sample_id: str, wallet: PersistentExperimentBudget) -> dict[str, object]:
    """Project only this case's model and web requests; never include another case's spend."""
    history = wallet.load_history()
    state = replay_history(history, wallet.manifest_snapshot())
    requests = {rid: row for rid, row in state.requests.items() if row.identity.get("sample_id") == sample_id}
    model_requests = sum(event["event_type"] == "dispatched" and event["request_id"] in requests
                         and requests[event["request_id"]].identity.get("group_id") != "live-research-web"
                         for event in history.events)
    settled = [row for row in requests.values() if row.state == "settled"]
    amount = sum((row.actual for row in settled), Decimal(0))
    known = all(row.state in {"settled", "cancelled"} for row in requests.values())
    data = {"known_cost_cny": str(amount), "cost_status": "known" if known else "partial" if settled else "unknown",
            "model_request_count": model_requests,
            "cumulative_known_cny": str(wallet.snapshot().known_cost_subtotal_cny),
            "cumulative_limit_cny": str(wallet.snapshot().max_cost_cny)}
    plan_path = wallet.storage_path.parent.parent / "continuation-plan.json"
    if wallet.storage_path.parent.name == "continuation" and plan_path.exists():
        plan = prepare_continuation(plan_path.parent, total_limit=Decimal(json.loads(plan_path.read_text(encoding="utf-8"))["total_limit_cny"]))
        prior = plan["predecessor"]
        data.update(cumulative_known_cny=str(Decimal(prior["known_cny"]) + wallet.snapshot().known_cost_subtotal_cny),
            cumulative_limit_cny=plan["total_limit_cny"], historical_unknown_reserved_cny=prior["unknown_reserved_cny"],
            cumulative_committed_cny=str(Decimal(prior["committed_cny"]) + wallet.snapshot().committed_cost_cny))
    (case / "cost.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def run_research_resume(case: Path, root: Path, source: Path, *, limit_cny: Decimal) -> None:
    """Re-run only model stages against verified archived evidence and the existing index."""
    source = source.resolve()
    if source.parent != root.resolve() or source == case.resolve():
        raise ValueError("resume_source_outside_run_root")
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("MODEL_CACHE_DIR"):
        raise ValueError("missing_resume_configuration")
    if Path(os.environ["MODEL_CACHE_DIR"]).resolve().drive.lower() != "d:":
        raise ValueError("model_cache_must_be_on_d_drive")
    request = ResearchRequest.model_validate_json((case / "request.json").read_bytes())
    original = ResearchRequest.model_validate_json((source / "request.json").read_bytes())
    if request != original:
        raise ValueError("resume_claim_mismatch")
    verify_research_index(source / "retrieval/evidence", source / "index", sample_id=source.name, claim=request.input_value)
    prepare_continuation(root, total_limit=limit_cny)
    rate = Decimal(os.environ["LIVE_RESEARCH_CNY_PER_CREDIT"])
    wallet = open_research_wallet(root, limit_cny=limit_cny, credit_cny=rate)
    fact_scope, propagation_scope = EvidenceWorkerScope(), EvidenceWorkerScope()
    try:
        reference = {"source_case": source.name, "original_request_sha256": sha256((source / "request.json").read_bytes()).hexdigest()}
        (case / "reuse.json").write_text(json.dumps(reference), encoding="utf-8")
        publish_stage(case, "复用已核验正文与原案索引（不再次搜索）", "completed")
        sample = ResearchSample(sample_id=source.name, claim=request.input_value)
        inputs = _case_inputs(case, sample, wallet, source / "index", fact_scope, propagation_scope)
        assembly, composition = assemble_research_case(request, inputs, source / "retrieval/evidence", external_calls_authorized=True)
        publish_stage(case, "Agent续查与仲裁", "running")
        output = assembly.invoke()
        result = research_snapshot(output, composition, execution_mode="live")
        save_research_snapshot(case, result)
        publish_stage(case, "Agent续查与仲裁", result.status)
    finally:
        save_case_cost(case, source.name, wallet)
        fact_scope.close()
        propagation_scope.close()
        wallet.close()
