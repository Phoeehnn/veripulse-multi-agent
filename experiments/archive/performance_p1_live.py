"""固定P1开发题单的单场Runner；逐槽终态、Trace和钱包对账。"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping
from uuid import uuid4

from src.yuqing import llm
from experiments.archive import performance_p1 as p1
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.preflight import propagation_contract_fingerprints, validate_runtime_config
from src.yuqing.harness.tools import EvidenceWorkerScope, ToolRegistry
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.trace import RunContext, TraceEvent
from src.yuqing.schemas import AgentSpec, Sample
from src.yuqing.settings import PROJECT_ROOT, get_settings, get_cost_config, get_policy_config, get_propagation_tools_config

AUTH = PROJECT_ROOT / "docs/cards/P1-live-authorization.json"
CAP = Decimal("3.84")
ACTIVE: ContextVar[tuple[str, str, str] | None] = ContextVar("p1_live_slot", default=None)
logger = logging.getLogger(__name__)


@dataclass
class Batch:
    """新场次的唯一钱包、冻结条件与完整36槽清单。"""

    root: Path
    wallet: ExperimentSession
    conditions: dict[str, str]
    items: tuple[p1.P1SelectionItem, ...]
    samples: tuple[Sample, ...]
    prompt: str
    rows: list[dict[str, object]]
    live: bool


def require_active_slot(ctx: RunContext, identity: Mapping[str, str]) -> None:
    """Agent只能使用当前Runner已登记的样本、组和run_id。"""
    if ACTIVE.get() != (ctx.run_id, ctx.sample_id, identity.get("group_id")):
        raise ValueError("p1_live_slot_unregistered")
    if identity.get("run_id") != ctx.run_id or identity.get("sample_id") != ctx.sample_id:
        raise ValueError("p1_live_identity_mismatch")


def conditions(live: bool) -> dict[str, str]:
    """冻结依赖、工具、模型和许可；不改原题单离线许可快照。"""
    items = p1.load_frozen_p1_selection()
    caps = p1.p1_group_caps(items)
    values = p1._p1_conditions(PROJECT_ROOT, items, p1.load_frozen_p1_prompt(), caps, CAP)
    values.update(propagation_contract_fingerprints())
    values.update(runner_sha256=p1._sha_file(Path(__file__)),
        authorization_sha256=p1._sha_file(AUTH), execution_mode="live" if live else "mock",
        provider_calls_authorized=str(live).lower(), paid_execution_authorized=str(live).lower())
    return values


def preflight(live: bool) -> None:
    """逐题逐组检查首请求/收尾容量、提示词许可和单场硬费用。"""
    from src.yuqing.agents import propagation

    validate_runtime_config()
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    expected = {"selection_sha256": p1.P1_SELECTION_SHA256, "prompt_sha256": p1.P1_PROMPT_SHA256,
        "max_cost_cny": "3.84", "slots": 36, "runs": 1, "approved": True}
    if any(auth.get(key) != value for key, value in expected.items()):
        raise ValueError("p1_authorization_mismatch")
    runtime = get_settings()
    if live and (auth.get("provider_destination_approved") is not True or
            auth.get("provider_base_url") != runtime.deepseek_base_url.rstrip("/") or
            auth.get("provider_model") != runtime.deepseek_model):
        raise ValueError("p1_provider_destination_not_approved")
    if live and auth.get("observation_sha256") != p1.P1_OBSERVATION_SHA256:
        raise ValueError("p1_observation_v2_not_authorized")
    contracts = json.loads((PROJECT_ROOT / "prompts/propagation/contracts.json").read_text())
    if not any(row.get("prompt_sha256") == p1.P1_PROMPT_SHA256 and row.get("approved_for_dev") is True
               for row in contracts):
        raise ValueError("p1_prompt_dev_permission_missing")
    if live and (llm._offline_replay.get() is not None or get_settings().deepseek_api_key is None):
        raise ValueError("p1_live_environment_invalid")
    items, samples = p1.load_frozen_p1_samples()
    for sample in samples:
        p1.observation_as_of(sample)
    spec = AgentSpec(agent="propagation_analyst", system_prompt=p1.load_frozen_p1_prompt())
    for item, sample in zip(items, samples):
        for group in item.order:
            profile = p1.group_profile(group)
            config = p1.config_for_group(p1.get_propagation_harness_config(), profile)
            budget = BudgetTracker(config, get_cost_config(), correction_enabled=profile.correction_enabled)
            registry = ToolRegistry()
            propagation.register_propagation_tools(registry, tools_config=get_propagation_tools_config(),
                contract_version="v1.1")
            ctx = RunContext(run_id=uuid4().hex, sample_id=sample.sample_id, agent=spec.agent)
            reason = propagation._startup_error(spec, sample, ctx, config, registry, get_policy_config(),
                budget, p1_cost_cap_cny=profile.slot_cap_cny)
            if reason:
                raise ValueError(f"p1_capacity_rejected:{group}:{reason}")


def create_batch(root: Path, live: bool) -> Batch:
    """全量预检后新建一次性场次；真实单场认领不可重置。"""
    preflight(live)
    frozen = conditions(live)
    items, samples = p1.load_frozen_p1_samples()
    target = p1._new_experiment_root(root)
    if live:
        p1._write_unique(PROJECT_ROOT / "traces/P1-dev12-live-v1.claim", {
            "root": str(target), "conditions": frozen, "authorized_on": "2026-11-03"})
    wallet = p1._create_p1_wallet(target, frozen, CAP, get_cost_config())
    rows = [{"sample_id": item.sample_id, "group": group, "status": "not_executed"}
            for item in items for group in item.order]
    batch = Batch(target, wallet, frozen, items, samples, p1.load_frozen_p1_prompt(), rows, live)
    p1._write_unique(target / "conditions.json", frozen)
    p1._write_unique(target / "observation.json", json.loads(
        (PROJECT_ROOT / "docs/cards/P1-observation-v2.json").read_bytes()))
    p1._write_unique(target / "inputs.json", [sample.model_dump(mode="json", exclude={
        "gold_veracity", "gold_risk"}) for sample in samples])
    save_state(batch)
    return batch


def save_state(batch: Batch) -> None:
    """每次保存新快照而非覆盖；未执行槽始终保留在分母。"""
    folder = batch.root / "states"
    folder.mkdir(exist_ok=True)
    p1._write_unique(folder / f"{len(list(folder.glob('*.json'))):03d}.json", {
        "slots": batch.rows, "wallet": batch.wallet.snapshot().model_dump(mode="json")})


def execute_slot(batch: Batch, row: dict[str, object], sample: Sample) -> object:
    """正式Agent共用三层钱包和原校验器；探针不会进入本场次。"""
    from src.yuqing.agents import propagation

    profile = p1.group_profile(str(row["group"]))
    ctx = RunContext(run_id=str(row["run_id"]), sample_id=sample.sample_id,
        agent="propagation_analyst", trace_root=batch.root / "traces")
    scope = EvidenceWorkerScope()
    token = ACTIVE.set((ctx.run_id, ctx.sample_id, profile.group))
    try:
        return propagation._run_propagation_p1_live(AgentSpec(agent=ctx.agent, system_prompt=batch.prompt),
            sample.model_copy(update={"gold_veracity": None, "gold_risk": None}), ctx,
            node_config=p1.config_for_group(p1.get_propagation_harness_config(), profile),
            worker_scope=scope, p1_profile=profile,
            coordinator_factory=batch.wallet.coordinator_factory(group_caps=p1.p1_group_caps(batch.items)),
            experiment_identity={"run_id": ctx.run_id, "sample_id": ctx.sample_id,
                "group_id": profile.group, "variant": f"p1-{profile.group}"})
    finally:
        ACTIVE.reset(token)
        scope.close()


def audit_slot(batch: Batch, row: dict[str, object]) -> dict[str, object]:
    """钱包请求与Trace逐一核对，任何未知/未结算和身份不符拒绝。"""
    paths = list((batch.root / "traces").glob(f"*/{row['run_id']}.jsonl"))
    if len(paths) != 1:
        raise ValueError("p1_trace_missing")
    events = [TraceEvent.model_validate_json(line) for line in paths[0].read_text().splitlines()]
    if any(e.run_id != row["run_id"] or e.sample_id != row["sample_id"] or
           e.agent != "propagation_analyst" for e in events):
        raise ValueError("p1_trace_identity")
    terminal = [e.data for e in events if e.type == "run_outcome"]
    if len(terminal) != 1 or events[-1].type != "run_outcome" or terminal[0].get("halt_unconfirmed"):
        raise ValueError("p1_terminal_invalid")
    history = batch.wallet.load_history().events
    reserved = [e for e in history if e.get("event_type") == "reserved" and e.get("run_id") == row["run_id"]]
    if any(e.get("group_id") != row["group"] or e.get("sample_id") != row["sample_id"] for e in reserved):
        raise ValueError("p1_wallet_identity")
    dispatched, total = set(), Decimal("0")
    for request in reserved:
        request_id = request["request_id"]
        actions = [e for e in history if e.get("request_id") == request_id]
        ends = [e for e in actions if e["event_type"] in {"settled", "cancelled", "unknown"}]
        if len(ends) != 1 or ends[0]["event_type"] == "unknown":
            raise ValueError("p1_unsettled_request")
        if any(e["event_type"] == "dispatched" for e in actions):
            dispatched.add(request_id)
        if ends[0]["event_type"] == "settled":
            total += Decimal(str(ends[0]["actual_cost_cny"]))
    calls = [e for e in events if e.type == "llm_call"]
    audit_archives_and_receipts(batch, events, history)
    requested = [e.data["request_id"] for e in events if e.type == "llm_request"]
    if len(requested) != len(dispatched) or set(requested) != dispatched or len(calls) != len(dispatched):
        raise ValueError("p1_request_receipt_mismatch")
    if {e.data.get("request_id") for e in calls} != dispatched or any(e.data.get("usage_known") is not True for e in calls):
        raise ValueError("p1_usage_unknown")
    if total > p1.group_profile(str(row["group"])).slot_cap_cny:
        raise ValueError("p1_slot_over_budget")
    if abs(Decimal(str(terminal[0]["cost_cny"])) - total) > Decimal("0.000000001"):
        raise ValueError("p1_terminal_cost_mismatch")
    return {"audit": "verified", "requests": len(calls), "cost_cny": str(total),
        "reason": terminal[0]["reason"], "trace_path": str(paths[0]),
        "correction_used": terminal[0].get("correction_used", False)}


def audit_archives_and_receipts(batch: Batch, events: list[TraceEvent],
                                history: tuple[Mapping[str, object], ...]) -> None:
    """核归档SHA与逐请求usage计价，不能用末尾总数掩盖回执错配。"""
    settled = {e["request_id"]: Decimal(str(e["actual_cost_cny"]))
               for e in history if e["event_type"] == "settled"}
    traces = (batch.root / "traces").resolve()
    for event in events:
        archive = event.data.get("archive")
        if archive is not None:
            path = (traces / archive["path"]).resolve()
            if not path.is_relative_to(traces) or p1._sha_file(path) != archive["sha256"]:
                raise ValueError("p1_archive_invalid")
        if event.type == "llm_call" and event.data.get("usage_known") is True:
            usage = llm.TokenUsage.model_validate({"input_tokens": event.data["input_tokens"],
                "output_tokens": event.data["output_tokens"]}, strict=True)
            cost = Decimal(str(llm.compute_cost_cny(usage, get_cost_config())))
            if settled.get(event.data["request_id"]) != cost:
                raise ValueError("p1_receipt_cost_mismatch")


def run_batch(batch: Batch, replay_factory: Callable[[Sample, str], p1.P1OfflineReplay] | None = None) -> dict[str, object]:
    """串行执行固定36槽；每槽审计与费用核验后才推进，无自动重跑。"""
    if batch.live == (replay_factory is not None):
        raise ValueError("p1_execution_mode_mismatch")
    samples = {sample.sample_id: sample for sample in batch.samples}
    try:
        for row in batch.rows:
            if row["status"] != "not_executed":
                raise ValueError("p1_slot_already_claimed")
            if conditions(batch.live) != batch.conditions:
                raise ValueError("p1_condition_drift")
            row.update(status="running", run_id=uuid4().hex)
            save_state(batch)
            started = perf_counter()
            try:
                result = run_slot_with_mode(batch, row, samples[str(row["sample_id"])], replay_factory)
                row.update(audit_slot(batch, row), status="qualified" if result else "rejected",
                    elapsed_s=perf_counter() - started, report=result.report.model_dump(mode="json") if result else None)
                if conditions(batch.live) != batch.conditions:
                    raise ValueError("p1_condition_drift")
                check_wallet(batch)
            except Exception as exc:
                row.update(status="audit_or_execution_failed", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                save_state(batch)
            logger.info("P1 %s %s %s cost=%s", row["group"], row["sample_id"], row["status"], row.get("cost_cny"))
    except Exception as exc:
        batch.wallet.halt("p1_batch_stopped")
        save_state(batch)
        return finish(batch, str(exc))
    return finish(batch, None)


def run_slot_with_mode(batch: Batch, row: dict[str, object], sample: Sample,
                       factory: Callable[[Sample, str], p1.P1OfflineReplay] | None) -> object:
    """真实场次没有Mock回调；离线场次必须显式回放。"""
    if factory is None:
        return execute_slot(batch, row, sample)
    with llm.use_offline_completion_replay(factory(sample, str(row["group"]))):
        return execute_slot(batch, row, sample)


def check_wallet(batch: Batch) -> None:
    """总额和分组累计额从同一个持久账本核验，不重置未知费用。"""
    snap = batch.wallet.snapshot()
    if snap.halted or snap.pending_requests or snap.unknown_requests or snap.cost_cny is None:
        raise ValueError("p1_wallet_not_clean")
    if snap.cost_cny > CAP:
        raise ValueError("p1_total_over_budget")
    totals: dict[str, Decimal] = {name: Decimal("0") for name in p1.P1_GROUPS}
    history = batch.wallet.load_history().events
    groups = {e["request_id"]: e.get("group_id") for e in history if e["event_type"] == "reserved"}
    for event in history:
        if event["event_type"] == "settled":
            totals[groups[event["request_id"]]] += Decimal(str(event["actual_cost_cny"]))
    if sum(totals.values()) != snap.cost_cny or any(totals[g] > cap for g, cap in p1.p1_group_caps(batch.items).items()):
        raise ValueError("p1_group_or_total_audit_failed")


def finish(batch: Batch, error: str | None) -> dict[str, object]:
    """保存全部36槽及整场对账，失败/未执行槽不从分母移除。"""
    history = batch.wallet.load_history().events
    known = {row.get("run_id") for row in batch.rows if row.get("run_id")}
    if any(e.get("run_id") not in known for e in history if e["event_type"] == "reserved"):
        raise ValueError("p1_orphan_wallet_run")
    summary = {"slots": batch.rows, "planned": 36, "execution_mode": "live" if batch.live else "mock",
        "stopped": error, "wallet": batch.wallet.snapshot().model_dump(mode="json"),
        "accuracy": None, "content_quality": "pending_blind_review",
        "complete": error is None and len(batch.rows) == 36 and all(
            row["status"] in {"qualified", "rejected"} for row in batch.rows)}
    p1._write_unique(batch.root / "summary.json", summary)
    return summary
