"""传播专家的确定性选样、离线运行记录与结果保存。"""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Callable, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.yuqing.agents.propagation import PropagationResult
from src.yuqing.agents import propagation
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.preflight import PreparedPropagationPrompt
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.tools import EvidenceWorkerScope, ToolRegistry
from src.yuqing.harness.trace import RunContext, RunUsage, TraceEvent, UsageReceipt, get_run_usage, use_run_context
from src.yuqing.schemas import AgentSpec
from src.yuqing.settings import (PROJECT_ROOT, CostConfig, PropagationHarnessConfig, get_cost_config,
                                 get_policy_config, get_propagation_tools_config)
from src.yuqing.schemas import Sample


class PropagationEvalOutcome(BaseModel):
    """一条传播样本的可追溯执行结果或明确失败。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    result: PropagationResult | None = None
    status: Literal["success", "failed", "not_run"] = "failed"
    stop_reason: str = "execution_failed"
    trace_path: str = ""
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_cny: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    exact_cost_cny: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    known_cost_subtotal_cny: float = Field(default=0, ge=0, allow_inf_nan=False)
    run_id: str = ""
    halt_remaining: bool = False


def prompt_fingerprint(prompt_path: Path) -> str:
    """计算提示词原始字节指纹，不授予模型调用许可。"""

    return sha256(prompt_path.read_bytes()).hexdigest()


def _input_fingerprint(sample: Sample) -> str:
    """对真实专家输入字段指纹化，同时排除标注和证据答案。"""

    payload = sample.model_dump(mode="json", exclude={"gold_veracity", "gold_risk", "evidence_ids"})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def select_propagation_samples(
    chef_dev: list[Sample], ced_pool_out: list[Sample], *, pool_ids: set[str],
    chef_test_ids: set[str], limit: int = 10,
) -> list[Sample]:
    """按sample_id稳定选出CHEF dev与池外CED等量样本。"""

    if limit != 10 or len(chef_dev) < 5 or len(ced_pool_out) < 5:
        raise ValueError("propagation_manifest_requires_five_plus_five")
    if any(item.source != "chef" for item in chef_dev) or any(item.source != "ced" for item in ced_pool_out):
        raise ValueError("propagation_manifest_wrong_source")
    chef = sorted((item for item in chef_dev if item.sample_id not in chef_test_ids), key=lambda item: item.sample_id)
    ced = sorted((item for item in ced_pool_out if item.sample_id not in pool_ids), key=lambda item: item.sample_id)
    if len(chef) < 5 or len(ced) < 5:
        raise ValueError("propagation_manifest_source_exhausted")
    selected = chef[:5] + ced[:5]
    ids = [item.sample_id for item in selected]
    if len(set(ids)) != 10 or set(ids) & pool_ids or set(ids) & chef_test_ids:
        raise ValueError("propagation_manifest_overlap")
    return selected


def manifest_rows(samples: list[Sample], prompt_sha256: str, max_cost_cny: float,
                  fingerprints: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    """生成不含标签的固定输入与提示词指纹清单。"""

    if len(samples) != 10 or len({item.sample_id for item in samples}) != 10 or \
            not isfinite(max_cost_cny) or max_cost_cny <= 0:
        raise ValueError("propagation_manifest_must_contain_ten_unique_samples")
    if any(item.source not in {"chef", "ced"} for item in samples):
        raise ValueError("propagation_manifest_wrong_source")
    hashes = dict(fingerprints or {})
    required = {"tools_sha256", "schema_sha256", "config_sha256", "code_sha256"}
    if hashes and (set(hashes) != required or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    )):
        raise ValueError("propagation_manifest_fingerprints_invalid")
    return [{"sample_id": item.sample_id, "source": item.source, "variant": "propagation_v1",
             "split": "dev", "as_of": "None", "input_sha256": _input_fingerprint(item),
             "prompt_sha256": prompt_sha256, "max_cost_cny": str(max_cost_cny),
             **{name: hashes.get(name, "") for name in sorted(required)}} for item in samples]


def _row(sample: Sample, outcome: PropagationEvalOutcome, prompt_sha256: str,
         fingerprints: Mapping[str, str], evaluation_group: str | None = None) -> dict[str, object]:
    """把传播结果投影为不含真实性或风险标签的验收行。"""

    result = outcome.result
    report = result.report if result is not None else None
    row = {"sample_id": sample.sample_id, "source": sample.source, "variant": "propagation_v1",
            "split": "dev", "as_of": "None", "run_id": result.run_id if result else outcome.run_id,
            "status": "success" if result else outcome.status,
            "stop_reason": outcome.stop_reason if outcome.halt_remaining else
                (result.stop_reason if result else outcome.stop_reason),
            "trace_path": result.trace_path if result else outcome.trace_path,
            "prompt_sha256": prompt_sha256,
            **fingerprints,
            "emotion_level": report.emotion_level if report else "",
            "emotion_signals": json.dumps([item.model_dump() for item in report.emotion_signals], ensure_ascii=False) if report else "",
            "spread_level": report.spread_level if report else "",
            "spread_basis": json.dumps(report.spread_basis, ensure_ascii=False) if report else "",
            "narrative": report.narrative if report else "",
            "narrative_basis": json.dumps(report.narrative_basis, ensure_ascii=False) if report else "",
            "leads": json.dumps([item.model_dump() for item in report.leads], ensure_ascii=False) if report else "",
            "stats": json.dumps(result.stats.model_dump() if result and result.stats else None, ensure_ascii=False),
            "tokens_total": result.input_output_tokens if result else "",
            "input_tokens": outcome.input_tokens if outcome.input_tokens is not None else "",
            "output_tokens": outcome.output_tokens if outcome.output_tokens is not None else "",
            "cost_cny": outcome.cost_cny if outcome.cost_cny is not None else "",
            "known_cost_subtotal_cny": outcome.known_cost_subtotal_cny,
            "submission_validation": "accepted" if report else "not_submitted"}
    if evaluation_group is not None:
        row["evaluation_group"] = evaluation_group
    return row


def _validate_outcome(sample: Sample, outcome: PropagationEvalOutcome) -> PropagationEvalOutcome:
    """确保执行器未串样本，且成功结果与费用回执一致。"""

    if type(outcome) is not PropagationEvalOutcome:
        raise ValueError("propagation_outcome_type_invalid")
    result = outcome.result
    if result is None and outcome.status == "success":
        return outcome.model_copy(update={"status": "failed", "stop_reason": "success_without_result"})
    if result is not None and result.sample_id != sample.sample_id:
        raise ValueError("propagation_outcome_sample_mismatch")
    if result is not None and outcome.cost_cny is None:
        return outcome.model_copy(update={"result": None, "status": "failed", "stop_reason": "usage_unknown"})
    expected_cost = outcome.exact_cost_cny or (
        Decimal(str(outcome.cost_cny)) if outcome.cost_cny is not None else None)
    if result is not None and expected_cost is not None and expected_cost != Decimal(str(result.cost_cny)):
        raise ValueError("propagation_outcome_cost_mismatch")
    return outcome


def _write_rows(rows: list[dict[str, object]], results_dir: Path) -> Path:
    """独占创建传播结果CSV并在正式results目录登记指纹。"""

    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{date.today().isoformat()}_propagation_{uuid4().hex}.csv"
    with target.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from src.yuqing.eval.run_eval import register_result

    register_result(target, "传播专家工程/离线验收记录，不代表完整系统风险评测")
    return target


def evaluate_propagation(
    samples: list[Sample], runner: Callable[[Sample, object], PropagationEvalOutcome], *,
    wallet: object, prompt_sha256: str, results_dir: Path, max_cost_cny: float,
    fingerprints: Mapping[str, str] | None = None, limit: int | None = None,
    sample_groups: Mapping[str, str] | None = None,
) -> Path:
    """逐题把同一共享钱包交给执行器并保留全部成功、失败与未运行行。"""

    chosen = samples if limit is None else samples[:limit]
    if wallet is None or not chosen or not isfinite(max_cost_cny) or max_cost_cny <= 0 or \
            (limit is not None and limit < 1) or \
            len({item.sample_id for item in chosen}) != len(chosen):
        raise ValueError("propagation_eval_invalid_inputs")
    if sample_groups is not None and (set(sample_groups) != {item.sample_id for item in chosen} or
            any(value not in {"regression", "new_validation"} for value in sample_groups.values())):
        raise ValueError("propagation_eval_groups_invalid")
    if any(item.source not in {"chef", "ced"} for item in chosen):
        raise ValueError("propagation_eval_wrong_source")
    return _evaluate_selected_samples(chosen, runner, wallet=wallet, prompt_sha256=prompt_sha256,
        results_dir=results_dir, max_cost_cny=max_cost_cny, fingerprints=fingerprints,
        sample_groups=sample_groups)


def _evaluate_selected_samples(samples: list[Sample],
        runner: Callable[[Sample, object], PropagationEvalOutcome], *, wallet: object,
        prompt_sha256: str, results_dir: Path, max_cost_cny: float,
        fingerprints: Mapping[str, str] | None, sample_groups: Mapping[str, str] | None) -> Path:
    """执行预先校验的样本并为全部题目留下终态行。"""

    rows: list[dict[str, object]] = []
    known_cost = Decimal(0)
    stopped_reason = ""
    prompt_sha = prompt_sha256
    hashes = dict(fingerprints or {})
    for sample in samples:
        if stopped_reason:
            rows.append(_row(sample, PropagationEvalOutcome(status="not_run", stop_reason=stopped_reason),
                              prompt_sha, hashes, sample_groups.get(sample.sample_id) if sample_groups else None))
            continue
        clean = sample.model_copy(update={"gold_veracity": None, "gold_risk": None, "evidence_ids": []})
        try:
            outcome = runner(clean, wallet)
        except Exception as exc:
            outcome = PropagationEvalOutcome(stop_reason=type(exc).__name__)
        try:
            outcome = _validate_outcome(sample, outcome)
        except ValueError as exc:
            outcome = _failed_outcome(outcome, str(exc))
            _halt_after_outcome_failure(wallet)
        if outcome.cost_cny is None:
            stopped_reason = "usage_unknown_stop_remaining"
        else:
            known_cost += outcome.exact_cost_cny or Decimal(str(outcome.cost_cny))
            if known_cost > Decimal(str(max_cost_cny)):
                stopped_reason = "cost_limit_stop_remaining"
                outcome = outcome.model_copy(update={"status": "failed", "result": None,
                                                     "stop_reason": "cost_limit_exceeded"})
        rows.append(_row(sample, outcome, prompt_sha, hashes,
                         sample_groups.get(sample.sample_id) if sample_groups else None))
        if outcome.halt_remaining and not stopped_reason:
            stopped_reason = outcome.stop_reason
    if len(rows) != len(samples):
        raise RuntimeError("propagation_eval_denominator_mismatch")
    return _write_rows(rows, results_dir)


def _failed_outcome(outcome: object, reason: str) -> PropagationEvalOutcome:
    """拒绝不一致结果时保留可信费用、运行身份和追踪路径。"""

    if type(outcome) is not PropagationEvalOutcome:
        return PropagationEvalOutcome(stop_reason=reason, halt_remaining=True)
    return outcome.model_copy(update={"result": None, "status": "failed", "stop_reason": reason,
                                      "halt_remaining": True})


def _halt_after_outcome_failure(wallet: object) -> None:
    """结果与费用账本不一致时持久停账，无法确认时关闭现有会话。"""

    if not isinstance(wallet, ExperimentSession):
        return
    try:
        wallet.halt("propagation_outcome_mismatch")
    except Exception:
        wallet.close()


def run_propagation_batch(
    samples: list[Sample], *, prompt: PreparedPropagationPrompt,
    node_config: PropagationHarnessConfig, session: ExperimentSession,
    results_dir: Path, max_cost_cny: float, manifest_path: Path,
    contract_version: Literal["v1.0", "v1.1"] = "v1.0",
) -> Path:
    """复核完整启动条件后运行5+5，并将同一持久钱包交给全部题目。"""

    fingerprints = preflight_propagation_batch(samples, prompt=prompt, node_config=node_config,
        manifest_path=manifest_path, max_cost_cny=max_cost_cny, contract_version=contract_version)
    snapshot = session.snapshot()
    if snapshot.halted or snapshot.pending_requests or snapshot.unknown_requests:
        raise ValueError("propagation_experiment_wallet_not_clean")
    if snapshot.max_cost_cny != Decimal(str(max_cost_cny)):
        raise ValueError("propagation_wallet_limit_mismatch")
    runner = lambda sample, wallet: _run_propagation_sample(
        sample, wallet, prompt=prompt, node_config=node_config, contract_version=contract_version)
    return evaluate_propagation(samples, runner, wallet=session, prompt_sha256=prompt.prompt_sha256,
                                results_dir=results_dir, max_cost_cny=max_cost_cny,
                                fingerprints=fingerprints, limit=10)


def preflight_propagation_batch(samples: list[Sample], *, prompt: PreparedPropagationPrompt,
        node_config: PropagationHarnessConfig, manifest_path: Path, max_cost_cny: float,
        contract_version: Literal["v1.0", "v1.1"] = "v1.0") -> dict[str, str]:
    """在创建持久钱包前复核许可、manifest、来源、费用和全部首请求容量。"""

    from src.yuqing.eval.preflight import validate_propagation_startup
    from src.yuqing import settings

    fresh = validate_propagation_startup(prompt.path)
    if fresh != prompt or node_config != settings.get_propagation_harness_config():
        raise ValueError("propagation_prepared_state_changed")
    _validate_selected_sources(samples)
    fingerprints = {name: getattr(prompt, name) for name in (
        "tools_sha256", "schema_sha256", "config_sha256", "code_sha256")}
    if _read_manifest(manifest_path) != manifest_rows(samples, prompt.prompt_sha256,
            max_cost_cny, fingerprints):
        raise ValueError("propagation_manifest_drift")
    _validate_batch_capacity(samples, prompt, node_config, contract_version=contract_version)
    return fingerprints


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """读取无重复键的冻结manifest，格式或内容异常一律拒绝。"""

    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("propagation_manifest_blank_line")
        row = json.loads(line, object_pairs_hook=_unique_json_pairs)
        if not isinstance(row, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                            for key, value in row.items()):
            raise ValueError("propagation_manifest_invalid_row")
        rows.append(row)
    if len(rows) != 10:
        raise ValueError("propagation_manifest_requires_ten_rows")
    return rows


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝重复JSON键，避免manifest字段被解析器静默覆盖。"""

    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise ValueError("propagation_manifest_duplicate_key")
        row[key] = value
    return row


def _validate_selected_sources(samples: list[Sample]) -> None:
    """拒绝非5+5来源、CHEF非dev或冻结池/测试题交集。"""

    if len(samples) != 10 or sum(item.source == "chef" for item in samples) != 5 or \
            sum(item.source == "ced" for item in samples) != 5:
        raise ValueError("propagation_batch_requires_five_plus_five")
    if any(item.source == "chef" and not item.sample_id.startswith("chef:dev:") for item in samples):
        raise ValueError("propagation_batch_chef_dev_only")
    root = PROJECT_ROOT / "eval"
    pool = {json.loads(line)["sample_id"] for line in (root / "label_pool_100.jsonl").read_text(
        encoding="utf-8").splitlines() if line}
    test = {json.loads(line)["sample_id"] for line in (root / "test_60.jsonl").read_text(
        encoding="utf-8").splitlines() if line}
    ids = {item.sample_id for item in samples}
    if len(ids) != len(samples) or ids & (pool | test):
        raise ValueError("propagation_batch_frozen_overlap_or_duplicate")


def _validate_batch_capacity(samples: list[Sample], prompt: PreparedPropagationPrompt,
                             node_config: PropagationHarnessConfig, *,
                             contract_version: Literal["v1.0", "v1.1"] = "v1.0") -> None:
    """用传播Agent正式启动检查逐题验证工具权限、首请求及收尾容量。"""

    from src.yuqing import settings

    policy = settings.get_policy_config()
    registry = ToolRegistry()
    propagation.register_propagation_tools(registry, tools_config=get_propagation_tools_config(),
                                            contract_version=contract_version)
    prices = get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        raise ValueError("propagation_prices_unknown")
    for sample in samples:
        context = RunContext(run_id=uuid4().hex, sample_id=sample.sample_id,
                             agent="propagation_analyst")
        budget = BudgetTracker(node_config, CostConfig.model_validate(prices.model_dump()))
        spec = AgentSpec(agent="propagation_analyst", system_prompt=prompt.text)
        reason = propagation._startup_error(spec, sample, context, node_config, registry, policy, budget)
        if reason:
            raise ValueError(f"propagation_startup_rejected:{reason}")


def _run_propagation_sample(
    sample: Sample, wallet: object, *, prompt: PreparedPropagationPrompt,
    node_config: PropagationHarnessConfig, trace_root: Path = PROJECT_ROOT / "traces",
    contract_version: Literal["v1.0", "v1.1"] = "v1.0",
) -> PropagationEvalOutcome:
    """运行一道已预检传播题并独立保留终局、用量和账本回执。"""

    if not isinstance(wallet, ExperimentSession):
        raise ValueError("propagation_requires_persistent_experiment_session")
    context = RunContext(run_id=uuid4().hex, sample_id=sample.sample_id,
                         agent="propagation_analyst", trace_root=trace_root)
    clean = sample.model_copy(update={"gold_veracity": None, "gold_risk": None, "evidence_ids": []})
    scope = EvidenceWorkerScope()
    snapshot = wallet.snapshot()
    identity = {"group_id": snapshot.experiment_id, "sample_id": sample.sample_id,
                "run_id": context.run_id}
    result, usage, usage_error, agent_error, reason, terminal_valid, halt_unconfirmed = \
        _execute_propagation_trace(sample, wallet, context, clean, prompt, node_config, scope, identity,
                                   contract_version=contract_version)
    exact_cost, known_subtotal, ledger_available = _run_cost_totals(wallet, context, usage)
    current = wallet.snapshot()
    unknown = exact_cost is None or current.unknown_requests or current.pending_requests
    if not terminal_valid:
        _halt_after_run_failure(wallet, "propagation_trace_audit_failed")
        current = wallet.snapshot()
    if agent_error and terminal_valid:
        _halt_after_run_failure(wallet, "propagation_runner_failed")
        current = wallet.snapshot()
    if agent_error:
        reason, result = "agent_execution_failed", None
    elif usage_error:
        reason, result = "usage_trace_unreadable", None
    elif usage is not None and (not usage.usage_known or usage.cost_cny is None):
        reason, result = "usage_unknown", None
    elif not terminal_valid:
        result = None
    return PropagationEvalOutcome(result=result if not unknown else None,
        status="success" if result is not None and terminal_valid and not unknown else "failed",
        stop_reason="usage_unknown" if unknown else reason, trace_path=str(context.trace_path),
        run_id=context.run_id, input_tokens=usage.input_tokens if usage else None, output_tokens=usage.output_tokens if usage else None,
        cost_cny=float(exact_cost) if exact_cost is not None else None, exact_cost_cny=exact_cost,
        known_cost_subtotal_cny=float(known_subtotal),
        halt_remaining=current.halted or halt_unconfirmed or not terminal_valid or usage_error or agent_error)


def _run_cost_totals(wallet: ExperimentSession, context: RunContext,
                     usage: RunUsage | None) -> tuple[Decimal | None, Decimal, bool]:
    """先读钱包可信费用；钱包不可用时仅对可信Trace用量做有限回退。"""
    try:
        exact, subtotal = _ledger_run_cost(wallet, context)
        return exact, subtotal, True
    except Exception:
        if usage is None:
            return None, Decimal(0), False
        try:
            exact, subtotal = _trace_run_cost(context)
            return exact, subtotal, False
        except Exception:
            return None, Decimal(0), False


def _execute_propagation_trace(sample: Sample, wallet: ExperimentSession, context: RunContext,
        clean: Sample, prompt: PreparedPropagationPrompt, node_config: PropagationHarnessConfig,
        scope: EvidenceWorkerScope, identity: Mapping[str, str], *,
        contract_version: Literal["v1.0", "v1.1"] = "v1.0"
        ) -> tuple[PropagationResult | None, RunUsage | None, bool, bool, str, bool, bool]:
    """运行生产传播Agent并分别捕获模型执行、用量汇总与终局解析故障。"""

    result: PropagationResult | None = None
    usage: RunUsage | None = None
    usage_error = False
    agent_error = False
    reason, terminal_valid, halt_unconfirmed = "execution_failed", False, False
    try:
        with use_run_context(context):
            try:
                result = propagation.run_propagation_agent(AgentSpec(agent="propagation_analyst",
                    system_prompt=prompt.text), clean, context, node_config=node_config,
                    worker_scope=scope, coordinator_factory=wallet.coordinator_factory(),
                    experiment_identity=identity, contract_version=contract_version)
            except Exception:
                agent_error = True
            try:
                usage = get_run_usage(context.run_id)
            except Exception:
                usage_error = True
            try:
                reason, terminal_valid, halt_unconfirmed = _terminal_status(context)
            except Exception:
                reason, terminal_valid, halt_unconfirmed = "terminal_trace_unreadable", False, True
    finally:
        try:
            scope.close()
        except Exception:
            agent_error = True
    return result, usage, usage_error, agent_error, reason, terminal_valid, halt_unconfirmed


def _ledger_run_cost(wallet: ExperimentSession, context: RunContext) -> tuple[Decimal | None, Decimal]:
    """仅汇总账本中身份匹配本题的request结算额，保留可信已结算小计。"""

    history = wallet.load_history()
    experiment_id = wallet.snapshot().experiment_id
    if history.manifest.get("experiment_id") != experiment_id:
        raise ValueError("propagation_ledger_identity_mismatch")
    candidates = [event for event in history.events
        if event.get("event_type") == "reserved" and event.get("run_id") == context.run_id and
        event.get("sample_id") == context.sample_id]
    if any(event.get("group_id") != experiment_id for event in candidates):
        raise ValueError("propagation_ledger_identity_mismatch")
    reservations = {str(event["request_id"]): event for event in candidates}
    subtotal = Decimal(0)
    complete = True
    for request_id in reservations:
        events = [event for event in history.events if event.get("request_id") == request_id and
                  event.get("event_type") in {"settled", "unknown", "cancelled"}]
        if len(events) != 1:
            complete = False
        elif events[0]["event_type"] == "settled":
            subtotal += Decimal(str(events[0]["actual_cost_cny"]))
        elif events[0]["event_type"] == "unknown":
            complete = False
    return (subtotal if complete else None), subtotal


def _trace_run_cost(context: RunContext) -> tuple[Decimal | None, Decimal]:
    """恢复账本已关闭时的逐请求已知回执，不把浮点总和当作精确值。"""

    events = [TraceEvent.model_validate_json(line) for line in
              context.trace_path.read_text(encoding="utf-8").splitlines() if line]
    if any(event.run_id != context.run_id or event.sample_id != context.sample_id or
           event.agent != context.agent for event in events):
        return None, Decimal(0)
    requests = [event.data.get("request_id") for event in events if event.type == "llm_request"]
    responses: dict[str, list[TraceEvent]] = {}
    for event in events:
        if event.type == "llm_call" and isinstance(event.data.get("request_id"), str):
            responses.setdefault(str(event.data["request_id"]), []).append(event)
    if len(requests) != len(set(requests)) or set(requests) != set(responses):
        return None, Decimal(0)
    subtotal, complete = Decimal(0), True
    for request_id in requests:
        items = responses[str(request_id)]
        if len(items) != 1 or items[0].data.get("usage_known") is not True:
            complete = False
            continue
        try:
            receipt = UsageReceipt.model_validate(items[0].data)
        except ValidationError:
            complete = False
            continue
        if receipt.cost_cny is None:
            complete = False
        else:
            subtotal += Decimal(str(receipt.cost_cny))
    return (subtotal if complete else None), subtotal


def _halt_after_run_failure(wallet: ExperimentSession, reason: str) -> None:
    """runner或审计失败时持久停账；无法确认停账则关闭已有会话。"""

    try:
        wallet.halt(reason)
    except Exception:
        wallet.close()


def _terminal_reason(context: RunContext) -> str:
    """兼容接口：读取已验证的本次传播终局原因。"""

    return _terminal_status(context)[0]


def _terminal_status(context: RunContext) -> tuple[str, bool, bool]:
    """返回经身份验证的原因、终局有效性和停账未确认状态。"""

    if not context.trace_path.is_file():
        return "terminal_trace_missing", False, True
    try:
        events = [TraceEvent.model_validate_json(line) for line in
                  context.trace_path.read_text(encoding="utf-8").splitlines() if line]
    except (UnicodeDecodeError, ValidationError):
        return "terminal_trace_invalid", False, True
    except OSError:
        return "terminal_trace_unreadable", False, True
    if any(event.run_id != context.run_id or event.sample_id != context.sample_id or
           event.agent != context.agent for event in events):
        return "terminal_trace_identity_mismatch", False, True
    terminal = [event for event in events if event.type == "run_outcome"]
    if len(terminal) != 1 or not events or events[-1].type != "run_outcome":
        return "terminal_trace_invalid", False, True
    data = terminal[-1].data
    reason, halt = data.get("reason"), data.get("halt_unconfirmed")
    if not isinstance(reason, str) or not reason or type(halt) is not bool:
        return "terminal_trace_invalid", False, True
    return reason, True, halt


def save_manifest(rows: list[dict[str, str]], path: Path) -> Path:
    """独占写入固定manifest，已有文件一律拒绝覆盖。"""

    if path.exists() or len(rows) != 10:
        raise ValueError("propagation_manifest_exists_or_invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
