"""有界事实调查循环；预算内最多一次只提交收尾，不压缩历史。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from html import escape
import logging
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Mapping, NoReturn

from pydantic import ValidationError

from src.yuqing import llm
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.feedback import format_retry_feedback
from src.yuqing.harness.fact_v2_messages import build_fact_v2_message_history
from src.yuqing.harness.context import EvidenceDelivery
from src.yuqing.harness.fact_v2_contract import FactRunContextV2, build_fact_run_context_v2
from src.yuqing.harness.initial_evidence import (InitialEvidenceReceiptItemV2,
    prepare_initial_evidence, read_initial_evidence_manifest)
from src.yuqing.harness.inherited_evidence import (followup_task_block_from_scope,
    inherited_evidence_block, register_inherited_evidence)
from src.yuqing.harness.ledger import EvidenceLedger
from src.yuqing.harness.policy import decide
from src.yuqing.harness.tools import (EvidenceToolExecutor, EvidenceWorkerScope,
    SubmissionToolExecutor, ToolRegistry, ToolSpec, register_evidence_tools, register_submission_tool,
    register_v2_submission_tool)
from src.yuqing.harness.trace import RunContext, record_event, require_run_context, use_run_context
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.schemas import (AgentResult, AgentResultV2, AgentSpec, FactVerdict,
    FactVerdictV2Envelope, SubmissionReview, TaskInput, ToolCallOutcome, ToolLLMResult,
    EvidenceToolPayload, ToolRequest, ToolExecutionResult, FollowupScopeV2, InheritedEvidencePlanV2)
from src.yuqing.settings import HarnessConfig, PolicyConfig, get_cost_config, get_harness_config, get_policy_config

if TYPE_CHECKING:
    from src.yuqing.eval.dual_budget import DualBudgetCoordinator

logger = logging.getLogger(__name__)
InputBound = Callable[[list[dict[str, object]], list[ToolSpec]], int]
FORMAT_RECOVERABLE = {"text_without_tools", "invalid_tool_arguments", "mixed_tool_batch"}


class _LoopOutcome:
    """保存本题出口与显式预算依赖，不从历史事件推测最终结果。"""

    def __init__(self) -> None:
        """初始化尚未产生可信结果的运行状态。"""
        self.data: dict[str, object] = {"reason": "no_valid_result"}
        self.step = 0
        self.coordinator: DualBudgetCoordinator | None = None
        self.experiment_identity: Mapping[str, str] | None = None
        self.shared_failure = False

    def stop(self, data: dict[str, object]) -> None:
        """先保存内存出口，再记录中间停止事件；审计异常继续向外传递。"""
        self.data = deepcopy(data)
        self.step = require_run_context().step
        try:
            record_event("stop", data)
        except Exception:
            self.shared_failure = True
            raise


def _record_run_outcome(ctx: RunContext, result: AgentResult | AgentResultV2 | None, state: _LoopOutcome,
                        budget: BudgetTracker | None) -> None:
    """运行出口独立记录一次终局，保留阶段原因但不把中间受阻当作失败。"""
    data = deepcopy(state.data)
    ordinary_steps = budget.steps if budget is not None else 0
    final_used = budget.final_used if budget is not None else False
    data.update(ordinary_steps=ordinary_steps, final_used=final_used,
        steps=ordinary_steps + int(final_used), step_count_version="budget-reservations-v1")
    data["outcome_class"] = "submitted_judgment" if result is not None else "no_valid_result"
    if result is not None:
        data["reason"] = result.stop_reason
    with use_run_context(ctx.model_copy(update={"step": state.step})):
        record_event("run_outcome", data)


class LoopStop(RuntimeError):
    """由程序固定原因触发的可解释停止，不保存模型或异常原文。"""

    def __init__(self, reason: Literal["run_identity_mismatch", "untrusted_response",
                 "duplicate_tool_call_id", "invalid_tool_batch", "submission_required",
                 "worker_busy", "worker_scope_closed", "tool_timeout_pending", "tool_error_limit",
                 "submission_unavailable", "incomplete_dual_budget_arguments",
                 "experiment_identity_mismatch", "coordinator_budget_mismatch",
                 "invalid_node_config", "node_prices_unknown", "node_capacity_insufficient",
                 "initial_evidence_capacity_exceeded", "inherited_evidence_capacity_exceeded",
                 "followup_budget_mismatch"]) -> None:
        """保存受控原因，供停止审计读取。"""
        self.reason = reason
        super().__init__(reason)


class FinalRequested(RuntimeError):
    """内部常规阶段出口，转入同一账本的一次收尾。"""

    def __init__(self, reason: str) -> None:
        """保存程序指定的收尾触发原因。"""
        self.reason = reason
        super().__init__(reason)


class LoopProgress:
    """每题独立记录实际工具尝试、无新增正文与错误，不替代预算。"""

    def __init__(self, budget: BudgetTracker) -> None:
        """使用配置确定的阈值，提醒状态与计数不跨运行共享。"""
        self.budget = budget
        self.config = budget.config
        self.format_nudges = 0
        self.queries: Counter[str] = Counter()
        self.no_new = 0
        self.errors = 0
        self.warned: set[tuple[str, str]] = set()
        self.pending_warnings: set[tuple[str, str]] = set()
        self.warnings: list[dict[str, object]] = []


    def confirm_warnings(self) -> None:
        """下一普通请求返回可信响应后确认提醒交付，不证明模型理解。"""
        for kind, _ in sorted(self.pending_warnings):
            record_event("loop_warning_delivered", {"kind": kind})
        self.warned.update(self.pending_warnings)
        self.pending_warnings.clear()

    def should_skip(self, request: ToolRequest) -> bool:
        """提醒尚未交付时暂停同批重复搜索，无新增提醒暂停所有剩余搜索。"""
        if request.name != "search_evidence":
            return False
        query = " ".join(request.arguments["query"].split())
        return (("repeat_query", query) in self.pending_warnings
                or ("no_new_evidence", "") in self.pending_warnings)

    def observe(self, request: ToolRequest, result: ToolExecutionResult, evidence: EvidenceToolExecutor,
                previous_attempts: int, previous_bodies: int) -> str | None:
        """只按实际执行更新计数；返回收尾原因，致命错误直接停止。"""
        payload = result.payload
        if payload.reason in {"tool_busy", "worker_scope_closed"}:
            raise LoopStop("worker_busy" if payload.reason == "tool_busy" else "worker_scope_closed")
        if evidence.budget.tool_calls[request.name] == previous_attempts:
            return None
        searching = request.name == "search_evidence"
        query = " ".join(request.arguments["query"].split()) if searching else ""
        if searching:
            self.queries[query] += 1
        if payload.status in {"error", "timeout"}:
            self.errors += 1
        elif payload.status == "success":
            self.errors = 0
            if searching:
                self.no_new = self.no_new + 1 if evidence.ledger.evidence_count() == previous_bodies else 0
        record_event("loop_progress", {"call_id": request.call_id, "tool": request.name,
            "repeat_count": self.queries[query] if searching else None,
            "no_new_evidence": self.no_new, "consecutive_tool_errors": self.errors})
        if payload.status == "timeout" and evidence.worker_scope.has_pending_work():
            raise LoopStop("tool_timeout_pending")
        if self.errors >= self.config.max_consecutive_tool_errors:
            raise LoopStop("tool_error_limit")
        triggers = {}
        if searching and self.queries[query] >= self.config.repeat_query_limit:
            triggers["repeat_query"] = self.queries[query]
        if searching and payload.status == "success" and self.no_new >= self.config.no_new_evidence_limit:
            triggers["no_new_evidence"] = self.no_new
        for kind in triggers:
            if (kind, query if kind == "repeat_query" else "") in self.warned:
                return kind
        for kind, count in triggers.items():
            self.pending_warnings.add((kind, query if kind == "repeat_query" else ""))
            self.warnings.append({"role": "user", "content": f"程序提醒：{kind}已达到阈值，请改变调查方式；再次触发将只允许最终提交。此提醒不是证据。"})
            record_event("loop_warning", {"kind": kind, "count": count})
        return None

    def drain_warnings(self) -> list[dict[str, object]]:
        """只在工具申请与回执配齐后取出程序提醒。"""
        warnings, self.warnings = self.warnings, []
        return warnings


@dataclass
class _LoopRuntime:
    """保存一次调查循环的工具、可信交付与消息状态。"""

    evidence: EvidenceToolExecutor
    delivery: EvidenceDelivery
    submission: SubmissionToolExecutor
    tools: list[ToolSpec]
    messages: list[dict[str, object]]



def _try_format_nudge(outcome: ToolCallOutcome, messages: list[dict[str, object]], progress: LoopProgress,
                      tools: list[ToolSpec | None]) -> bool:
    """只为获准格式错误排队一次程序提醒，坏响应不进入消息历史。"""
    status = progress.budget.status()
    if (outcome.status != "invalid_response" or outcome.reason not in FORMAT_RECOVERABLE
            or status.halted or status.pending or progress.budget.final_used
            or progress.format_nudges >= progress.config.max_format_nudges):
        return False
    try:
        record_event("format_nudge", {"reason": outcome.reason, "request_id": outcome.request_id,
            "count": progress.format_nudges + 1, "state": "queued"})
    except Exception:
        progress.budget.halt("format_nudge_audit_failed")
        raise
    progress.format_nudges += 1
    schemas = {tool.name: tool.args_model for tool in tools if tool is not None}
    content = format_retry_feedback(outcome.reason, outcome.validation_issues, schemas,
                                    progress.budget.config.max_tool_result_chars)
    messages.append({"role": "user", "content": content})
    return True


def _handle_failed_call(outcome: ToolCallOutcome, messages: list[dict[str, object]],
                        progress: LoopProgress, state: _LoopOutcome, tools: list[ToolSpec | None]) -> bool:
    """决定停止、排队提醒或转原收尾；返回True仅表示可进入下一普通轮。"""
    budget, status = progress.budget, progress.budget.status()
    if not status.halted and not status.pending and not budget.final_used:
        if outcome.status == "budget_denied" and outcome.reason in {"step_limit", "token_limit", "cost_limit", "per_call_limit"}:
            raise FinalRequested(outcome.reason)
        recoverable = outcome.status == "invalid_response" and outcome.reason in FORMAT_RECOVERABLE
        if recoverable and progress.format_nudges < progress.config.max_format_nudges:
            if budget.steps >= budget.config.max_steps:
                raise FinalRequested("step_limit")
            if _try_format_nudge(outcome, messages, progress, tools):
                return True
    data: dict[str, object] = {"reason": status.halt_reason or outcome.reason, "status": outcome.status}
    if (not status.halted and outcome.status == "invalid_response" and outcome.reason in FORMAT_RECOVERABLE
            and progress.config.max_format_nudges > 0 and progress.format_nudges >= progress.config.max_format_nudges):
        data.update(reason="format_nudge_limit", raw_reason=outcome.reason, format_nudges=progress.format_nudges)
    state.stop(data)
    return False


def _refuse_tool(request: ToolRequest, messages: list[dict[str, object]], reason: str) -> None:
    """补一张拒绝回执并审计，未执行的申请不扣工具次数。"""
    messages.append({"role": "tool", "tool_call_id": request.call_id,
        "content": EvidenceToolPayload(status="denied", reason=reason).model_dump_json()})
    record_event("tool_skipped", {"call_id": request.call_id, "tool": request.name,
        "reason": reason, "executed": False})


def _skip_pending(response: ToolLLMResult, messages: list[dict[str, object]]) -> None:
    """给尚未执行的申请补拒绝回执，不执行或计次，保持消息配对。"""
    done = {message["tool_call_id"] for message in messages if message["role"] == "tool"}
    for request in response.tool_calls:
        if request.call_id not in done:
            _refuse_tool(request, messages, "final_requested")


def _failure(exc: Exception, budget: BudgetTracker | None, state: _LoopOutcome | None = None) -> None:
    """停止预算并尽量记录明确原因；未知异常只记类型。"""
    reason = exc.reason if isinstance(exc, LoopStop) else "error"
    if state is not None and not isinstance(exc, LoopStop):
        state.shared_failure = True
    if budget is not None:
        reason = budget.status().halt_reason or reason
        budget.halt(reason)
    logger.error("调查循环失败 type=%s", type(exc).__name__)
    try:
        (state or _LoopOutcome()).stop({"reason": reason, "error_type": type(exc).__name__})
    except Exception as audit_exc:
        logger.error("循环停止审计失败 type=%s", type(audit_exc).__name__)


def _check_dispatch_batch(response: ToolLLMResult, messages: list[dict[str, object]]) -> None:
    """全部编号与提交批次在执行前复检，冲突时不执行任何工具。"""
    previous = {call["id"] for message in messages for call in message.get("tool_calls", [])}
    ids = [request.call_id for request in response.tool_calls]
    if len(ids) != len(set(ids)) or previous.intersection(ids):
        raise LoopStop("duplicate_tool_call_id")
    if not ids:
        raise LoopStop("submission_required")
    if any(request.name == "submit_verdict" for request in response.tool_calls) and len(ids) != 1:
        raise LoopStop("invalid_tool_batch")


def _run_evidence(request: ToolRequest, messages: list[dict[str, object]], evidence: EvidenceToolExecutor,
                  delivery: EvidenceDelivery, agent: str, progress: LoopProgress | None) -> str | None:
    """顺序执行取证或补提醒拒绝回执，返回需要转收尾的原因。"""
    if progress and progress.should_skip(request):
        _refuse_tool(request, messages, "warning_pending")
        return None
    before = evidence.budget.tool_calls[request.name], evidence.ledger.evidence_count()
    tool = evidence.execute(request, agent=agent)
    messages.append({"role": "tool", "tool_call_id": request.call_id,
                     "content": tool.payload.model_dump_json()})
    if tool.name == "read_evidence" and tool.payload.status == "success":
        delivery.stage_read(tool, evidence.ledger)
    trigger = progress.observe(request, tool, evidence, *before) if progress else None
    return trigger or ("tool_limit" if tool.payload.status == "denied" and tool.payload.reason == "tool_limit" else None)


def _dispatch(response: ToolLLMResult, messages: list[dict[str, object]],
              evidence: EvidenceToolExecutor, submission: SubmissionToolExecutor,
              delivery: EvidenceDelivery, agent: str, progress: LoopProgress | None = None
              ) -> FactVerdict | SubmissionReview | None:
    """只分派当前获许可响应，提交独占一批，正文由可信交付记录核对。"""
    authorized = evidence.budget.authorize_tool_response(response)
    if not authorized.allowed:
        raise LoopStop("untrusted_response")
    _check_dispatch_batch(response, messages)
    if progress:
        progress.confirm_warnings()
    messages.append(deepcopy(response.assistant_message))
    for request in response.tool_calls:
        if request.name == "submit_verdict":
            result = submission.execute(response, agent=agent)
            messages.append({"role": "tool", "tool_call_id": request.call_id,
                             "content": result.feedback.model_dump_json()})
            if result.feedback.accepted and result.review is not None:
                if submission.fact_contract is not None:
                    return result.review
                return FactVerdict.model_validate_json(result.review.verdict_json)
            if result.review is None or result.reason != "validation_failed":
                raise ValueError(result.reason)
        else:
            trigger = _run_evidence(request, messages, evidence, delivery, agent, progress)
            if trigger:
                _skip_pending(response, messages)
                if progress:
                    messages.extend(progress.drain_warnings())
                raise FinalRequested(trigger)
    if progress:
        messages.extend(progress.drain_warnings())
    return None


def _completed(verdict: FactVerdict, ctx: RunContext, evidence: EvidenceToolExecutor,
               reason: Literal["submitted", "forced_final"], state: _LoopOutcome) -> AgentResult:
    """仅把正式提交通过的判断、引用和原账本消费组成成功结果。"""
    budget, ledger = evidence.budget, evidence.ledger
    ids = dict.fromkeys(eid for item in verdict.key_facts + verdict.reasons for eid in item.evidence_ids)
    state.stop({"reason": reason, "phase": "final" if budget.final_used else "ordinary"})
    return AgentResult(run_id=ctx.run_id, verdict=verdict, evidence=[ledger.get(eid) for eid in ids],
        stop_reason=reason, steps=budget.steps + int(budget.final_used), input_output_tokens=budget.tokens,
        cost_cny=float(budget.cost_cny), trace_path=str(ctx.trace_path))


def _completed_v2(review: SubmissionReview, ctx: RunContext, evidence: EvidenceToolExecutor,
                  budget: BudgetTracker, fact_contract: FactRunContextV2,
                  state: _LoopOutcome,
                  reason: Literal["submitted", "forced_final"] = "submitted") -> AgentResultV2:
    """只发布当前正式V2提交的完整报告、程序核验和实际引用证据。"""
    if review.v2_validation is None or review.v2_report_json is None or not review.accepted:
        raise ValueError("fact_v2_review_incomplete")
    envelope = FactVerdictV2Envelope.model_validate_json(review.v2_report_json, strict=True)
    ids = dict.fromkeys(evidence_id for item in envelope.key_facts + envelope.reasons
                        for evidence_id in item.get("evidence_ids", []))
    state.stop({"reason": reason, "phase": "final" if budget.final_used else "ordinary",
                "schema_sha256": fact_contract.schema_sha256})
    return AgentResultV2(run_id=ctx.run_id, report_json=review.v2_report_json,
        validation=review.v2_validation, evidence=[evidence.ledger.get(eid) for eid in ids],
        stop_reason=reason, steps=budget.steps + int(budget.final_used),
        input_output_tokens=budget.tokens, cost_cny=float(budget.cost_cny),
        trace_path=str(ctx.trace_path))


def _fact_v2_fingerprints(system_prompt: str, fact_contract: FactRunContextV2,
                          definitions: list[dict[str, object]]) -> dict[str, object]:
    """记录完整系统消息、模式Schema和实际工具结构的SHA指纹。"""
    tool_payload = json.dumps(definitions, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
    limits_payload = json.dumps(fact_contract.limits.model_dump(mode="json"),
                                ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"mode": fact_contract.mode,
        "contract_sha256": fact_contract.contract_sha256,
        "system_sha256": sha256(system_prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": fact_contract.schema_sha256,
        "tool_schema_sha256": sha256(tool_payload.encode("utf-8")).hexdigest(),
        "limits_sha256": sha256(limits_payload.encode("utf-8")).hexdigest()}


def _validate_fact_run_context(fact_contract: FactRunContextV2) -> None:
    """重建已批准配置对应的Schema，拒绝被调用方替换的模式或限制。"""
    expected = build_fact_run_context_v2(fact_contract.mode,
        assigned_issue_ids=fact_contract.assigned_issue_ids,
        initial_manifest_path=fact_contract.initial_manifest_path,
        scope=fact_contract.scope, inherited_plan=fact_contract.inherited_plan,
        action_claim=fact_contract.action_claim)
    if (fact_contract.schema_sha256 != expected.schema_sha256
            or fact_contract.limits.model_dump() != expected.limits.model_dump()
            or fact_contract.plan_max_text_chars != expected.plan_max_text_chars):
        raise ValueError("fact_v2_contract_mismatch")


def _preflight_inherited_capacity(runtime: _LoopRuntime, input_bound: InputBound | None,
                                  budget: BudgetTracker, state: _LoopOutcome) -> bool:
    """Refuse an oversized full inherited set before the first model request."""
    messages = runtime.delivery.assemble(runtime.messages)
    available = runtime.delivery.inspect(messages)
    messages.append(_runtime_status_message(budget, runtime.tools, available))
    messages = runtime.delivery.assemble(messages)
    definitions = runtime.evidence.registry.model_tools(require_run_context().agent, runtime.submission.policy)
    estimate = (llm.estimate_request_input(messages, definitions, budget.config.input_bound_margin_tokens)
                if input_bound is None else input_bound(deepcopy(messages), list(runtime.tools)))
    decision = budget.check_model_call(estimate, budget.config.max_output_tokens)
    if decision.allowed:
        record_event("followup_capacity_check", {"status": "fits", "input_token_estimate": estimate})
        return True
    record_event("followup_capacity_check", {"status": "rejected", "reason": decision.reason,
        "input_token_estimate": estimate})
    state.stop({"reason": "inherited_evidence_capacity_exceeded", "capacity_reason": decision.reason})
    return False


def _submission_tools(registry: ToolRegistry, agent: str, policy: PolicyConfig) -> list[ToolSpec]:
    """启动及收尾都复用展示权限，缺少完成任务的提交能力时在付费前拒绝。"""
    visible = {item["function"]["name"] for item in registry.model_tools(agent, policy)}
    tool = registry.get("submit_verdict")
    if tool is None or tool.name not in visible:
        decision = decide(registry, agent, "submit_verdict", caller="model", config=policy)
        record_event("loop_policy_denied", {"tool": "submit_verdict", "reason": decision.reason})
        raise LoopStop("submission_unavailable")
    return [tool]


def _runtime_status_message(budget: BudgetTracker, tools: list[ToolSpec],
                            evidence_ids: frozenset[str], *, final: bool = False) -> dict[str, str]:
    """向本次模型消息报告真实剩余额度和已附带的已读编号。"""
    quotas = []
    for tool in tools:
        limit = budget.config.max_tool_calls.get(tool.name)
        if limit is not None:
            used = budget.tool_calls[tool.name]
            quotas.append(f"{tool.name}剩余{limit - used}/{limit}")
    text = "程序状态（不是证据）：" + ("；".join(quotas) or "当前没有调查工具额度")
    readable = "、".join(sorted(evidence_ids)) or "无"
    if final:
        text += f"\n常规阶段已结束，只能提交。本次输入已附带且成功阅读的证据编号：{readable}。"
        text += "不得引用其他搜索预览或猜测正文；没有合格证据时按规则说明不足。"
    else:
        text += f"\n本次请求已附带且成功阅读的证据编号：{readable}。搜索预览不属于可引用正文。"
    return {"role": "user", "content": text}


def _initial_message_history(spec: AgentSpec, fact_contract: FactRunContextV2 | None,
                             initial_message: str, program_messages: tuple[str, ...]
                             ) -> list[dict[str, object]]:
    """Build the initial history without changing the legacy V1 message path."""
    if fact_contract is None:
        messages: list[dict[str, object]] = [{"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": initial_message}]
        messages.extend({"role": "user", "content": item} for item in program_messages)
        return messages
    return build_fact_v2_message_history(spec.system_prompt, initial_message, program_messages)


def _finalize(reason: str, ctx: RunContext, agent: str, messages: list[dict[str, object]],
              evidence: EvidenceToolExecutor, submission: SubmissionToolExecutor,
              delivery: EvidenceDelivery, input_bound: InputBound | None,
              state: _LoopOutcome | None = None) -> AgentResult | AgentResultV2 | None:
    """同一账本只尝试一次最终提交，失败不重试，不删正文来挤预算。"""
    budget, state = evidence.budget, state or _LoopOutcome()
    with use_run_context(ctx.model_copy(update={"step": budget.steps})):
        try:
            record_event("finalization", {"trigger": reason, "phase": "final"})
            if budget.status().halted or budget.status().pending or budget.final_used:
                state.stop({"reason": "final_unavailable", "trigger": reason, "phase": "final"})
                return None
            tools = _submission_tools(evidence.registry, agent, submission.policy)
            current = delivery.assemble(messages)
            available = delivery.inspect(current)
            messages.append(_runtime_status_message(budget, tools, available, final=True))
            current = delivery.assemble(messages)
            bound = input_bound(deepcopy(current), list(tools)) if input_bound is not None else None
            outcome = llm.call_with_tools_outcome(current, tools, budget,
                input_token_upper_bound=bound, final=True, delivery=delivery,
                **_gateway_kwargs(state.coordinator, state.experiment_identity))
            if outcome.result is None:
                # 网关输出预占固定等于配置上限，因此此处per_call_limit只能是输入超限。
                input_large = outcome.status == "budget_denied" and outcome.reason == "per_call_limit"
                state.stop({"reason": "final_input_too_large" if input_large else outcome.reason,
                    "raw_reason": outcome.reason, "status": outcome.status, "trigger": reason, "phase": "final"})
                return None
            verdict = _dispatch(outcome.result, messages, evidence, submission, delivery, agent)
            if verdict is None:
                state.stop({"reason": "final_submission_rejected", "trigger": reason, "phase": "final"})
                return None
            if isinstance(verdict, SubmissionReview):
                if submission.fact_contract is None:
                    raise ValueError("fact_v2_contract_missing")
                return _completed_v2(verdict, ctx, evidence, budget, submission.fact_contract,
                                     state, "forced_final")
            return _completed(verdict, ctx, evidence, "forced_final", state)
        except Exception as exc:
            _failure(exc, budget, state)
            return None


def _iterate(spec: AgentSpec, task: TaskInput, ctx: RunContext, index: BM25Index,
             input_bound: InputBound | None, worker_scope: EvidenceWorkerScope, budget: BudgetTracker,
             store_path: Path, state: _LoopOutcome,
             fact_contract: FactRunContextV2 | None = None) -> AgentResult | AgentResultV2 | None:
    """常规调查有界推进，只有可信预算边界才进入一次收尾。"""
    runtime = _prepare_loop_runtime(spec, task, ctx, index, budget, store_path,
                                    worker_scope, fact_contract, state)
    if fact_contract is not None and isinstance(fact_contract.scope, FollowupScopeV2):
        if not _preflight_inherited_capacity(runtime, input_bound, budget, state):
            return None
    progress = LoopProgress(budget)
    for step in range(budget.config.max_steps):
        with use_run_context(ctx.model_copy(update={"step": step})):
            try:
                current = runtime.delivery.assemble(runtime.messages)
                available = runtime.delivery.inspect(current)
                runtime.messages.append(_runtime_status_message(budget, runtime.tools, available))
                current = runtime.delivery.assemble(runtime.messages)
                bound = input_bound(deepcopy(current), list(runtime.tools)) if input_bound is not None else None
                outcome = llm.call_with_tools_outcome(current, runtime.tools, budget,
                    input_token_upper_bound=bound, delivery=runtime.delivery,
                    **_gateway_kwargs(state.coordinator, state.experiment_identity))
                if outcome.result is None:
                    if _handle_failed_call(outcome, runtime.messages, progress, state, runtime.tools):
                        continue
                    return None
                verdict = _dispatch(outcome.result, runtime.messages, runtime.evidence,
                    runtime.submission, runtime.delivery, spec.agent, progress)
                if verdict is not None:
                    return _complete_verdict(verdict, ctx, runtime, budget, fact_contract, state)
                if step + 1 == budget.config.max_steps:
                    return _finalize("step_limit", ctx, spec.agent, runtime.messages, runtime.evidence,
                        runtime.submission, runtime.delivery, input_bound, state)
            except FinalRequested as final:
                return _finalize(final.reason, ctx, spec.agent, runtime.messages, runtime.evidence,
                    runtime.submission, runtime.delivery, input_bound, state)
            except Exception as exc:
                _failure(exc, budget, state)
                return None
    return None


def _prepare_loop_runtime(spec: AgentSpec, task: TaskInput, ctx: RunContext, index: BM25Index,
                          budget: BudgetTracker, store_path: Path, worker_scope: EvidenceWorkerScope,
                          fact_contract: FactRunContextV2 | None, state: _LoopOutcome) -> _LoopRuntime:
    """一次性准备工具、首轮证据正文和网关消息。"""
    registry, ledger = ToolRegistry(), EvidenceLedger()
    register_evidence_tools(registry, config=budget.config)
    initial_message = f"<claim>{escape(task.claim, quote=False)}</claim>"
    initial_ids: frozenset[str] = frozenset()
    initial_expected: tuple[InitialEvidenceReceiptItemV2, ...] = ()
    program_messages: tuple[str, ...] = ()
    inherited_plan = None
    manifest = None
    if fact_contract is not None:
        if fact_contract.mode == "initial":
            if fact_contract.initial_manifest_path is None:
                raise ValueError("fact_v2_initial_manifest_missing")
            manifest, manifest_sha = read_initial_evidence_manifest(
                fact_contract.initial_manifest_path, task.sample_id, task.claim)
            record_event("fact_v2_initial_manifest", {"status": manifest.retrieval_status,
                "retrieval_error_type": manifest.retrieval_error_type, "candidate_count": len(manifest.entries),
                "manifest_sha256": manifest_sha})
            if manifest.retrieval_status == "request_capacity_exceeded":
                raise LoopStop("initial_evidence_capacity_exceeded")
        elif isinstance(fact_contract.scope, FollowupScopeV2):
            inherited_plan, program_messages = _prepare_followup_messages(fact_contract, task, ledger)
        register_v2_submission_tool(registry, fact_contract.schema, fact_contract.response_schema)
    else:
        register_submission_tool(registry)
    policy = get_policy_config()
    _submission_tools(registry, spec.agent, policy)
    if fact_contract is not None and manifest is not None:
        block, initial_ids, initial_expected = prepare_initial_evidence(manifest, ledger)
        initial_message = f"<claim>{escape(task.claim, quote=False)}</claim>\n{block}"
    evidence = EvidenceToolExecutor(registry, ledger, budget, policy, index, store_path, worker_scope=worker_scope)
    delivery = EvidenceDelivery(initial_message=initial_message if manifest is not None else None,
        initial_evidence_ids=initial_ids, initial_expected=initial_expected,
        program_messages=program_messages, inherited_plan=inherited_plan)
    submission = SubmissionToolExecutor(registry, ledger, budget, policy,
        fact_contract=fact_contract, delivery=delivery)
    messages = _initial_message_history(spec, fact_contract, initial_message, program_messages)
    # 只展示本卡已接入的三个工具，不展示配置中尚未实现的load_skill。
    definitions = registry.model_tools(spec.agent, policy)
    tools = [registry.get(item["function"]["name"]) for item in definitions]
    if fact_contract is not None:
        record_event("fact_v2_request_contract", _fact_v2_fingerprints(
            spec.system_prompt, fact_contract, definitions))
    return _LoopRuntime(evidence, delivery, submission, tools, messages)


def _prepare_followup_messages(contract: FactRunContextV2, task: TaskInput,
                               ledger: EvidenceLedger) -> tuple[InheritedEvidencePlanV2, tuple[str, ...]]:
    """校验继承计划、重登记台账并构造补查数据消息。"""
    plan = contract.inherited_plan
    if plan is None or not isinstance(contract.scope, FollowupScopeV2):
        raise ValueError("fact_v2_followup_inheritance_missing")
    register_inherited_evidence(plan, ledger)
    messages = (followup_task_block_from_scope(task, contract.scope), inherited_evidence_block(plan))
    from src.yuqing.harness.inherited_evidence import followup_leads_block
    leads = followup_leads_block(contract.action_claim)
    if leads is not None:
        messages += (leads,)
    record_event("fact_v2_inherited_plan", {
        "source_run_id": plan.source_scope.run_id,
        "source_fact_round_no": plan.source_scope.fact_round_no,
        "target_action_id": contract.scope.action_id,
        "target_action_round": contract.scope.action_round,
        "origin": "inherited",
        "inherited_count": len(plan.entries),
        "plan_sha256": sha256(plan.model_dump_json().encode("utf-8")).hexdigest()})
    return plan, messages


def _complete_verdict(verdict: FactVerdict | SubmissionReview, ctx: RunContext,
                      runtime: _LoopRuntime, budget: BudgetTracker,
                      fact_contract: FactRunContextV2 | None,
                      state: _LoopOutcome) -> AgentResult | AgentResultV2:
    """按显式合同选择 V2 或既有结果转换。"""
    if isinstance(verdict, SubmissionReview):
        if fact_contract is None:
            raise ValueError("fact_v2_contract_missing")
        return _completed_v2(verdict, ctx, runtime.evidence, budget, fact_contract, state)
    return _completed(verdict, ctx, runtime.evidence, "submitted", state)


def _gateway_kwargs(coordinator: DualBudgetCoordinator | None,
                    identity: Mapping[str, str] | None) -> dict[str, object]:
    """旧路径不追加参数，新路径始终把同一个协调器与身份传入网关。"""
    return {} if coordinator is None else {"coordinator": coordinator, "experiment_identity": identity}


def _bind_experiment(state: _LoopOutcome, budget: BudgetTracker, ctx: RunContext,
                     factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None,
                     identity: Mapping[str, str] | None) -> None:
    """启动核对成对依赖与运行身份，仅创建一次绑定当前单题的协调器。"""
    if (factory is None) != (identity is None):
        raise LoopStop("incomplete_dual_budget_arguments")
    if factory is None:
        return
    tags = dict(identity)
    if tags.get("sample_id") != ctx.sample_id or tags.get("run_id") != ctx.run_id or not tags.get("group_id"):
        raise LoopStop("experiment_identity_mismatch")
    coordinator = factory(budget)
    if coordinator is None or coordinator.task is not budget:
        raise LoopStop("coordinator_budget_mismatch")
    state.coordinator, state.experiment_identity = coordinator, tags


def _stop_shared(state: _LoopOutcome, budget: BudgetTracker | None, reason: str | None = None) -> None:
    """审计或意外异常停用整场；正常本题出口不虚构未知消费。"""
    if state.coordinator is None or not (state.shared_failure or reason is not None):
        return
    reason = reason or (budget.status().halt_reason if budget is not None else None) or "loop_audit_failed"
    try:
        if not state.coordinator.experiment.snapshot().halted:
            state.coordinator.halt(reason)
    except Exception as exc:
        logger.error("循环整场停止未确认 type=%s", type(exc).__name__)


def _reject_node_config(ctx: RunContext, reason: Literal["invalid_node_config",
                        "node_prices_unknown", "node_capacity_insufficient"], detail: str) -> NoReturn:
    """只记录可信节点名与受控拒绝原因，在绑定共享钱包或调用模型前停止。"""
    record_event("node_config_rejected", {"node": ctx.agent, "reason": reason, "detail": detail})
    raise LoopStop(reason)


def _node_budget(config: HarnessConfig, ctx: RunContext) -> BudgetTracker:
    """复制并复检显式节点配置，复用预算检查保证普通请求及一次收尾容量。"""
    try:
        validated = HarnessConfig.model_validate(vars(config))
    except ValidationError:
        _reject_node_config(ctx, "invalid_node_config", "schema_validation_failed")
    prices = get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        _reject_node_config(ctx, "node_prices_unknown", "prices_unknown")
    try:
        budget = BudgetTracker(validated, prices)
    except ValueError:
        _reject_node_config(ctx, "node_capacity_insufficient", "cost_limit")
    capacity = budget.check_model_call(validated.token_budget_per_call, validated.max_output_tokens)
    if not capacity.allowed:
        _reject_node_config(ctx, "node_capacity_insufficient", capacity.reason)
    return budget


def _run_budget(config: HarnessConfig | None, ctx: RunContext,
                contract: FactRunContextV2 | None) -> BudgetTracker:
    """Keep old defaults intact and cap scoped follow-up spending by the authorized brief."""
    scoped = contract is not None and isinstance(contract.scope, FollowupScopeV2)
    budget = (_node_budget(config or get_harness_config(), ctx) if config is not None or scoped
              else BudgetTracker(get_harness_config(), get_cost_config()))
    if scoped and Decimal(str(budget.config.max_cost_cny)) > contract.scope.budget_cny:
        record_event("followup_budget_rejected", {"node_cap_cny": str(budget.config.max_cost_cny),
            "authorized_cny": str(contract.scope.budget_cny)})
        raise LoopStop("followup_budget_mismatch")
    return budget


def _check_fact_context(contract: FactRunContextV2 | None, task: TaskInput, ctx: RunContext) -> None:
    """Validate the complete V2 contract before creating a budget or binding a shared wallet."""
    if contract is None:
        return
    _validate_fact_run_context(contract)
    scope = contract.scope
    if scope is not None:
        if (scope.run_id != ctx.run_id or scope.sample_id != task.sample_id or scope.agent != ctx.agent
                or scope.claim_sha256 != sha256(task.claim.encode("utf-8")).hexdigest()):
            raise LoopStop("run_identity_mismatch")
        record_event("fact_v2_scope", {**scope.model_dump(mode="json"),
            "schema_sha256": contract.schema_sha256, "contract_sha256": contract.contract_sha256})


def run_agent(spec: AgentSpec, task: TaskInput, ctx: RunContext, *, index: BM25Index,
              input_bound: InputBound | None = None, worker_scope: EvidenceWorkerScope, store_path: Path,
              coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None = None,
              experiment_identity: Mapping[str, str] | None = None,
              node_config: HarnessConfig | None = None,
              fact_contract: FactRunContextV2 | None = None) -> AgentResult | AgentResultV2 | None:
    """运行最小调查循环；依赖显式注入，失败返回None并尽量记录安全停止事件。"""
    budget: BudgetTracker | None = None
    state, result = _LoopOutcome(), None
    try:
        with use_run_context(ctx):
            try:
                if ctx.agent != spec.agent or ctx.sample_id != task.sample_id:
                    raise LoopStop("run_identity_mismatch")
                _check_fact_context(fact_contract, task, ctx)
                availability = worker_scope.availability()
                if availability != "ready":
                    raise LoopStop("worker_busy" if availability == "busy" else "worker_scope_closed")
                budget = _run_budget(node_config, ctx, fact_contract)
                _bind_experiment(state, budget, ctx, coordinator_factory, experiment_identity)
                result = _iterate(spec, task, ctx, index, input_bound, worker_scope, budget,
                                  store_path, state, fact_contract)
            except Exception as exc:
                _failure(exc, budget, state)
            _stop_shared(state, budget)
            try:
                _record_run_outcome(ctx, result, state, budget)
            except Exception as audit_exc:
                if budget is not None:
                    budget.halt("run_outcome_audit_failed")
                _stop_shared(state, budget, "run_outcome_audit_failed")
                logger.error("运行终局审计失败 type=%s", type(audit_exc).__name__)
                return None
            return result
    except Exception as exc:
        logger.error("循环上下文建立或恢复失败 type=%s", type(exc).__name__)
        return None
