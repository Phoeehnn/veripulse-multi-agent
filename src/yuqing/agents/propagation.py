"""传播专家的确定性报告验收逻辑。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from html import escape
import logging
from typing import Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask, PublicSignalReport, PublicSignalResult
from src.yuqing import llm
from src.yuqing.eval.dual_budget import DualBudgetCoordinator
from src.yuqing.features.propagation import CommentItem, SpreadStats
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.feedback import format_retry_feedback, propagation_submission_feedback
from src.yuqing.harness.policy import decide
from src.yuqing.harness.tools import EvidenceWorkerScope, ToolRegistry, ToolSpec, WorkerUnavailable
from src.yuqing.harness.trace import RunContext, record_event, require_run_context, use_run_context, write_archive
from src.yuqing.agents.propagation_v11 import PropagationStatsReceipt
from src.yuqing.schemas import (
    Lead,
    AgentSpec,
    PropagationRaw,
    PropagationReport,
    PropagationSubmissionFeedback,
    PropagationReviewTask,
    PropagationReviewRun,
    ReviewResult,
    Sample,
    ToolCallOutcome,
    ToolLLMResult,
    ToolRequest,
    TaskInput,
)
from src.yuqing.settings import (
    PolicyConfig,
    CostConfig,
    PropagationHarnessConfig,
    PropagationReviewHarnessConfig,
    PropagationToolsConfig,
    get_cost_config,
    get_m6c_case_budget_config,
    get_policy_config,
    get_propagation_limits_config,
    get_propagation_tools_config,
)

logger = logging.getLogger(__name__)


class PropagationToolPayload(BaseModel):
    """已序列化并完整受长度上限约束的传播工具回执。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["success", "denied", "error", "timeout"]
    reason: str
    stats: SpreadStats | None = None
    stats_receipt: PropagationStatsReceipt | None = None
    comments: list[CommentItem] = Field(default_factory=list)
    availability_note: str | None = None
    feedback: PropagationSubmissionFeedback | None = None
    truncated: bool = False
    public_data: dict[str, object] | None = None


class PropagationResult(BaseModel):
    """程序验收通过的传播结果，不代表语义判断正确。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    run_id: str
    report: PropagationReport
    leads_by_id: dict[str, Lead]
    comments: list[CommentItem]
    stats: SpreadStats | None
    stop_reason: Literal["submitted", "forced_final"]
    steps: int = Field(ge=0, strict=True)
    input_output_tokens: int = Field(ge=0, strict=True)
    cost_cny: float = Field(ge=0, allow_inf_nan=False)
    trace_path: str

    @classmethod
    def detached(cls, sample_id: str, run_id: str, report: PropagationReport,
                 comments: Mapping[str, CommentItem], stats: SpreadStats | None,
                 budget: BudgetTracker, stop_reason: Literal["submitted", "forced_final"],
                 trace_path: str) -> PropagationResult:
        """按当前报告分配L编号并深复制所有可信结果。"""
        copied = PropagationReport.model_validate_json(report.model_dump_json())
        leads = {f"L{index}": Lead.model_validate(item.model_dump())
                 for index, item in enumerate(copied.leads, start=1)}
        result = cls(sample_id=sample_id, run_id=run_id, report=copied, leads_by_id=leads,
            comments=[CommentItem.model_validate(item.model_dump()) for item in comments.values()],
            stats=SpreadStats.model_validate(stats.model_dump()) if stats is not None else None,
            stop_reason=stop_reason, steps=budget.steps + int(budget.final_used) + int(budget.correction_used),
            input_output_tokens=budget.tokens, cost_cny=float(budget.cost_cny), trace_path=trace_path)
        return cls.model_validate(result.model_dump(mode="python"))


class EmptyPropagationArgs(BaseModel):
    """无参传播统计工具的严格输入模型。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class SampleCommentsArgs(BaseModel):
    """抽样工具参数，条数上限与传播数据工具一致。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    order: Literal["earliest", "latest", "seeded_random"]
    k: int = Field(gt=0)


class SearchCommentsArgs(BaseModel):
    """评论检索参数，实际上界绑定传播数据工具配置。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, pattern=r"\S")
    k: int = Field(gt=0)


class _ToolCounter:
    """保存主循环与执行器共同使用的传播运行状态。"""

    def __init__(self) -> None:
        """初始化尚未向模型交付的评论和统计快照。"""
        self.comments: dict[str, CommentItem] = {}
        self.stats: SpreadStats | None = None
        self.errors = 0
        self.nudges = 0
        self.reason = "no_valid_result"
        self.budget: BudgetTracker | None = None
        self.coordinator: DualBudgetCoordinator | None = None
        self.phase: Literal["startup", "ordinary", "final", "correction"] = "startup"
        self.stage: Literal["validation", "model", "tools", "submission", "tool_archive", "tool_event", "terminal"] = "validation"
        self.request_step = 0
        self.raw_reason: str | None = None
        self.final_trigger: str | None = None
        self.halt_unconfirmed = False
        self.outcome_recorded = False
        self.contract_version: Literal["v1.0", "v1.1"] = "v1.0"
        self.empty_queries: set[tuple[str, int]] = set()
        self.no_visible_comments = False
        self.final_feedback: PropagationSubmissionFeedback | None = None
        self.p1_group: str | None = None


class _PropagationFailure(RuntimeError):
    """只携带程序定义的传播启动或终止原因。"""

    def __init__(self, reason: str) -> None:
        """保留可写入追踪的受控原因。"""
        self.reason = reason
        super().__init__(reason)


class _PropagationAuditFailure(RuntimeError):
    """只标记回执归档或终局审计的完整性失败。"""

    def __init__(self, reason: str, raw_reason: str) -> None:
        """保存受控机器原因，不携带外部文本。"""
        self.reason, self.raw_reason = reason, raw_reason
        super().__init__(reason)


def _serialize_tool_payload(payload: PropagationToolPayload) -> str:
    """统一序列化实际送入模型历史的传播回执。"""
    excluded = set()
    if not payload.comments:
        excluded.add("comments")
    if not payload.truncated:
        excluded.add("truncated")
    if payload.stats_receipt is not None:
        excluded.add("stats")
    content = payload.model_dump_json(exclude_none=True, exclude=excluded)
    return content.replace("<", "\\u003c").replace(">", "\\u003e") if payload.public_data is not None else content


def _fit_tool_payload(payload: PropagationToolPayload, *, max_chars: int) -> PropagationToolPayload:
    """把所有回执压到完整JSON上限内，保留拒绝状态和完整统计值。"""
    if max_chars < 1:
        raise ValueError("tool_payload_capacity_error")
    if len(_serialize_tool_payload(payload)) <= max_chars:
        return payload
    if payload.feedback is not None:
        issues = list(payload.feedback.issues)
        while len(issues) > 1:
            candidate = payload.model_copy(update={"feedback": PropagationSubmissionFeedback.reviewed(
                tuple(issues), payload.feedback.source_issues), "truncated": True})
            if len(_serialize_tool_payload(candidate)) <= max_chars:
                return candidate
            issues.pop()
        for issue in (issues[0] if issues else "validation_failed", "validation_failed"):
            safe_issue = issue if issue != "validation_failed" else "按Schema和本次工具回执修正。"
            candidate = payload.model_copy(update={"feedback": PropagationSubmissionFeedback.reviewed(
                (safe_issue,), payload.feedback.source_issues), "truncated": True})
            if len(_serialize_tool_payload(candidate)) <= max_chars:
                return candidate
        raise ValueError("tool_payload_capacity_error")
    if payload.stats is not None:
        candidate = PropagationToolPayload(status="error", reason="result_limit")
        if len(_serialize_tool_payload(candidate)) <= max_chars:
            return candidate
        raise ValueError("tool_payload_capacity_error")
    comments = list(payload.comments)
    while comments:
        candidate = payload.model_copy(update={"comments": comments, "truncated": True})
        if len(_serialize_tool_payload(candidate)) <= max_chars:
            return candidate
        comments.pop()
    candidate = payload.model_copy(update={"comments": [], "truncated": bool(payload.comments)})
    if len(_serialize_tool_payload(candidate)) <= max_chars:
        return candidate
    raise ValueError("tool_payload_capacity_error")


def _minimum_tool_payload_capacity() -> int:
    """计算必要回执结构中的最大最小序列化长度。"""
    denied = ("unknown_tool", "invalid_caller", "not_allowed", "risk_mismatch", "tool_limit",
        "budget_unavailable", "final_only", "experiment_cost_limit", "experiment_halted", "batch_stopped")
    payloads = [PropagationToolPayload(status="success", reason="ok")]
    payloads.extend(PropagationToolPayload(status="denied", reason=reason) for reason in denied)
    payloads.extend([
        PropagationToolPayload(status="error", reason="result_limit"),
        PropagationToolPayload(status="timeout", reason="tool_timeout"),
        PropagationToolPayload(status="success", reason="ok", truncated=True),
        PropagationToolPayload(status="success", reason="validation_failed",
            feedback=PropagationSubmissionFeedback(issues=("按Schema和本次工具回执修正。",))),
        PropagationToolPayload(status="success", reason="accepted",
            feedback=PropagationSubmissionFeedback(issues=())),
        PropagationToolPayload(status="denied", reason="experiment_cost_limit",
            feedback=PropagationSubmissionFeedback(issues=("experiment_cost_limit",))),
        PropagationToolPayload(status="denied", reason="batch_stopped"),
    ])
    return max(len(_serialize_tool_payload(payload)) for payload in payloads)


def _valid_comment(
    comment_id: str,
    quote: str,
    raw: PropagationRaw | None,
    returned: Mapping[str, CommentItem],
    as_of: datetime | None,
) -> bool:
    """校验引用来自本案实际返回且未超时截断的原始评论。"""
    item = returned.get(comment_id)
    if item is None or raw is None or not quote or quote not in item.text:
        return False
    if not _valid_comment_id(comment_id, raw, returned, as_of):
        return False
    index = int(comment_id[1:])
    node = raw.nodes[index - 1]
    limit = get_propagation_tools_config().comment_preview_chars
    return quote in node.text[:limit]


def _valid_comment_id(comment_id: str, raw: PropagationRaw | None,
                      returned: Mapping[str, CommentItem], as_of: datetime | None) -> bool:
    """校验C编号指向本案返回的可见原始节点。"""
    item = returned.get(comment_id)
    if item is None or raw is None or not comment_id.startswith("C"):
        return False
    suffix = comment_id[1:]
    if not suffix.isdigit() or int(suffix) < 1 or int(suffix) > len(raw.nodes):
        return False
    node = raw.nodes[int(suffix) - 1]
    limit = get_propagation_tools_config().comment_preview_chars
    return (item.date == node.date and node.text.strip() != "" and
            (as_of is None or node.date <= as_of) and item.text == node.text[:limit])


def _valid_stats_basis(basis: list[str], stats: SpreadStats | None) -> bool:
    """只允许引用已交付统计中的业务字段和值。"""
    if stats is None or not stats.has_data:
        return not basis
    return all(name in _available_stats_fields(stats) for name in basis)


def _available_stats_fields(stats: SpreadStats | None) -> tuple[str, ...]:
    """从本次真实返回且有值的统计中生成spread_basis可用字段。"""
    if stats is None or not stats.has_data:
        return ()
    excluded = {"has_data", "visible_nodes", "unavailable_fields"}
    unavailable = set(stats.unavailable_fields)
    return tuple(name for name, value in stats.model_dump().items()
                 if name not in excluded and name not in unavailable and value is not None)


def _has_visible_comment(raw: PropagationRaw | None, as_of: datetime | None) -> bool:
    """按本案截止时间判断是否存在可供工具读取的非空评论。"""

    return bool(raw and any(node.text.strip() and (as_of is None or node.date <= as_of)
                            for node in raw.nodes))


def _valid_lead_source(lead: Lead, raw: PropagationRaw | None,
                       returned: Mapping[str, CommentItem], as_of: datetime | None) -> bool:
    """线索必须逐字引用已返回评论。"""
    return _valid_comment(lead.source, lead.quote, raw, returned, as_of)


def review_propagation_report(
    report: PropagationReport,
    *,
    claim: str,
    raw: PropagationRaw | None,
    returned_comments: Mapping[str, CommentItem],
    stats: SpreadStats | None,
    as_of: datetime | None = None,
) -> PropagationSubmissionFeedback:
    """独立检查传播报告的逐字来源、统计依据与必要字段。"""
    issues: list[str] = []
    if report.emotion_level in {"中", "高"} and not report.emotion_signals:
        issues.append("emotion_signal_required")
    for signal in report.emotion_signals:
        valid = signal.quote in claim if signal.source == "claim" else _valid_comment(
            signal.source, signal.quote, raw, returned_comments, as_of)
        if not valid:
            issues.append("emotion_quote_invalid")
    if not _valid_stats_basis(report.spread_basis, stats):
        issues.append("spread_basis_invalid")
    if (stats is None or not stats.has_data) and report.spread_level != "未知":
        issues.append("spread_must_be_unknown")
    if report.spread_level != "未知" and not report.spread_basis:
        issues.append("spread_basis_required")
    if raw is None:
        if report.narrative != "无法判断" or report.narrative_basis:
            issues.append("narrative_requires_propagation_data")
    elif report.narrative != "无法判断" and not report.narrative_basis:
        issues.append("narrative_basis_required")
    if any(not _valid_comment_id(cid, raw, returned_comments, as_of)
           for cid in report.narrative_basis):
        issues.append("narrative_basis_invalid")
    for lead in report.leads:
        if not lead.quote.strip() or not lead.claim_to_verify.strip():
            issues.append("lead_quote_invalid")
            continue
        if not _valid_lead_source(lead, raw, returned_comments, as_of):
            issues.append("lead_quote_invalid")
    for signal in report.emotion_signals:
        if not signal.quote.strip():
            issues.append("emotion_quote_invalid")
    public_issues = propagation_submission_feedback(issues,
        available_fields=_available_stats_fields(stats), returned_comment_ids=returned_comments)
    return PropagationSubmissionFeedback.reviewed(public_issues, tuple(issues))


def build_sample_comments_args(config: PropagationToolsConfig) -> type[BaseModel]:
    """将抽样参数Schema与本次传播工具配置绑定。"""
    return create_model("ConfiguredSampleCommentsArgs", __base__=SampleCommentsArgs,
                        k=(int, Field(gt=0, le=config.sample_max_k)))


def build_search_comments_args(config: PropagationToolsConfig) -> type[BaseModel]:
    """将评论查询长度与结果条数限制写入工具Schema。"""
    return create_model("ConfiguredSearchCommentsArgs", __base__=SearchCommentsArgs,
        query=(str, Field(min_length=1, max_length=config.search_query_max_chars, pattern=r"\S")),
        k=(int, Field(gt=0, le=config.search_max_k)))


def register_propagation_tools(registry: ToolRegistry, *, tools_config: PropagationToolsConfig,
                               contract_version: Literal["v1.0", "v1.1"] = "v1.0",
                               review_mode: bool = False) -> None:
    """登记相同只读传播工具，并按模式只开放一种互斥提交工具。"""
    from src.yuqing.schemas import PropagationReport

    submit_model = PropagationReport
    if contract_version == "v1.1":
        from src.yuqing.agents.propagation_v11 import submission_model

        submit_model = submission_model(())

    submission = (("submit_review", ReviewResult,
        "提交传播交叉复核；程序验证后才更新指定解释字段。") if review_mode else
        ("submit_propagation_report", submit_model,
         "独占提交传播分析报告，程序检查来源和依据。"))
    definitions = (
        ("get_spread_stats", EmptyPropagationArgs, "查看本案已提供的传播统计。"),
        ("sample_comments", build_sample_comments_args(tools_config), "按稳定编号抽取本案评论。"),
        ("search_comments", build_search_comments_args(tools_config), "按关键词检索本案可见评论。"),
        submission,
    )
    for name, model, description in definitions:
        registry.register(ToolSpec(name=name, args_model=model, description=description, risk="R0"))


class PropagationToolExecutor:
    """在可信本案快照上执行工具，并只公布完整归档的回执。"""

    def __init__(self, registry: ToolRegistry, budget: BudgetTracker, policy: PolicyConfig,
                 worker_scope: EvidenceWorkerScope, sample: Sample, raw: PropagationRaw | None,
                 as_of: datetime | None, state: _ToolCounter, *,
                 review_task: PropagationReviewTask | None = None,
                 current_report: PropagationReport | None = None,
                 allowed_peer_refs: frozenset[str] = frozenset()) -> None:
        """复制输入快照并保留本次实际交付的C集合和统计。"""
        self.registry, self.budget = registry, budget
        self.policy = PolicyConfig.model_validate(policy.model_dump())
        self.worker_scope, self.sample_id, self.raw, self.as_of = worker_scope, sample.sample_id, raw, as_of
        self._claim_text = sample.claim
        self.counter = state
        self.review_task, self.current_report = review_task, current_report
        self.allowed_peer_refs = allowed_peer_refs
        self.submitted: PropagationReport | ReviewResult | None = None
        self._calls: set[str] = set()

    def _operation(self, request: ToolRequest) -> Callable[[], object]:
        """把已校验参数绑定到可信样本和时间快照。"""
        from src.yuqing.features.propagation import get_spread_stats, sample_comments, search_comments

        if request.name == "get_spread_stats":
            return lambda: get_spread_stats(self.raw, as_of=self.as_of)
        if request.name == "sample_comments":
            return lambda: sample_comments(self.raw, sample_id=self.sample_id, as_of=self.as_of, **request.arguments)
        if request.name == "search_comments":
            return lambda: search_comments(self.raw, as_of=self.as_of, **request.arguments)
        raise ValueError("unsupported_propagation_tool")

    def _fit_payload(self, payload: PropagationToolPayload) -> PropagationToolPayload:
        """只减少评论条目，不拆分JSON或评论；统计结构必须完整容纳。"""
        maximum = self.budget.config.max_tool_result_chars
        return _fit_tool_payload(payload, max_chars=maximum)

    def _tool(self, request: ToolRequest) -> PropagationToolPayload:
        """验证权限与用量后锁内占额，迟到结果不注册为已交付。"""
        decision = decide(self.registry, "propagation_analyst", request.name, caller="model", config=self.policy)
        if decision.action != "allow":
            return self._archive_tool(request, PropagationToolPayload(status="denied", reason=decision.reason))
        denied = self._v11_empty_search_denial(request)
        if denied is not None:
            return self._archive_tool(request, denied)
        operation = self._operation(request)

        def authorize() -> None:
            """工作位可用时再扣限次，启动失败不退款真实尝试。"""
            reserved = self.budget.reserve_tool_call(request.name)
            if not reserved.allowed:
                raise WorkerUnavailable(reserved.reason)

        try:
            future = self.worker_scope.submit_authorized(operation, authorize)
            result = future.result(timeout=self.budget.config.tool_timeout_s[request.name])
        except TimeoutError:
            return self._archive_tool(request, PropagationToolPayload(status="timeout", reason="tool_timeout"))
        except WorkerUnavailable as exc:
            return self._archive_tool(request, PropagationToolPayload(status="denied", reason=str(exc)))
        except (ValueError, TypeError, OSError):
            return self._archive_tool(request, PropagationToolPayload(status="error", reason="tool_error"))
        except Exception as exc:
            logger.error("传播工具失败 type=%s", type(exc).__name__)
            return self._archive_tool(request, PropagationToolPayload(status="error", reason="tool_error"))
        payload = self._result_payload(request, result)
        archived = self._archive_tool(request, self._fit_payload(payload))
        self._remember_empty_result(request, result)
        return archived

    def _v11_empty_search_denial(self, request: ToolRequest) -> PropagationToolPayload | None:
        """拒绝已确认无评论或同快照内重复的空搜索，且不占工具次数。"""
        if self.counter.contract_version != "v1.1" or request.name not in {"sample_comments", "search_comments"}:
            return None
        if self.counter.no_visible_comments:
            missing = self.raw is None
            return PropagationToolPayload(status="denied",
                reason="no_raw_coverage" if missing else "no_comments_in_snapshot",
                availability_note="本案没有传播资料覆盖。" if missing else "本案当前快照没有可用评论文本。")
        key = (str(request.arguments.get("query", "")), int(request.arguments.get("k", 0)))
        if request.name == "search_comments" and key in self.counter.empty_queries:
            return PropagationToolPayload(status="denied", reason="repeated_empty_query",
                availability_note="相同查询在本案快照中已无匹配结果。")
        return None

    def _result_payload(self, request: ToolRequest, result: object) -> PropagationToolPayload:
        """按版本生成工具的公开回执，同时把完整统计留在内部审计封套。"""
        if isinstance(result, SpreadStats):
            fields: dict[str, object] = {"stats": result}
            if self.counter.contract_version == "v1.1":
                from src.yuqing.agents.propagation_v11 import stats_receipt

                fields["stats_receipt"] = stats_receipt(result)
            return PropagationToolPayload(status="success", reason="ok", **fields)
        if self.counter.contract_version != "v1.1" or result:
            return PropagationToolPayload(status="success", reason="ok", comments=result)
        has_visible = _has_visible_comment(self.raw, self.as_of)
        has_coverage = self.raw is not None
        from src.yuqing.agents.propagation_v11 import empty_comment_note

        reason = "no_raw_coverage" if not has_coverage else "no_query_match" if has_visible else "no_comments_in_snapshot"
        return PropagationToolPayload(status="success", reason=reason, comments=[],
            availability_note=empty_comment_note(has_coverage=has_coverage,
                                                has_visible_comments=has_visible))

    def _remember_empty_result(self, request: ToolRequest, result: object) -> None:
        """只在空结果回执归档成功后记录本案不可用状态或重复查询键。"""
        if self.counter.contract_version != "v1.1" or result or request.name not in {"sample_comments", "search_comments"}:
            return
        if not _has_visible_comment(self.raw, self.as_of):
            self.counter.no_visible_comments = True
        elif request.name == "search_comments":
            key = (str(request.arguments.get("query", "")), int(request.arguments.get("k", 0)))
            self.counter.empty_queries.add(key)

    def _archive_tool(self, request: ToolRequest, payload: PropagationToolPayload) -> PropagationToolPayload:
        """归档完整回执后再允许循环登记可见材料。"""
        self.counter.stage = "tool_archive"
        detailed_issues = (list(payload.feedback.source_issues or payload.feedback.issues)
                           if payload.feedback is not None else None)
        try:
            fitted = self._fit_payload(payload)
            content = _serialize_tool_payload(fitted)
            if len(content) > self.budget.config.max_tool_result_chars:
                raise ValueError("tool_payload_capacity_error")
        except Exception as exc:
            self.counter.reason = "tool_payload_capacity_error"
            self.counter.raw_reason = type(exc).__name__
            raise _PropagationAuditFailure(self.counter.reason, self.counter.raw_reason) from exc
        try:
            archive = write_archive("tool_result", {"request": request.model_dump(mode="json"),
                "payload": fitted.model_dump(mode="json"), "content": content,
                "program_metadata": {"detailed_feedback_issues": detailed_issues}})
        except Exception as exc:
            self.counter.reason = "tool_result_archive_failed"
            self.counter.raw_reason = type(exc).__name__
            raise _PropagationAuditFailure(self.counter.reason, self.counter.raw_reason) from exc
        self.counter.stage = "tool_event"
        try:
            record_event("tool_call", {"call_id": request.call_id, "tool": request.name,
                "risk": self.registry.get(request.name).risk, "status": fitted.status,
                "reason": fitted.reason, "archive": archive, "result_chars": len(content),
                "request_step": self.counter.request_step, "phase": self.counter.phase})
        except Exception as exc:
            self.counter.reason = "tool_event_audit_failed"
            self.counter.raw_reason = type(exc).__name__
            raise _PropagationAuditFailure(self.counter.reason, self.counter.raw_reason) from exc
        self.counter.stage = "submission" if request.name == self.budget.config.submission_tool else "tools"
        return fitted

    def execute(self, response: ToolLLMResult, *, final: bool) -> tuple[list[dict[str, str]], str | None]:
        """检查整批身份与提交互斥，生成一对一回执并更新本轮提交。"""
        from src.yuqing.harness.verifier import check_submission_batch

        if not self.budget.authorize_tool_response(response).allowed:
            raise ValueError("untrusted_response")
        calls = response.tool_calls
        if any(call.call_id in self._calls for call in calls):
            raise ValueError("duplicate_tool_call_id")
        if check_submission_batch(calls, submission_tool=self.budget.config.submission_tool):
            raise ValueError("invalid_tool_batch")
        if final and (len(calls) != 1 or calls[0].name != self.budget.config.submission_tool):
            raise ValueError("final_submission_mismatch")
        messages: list[dict[str, object]] = [response.assistant_message]
        trigger = None
        self._calls.update(call.call_id for call in calls)
        for request in calls:
            if trigger is not None:
                skipped = self._archive_tool(request, PropagationToolPayload(
                    status="denied", reason="batch_stopped"))
                messages.append({"role": "tool", "tool_call_id": request.call_id,
                                 "content": _serialize_tool_payload(skipped)})
                continue
            if request.name == self.budget.config.submission_tool:
                trigger = self._submit(response, request, messages)
            else:
                payload = self._tool(request)
                messages.append({"role": "tool", "tool_call_id": request.call_id,
                                 "content": _serialize_tool_payload(payload)})
                if payload.status == "success":
                    self.counter.errors = 0
                    self._record_delivered(payload)
                elif payload.status in {"error", "timeout"}:
                    self.counter.errors += 1
                    if self.counter.errors >= self.budget.config.max_consecutive_tool_errors:
                        trigger = "tool_error_limit"
                    if payload.status == "timeout" and self.worker_scope.has_pending_work():
                        trigger = "tool_timeout_pending"
                if payload.status == "denied" and payload.reason == "tool_limit":
                    trigger = "tool_limit"
        return messages, trigger

    def _submit(self, response: ToolLLMResult, request: ToolRequest,
                messages: list[dict[str, object]]) -> str | None:
        """以当前已交付材料复核提交，并配对返回固定问题代码。"""
        self.counter.stage = "submission"
        reserved = self.budget.reserve_tool_call(request.name)
        if not reserved.allowed:
            denied = PropagationToolPayload(status="denied", reason=reserved.reason,
                feedback=PropagationSubmissionFeedback(issues=(reserved.reason,)))
            denied = self._archive_tool(request, denied)
            messages.append({"role": "tool", "tool_call_id": request.call_id,
                             "content": _serialize_tool_payload(denied)})
            return reserved.reason
        submitted, feedback = self._check_submission(request.arguments)
        payload = PropagationToolPayload(status="success",
            reason="accepted" if feedback.accepted else "validation_failed", feedback=feedback)
        payload = self._archive_tool(request, payload)
        messages.append({"role": "tool", "tool_call_id": request.call_id,
                         "content": _serialize_tool_payload(payload)})
        if feedback.accepted:
            self.submitted = submitted
            return "accepted"
        if self.counter.phase in {"final", "correction"}:
            self.counter.final_feedback = feedback
        return None

    def _check_submission(self, arguments: dict[str, object]) -> tuple[
        PropagationReport | ReviewResult, PropagationSubmissionFeedback]:
        """Validate one submission against this run's delivered references before archiving feedback."""
        if self.review_task is not None:
            from src.yuqing.collab.review import validate_review_result

            report, result = self.current_report, ReviewResult.model_validate(arguments, strict=True)
            task_fields = frozenset(item.field for item in self.review_task.items)
            issues = ("review_task_missing_report",) if report is None else validate_review_result(
                result, action_id=self.review_task.action_id, allowed_fields=task_fields,
                current_report=report, returned_comment_ids=frozenset(self.counter.comments),
                available_stats_fields=frozenset(_available_stats_fields(self.counter.stats)),
                allowed_peer_refs=self.allowed_peer_refs)
            feedback = PropagationSubmissionFeedback.reviewed(issues, issues)
            submitted = result
        else:
            submitted = PropagationReport.model_validate(arguments, strict=True)
            feedback = review_propagation_report(submitted, claim=self._claim,
                raw=self.raw, returned_comments=self.counter.comments,
                stats=self.counter.stats, as_of=self.as_of)
        report = submitted if isinstance(submitted, PropagationReport) else None
        if report is not None and self.counter.contract_version == "v1.1" and "emotion_quote_invalid" in feedback.source_issues:
            from src.yuqing.agents.propagation_v11 import v11_emotion_feedback

            allowed = ("claim", *sorted(self.counter.comments))
            details = []
            for index, signal in enumerate(report.emotion_signals):
                valid = signal.quote in self._claim if signal.source == "claim" else _valid_comment(
                    signal.source, signal.quote, self.raw, self.counter.comments, self.as_of)
                if not valid:
                    details.append(v11_emotion_feedback(index, allowed))
            remaining = propagation_submission_feedback(
                tuple(issue for issue in feedback.source_issues if issue != "emotion_quote_invalid"),
                available_fields=_available_stats_fields(self.counter.stats),
                returned_comment_ids=self.counter.comments)
            feedback = PropagationSubmissionFeedback.reviewed(
                (*remaining, *details) if details else feedback.issues, feedback.source_issues)
        if report is not None and self.counter.contract_version == "v1.1" and "spread_basis_invalid" in feedback.source_issues:
            from src.yuqing.agents.propagation_v11 import available_fields, v11_submission_feedback

            detail = v11_submission_feedback(report.spread_basis,
                                             available_fields(self.counter.stats))
            issues = tuple(detail if "spread_basis" in issue else issue for issue in feedback.issues)
            feedback = PropagationSubmissionFeedback.reviewed(issues, feedback.source_issues)
        return submitted, feedback

    def _record_delivered(self, payload: PropagationToolPayload) -> None:
        """只有归档且能序列化进请求历史的成功内容可被终稿引用。"""
        if payload.stats is not None:
            self.counter.stats = payload.stats.model_copy(deep=True)
            if (self.review_task is None and self.counter.contract_version == "v1.1"
                    and payload.stats_receipt is not None):
                from src.yuqing.agents.propagation_v11 import submission_model

                spec = self.registry.get("submit_propagation_report")
                fields = tuple(payload.stats_receipt.allowed_spread_basis)
                self.registry._tools["submit_propagation_report"] = spec.model_copy(
                    update={"args_model": submission_model(fields)})
                self.counter.no_visible_comments = (
                    not payload.stats.has_data or payload.stats.visible_comments == 0)
        self.counter.comments.update({item.comment_id: item.model_copy(deep=True) for item in payload.comments})

    @property
    def _claim(self) -> str:
        """从绑定的可信样本取声明，不接受工具参数覆盖。"""
        return self._claim_text


def _startup_error(
    spec: AgentSpec,
    sample: Sample,
    ctx: RunContext,
    node_config: PropagationHarnessConfig,
    registry: ToolRegistry,
    policy: PolicyConfig,
    budget: BudgetTracker,
    *, review_mode: bool = False, review_task: PropagationReviewTask | None = None,
    p1_cost_cap_cny: Decimal | None = None,
) -> str | None:
    """核对运行身份、规格上界、工具权限和首轮/收尾输入容量。"""
    limits = get_propagation_limits_config()
    if spec.agent != "propagation_analyst" or ctx.agent != "propagation_analyst" or sample.sample_id != ctx.sample_id:
        return "run_identity_mismatch"
    if not sample.claim.strip() or not spec.system_prompt.strip():
        return "invalid_start_input"
    cap = (get_m6c_case_budget_config().action_caps["propagation_review"] if review_mode
           else limits.round1_max_cost_cny)
    cost_cap = float(p1_cost_cap_cny) if p1_cost_cap_cny is not None else cap
    if node_config.max_steps > limits.round1_max_steps or node_config.max_cost_cny > cost_cap:
        return "propagation_limits_exceeded"
    if node_config.max_tool_calls["sample_comments"] > limits.sample_max_calls or node_config.max_tool_calls["search_comments"] > limits.search_max_calls:
        return "propagation_limits_exceeded"
    if registry_names := set(registry._tools):
        expected = {"get_spread_stats", "sample_comments", "search_comments", node_config.submission_tool}
        if registry_names != expected:
            return "propagation_tools_mismatch"
    else:
        return "propagation_tools_missing"
    tools = [registry.get(name) for name in registry._tools]
    if any(tool is None or tool.risk != "R0" for tool in tools):
        return "propagation_tool_risk_mismatch"
    visible = registry.model_tools("propagation_analyst", policy)
    if {item["function"]["name"] for item in visible} != registry_names:
        return "propagation_tools_denied"
    initial = _build_initial_messages(spec, sample.claim, review_task)
    ordinary = _build_request_messages(initial, budget=budget, registry=registry,
                                       state=_ToolCounter(), final=False)
    final = _build_request_messages(initial, budget=budget, registry=registry,
                                    state=_ToolCounter(), final=True)
    return _startup_capacity_error(ordinary, final, node_config, registry, budget)


def _build_initial_messages(spec: AgentSpec, claim: str,
                            review_task: PropagationReviewTask | None = None) -> list[dict[str, object]]:
    """构造首请求使用的完整系统、任务和转义声明消息。"""
    task = "propagation_review" if review_task is not None else "propagation_analysis"
    content = f"<task>{task}</task>\n<claim>{escape(claim, quote=False)}</claim>"
    if review_task is not None:
        content += '\n<review_task encoding="json">' + escape(
            review_task.model_dump_json(), quote=False) + "</review_task>"
    return [{"role": "system", "content": spec.system_prompt}, {"role": "user", "content": content}]


def _build_request_messages(history: list[dict[str, object]], *, budget: BudgetTracker,
                            registry: ToolRegistry, state: _ToolCounter,
                            final: bool, correction: bool = False) -> list[dict[str, object]]:
    """复制历史并只追加一条按当前状态生成的真实程序消息。"""
    messages = deepcopy(history)
    messages.append(_status_message(budget, registry, state, final=final, correction=correction))
    return messages


def _startup_capacity_error(ordinary_messages: list[dict[str, object]],
                            final_messages: list[dict[str, object]], node_config: PropagationHarnessConfig,
                             registry: ToolRegistry, budget: BudgetTracker) -> str | None:
    """用正式首请求和网关同一估算器检查普通、收尾与回执容量。"""
    if node_config.max_tool_result_chars < _minimum_tool_payload_capacity():
        return "tool_result_capacity_insufficient"
    try:
        tools = [registry.get(name) for name in registry._tools]
        definitions = llm._tool_definitions(tools, False, submission_tool=node_config.submission_tool)
        ordinary_bound = llm.estimate_request_input(ordinary_messages, definitions, node_config.input_bound_margin_tokens)
        final_tool = [registry.get(node_config.submission_tool)]
        final_defs = llm._tool_definitions(final_tool, True, submission_tool=node_config.submission_tool)
        final_bound = llm.estimate_request_input(final_messages, final_defs, node_config.input_bound_margin_tokens)
    except (TypeError, ValueError, AttributeError):
        return "startup_capacity_error"
    if not budget.check_model_call(ordinary_bound, node_config.max_output_tokens).allowed:
        return "ordinary_capacity_insufficient"
    if not budget.check_model_call(final_bound, node_config.max_output_tokens, final=True).allowed:
        return "final_capacity_insufficient"
    return None


def _status_message(budget: BudgetTracker, registry: ToolRegistry,
                    counter: _ToolCounter, *, final: bool = False,
                    correction: bool = False) -> dict[str, str]:
    """向模型声明程序计算的额度及实际已交付的评论和统计状态。"""
    if hasattr(counter, "public_status"):
        return {"role": "user", "content": counter.public_status(budget, final=final)}
    quotas = [f"{name}:{budget.config.max_tool_calls[name] - budget.tool_calls[name]}/{budget.config.max_tool_calls[name]}"
              for name in ("sample_comments", "search_comments", "get_spread_stats")]
    visible = "、".join(sorted(counter.comments)) or "无"
    stats = "已交付" if counter.stats is not None else "未交付"
    phase = ("仅可根据已交付原文和程序退卷反馈修正提交内容并重交；禁止调查或增加材料"
             if correction else "只允许最终提交" if final else "可调用获准传播工具")
    content = f"程序状态（不是证据）：{phase}；额度{'；'.join(quotas)}；统计{stats}；已返回评论C：{visible}。"
    if counter.contract_version == "v1.1" and counter.stats is not None:
        from src.yuqing.agents.propagation_v11 import available_fields

        fields = "、".join(available_fields(counter.stats)) or "无"
        content += f"可引用统计字段：{fields}。"
    return {"role": "user", "content": content}


def _final_request(
    spec: AgentSpec,
    messages: list[dict[str, object]],
    ctx: RunContext,
    budget: BudgetTracker,
    registry: ToolRegistry,
    executor: PropagationToolExecutor,
    state: _ToolCounter,
    coordinator: DualBudgetCoordinator | None,
    identity: Mapping[str, str] | None,
    trigger: str,
) -> tuple[PropagationReport | PublicSignalReport | ReviewResult | None, str]:
    """发送一次强制提交；仅C档获准对可信退卷执行单次纠错。"""
    submit = registry.get(budget.config.submission_tool)
    state.phase, state.stage = "final", "model"
    state.request_step, state.final_trigger = budget.steps, trigger
    request_messages = _build_request_messages(messages, budget=budget, registry=registry,
                                               state=state, final=True)
    messages[:] = request_messages
    with use_run_context(ctx.model_copy(update={"step": state.request_step})):
        record_event("finalization", {"trigger": trigger, "phase": "final"})
        outcome = llm.call_with_tools_outcome(request_messages, [submit], budget, final=True,
            **_coordinator_kwargs(coordinator, identity))
        if outcome.result is None:
            state.raw_reason = outcome.reason
            if _append_final_format_feedback(outcome, messages, registry, budget):
                return _correction_request(messages, ctx, budget, submit, executor, state,
                                           coordinator, identity)
            return None, outcome.reason
        state.stage = "submission"
        replies, _ = executor.execute(outcome.result, final=True)
        messages.extend(replies)
    if executor.submitted is None:
        if _trusted_feedback_can_correct(state, budget):
            return _correction_request(messages, ctx, budget, submit, executor, state,
                                       coordinator, identity)
        return None, "final_submission_rejected"
    return executor.submitted, "forced_final"


def _append_final_format_feedback(outcome: ToolCallOutcome, messages: list[dict[str, object]],
                                  registry: ToolRegistry, budget: BudgetTracker) -> bool:
    """只将网关识别的提交Schema错误转成受控反馈，不回显模型原文。"""
    if (not budget.correction_enabled or budget.status().halted or outcome.status != "invalid_response"
            or outcome.reason not in {"invalid_tool_arguments", "text_without_tools"}):
        return False
    schemas = {budget.config.submission_tool: registry.get(budget.config.submission_tool).args_model}
    messages.append({"role": "user", "content": format_retry_feedback(outcome.reason,
        outcome.validation_issues, schemas, budget.config.max_tool_result_chars)})
    return True


def _trusted_feedback_can_correct(state: _ToolCounter, budget: BudgetTracker) -> bool:
    """只对已归档、非空且来自传播提交校验器的拒绝开放C档纠错。"""
    feedback = state.final_feedback
    return (budget.correction_enabled and not budget.status().halted and feedback is not None
            and not feedback.accepted and bool(feedback.source_issues))


def _correction_request(messages: list[dict[str, object]], ctx: RunContext, budget: BudgetTracker,
                        submit: ToolSpec, executor: PropagationToolExecutor, state: _ToolCounter,
                        coordinator: DualBudgetCoordinator | None,
                        identity: Mapping[str, str] | None) -> tuple[PropagationReport | ReviewResult | None, str]:
    """沿用原预算和钱包只开放一次提交工具，纠错后立即终局。"""
    state.phase, state.stage = "correction", "model"
    state.request_step = budget.steps + int(budget.final_used)
    request_messages = _build_request_messages(messages, budget=budget, registry=executor.registry,
                                               state=state, final=True, correction=True)
    messages[:] = request_messages
    with use_run_context(ctx.model_copy(update={"step": state.request_step})):
        record_event("finalization", {"trigger": state.final_trigger, "phase": "correction"})
        outcome = llm.call_with_tools_outcome(request_messages, [submit], budget, final=True, correction=True,
            **_coordinator_kwargs(coordinator, identity))
        if outcome.result is None:
            state.raw_reason = outcome.reason
            return None, outcome.reason
        state.stage = "submission"
        replies, _ = executor.execute(outcome.result, final=True)
        messages.extend(replies)
    return (executor.submitted, "forced_final") if executor.submitted is not None else (
        None, "correction_submission_rejected")


def _coordinator_kwargs(coordinator: DualBudgetCoordinator | None,
                        identity: Mapping[str, str] | None) -> dict[str, object]:
    """无整场钱包时不传参，有钱包时每次都复用同一个身份。"""
    return {} if coordinator is None else {"coordinator": coordinator, "experiment_identity": identity}


def _prepare_coordinator(factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None,
                         identity: Mapping[str, str] | None, budget: BudgetTracker,
                         ctx: RunContext, state: _ToolCounter) -> DualBudgetCoordinator | None:
    """验证整场钱包依赖成对且绑定本次题目和预算。"""
    if (factory is None) != (identity is None):
        raise ValueError("incomplete_dual_budget_arguments")
    if factory is None:
        return None
    tags = dict(identity)
    if tags.get("sample_id") != ctx.sample_id or tags.get("run_id") != ctx.run_id or not tags.get("group_id"):
        raise ValueError("experiment_identity_mismatch")
    coordinator = factory(budget)
    if coordinator is None or coordinator.task is not budget:
        raise ValueError("coordinator_budget_mismatch")
    state.coordinator = coordinator
    return coordinator


def _new_result(sample: Sample | PublicPropagationTask, ctx: RunContext, report: PropagationReport | PublicSignalReport,
                executor: PropagationToolExecutor, budget: BudgetTracker,
                stop_reason: Literal["submitted", "forced_final"]) -> PropagationResult | PublicSignalResult:
    """从同一预算与最后已交付材料组装可信传播结果。"""
    if isinstance(report, PublicSignalReport):
        return executor.public_result(sample, ctx, report, budget, stop_reason)
    return PropagationResult.detached(sample.sample_id, ctx.run_id, report,
        executor.counter.comments, executor.counter.stats, budget, stop_reason, str(ctx.trace_path))


def _new_review_run(sample: Sample, ctx: RunContext, review: ReviewResult,
                    executor: PropagationToolExecutor, budget: BudgetTracker,
                    stop_reason: Literal["submitted", "forced_final"]) -> PropagationReviewRun:
    """保存校验通过的复核提交及本次确实交付的传播材料。"""
    return PropagationReviewRun(sample_id=sample.sample_id, run_id=ctx.run_id,
        review=ReviewResult.model_validate(review.model_dump()),
        comments=[item.model_dump(mode="json") for item in executor.counter.comments.values()],
        stats=executor.counter.stats.model_dump(mode="json") if executor.counter.stats else None,
        stop_reason=stop_reason, steps=budget.steps + int(budget.final_used) + int(budget.correction_used),
        input_output_tokens=budget.tokens, cost_cny=float(budget.cost_cny),
        trace_path=str(ctx.trace_path))


def _format_retry(outcome: ToolCallOutcome, messages: list[dict[str, object]],
                  state: _ToolCounter, budget: BudgetTracker, registry: ToolRegistry) -> bool:
    """只对三类可恢复格式问题使用共享提醒额度。"""
    recoverable = {"text_without_tools", "invalid_tool_arguments", "mixed_tool_batch"}
    if outcome.status != "invalid_response" or outcome.reason not in recoverable:
        return False
    if state.nudges >= budget.config.max_format_nudges or budget.status().halted:
        return False
    state.nudges += 1
    schemas = {name: spec.args_model for name in registry._tools
               if (spec := registry.get(name)) is not None}
    messages.append({"role": "user", "content": format_retry_feedback(
        outcome.reason, outcome.validation_issues, schemas, budget.config.max_tool_result_chars)})
    record_event("format_nudge", {"reason": outcome.reason, "count": state.nudges})
    return True


def _record_run_outcome(ctx: RunContext, result: PropagationResult | PublicSignalResult | PropagationReviewRun | None,
                        state: _ToolCounter) -> None:
    """记录唯一终局事件及本次预算账本中的实际调用量。"""
    budget = state.budget
    step = budget.steps + int(budget.final_used) + int(budget.correction_used) if budget is not None else 0
    if state.reason not in {"tool_result_archive_failed", "tool_event_audit_failed",
                             "run_outcome_audit_failed", "tool_payload_capacity_error"}:
        state.stage = "terminal"
    outcome_class = ("submitted_review" if isinstance(result, PropagationReviewRun) else
                     "submitted_report" if result else "no_valid_result")
    data: dict[str, object] = {"outcome_class": outcome_class,
        "reason": state.reason, "raw_reason": state.raw_reason,
        "phase": state.phase, "stage": state.stage, "request_step": state.request_step,
        "final_trigger": state.final_trigger, "halt_unconfirmed": state.halt_unconfirmed, "steps": step,
        "input_output_tokens": budget.tokens if budget is not None else 0,
        "cost_cny": float(budget.cost_cny) if budget is not None else None,
        "usage_known": not (budget.status().halted and budget.status().halt_reason == "usage_unknown")
                       if budget is not None else True,
        "stop_reason": result.stop_reason if result else None, "trace_path": str(ctx.trace_path)}
    if state.p1_group is not None:
        data.update({"p1_group": state.p1_group,
                     "correction_used": budget.correction_used if budget is not None else False})
    with use_run_context(ctx.model_copy(update={"step": step})):
        record_event("run_outcome", data)
    state.outcome_recorded = True


def _ordinary_request(ctx: RunContext, messages: list[dict[str, object]], registry: ToolRegistry,
                      executor: PropagationToolExecutor, budget: BudgetTracker,
                      coordinator: DualBudgetCoordinator | None,
                      identity: Mapping[str, str] | None) -> tuple[ToolCallOutcome, str | None]:
    """发送一个普通请求并在原请求步内处理工具审计回执。"""
    state = executor.counter
    state.phase, state.stage, state.request_step = "ordinary", "model", budget.steps
    tools = [registry.get(name) for name in registry._tools]
    request_messages = _build_request_messages(messages, budget=budget, registry=registry,
                                               state=state, final=False)
    messages[:] = request_messages
    with use_run_context(ctx.model_copy(update={"step": state.request_step})):
        outcome = llm.call_with_tools_outcome(request_messages, tools, budget,
            **_coordinator_kwargs(coordinator, identity))
        if outcome.result is None:
            state.raw_reason = outcome.reason
            return outcome, None
        state.stage = "tools"
        replies, trigger = executor.execute(outcome.result, final=False)
        messages.extend(replies)
    return outcome, trigger


def _halt_after_audit_failure(state: _ToolCounter, *, reason: str) -> bool:
    """停用单题与共享账本；持久停账失败时关闭当前钱包对象。"""
    budget, coordinator = state.budget, state.coordinator
    if budget is not None:
        budget.halt(reason)
    if coordinator is None:
        return True
    experiment = coordinator.experiment
    try:
        if not experiment.snapshot().halted:
            coordinator.halt(reason)
        return True
    except Exception as exc:
        state.halt_unconfirmed = True
        logger.error("传播共享停账未确认 reason=%s type=%s", reason, type(exc).__name__)
        try:
            experiment.close()
        except Exception as close_exc:
            logger.error("传播钱包封闭失败 reason=%s type=%s", reason, type(close_exc).__name__)
        return False


def _handle_run_failure(state: _ToolCounter, exc: Exception) -> None:
    """将入口异常压成受控原因并停止可能受损的账本。"""
    budget = state.budget
    if isinstance(exc, _PropagationAuditFailure):
        state.reason, state.raw_reason = exc.reason, exc.raw_reason
        _halt_after_audit_failure(state, reason=exc.reason)
    elif isinstance(exc, _PropagationFailure):
        state.reason, state.raw_reason = exc.reason, None
        if budget is not None:
            budget.halt(exc.reason)
    else:
        state.reason = budget.status().halt_reason if budget and budget.status().halted else "unexpected_propagation_error"
        state.raw_reason = type(exc).__name__
        _halt_after_audit_failure(state, reason=state.reason)
    logger.error("传播分析失败 type=%s", type(exc).__name__)


def _prepare_runtime(
    spec: AgentSpec, sample: Sample, ctx: RunContext,
    node_config: PropagationHarnessConfig, worker_scope: EvidenceWorkerScope,
    as_of: datetime | None, policy_config: PolicyConfig | None,
    coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None,
    experiment_identity: Mapping[str, str] | None, state: _ToolCounter,
    contract_version: Literal["v1.0", "v1.1"],
    *, review_task: PropagationReviewTask | None = None, current_report: PropagationReport | None = None,
    allowed_peer_refs: frozenset[str] = frozenset(),
    p1_profile: object | None = None,
) -> tuple[Sample, PropagationHarnessConfig, ToolRegistry, PropagationToolExecutor, DualBudgetCoordinator | None, list[dict[str, object]]]:
    """复制可信输入并在首个模型请求前验证所有依赖和配置。"""
    trusted_sample = Sample.model_validate(sample.model_dump())
    review_mode = review_task is not None
    allowed_configs = {PropagationHarnessConfig, PropagationReviewHarnessConfig}
    if type(node_config) not in allowed_configs or (review_mode != isinstance(node_config, PropagationReviewHarnessConfig)):
        raise _PropagationFailure("invalid_node_config")
    config = type(node_config).model_validate(node_config.model_dump())
    profile = _validate_p1_profile(p1_profile, config, review_mode)
    raw = PropagationRaw.model_validate(trusted_sample.propagation.model_dump()) if trusted_sample.propagation else None
    if as_of is not None and (as_of.tzinfo is None or as_of.utcoffset() is None):
        raise _PropagationFailure("as_of_timezone_required")
    if worker_scope.availability() != "ready":
        raise _PropagationFailure("worker_not_ready")
    prices = get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        raise _PropagationFailure("prices_unknown")
    try:
        state.budget = _create_runtime_budget(config, prices,
            profile.correction_enabled if profile is not None else False)
    except ValueError as exc:
        raise _PropagationFailure("node_capacity_insufficient") from exc
    policy = PolicyConfig.model_validate((policy_config or get_policy_config()).model_dump())
    registry = _prepare_registry(spec, trusted_sample, ctx, config, policy, state.budget,
        contract_version, review_mode, review_task,
        profile.slot_cap_cny if profile is not None else None)
    state.contract_version = contract_version
    try:
        coordinator = _prepare_coordinator(coordinator_factory, experiment_identity, state.budget, ctx, state)
    except ValueError as exc:
        raise _PropagationFailure(str(exc)) from exc
    state.coordinator = coordinator
    state.p1_group = profile.group if profile is not None else None
    executor = PropagationToolExecutor(registry, state.budget, policy, worker_scope,
        trusted_sample, raw, as_of, state, review_task=review_task,
        current_report=current_report, allowed_peer_refs=allowed_peer_refs)
    messages = _build_initial_messages(spec, trusted_sample.claim, review_task)
    return trusted_sample, config, registry, executor, coordinator, messages


def _prepare_registry(spec: AgentSpec, sample: Sample, ctx: RunContext,
                      config: PropagationHarnessConfig, policy: PolicyConfig, budget: BudgetTracker,
                      contract_version: Literal["v1.0", "v1.1"], review_mode: bool,
                      review_task: PropagationReviewTask | None,
                      p1_cost_cap_cny: Decimal | None) -> ToolRegistry:
    """用正式传播工具注册表完成版本、权限和首请求容量预检。"""
    if contract_version not in {"v1.0", "v1.1"}:
        raise _PropagationFailure("propagation_contract_version_invalid")
    registry = ToolRegistry()
    register_propagation_tools(registry, tools_config=get_propagation_tools_config(),
        contract_version=contract_version, review_mode=review_mode)
    failure = _startup_error(spec, sample, ctx, config, registry, policy, budget,
        review_mode=review_mode, review_task=review_task, p1_cost_cap_cny=p1_cost_cap_cny)
    if failure:
        raise _PropagationFailure(failure)
    return registry


def _validate_p1_profile(profile: object | None, config: PropagationHarnessConfig,
                         review_mode: bool) -> object | None:
    """实验档只接受Runner登记的A/B/C预算指纹，普通Agent路径保持关闭。"""
    if profile is None:
        return None
    from experiments.archive.performance_p1 import validate_group_profile

    if review_mode:
        raise _PropagationFailure("p1_profile_not_allowed_for_review")
    try:
        return validate_group_profile(profile, config)
    except (TypeError, ValueError) as exc:
        raise _PropagationFailure("p1_profile_invalid") from exc


def _create_runtime_budget(config: PropagationHarnessConfig, prices: CostConfig,
                           correction_enabled: bool) -> BudgetTracker:
    """默认构造保持原调用签名，仅P1-C显式开启额外预算预留。"""
    if correction_enabled:
        return BudgetTracker(config, prices, correction_enabled=True)
    return BudgetTracker(config, prices)


def run_propagation_agent(
    spec: AgentSpec,
    sample: Sample | TaskInput,
    ctx: RunContext,
    *,
    node_config: PropagationHarnessConfig,
    worker_scope: EvidenceWorkerScope,
    as_of: datetime | None = None,
    policy_config: PolicyConfig | None = None,
    coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None = None,
    experiment_identity: Mapping[str, str] | None = None,
    contract_version: Literal["v1.0", "v1.1"] = "v1.0",
    public_task: PublicPropagationTask | None = None,
) -> PropagationResult | PublicSignalResult | None:
    """运行共享预算、只读工具和单次收尾组成的传播短循环。"""
    if public_task is not None:
        from src.yuqing.serve.public_propagation import run_public_propagation

        return run_public_propagation(spec, sample, ctx, public_task=public_task,
            node_config=node_config, worker_scope=worker_scope, as_of=as_of,
            policy_config=policy_config, coordinator_factory=coordinator_factory,
            experiment_identity=experiment_identity, contract_version=contract_version)
    return _run_propagation_agent(spec, sample, ctx, node_config=node_config,
        worker_scope=worker_scope, as_of=as_of, policy_config=policy_config,
        coordinator_factory=coordinator_factory, experiment_identity=experiment_identity,
        contract_version=contract_version, p1_profile=None)


def _run_propagation_p1(
    spec: AgentSpec, sample: Sample, ctx: RunContext, *, node_config: PropagationHarnessConfig,
    worker_scope: EvidenceWorkerScope, p1_profile: object,
    coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None,
    experiment_identity: Mapping[str, str],
) -> PropagationResult | None:
    """仅离线Runner可调用的A/B/C路径，仍进入正式Agent校验与预算循环。"""
    from experiments.archive.performance_p1 import P1OfflineReplay, observation_as_of

    if type(llm._offline_replay.get()) is not P1OfflineReplay:
        raise ValueError("p1_runner_requires_offline_replay")
    return _run_propagation_agent(spec, sample, ctx, node_config=node_config,
        worker_scope=worker_scope, as_of=observation_as_of(sample), policy_config=None,
        coordinator_factory=coordinator_factory, experiment_identity=experiment_identity,
        contract_version="v1.1", p1_profile=p1_profile)


def _run_propagation_p1_live(
    spec: AgentSpec, sample: Sample, ctx: RunContext, *, node_config: PropagationHarnessConfig,
    worker_scope: EvidenceWorkerScope, p1_profile: object,
    coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator],
    experiment_identity: Mapping[str, str],
) -> PropagationResult | None:
    """已许可P1批次的私有入口；运行身份必须与Runner当前槽一致。"""
    from experiments.archive.performance_p1_live import require_active_slot
    from experiments.archive.performance_p1 import observation_as_of

    require_active_slot(ctx, experiment_identity)
    return _run_propagation_agent(spec, sample, ctx, node_config=node_config,
        worker_scope=worker_scope, as_of=observation_as_of(sample), policy_config=None,
        coordinator_factory=coordinator_factory, experiment_identity=experiment_identity,
        contract_version="v1.1", p1_profile=p1_profile)


def _run_propagation_agent(
    spec: AgentSpec, sample: Sample, ctx: RunContext, *, node_config: PropagationHarnessConfig,
    worker_scope: EvidenceWorkerScope, as_of: datetime | None, policy_config: PolicyConfig | None,
    coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None,
    experiment_identity: Mapping[str, str] | None,
    contract_version: Literal["v1.0", "v1.1"], p1_profile: object | None,
) -> PropagationResult | None:
    """运行默认或显式P1档案；审计、工具执行和终局记录共用原路径。"""
    result: PropagationResult | None = None
    state = _ToolCounter()
    try:
        with use_run_context(ctx):
            if type(spec) is not AgentSpec:
                raise _PropagationFailure("invalid_agent_spec")
            runtime = _prepare_runtime(spec, sample, ctx, node_config, worker_scope, as_of,
                policy_config, coordinator_factory, experiment_identity, state, contract_version,
                p1_profile=p1_profile)
            trusted_sample, config, registry, executor, coordinator, messages = runtime
            budget = state.budget
            if budget is None:
                raise ValueError("budget_initialization_failed")
            result, state.reason, state.phase = _iterate_propagation(
                spec, trusted_sample, ctx, config, registry, executor, budget, messages, coordinator,
                experiment_identity)
    except Exception as exc:
        result = None
        _handle_run_failure(state, exc)
    try:
        _record_run_outcome(ctx, result, state)
    except Exception as exc:
        state.reason = "run_outcome_audit_failed"
        state.raw_reason = type(exc).__name__
        state.stage = "terminal"
        _halt_after_audit_failure(state, reason=state.reason)
        logger.error("传播终局审计失败 stage=terminal type=%s", type(exc).__name__)
        return None
    return result


def run_propagation_review(
    spec: AgentSpec, sample: Sample, ctx: RunContext, *,
    node_config: PropagationReviewHarnessConfig, worker_scope: EvidenceWorkerScope,
    review_task: PropagationReviewTask, current_report: PropagationReport,
    allowed_peer_refs: frozenset[str], as_of: datetime | None = None,
    policy_config: PolicyConfig | None = None,
    coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None = None,
    experiment_identity: Mapping[str, str] | None = None,
    contract_version: Literal["v1.0", "v1.1"] = "v1.0",
) -> PropagationReviewRun | None:
    """Run an explicit M6c review request through the existing propagation loop."""
    result, state = None, _ToolCounter()
    try:
        with use_run_context(ctx):
            if type(spec) is not AgentSpec:
                raise _PropagationFailure("invalid_agent_spec")
            task = PropagationReviewTask.model_validate(review_task.model_dump())
            report = PropagationReport.model_validate(current_report.model_dump())
            runtime = _prepare_runtime(spec, sample, ctx, node_config, worker_scope, as_of,
                policy_config, coordinator_factory, experiment_identity, state, contract_version,
                review_task=task, current_report=report, allowed_peer_refs=allowed_peer_refs)
            trusted, config, registry, executor, coordinator, messages = runtime
            if state.budget is None:
                raise ValueError("budget_initialization_failed")
            result, state.reason, state.phase = _iterate_propagation(spec, trusted, ctx, config,
                registry, executor, state.budget, messages, coordinator, experiment_identity)
            if not isinstance(result, PropagationReviewRun):
                result = None
                state.reason = "review_submission_missing"
    except Exception as exc:
        result = None
        _handle_run_failure(state, exc)
    try:
        _record_run_outcome(ctx, result, state)
    except Exception as exc:
        state.reason, state.raw_reason, state.stage = "run_outcome_audit_failed", type(exc).__name__, "terminal"
        _halt_after_audit_failure(state, reason=state.reason)
        logger.error("传播复核终局审计失败 type=%s", type(exc).__name__)
        return None
    return result


def _iterate_propagation(
    spec: AgentSpec,
    sample: Sample,
    ctx: RunContext,
    config: PropagationHarnessConfig,
    registry: ToolRegistry,
    executor: PropagationToolExecutor,
    budget: BudgetTracker,
    messages: list[dict[str, object]],
    coordinator: DualBudgetCoordinator | None,
    identity: Mapping[str, str] | None,
) -> tuple[PropagationResult | PublicSignalResult | PropagationReviewRun | None, str, str]:
    """驱动普通请求、可恢复提醒与最多一次最终提交。"""
    reason, phase = "step_limit", "ordinary"
    while budget.steps < config.max_steps:
        outcome, trigger = _ordinary_request(ctx, messages, registry, executor, budget,
                                              coordinator, identity)
        if outcome.result is None:
            if outcome.status == "budget_denied" and not budget.status().halted:
                reason = outcome.reason
                break
            if _format_retry(outcome, messages, executor.counter, budget, registry):
                reason = outcome.reason
                continue
            return None, outcome.reason, "ordinary"
        if executor.submitted is not None:
            if isinstance(executor.submitted, ReviewResult):
                return _new_review_run(sample, ctx, executor.submitted, executor, budget,
                    "submitted"), "submitted", "ordinary"
            return _new_result(sample, ctx, executor.submitted, executor, budget, "submitted"), "submitted", "ordinary"
        if trigger:
            if trigger in {"tool_error_limit", "tool_timeout_pending"}:
                budget.halt(trigger)
                return None, trigger, "ordinary"
            reason = trigger
            break
    if budget.status().halted or budget.final_used:
        return None, budget.status().halt_reason or reason, "final"
    report, final_reason = _final_request(spec, messages, ctx, budget, registry, executor,
                                           executor.counter, coordinator, identity, reason)
    terminal_phase = "correction" if budget.correction_used else "final"
    if report is None:
        return None, final_reason, terminal_phase
    if isinstance(report, ReviewResult):
        return _new_review_run(sample, ctx, report, executor, budget, "forced_final"), "forced_final", terminal_phase
    return _new_result(sample, ctx, report, executor, budget, "forced_final"), "forced_final", terminal_phase
