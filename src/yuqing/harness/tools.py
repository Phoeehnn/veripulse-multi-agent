"""登记工具规格并生成当前专家可以看到的工具说明。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import timedelta
from hashlib import sha256
import json
import logging
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from src.yuqing.settings import HarnessConfig, PolicyConfig, get_harness_config
from src.yuqing.harness.fact_v2_contract import FactRunContextV2
from src.yuqing.harness.ledger import EvidenceLedger
from src.yuqing.harness.feedback import fact_submission_feedback
from src.yuqing.harness.trace import record_event, require_run_context, write_archive
from src.yuqing.harness.verifier import check_submission_batch, review_submission
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.retrieval.search import search
from src.yuqing.schemas import (EvidenceChunk, EvidenceToolPayload, FactVerdict, SubmissionFeedback,
    SubmissionReview, SubmissionToolResult, ToolEvidence, ToolExecutionResult, ToolLLMResult, ToolRequest)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.yuqing.harness.budget import BudgetTracker
    from src.yuqing.harness.context import EvidenceDelivery

RiskTier = Literal["R0", "R1", "R2"]
WorkerResult = TypeVar("WorkerResult")


class ToolSpec(BaseModel):
    """一个工具的名称、用途、参数结构和风险等级；超时统一读取配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    args_model: type[BaseModel]
    risk: RiskTier
    response_model: type[BaseModel] | None = None


class ToolRegistry:
    """只保存显式登记的工具，不执行工具或推断权限。"""

    def __init__(self) -> None:
        """建立空登记册，每次运行由程序显式注册所需工具。"""
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """登记唯一工具名，禁止同名替换已有规格。"""
        if spec.name in self._tools:
            raise ValueError(f"工具已登记：{spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """按名称读取规格，未登记返回空值。"""
        return self._tools.get(name)

    def model_tools(self, agent: str, config: PolicyConfig) -> list[dict[str, object]]:
        """复用权限闸门，仅向模型暴露当前专家获准使用的工具结构。"""
        from src.yuqing.harness.policy import decide

        visible = []
        for name, spec in self._tools.items():
            decision = decide(self, agent, name, caller="model", config=config)
            if decision.action == "allow":
                visible.append({"type": "function", "function": {
                    "name": name, "description": spec.description,
                    "parameters": spec.args_model.model_json_schema(),
                }})
        return visible


class SearchEvidenceArgs(BaseModel):
    """搜索关键词与可选条数，条数不得超过运行配置上限。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, pattern=r"\S")
    k: int | None = Field(default=None, gt=0)


class ReadEvidenceArgs(BaseModel):
    """只打开本次台账的E编号，不接受网址、路径或已读声明。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")


def build_search_args_model(config: HarnessConfig) -> type[SearchEvidenceArgs]:
    """让模型参数结构与Pydantic解析共同使用本次配置的搜索上限。"""
    return create_model("ConfiguredSearchEvidenceArgs", __base__=SearchEvidenceArgs,
        k=(int | None, Field(default=None, gt=0, le=config.search_k)))


def register_evidence_tools(registry: ToolRegistry, *, config: HarnessConfig | None = None) -> None:
    """显式登记只读搜索与正文阅读工具，不开放台账写入接口。"""
    registry.register(ToolSpec(name="search_evidence", risk="R0", args_model=build_search_args_model(config or get_harness_config()),
        description="检索声明相关资料，返回E编号和未读预览；引用前须read_evidence，不抓取任意网址。"))
    registry.register(ToolSpec(name="read_evidence", risk="R0", args_model=ReadEvidenceArgs,
        description="读取已搜索的E编号对应当前片段，返回原文及截断状态；不返回邻段或整篇文章。"))


def register_submission_tool(registry: ToolRegistry) -> None:
    """提交只接判断表，不接模型声称的已读、验收通过或运行身份。"""
    registry.register(ToolSpec(name="submit_verdict", risk="R0", args_model=FactVerdict,
        description="单独提交关键事实、理由、标签和置信度；引用须此前读过且已收到正文，程序独立验收。"))


def register_v2_submission_tool(registry: ToolRegistry, schema: type[BaseModel],
                               response_schema: type[BaseModel]) -> None:
    """向模型展示严格V2 Schema，同时由网关将可选附录保留为原始数据。"""
    registry.register(ToolSpec(name="submit_verdict", risk="R0", args_model=schema,
        response_model=response_schema,
        description="提交事实V2报告；程序单独复核核心、引用和可选建议附录。"))


class SubmissionToolExecutor:
    """仅接受当前真实响应的一次提交，完整复核留内部、反馈受配置限长。"""

    def __init__(self, registry: ToolRegistry, ledger: EvidenceLedger, budget: BudgetTracker,
                 policy: PolicyConfig, *, fact_contract: FactRunContextV2 | None = None,
                 delivery: EvidenceDelivery | None = None) -> None:
        """注入本运行工具、台账和预算，不自行调用模型或构造交付资格。"""
        self.registry, self.ledger, self.budget = registry, ledger, budget
        self.policy = PolicyConfig.model_validate(policy.model_dump())
        self.fact_contract, self.delivery = fact_contract, delivery
        self._audit_failed = False

    def _evaluate(self, response: ToolLLMResult, agent: str) -> tuple[str, SubmissionReview | None]:
        """身份、权限、可信响应及预算通过才复核提交，失败不偷偷改判断。"""
        from src.yuqing.harness.policy import decide
        if require_run_context().agent != agent:
            return "run_identity_mismatch", None
        calls = response.tool_calls
        if check_submission_batch(calls) or len(calls) != 1 or calls[0].name != "submit_verdict":
            return "invalid_submission_batch", None
        decision = decide(self.registry, agent, "submit_verdict", caller="model", config=self.policy)
        if decision.action != "allow":
            return decision.reason, None
        submission_spec = self.registry.get("submit_verdict")
        if self.fact_contract is None and submission_spec.args_model is not FactVerdict:
            return "invalid_submission_spec", None
        if self.fact_contract is not None and submission_spec.args_model is not self.fact_contract.schema:
            return "invalid_submission_spec", None
        if (self.fact_contract is not None
                and submission_spec.response_model is not self.fact_contract.response_schema):
            return "invalid_submission_spec", None
        authorized = self.budget.authorize_tool_response(response)
        if not authorized.allowed:
            return authorized.reason, None
        reserved = self.budget.reserve_tool_call("submit_verdict")
        if not reserved.allowed:
            return reserved.reason, None
        delivered_ids = self.budget.delivered_for(response)
        if self.fact_contract is not None:
            review = self._review_v2(calls[0].arguments, delivered_ids)
        else:
            verdict = FactVerdict.model_validate(calls[0].arguments, strict=True)
            review = review_submission(verdict, self.ledger, delivered_evidence_ids=delivered_ids)
        return "accepted" if review.accepted else "validation_failed", review

    def _review_v2(self, arguments: dict[str, object], delivered_ids: frozenset[str]) -> SubmissionReview:
        """隔离可选建议附录，再严格校验V2核心及当前请求交付引用。"""
        from src.yuqing.harness.fact_v2_contract import DeliveredEvidenceV2, validate_fact_v2_core

        if self.delivery is None:
            raise ValueError("fact_v2_delivery_missing")
        delivered = {}
        for evidence_id in delivered_ids:
            entry = self.ledger.get(evidence_id)
            if entry is not None:
                delivered[evidence_id] = DeliveredEvidenceV2(entry.text, entry.read_end)
        checked = validate_fact_v2_core(arguments, delivered=delivered,
            limits=self.fact_contract.limits, plan_max_text_chars=self.fact_contract.plan_max_text_chars,
            mode=self.fact_contract.mode, assigned_issue_ids=self.fact_contract.assigned_issue_ids)
        verdict_json = checked.report.model_dump_json() if checked.report is not None else "{}"
        report_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return SubmissionReview(verdict_json=verdict_json,
                               issues=tuple(checked.validation.issues),
                               v2_validation=checked.validation, v2_report_json=report_json)

    def _feedback(self, reason: str, review: SubmissionReview | None) -> SubmissionFeedback:
        """只缩短模型可见问题，内部复核原文及问题清单保持完整。"""
        parsed_report = _trusted_v2_feedback_report(review) if review is not None else None
        issues = (fact_submission_feedback(review.issues, parsed_report=parsed_report)
                  if review is not None and reason != "tool_timeout"
                  else ("工具未完成复核；请按当前工具回执和 Schema 修正。",))
        feedback = SubmissionFeedback(accepted=reason == "accepted", issues=issues, truncated=False)
        while len(feedback.model_dump_json()) > self.budget.config.max_tool_result_chars:
            if not feedback.issues:
                raise ValueError("submission_feedback_limit")
            feedback = feedback.model_copy(update={"issues": feedback.issues[:-1], "truncated": True})
        if not feedback.issues and issues:
            feedback = feedback.model_copy(update={"issues": ("按 Schema 和已读证据规则修正。",)})
            if len(feedback.model_dump_json()) > self.budget.config.max_tool_result_chars:
                raise ValueError("submission_feedback_limit")
        return feedback

    def execute(self, response: ToolLLMResult, *, agent: str) -> SubmissionToolResult:
        """执行一次提交并写审计，审计失败抛出异常且禁止执行器继续使用。"""
        require_run_context()
        if self._audit_failed:
            raise RuntimeError("executor_audit_failed")
        started = monotonic()
        try:
            reason, review = self._evaluate(response, agent)
            latency = monotonic() - started
            if review is not None and latency > self.budget.config.tool_timeout_s["submit_verdict"]:
                reason = "tool_timeout"
            result = SubmissionToolResult(request_id=response.request_id,
                call_id=response.tool_calls[0].call_id if response.tool_calls else "", reason=reason,
                feedback=self._feedback(reason, review), review=review)
            spec = self.registry.get("submit_verdict")
            archive = write_archive("submission", result.model_dump(mode="json"))
            record_event("tool_call", {"tool": "submit_verdict", "archive": archive, "risk": spec.risk if spec else None,
                "decision": "allow" if review is not None else "deny",
                "args_digest": sha256(response.tool_calls[0].model_dump_json(include={"arguments"}).encode("utf-8")).hexdigest() if response.tool_calls else None,
                "request_id": result.request_id, "call_id": result.call_id,
                "accepted": result.feedback.accepted, "reason": reason,
                "status": "timeout" if reason == "tool_timeout" else "success" if review is not None else "denied",
                "errors": [_issue_audit_fields(issue) for issue in review.issues] if review else [],
                "verdict_sha256": sha256(review.verdict_json.encode("utf-8")).hexdigest() if review else None,
                "result_chars": len(result.feedback.model_dump_json()),
                "latency_ms": int(latency / timedelta(milliseconds=1).total_seconds())})
            return result
        except Exception:
            self._audit_failed = True
            self.budget.halt()
            raise


def _trusted_v2_feedback_report(review: SubmissionReview) -> object:
    """只解析Pydantic封套中的V2报告以生成数字反馈。"""
    if not review.v2_report_json:
        return None
    from src.yuqing.schemas import FactVerdictV2Envelope

    try:
        return FactVerdictV2Envelope.model_validate_json(review.v2_report_json, strict=True)
    except ValidationError:
        return None


def _issue_audit_fields(issue: str) -> dict[str, str]:
    """保留校验问题的字段位置和原因码，不记录提交正文。"""
    location, separator, detail = issue.partition(":")
    error_type = detail.strip().split(maxsplit=1)[0] if separator and detail.strip() else "invalid_value"
    return {"loc": location, "type": error_type}


class WorkerUnavailable(RuntimeError):
    """工作位或预算拒绝启动，异常文本仅为程序定义的原因。"""


class EvidenceWorkerScope:
    """连续题目共用一个后台工作位；不同专家可持有独立工作位。"""

    def __init__(self) -> None:
        """建立锁和任务记录，不预先创建后台线程。"""
        self._lock = Lock()
        self._future: Future[object] | None = None
        self._closed = False

    def has_pending_work(self) -> bool:
        """仅以真实任务完成状态判断后台是否仍占用。"""
        with self._lock:
            return self._future is not None and not self._future.done()

    def availability(self) -> Literal["ready", "busy", "closed"]:
        """在锁内读取工作位状态，不启动、取消或等待后台任务。"""
        with self._lock:
            if self._closed:
                return "closed"
            return "busy" if self._future is not None and not self._future.done() else "ready"

    def wait_until_idle(self, timeout_s: float) -> bool:
        """仅等待已有任务完成，不领次数、不取消任务、不登记迟到正文。"""
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (float, int)) or not 0 < timeout_s < float("inf"):
            raise ValueError("invalid_worker_wait_timeout")
        with self._lock:
            if self._closed:
                raise WorkerUnavailable("worker_scope_closed")
            future = self._future
        return future is None or bool(wait([future], timeout=timeout_s).done)

    def close(self) -> None:
        """永久拒绝新任务，不等待或声称已取消正在运行的线程。"""
        with self._lock:
            self._closed = True

    def submit(self, work: Callable[[], WorkerResult]) -> Future[WorkerResult]:
        """原子启动不带预算的内部计算；正式工具使用同锁下的预算检查。"""
        return self._submit(work, None)

    def submit_authorized(self, work: Callable[[], WorkerResult],
                          authorize: Callable[[], None]) -> Future[WorkerResult]:
        """在同一锁区内完成许可与启动，确保拒绝调用不会占用工作位。"""
        return self._submit(work, authorize)

    def _submit(self, work: Callable[[], WorkerResult],
                authorize: Callable[[], None] | None) -> Future[WorkerResult]:
        """检查占用、预占工具次数和发布任务在同一个临界区完成。"""
        with self._lock:
            if self._closed:
                raise WorkerUnavailable("worker_scope_closed")
            if self._future is not None and not self._future.done():
                raise WorkerUnavailable("tool_busy")
            if authorize is not None:
                authorize()
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(work)
                self._future = future
                return future
            finally:
                pool.shutdown(wait=False, cancel_futures=False)


class EvidenceToolExecutor:
    """单运行串行执行器；后台只计算，台账写入仅发生在及时成功后。"""

    def __init__(self, registry: ToolRegistry, ledger: EvidenceLedger, budget: BudgetTracker,
                 policy: PolicyConfig, index: BM25Index, store_path: Path | None = None,
                 worker_scope: EvidenceWorkerScope | None = None) -> None:
        """注入可信依赖，每题建立新实例，不共享台账或在同实例并行调用。"""
        self.registry, self.ledger, self.budget = registry, ledger, budget
        self.policy = PolicyConfig.model_validate(policy.model_dump())
        self.index, self.store_path = index, store_path
        self.worker_scope = worker_scope if worker_scope is not None else EvidenceWorkerScope()
        self._future: Future[list[EvidenceChunk]] | None = None
        self._identity: tuple[str, str, str, Path] | None = None
        self._audit_failed = False

    def _compute(self, args: SearchEvidenceArgs | ReadEvidenceArgs) -> list[EvidenceChunk]:
        """后台仅读取或检索，禁止更新台账和追踪。"""
        if isinstance(args, SearchEvidenceArgs):
            return search(args.query, self.index, k=args.k or self.budget.config.search_k,
                          path=self.store_path)
        entry = self.ledger.get(args.evidence_id)
        if entry is None:
            raise ValueError("unknown_evidence")
        return [entry.sources[0].model_copy(update={"text": entry.text})]

    def _payload(self, chunks: list[EvidenceChunk], ids: list[str], reading: bool) -> EvidenceToolPayload:
        """按完整JSON长度缩短原文前缀，保证结构完整并明确未返回尾部。"""
        config = self.budget.config
        limit = config.max_tool_result_chars if reading else config.gist_chars
        items = [ToolEvidence(evidence_id=eid, text=chunk.text[:limit],
                 truncated=len(chunk.text) > limit, read_end=min(len(chunk.text), limit) if reading else 0)
                 for eid, chunk in dict(zip(ids, chunks)).items()]
        payload = EvidenceToolPayload(status="success", reason="ok", items=items)
        items = payload.items
        while len(payload.model_dump_json()) > config.max_tool_result_chars:
            item = max(items, key=lambda value: len(value.text))
            if not item.text:
                if reading:
                    raise ValueError("result_limit")
                items.pop()
                continue
            position, low, high = items.index(item), 0, len(item.text)
            while low < high:
                middle = (low + high + 1) >> 1
                items[position] = item.model_copy(update={"text": item.text[:middle],
                    "truncated": True, "read_end": middle if reading else 0})
                if len(payload.model_dump_json()) <= config.max_tool_result_chars:
                    low = middle
                else:
                    high = middle - 1
            items[position] = item.model_copy(update={"text": item.text[:low],
                "truncated": True, "read_end": low if reading else 0})
        if reading and not items[0].text.strip():
            raise ValueError("result_limit")
        return payload

    def _run(self, args: SearchEvidenceArgs | ReadEvidenceArgs, name: str) -> tuple[EvidenceToolPayload, EvidenceLedger]:
        """限时取得结果并准备临时台账，迟到或格式失败不修改正式状态。"""
        def authorize() -> None:
            """工作位空闲后才计工具尝试，被拒绝不占次数。"""
            reserved = self.budget.reserve_tool_call(name)
            if not reserved.allowed:
                raise WorkerUnavailable(reserved.reason)

        self._future = self.worker_scope._submit(lambda: self._compute(args), authorize)
        chunks = self._future.result(timeout=self.budget.config.tool_timeout_s[name])
        reading = isinstance(args, ReadEvidenceArgs)
        staged = self.ledger.fork()
        ids = [args.evidence_id] if reading else staged.add(chunks)
        payload = self._payload(chunks, ids, reading)
        if reading:
            staged.record_read(args.evidence_id, payload.items[0].text)
        return payload, staged

    def _execute_allowed(self, request: ToolRequest) -> tuple[EvidenceToolPayload, EvidenceLedger | None]:
        """参数、后台占用和预算都通过后才计算，不在此提交正式台账。"""
        models = {"search_evidence": SearchEvidenceArgs, "read_evidence": ReadEvidenceArgs}
        if request.name not in models:
            return EvidenceToolPayload(status="denied", reason="unsupported_tool"), None
        args = models[request.name].model_validate(request.arguments, strict=True)
        if isinstance(args, SearchEvidenceArgs) and args.k is not None and args.k > self.budget.config.search_k:
            return EvidenceToolPayload(status="denied", reason="invalid_k"), None
        try:
            return self._run(args, request.name)
        except WorkerUnavailable as exc:
            return EvidenceToolPayload(status="denied", reason=str(exc)), None

    def _audit_evidence(self, payload: EvidenceToolPayload, staged: EvidenceLedger | None) -> list[dict[str, object]]:
        """保存编号、原文指纹、来源及实际返回范围，常规审计不写正文。"""
        if staged is None:
            return []
        records: list[dict[str, object]] = []
        for item in payload.items:
            entry = staged.get(item.evidence_id)
            if entry is None:
                raise ValueError("missing_staged_evidence")
            records.append({"evidence_id": item.evidence_id,
                "body_sha256": sha256(entry.text.encode("utf-8")).hexdigest(),
                "returned_sha256": sha256(item.text.encode("utf-8")).hexdigest(),
                "returned_start": 0, "returned_end": len(item.text), "read_end": item.read_end,
                "truncated": item.truncated, "sources": [{"chunk_id": source.chunk_id,
                    "source_doc": source.source_doc, "position": source.position} for source in entry.sources]})
        return records

    def _finish(self, request: ToolRequest, payload: EvidenceToolPayload, staged: EvidenceLedger | None,
                metadata: dict[str, object], started: float) -> ToolExecutionResult:
        """先构造并审计待返回结果，再提交台账；失败向外抛出且禁止继续执行。"""
        try:
            result = ToolExecutionResult(call_id=request.call_id, name=request.name, payload=payload)
            archive = write_archive("tool_result", {"request": request.model_dump(mode="json"),
                "result": result.model_dump(mode="json"), "ledger_update": "prepared" if staged else "none",
                "evidence": [staged.get(item.evidence_id).model_dump(mode="json")
                             for item in payload.items] if staged else []})
            record_event("tool_call", {**metadata, "archive": archive, "call_id": request.call_id, "tool": request.name,
                "reason": payload.reason, "status": payload.status,
                "latency_ms": int((monotonic() - started) / timedelta(milliseconds=1).total_seconds()),
                "result_chars": len(payload.model_dump_json()), "ledger_update": "prepared" if staged else "none",
                "evidence": self._audit_evidence(payload, staged)})
            if staged is not None:
                self.ledger.commit_from(staged)
            return result
        except Exception:
            self._audit_failed = True
            self.budget.halt()
            raise

    def execute(self, request: ToolRequest, *, agent: str) -> ToolExecutionResult:
        """绑定真实身份再许可执行，审计成功才让台账生效并返回结果。"""
        from src.yuqing.harness.policy import decide

        context = require_run_context()
        if self._audit_failed:
            raise RuntimeError("executor_audit_failed")
        identity = (context.run_id, context.sample_id, context.agent, context.trace_path)
        if self._identity is None:
            self._identity = identity
        started = monotonic()
        spec = self.registry.get(request.name)
        digest = sha256(request.model_dump_json(include={"arguments"}).encode("utf-8")).hexdigest()
        decision = decide(self.registry, agent, request.name, caller="model", config=self.policy)
        payload = EvidenceToolPayload(status="denied", reason=decision.reason)
        errors: list[dict[str, object]] = []
        staged: EvidenceLedger | None = None
        try:
            if identity != self._identity or context.agent != agent:
                payload = EvidenceToolPayload(status="denied", reason="run_identity_mismatch")
            elif decision.action == "allow":
                payload, staged = self._execute_allowed(request)
        except ValidationError as exc:
            errors = [{"loc": list(item["loc"]), "type": item["type"]} for item in exc.errors()]
            payload = EvidenceToolPayload(status="denied", reason="invalid_arguments")
        except TimeoutError:
            payload = EvidenceToolPayload(status="timeout", reason="tool_timeout")
        except Exception as exc:
            logger.error("证据工具失败：%s", type(exc).__name__)
            errors = [{"type": type(exc).__name__}]
            payload = EvidenceToolPayload(status="error", reason="tool_error")
        return self._finish(request, payload, staged, {"risk": spec.risk if spec else None,
            "args_digest": digest, "decision": decision.action, "error": errors}, started)
