"""用独立运行身份保存事件，按运行汇总用量而不依赖进程计数。"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from typing import Iterator, Literal, Mapping
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError

from src.yuqing.settings import PROJECT_ROOT


class RunContext(BaseModel):
    """由程序提供的题目、专家、步骤与追踪目录。"""

    run_id: str
    sample_id: str
    agent: str
    step: int = Field(default=0, ge=0)
    trace_root: Path = PROJECT_ROOT / "traces"
    trace_day: date = Field(default_factory=lambda: datetime.now(timezone.utc).date())

    @property
    def trace_path(self) -> Path:
        """验证运行编号，防止编号成为任意路径。"""
        identifier = UUID(self.run_id).hex
        return self.trace_root / self.trace_day.isoformat() / f"{identifier}.jsonl"


class RunUsage(BaseModel):
    """本次运行的已知消耗及是否存在未知消耗。"""

    calls: int = 0
    successful_responses: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_cny: float | None = 0
    usage_known: bool = True
    known_cost_subtotal_cny: float = 0
    issues: list[str] = Field(default_factory=list)


class UsageReceipt(BaseModel):
    """严格校验单次响应的消耗，不把字符串、布尔值或负数转换成用量。"""

    input_tokens: int = Field(strict=True, ge=0)
    output_tokens: int = Field(strict=True, ge=0)
    cost_cny: float | None = Field(strict=True, ge=0, allow_inf_nan=False)


class TraceEvent(BaseModel):
    """一个独立事件；数据只接收程序挑选的元信息。"""

    ts: str
    run_id: str
    sample_id: str
    agent: str
    step: int
    type: str
    data: dict[str, object]
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_cny: float | None = None
    latency_ms: int | None = None


_context: ContextVar[RunContext | None] = ContextVar("yuqing_run", default=None)
_write_lock = Lock()


@contextmanager
def use_run_context(context: RunContext) -> Iterator[RunContext]:
    """进入独立上下文，正常退出或异常时都恢复上层身份。"""
    context.trace_path.parent.mkdir(parents=True, exist_ok=True)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def require_run_context() -> RunContext:
    """新工具路径必须绑定运行身份，不能悄悄丢弃审计事件。"""
    context = _context.get()
    if context is None:
        raise RuntimeError("missing_run_context")
    return context.model_copy(deep=True)


def write_archive(kind: Literal["model_request", "model_response", "tool_result", "submission"],
                  payload: Mapping[str, object]) -> dict[str, str]:
    """独占写入本运行归档，完整持久化后才发布JSON；引用须由调用方写入审计。"""
    context = require_run_context()
    if kind not in {"model_request", "model_response", "tool_result", "submission"}:
        raise ValueError("invalid_archive_kind")
    root = context.trace_root.resolve()
    folder = context.trace_path.with_suffix("").resolve()
    if not folder.is_relative_to(root) or folder == root:
        raise ValueError("archive_path_escape")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{UUID(uuid4().hex).hex}_{kind}.json"
    temporary = path.with_suffix(".partial")
    document = {"kind": kind, "run_id": context.run_id, "sample_id": context.sample_id,
                "agent": context.agent, "step": context.step, "payload": dict(payload)}
    data = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "nt":
        temporary.rename(path)
    else:
        os.link(temporary, path)
        temporary.unlink()
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(data).hexdigest()}


def record_event(event_type: str, data: Mapping[str, object]) -> None:
    """有运行上下文时追加事件，未绑定身份的旧入口保持兼容。"""
    context = _context.get()
    if context is None:
        return
    event = TraceEvent(ts=datetime.now(timezone.utc).isoformat(), run_id=context.run_id,
                       sample_id=context.sample_id, agent=context.agent, step=context.step,
                       type=event_type, data=dict(data), tokens_in=data.get("input_tokens"),
                       tokens_out=data.get("output_tokens"), cost_cny=data.get("cost_cny"),
                       latency_ms=data.get("latency_ms"))
    with _write_lock, context.trace_path.open("a", encoding="utf-8") as stream:
        stream.write(event.model_dump_json() + "\n")
        stream.flush()


def _mark_usage_unknown(usage: RunUsage, issue: str) -> None:
    """保留已知小计，同时明确总用量或总费用不能确认。"""
    usage.usage_known = False
    usage.cost_cny = None
    if issue not in usage.issues:
        usage.issues.append(issue)


def _collect_usage_events(context: RunContext) -> tuple[RunUsage, dict[str, int], dict[str, list[TraceEvent]]]:
    """按请求编号收集事件，并核对运行身份和请求记录完整性。"""
    usage = RunUsage()
    requests: dict[str, int] = {}
    responses: dict[str, list[TraceEvent]] = {}
    if not context.trace_path.exists():
        return usage, requests, responses
    for line in context.trace_path.read_text(encoding="utf-8").splitlines():
        event = TraceEvent.model_validate_json(line)
        if event.run_id != context.run_id or event.sample_id != context.sample_id or event.agent != context.agent:
            raise ValueError("追踪文件混入其他运行身份")
        if event.type not in {"llm_request", "llm_call"}:
            continue
        if event.type == "llm_request":
            usage.calls += 1
        identifier = event.data.get("request_id")
        if not isinstance(identifier, str) or not identifier.strip():
            _mark_usage_unknown(usage, "missing_request_id")
            continue
        if event.type == "llm_request":
            requests[identifier] = requests.get(identifier, 0) + 1
        else:
            if identifier not in requests:
                _mark_usage_unknown(usage, "response_before_request")
            responses.setdefault(identifier, []).append(event)
    return usage, requests, responses


def _add_usage_receipt(usage: RunUsage, data: dict[str, object]) -> None:
    """只累加可信响应，费用缺失时保留可确认的 token 与费用小计。"""
    if data.get("usage_known") is not True:
        _mark_usage_unknown(usage, "usage_unknown")
        return
    try:
        receipt = UsageReceipt.model_validate(data)
    except ValidationError:
        _mark_usage_unknown(usage, "invalid_usage")
        return
    usage.successful_responses += 1
    usage.input_tokens += receipt.input_tokens
    usage.output_tokens += receipt.output_tokens
    if receipt.cost_cny is None:
        usage.cost_cny = None
        if "cost_unknown" not in usage.issues:
            usage.issues.append("cost_unknown")
    else:
        usage.known_cost_subtotal_cny += receipt.cost_cny
        if usage.cost_cny is not None:
            usage.cost_cny += receipt.cost_cny


def _settle_usage_request(usage: RunUsage, count: int, responses: list[TraceEvent]) -> None:
    """一份请求只结算一次；冲突回执或重复请求不产生可信消耗。"""
    if count != 1:
        _mark_usage_unknown(usage, "duplicate_request")
        return
    if not responses:
        _mark_usage_unknown(usage, "missing_response")
        return
    first = responses[0].data
    if len(responses) > 1:
        _mark_usage_unknown(usage, "duplicate_response")
        if any(event.data != first for event in responses[1:]):
            _mark_usage_unknown(usage, "conflicting_response")
            return
    _add_usage_receipt(usage, first)


def get_run_usage(run_id: str) -> RunUsage:
    """按 request_id 配对汇总，异常回执不重复收费且总消耗保持未知。"""
    context = _context.get()
    if context is None or context.run_id != run_id:
        raise ValueError("必须在对应运行上下文内汇总用量")
    usage, requests, responses = _collect_usage_events(context)
    if responses.keys() - requests.keys():
        _mark_usage_unknown(usage, "orphan_response")
    for identifier, count in requests.items():
        _settle_usage_request(usage, count, responses.get(identifier, []))
    return usage
