"""只读核验循环Trace的身份、终局与请求登记，不判断消费或许可执行。"""

from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from experiments.archive.e2_artifacts import Fingerprint
from experiments.archive.e2_checkpoint import CheckpointRun, RequestId
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.settings import PROJECT_ROOT

Outcome = Literal["submitted_judgment", "no_valid_result"]
SINGLE_GROUPS = {"single_shot", "single_v3", "single_v4"}
LOOP_GROUPS = {"iterative", "iterative_v2"}


class TraceAudit(BaseModel):
    """核验报告不代表事实正确、请求已发送或钱包可继续消费。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    consistent: bool
    issues: tuple[str, ...] = ()
    outcome_class: Outcome | None = None
    reason: str | None = None
    request_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()


def _event(value: Mapping[str, object] | TraceEvent) -> TraceEvent:
    """重新严格检查事件，不静默丢弃额外栏目或转换布尔步数。"""
    data = value.model_dump() if isinstance(value, TraceEvent) else dict(value)
    if set(data) - TraceEvent.model_fields.keys():
        raise ValueError("unexpected_trace_fields")
    result = TraceEvent.model_validate(data, strict=True)
    if result.step < 0:
        raise ValueError("invalid_trace_step")
    return result


def _requests(events: Sequence[TraceEvent]) -> tuple[list[str], list[str], list[str]]:
    """只核对尝试与回执编号，不把缺回执推断为免费或实际发送。"""
    attempts: list[str] = []
    receipts: list[str] = []
    issues: list[str] = []
    for event in events:
        if event.type not in {"llm_request", "llm_call"}:
            continue
        identifier = TypeAdapter(RequestId).validate_python(event.data.get("request_id"), strict=True)
        if event.type == "llm_request":
            if identifier in attempts:
                issues.append("duplicate_trace_request")
            attempts.append(identifier)
        else:
            if identifier not in attempts:
                issues.append("receipt_without_prior_request")
            if identifier in receipts:
                issues.append("duplicate_trace_receipt")
            receipts.append(identifier)
    return attempts, receipts, issues


def _audit_trace(run: CheckpointRun, events: Sequence[Mapping[str, object] | TraceEvent], *,
                 expected_agent: str, expected_outcome: Outcome, expected_reason: str, single_shot: bool) -> TraceAudit:
    """复用两组的身份与终局规则，单次只能由明确入口且fact身份认证。"""
    try:
        checked = CheckpointRun.model_validate(run)
        if not isinstance(expected_agent, str) or not expected_agent.strip() or not isinstance(expected_reason, str) or not expected_reason.strip():
            raise ValueError("invalid_expected_trace_identity")
        TypeAdapter(Outcome).validate_python(expected_outcome, strict=True)
        parsed = [_event(value) for value in events]
        attempts, receipts, issues = _requests(parsed)
    except (ValueError, TypeError, AttributeError, ValidationError):
        return TraceAudit(consistent=False, issues=("invalid_trace_input",))
    allowed_groups = SINGLE_GROUPS if single_shot else LOOP_GROUPS
    if checked.group_id not in allowed_groups or checked.execution_status != "finished" or (single_shot and expected_agent != "fact"):
        issues.append("unsupported_trace_run")
    if any((e.run_id, e.sample_id, e.agent) != (checked.run_id, checked.sample_id, expected_agent) for e in parsed):
        issues.append("trace_identity_mismatch")
    terminals = [index for index, event in enumerate(parsed) if event.type == "run_outcome"]
    outcome, reason = None, None
    if len(terminals) != 1:
        issues.append("terminal_count_mismatch")
    else:
        terminal = parsed[terminals[0]].data
        outcome, reason = terminal.get("outcome_class"), terminal.get("reason")
        if outcome != expected_outcome or reason != expected_reason:
            issues.append("terminal_outcome_mismatch")
        if terminals[0] != len(parsed) - 1:
            issues.append("events_after_terminal")
    if len(set(checked.request_ids)) != len(checked.request_ids) or set(attempts) != set(checked.request_ids):
        issues.append("trace_request_registration_mismatch")
    if outcome == "submitted_judgment" and not attempts:
        issues.append("submitted_without_request")
    return TraceAudit(consistent=not issues, issues=tuple(dict.fromkeys(issues)),
        outcome_class=outcome if isinstance(outcome, str) and outcome in {"submitted_judgment", "no_valid_result"} else None,
        reason=reason if isinstance(reason, str) else None, request_ids=tuple(attempts), receipt_ids=tuple(receipts))


def audit_run_trace(run: CheckpointRun, events: Sequence[Mapping[str, object] | TraceEvent], *,
                    expected_agent: str, expected_outcome: Outcome, expected_reason: str) -> TraceAudit:
    """保持旧入口仅认证循环，旧单次或stop不能被自动放行。"""
    return _audit_trace(run, events, expected_agent=expected_agent, expected_outcome=expected_outcome,
        expected_reason=expected_reason, single_shot=False)


def audit_single_trace(run: CheckpointRun, events: Sequence[Mapping[str, object] | TraceEvent], *,
                       expected_outcome: Outcome, expected_reason: str) -> TraceAudit:
    """明确认证新单次fact终局，不接受仅含stop的旧日志。"""
    return _audit_trace(run, events, expected_agent="fact", expected_outcome=expected_outcome,
        expected_reason=expected_reason, single_shot=True)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝任意层级重复JSON字段。"""
    data: dict[str, object] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("duplicate_trace_key")
        data[key] = value
    return data


def _invalid_constant(value: str) -> None:
    """拒绝JSON非标准NaN和无穷值，不把它们带入核验。"""
    raise ValueError("invalid_trace_constant")


def read_and_audit_trace(path: Path, *, expected_sha256: str, run: CheckpointRun,
                         expected_agent: str, expected_outcome: Outcome, expected_reason: str,
                         single_shot: bool = False) -> TraceAudit:
    """只读取项目traces内文件一次，对同一份字节核验指纹并解析全部行。"""
    audit, _ = read_and_audit_trace_events(path, expected_sha256=expected_sha256,
        run=run, expected_agent=expected_agent, expected_outcome=expected_outcome,
        expected_reason=expected_reason, single_shot=single_shot)
    return audit


def read_and_audit_trace_events(path: Path, *, expected_sha256: str, run: CheckpointRun,
                                expected_agent: str, expected_outcome: Outcome,
                                expected_reason: str, single_shot: bool = False
                                ) -> tuple[TraceAudit, tuple[TraceEvent, ...]]:
    """返回同一份严格读取字节经身份、终局和指纹校验后的事件。"""
    try:
        if type(single_shot) is not bool or (single_shot and expected_agent != "fact"):
            raise ValueError("invalid_trace_mode")
        TypeAdapter(Fingerprint).validate_python(expected_sha256, strict=True)
        project = PROJECT_ROOT.resolve()
        root, target = (project / "traces").resolve(), Path(path).resolve()
        if not root.is_relative_to(project) or target == root or not target.is_relative_to(root):
            return TraceAudit(consistent=False, issues=("trace_path_escape",)), ()
        data = target.read_bytes()
        if sha256(data).hexdigest() != expected_sha256:
            return TraceAudit(consistent=False, issues=("trace_digest_mismatch",)), ()
        events = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            for line in data.decode("utf-8").splitlines()]
        audit = _audit_trace(run, events, expected_agent=expected_agent,
            expected_outcome=expected_outcome, expected_reason=expected_reason, single_shot=single_shot)
        if not audit.consistent:
            return audit, ()
        return audit, tuple(_event(value) for value in events)
    except (OSError, ValueError, TypeError, AttributeError, ValidationError):
        return TraceAudit(consistent=False, issues=("invalid_trace_file",)), ()
