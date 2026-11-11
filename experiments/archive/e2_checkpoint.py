"""纯核对E2答卷与已校验账本历史，不写文件、不许可模型执行。"""

from collections import Counter
from hashlib import sha256
import json
from typing import Annotated, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from src.yuqing.eval.experiment_budget_recovery import RecoveredRequest, replay_history
from src.yuqing.eval.experiment_budget_store import LedgerHistory

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:/-]{1,128}$")]
RequestId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
GroupId = Literal["single_shot", "iterative", "single_v3", "single_v4", "iterative_v2"]
PAIR_GROUPS = ("single_shot", "iterative")
THREE_ARM_GROUPS = ("single_v3", "single_v4", "iterative_v2")
FINAL_RECOVERY_FREEZE_SHA256 = "3d97b52c02c27b2ab2a6f5bdf89388a34152cc121932ed63e08b8f92205340d6"
FINAL_RECOVERY_PLAN_SHA256 = "e007ab455dfead28ae362292101551e1c32be219504bc9dee2f8d9227cb386d5"


class PlannedRun(BaseModel):
    """本场固定的一题一组一运行编号，不含模型判断或成绩。"""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, revalidate_instances="always")
    sample_id: Identifier
    group_id: GroupId
    run_id: RequestId


class CheckpointRun(PlannedRun):
    """已保存运行元信息，finished包括有答案或无合格结果。"""

    execution_status: Literal["finished", "not_executed"]
    request_ids: tuple[RequestId, ...] = ()


class CheckpointIssue(BaseModel):
    """只包含程序原因码与关联编号，不记录模型正文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    code: str
    run_id: str | None = None
    request_id: str | None = None


class CheckpointAudit(BaseModel):
    """元信息核对结果；计数不是成绩，一致也不是付费运行许可。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    issues: tuple[CheckpointIssue, ...] = ()
    finished_runs: int = 0
    completed_pairs: int = 0
    half_pairs: tuple[str, ...] = ()
    next_run_id: str | None = None

    @computed_field
    @property
    def consistent(self) -> bool:
        """无阻塞仅表示本卡检查一致，不代表Trace、行指纹和合同已验证。"""
        return not self.issues


def _layout_issues(plan: Sequence[PlannedRun], rows: Sequence[CheckpointRun]) -> list[CheckpointIssue]:
    """只接受原双组或冻结三臂的固定顺序，答卷必须是计划前缀。"""
    issues: list[CheckpointIssue] = []
    ids = [item.run_id for item in plan]
    samples = list(dict.fromkeys(item.sample_id for item in plan))
    groups = tuple(dict.fromkeys(item.group_id for item in plan))
    allowed = groups in (PAIR_GROUPS, THREE_ARM_GROUPS)
    expected = [(sample, group) for sample in samples for group in groups]
    if not plan or not allowed or len(set(ids)) != len(ids) or [
            (p.sample_id, p.group_id) for p in plan] != expected:
        issues.append(CheckpointIssue(code="invalid_pair_plan"))
    if [row.run_id for row in rows] != ids[:len(rows)] or len(rows) > len(plan):
        issues.append(CheckpointIssue(code="row_order_mismatch"))
    slots = {p.run_id: p for p in plan}
    stopped = False
    for row in rows:
        slot = slots.get(row.run_id)
        if slot is None or (slot.sample_id, slot.group_id) != (row.sample_id, row.group_id):
            issues.append(CheckpointIssue(code="row_identity_mismatch", run_id=row.run_id))
        if stopped and row.execution_status == "finished":
            issues.append(CheckpointIssue(code="finished_after_not_executed", run_id=row.run_id))
        stopped = stopped or row.execution_status == "not_executed"
    return issues


def _final_recovery_layout_issues(plan: Sequence[PlannedRun], rows: Sequence[CheckpointRun], *,
                                  source_sha256: str, plan_sha256: str) -> list[CheckpointIssue]:
    """只放行固定来源哈希绑定的334循环起始与其后完整题对。"""
    issues: list[CheckpointIssue] = []
    semantic = [(slot.sample_id, slot.group_id) for slot in plan]
    canonical = json.dumps([slot.model_dump(mode="json") for slot in plan], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    samples = list(dict.fromkeys(sample for sample, _ in semantic))
    expected = ([(samples[0], "iterative")] +
        [(sample, group) for sample in samples[1:] for group in ("single_shot", "iterative")]) if samples else []
    if (source_sha256 != FINAL_RECOVERY_FREEZE_SHA256 or sha256(canonical).hexdigest() != plan_sha256 or
            len(plan) != 157 or semantic != expected or not samples or samples[0] != "chef:dev:334" or
            len({slot.run_id for slot in plan}) != len(plan)):
        issues.append(CheckpointIssue(code="invalid_final_recovery_plan"))
    if [row.run_id for row in rows] != [slot.run_id for slot in plan[:len(rows)]] or len(rows) > len(plan):
        issues.append(CheckpointIssue(code="row_order_mismatch"))
    slots = {slot.run_id: slot for slot in plan}
    for row in rows:
        slot = slots.get(row.run_id)
        if slot is None or (slot.sample_id, slot.group_id) != (row.sample_id, row.group_id):
            issues.append(CheckpointIssue(code="row_identity_mismatch", run_id=row.run_id))
    return issues


def _matches(request: RecoveredRequest, run: PlannedRun) -> bool:
    """钱包必须明确记录完整运行身份，缺字段不能靠请求编号猜测。"""
    return all(request.identity.get(key) == getattr(run, key) for key in ("sample_id", "group_id", "run_id"))


def _request_issues(plan: Sequence[PlannedRun], rows: Sequence[CheckpointRun],
                    requests: Mapping[str, RecoveredRequest]) -> list[CheckpointIssue]:
    """账本到答卷逐项检查，收费或取消但无答卷都禁止静默补跑。"""
    issues: list[CheckpointIssue] = []
    slots, answers = {p.run_id: p for p in plan}, {r.run_id: r for r in rows}
    for identifier, request in requests.items():
        run_id = str(request.identity.get("run_id", ""))
        slot, row = slots.get(run_id), answers.get(run_id)
        if slot is None or not _matches(request, slot):
            code = "request_identity_mismatch"
        elif row is None:
            code = "settled_run_without_row" if request.state == "settled" else "ledger_run_without_row"
        elif row.execution_status == "not_executed":
            code = "not_executed_has_request"
        elif identifier not in row.request_ids:
            code = "request_missing_from_row"
        else:
            continue
        issues.append(CheckpointIssue(code=code, run_id=run_id, request_id=identifier))
    return issues


def _row_request_issues(rows: Sequence[CheckpointRun], requests: Mapping[str, RecoveredRequest]) -> list[CheckpointIssue]:
    """反向核对答卷声明的请求，零调用完成必须另查终局Trace。"""
    issues: list[CheckpointIssue] = []
    seen: set[str] = set()
    for row in rows:
        if row.execution_status == "not_executed" and row.request_ids:
            issues.append(CheckpointIssue(code="not_executed_has_request", run_id=row.run_id))
        for identifier in row.request_ids:
            request = requests.get(identifier)
            code = ("duplicate_row_request" if identifier in seen else "unknown_row_request" if request is None
                    else "row_request_identity_mismatch" if not _matches(request, row) else None)
            seen.add(identifier)
            if code:
                issues.append(CheckpointIssue(code=code, run_id=row.run_id, request_id=identifier))
        if row.execution_status == "finished" and not any(
            identifier in requests and requests[identifier].state != "cancelled" for identifier in row.request_ids):
            issues.append(CheckpointIssue(code="zero_call_run_requires_trace", run_id=row.run_id))
    return issues


def audit_e2_checkpoint(plan: Sequence[PlannedRun], rows: Sequence[CheckpointRun], history: LedgerHistory,
                        expected_manifest: Mapping[str, object], *, recovery_source_sha256: str | None = None,
                        recovery_plan_sha256: str | None = None) -> CheckpointAudit:
    """仅接受存储load已核验的历史，复用语义恢复核对元信息，不执行或保存。"""
    try:
        clean_plan = [PlannedRun.model_validate(p) for p in plan]
        clean_rows = [CheckpointRun.model_validate(r) for r in rows]
        state = replay_history(history, expected_manifest)
    except (ValueError, ValidationError, TypeError, AttributeError):
        return CheckpointAudit(issues=(CheckpointIssue(code="invalid_checkpoint_input"),))
    if recovery_source_sha256 is None:
        issues = _layout_issues(clean_plan, clean_rows)
    elif recovery_plan_sha256 is None:
        issues = [CheckpointIssue(code="invalid_final_recovery_plan")]
    else:
        issues = _final_recovery_layout_issues(clean_plan, clean_rows,
            source_sha256=recovery_source_sha256, plan_sha256=recovery_plan_sha256)
    if issues:
        return CheckpointAudit(issues=tuple(issues))
    if state.halt_reason is not None:
        issues.append(CheckpointIssue(code="ledger_halted"))
    for identifier, request in state.requests.items():
        if request.state in {"prepared", "reserved", "dispatched", "unknown"}:
            issues.append(CheckpointIssue(code="unresolved_request", request_id=identifier))
    issues.extend(_request_issues(clean_plan, clean_rows, state.requests))
    issues.extend(_row_request_issues(clean_rows, state.requests))
    finished = [r for r in clean_rows if r.execution_status == "finished"]
    counts = Counter(r.sample_id for r in finished)
    planned_group_count = len(set(slot.group_id for slot in clean_plan))
    next_id = clean_plan[len(finished)].run_id if not issues and len(finished) < len(clean_plan) else None
    return CheckpointAudit(issues=tuple(dict.fromkeys(issues)), finished_runs=len(finished),
        completed_pairs=sum(count == planned_group_count for count in counts.values()),
        half_pairs=tuple(sample for sample, count in counts.items() if 0 < count < planned_group_count),
        next_run_id=next_id)
