"""组合答卷、终局和钱包核验，不执行恢复或许可付费。"""

from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from experiments.archive.e2_artifacts import E2CheckpointRow, Fingerprint, validate_checkpoint_artifacts
from experiments.archive.e2_checkpoint import CheckpointAudit, CheckpointIssue, PlannedRun, audit_e2_checkpoint
from src.yuqing.eval.experiment_budget_store import LedgerHistory
from src.yuqing.eval.experiment_budget_recovery import replay_history
from experiments.archive.e2_trace import Outcome, read_and_audit_trace
from src.yuqing.settings import PROJECT_ROOT


class TraceReference(BaseModel):
    """答卷payload.trace内受整行指纹保护的终局文件引用。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    path: str = Field(min_length=1)
    sha256: Fingerprint
    outcome_class: Outcome
    reason: str = Field(min_length=1)


class ResumeAudit(CheckpointAudit):
    """三层只读核验报告；next_run_id仅为候选，不是调用模型许可。"""


def _trace_checks(items: Sequence[E2CheckpointRow]) -> tuple[list[CheckpointIssue], set[str], set[str]]:
    """逐行验证引用和实际终局，只有合格无回执失败可补足零调用证明。"""
    issues: list[CheckpointIssue] = []
    zero_verified: set[str] = set()
    receipts: set[str] = set()
    for item in items:
        run = item.run
        try:
            if run.execution_status == "not_executed":
                if "trace" in item.payload or item.payload.get("success", False) is not False:
                    raise ValueError("unexecuted_trace_or_success")
                continue
            ref = TraceReference.model_validate(item.payload.get("trace"))
            success = item.payload.get("success")
            if type(success) is not bool or success != (ref.outcome_class == "submitted_judgment"):
                raise ValueError("trace_success_mismatch")
            if Path(ref.path).is_absolute():
                raise ValueError("absolute_trace_reference")
            single = run.group_id in {"single_shot", "single_v3", "single_v4"}
            report = read_and_audit_trace(PROJECT_ROOT / ref.path, expected_sha256=ref.sha256, run=run,
                expected_agent="fact" if single else "fact_investigator",
                expected_outcome=ref.outcome_class, expected_reason=ref.reason, single_shot=single)
            issues.extend(CheckpointIssue(code=code, run_id=run.run_id) for code in report.issues)
            receipts.update(report.receipt_ids)
            if report.consistent and ref.outcome_class == "no_valid_result" and not report.receipt_ids:
                zero_verified.add(run.run_id)
        except (ValueError, TypeError, AttributeError):
            issues.append(CheckpointIssue(code="invalid_trace_reference", run_id=run.run_id))
    return issues, zero_verified, receipts


def audit_e2_resume(plan: Sequence[PlannedRun], rows: Sequence[Mapping[str, object] | E2CheckpointRow | str],
                     history: LedgerHistory, expected_manifest: Mapping[str, object], *,
                     recovery_source_sha256: str | None = None,
                     recovery_plan_sha256: str | None = None) -> ResumeAudit:
    """完整批次先校验答卷，再查Trace与已load的钱包，仅解除对应零调用待验证项。"""
    try:
        fixed_plan = [PlannedRun.model_validate(slot) for slot in plan]
        items = validate_checkpoint_artifacts(rows, experiment_id=expected_manifest["experiment_id"],
            condition_sha256=expected_manifest["conditions_fingerprint"])
        trace_issues, zero_verified, receipts = _trace_checks(items)
        wallet = audit_e2_checkpoint(fixed_plan, [item.run for item in items], history, expected_manifest,
            recovery_source_sha256=recovery_source_sha256, recovery_plan_sha256=recovery_plan_sha256)
        state = replay_history(history, expected_manifest)
        for identifier, request in state.requests.items():
            if request.state == "settled" and identifier not in receipts:
                trace_issues.append(CheckpointIssue(code="settled_request_missing_trace_receipt", request_id=identifier))
            if request.state == "cancelled" and identifier in receipts:
                trace_issues.append(CheckpointIssue(code="cancelled_request_has_trace_receipt", request_id=identifier))
    except (ValueError, TypeError, AttributeError, KeyError):
        return ResumeAudit(issues=(CheckpointIssue(code="invalid_resume_input"),))
    issues = trace_issues + [issue for issue in wallet.issues if not (
        issue.code == "zero_call_run_requires_trace" and issue.run_id in zero_verified)]
    candidate = fixed_plan[wallet.finished_runs].run_id if not issues and wallet.finished_runs < len(fixed_plan) else None
    return ResumeAudit(issues=tuple(dict.fromkeys(issues)), finished_runs=wallet.finished_runs,
        completed_pairs=wallet.completed_pairs, half_pairs=wallet.half_pairs, next_run_id=candidate)
