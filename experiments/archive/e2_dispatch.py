"""单槽位调度控制，不包含真实Agent适配、CLI或付费许可。"""

import logging
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict

from src.yuqing.eval.dual_budget import DualBudgetCoordinator
from experiments.archive.e2_artifacts import E2CheckpointRow
from experiments.archive.e2_checkpoint import PlannedRun
from experiments.archive.e2_journal import AnswerJournal
from experiments.archive.e2_resume import ResumeAudit
from experiments.archive.e2_workspace import E2Workspace
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.harness.budget import BudgetTracker

logger = logging.getLogger(__name__)
CoordinatorFactory = Callable[[BudgetTracker], DualBudgetCoordinator]
SlotExecutor = Callable[[PlannedRun, CoordinatorFactory], E2CheckpointRow]


class DispatchResult(BaseModel):
    """一次控制动作的结果，不代表事实正确或整场正式评测完成。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["saved", "blocked", "finished", "failed"]
    reason: str
    run_id: str | None = None
    audit: ResumeAudit | None = None


def _fault(session: ExperimentSession, reason: str, run_id: str | None) -> DispatchResult:
    """故障停止整场，不重试/退款；停账未确认则关闭会话并明确报告。"""
    try:
        if not session.snapshot().halted:
            session.halt(reason)
    except Exception as exc:
        logger.error("调度停账未确认 type=%s", type(exc).__name__)
        try:
            session.close()
        except Exception as close_exc:
            logger.error("调度会话关闭未确认 type=%s", type(close_exc).__name__)
        return DispatchResult(status="failed", reason="halt_unconfirmed", run_id=run_id)
    return DispatchResult(status="failed", reason=reason, run_id=run_id)


def dispatch_next(workspace: E2Workspace, journal: AnswerJournal, session: ExperimentSession, *,
                   check_conditions: Callable[[], None], execute_slot: SlotExecutor) -> DispatchResult:
    """核验后仅执行下一槽位，保存失败停账，检查报告不自动许可真实模型。"""
    try:
        workspace.require_open()
        journal.require_workspace(workspace)
        history = session.load_history()
        if dict(history.manifest) != workspace.manifest:
            return DispatchResult(status="blocked", reason="session_manifest_mismatch")
    except Exception as exc:
        logger.error("调度绑定检查失败 type=%s", type(exc).__name__)
        return DispatchResult(status="blocked", reason="dispatch_binding_error")
    run_id: str | None = None
    try:
        if not callable(check_conditions) or not callable(execute_slot):
            raise ValueError("invalid_dispatch_dependencies")
        check_conditions()
        before = journal.audit_saved_answers(history)
        if not before.consistent:
            return DispatchResult(status="blocked", reason="resume_audit_blocked", audit=before)
        if before.next_run_id is None:
            return DispatchResult(status="finished", reason="plan_finished", audit=before)
        slot = next(slot for slot in workspace.plan if slot.run_id == before.next_run_id)
        run_id = slot.run_id
        row = execute_slot(slot, session.coordinator_factory())
        if (row.run.sample_id, row.run.group_id, row.run.run_id) != (slot.sample_id, slot.group_id, slot.run_id):
            raise ValueError("executor_returned_wrong_slot")
        journal.append_answer(row)
        check_conditions()
        after = journal.audit_saved_answers(session.load_history())
        if not after.consistent:
            stopped = _fault(session, "post_run_audit_blocked", run_id)
            if stopped.reason == "halt_unconfirmed":
                return stopped
            return DispatchResult(status="blocked", reason="post_run_audit_blocked", run_id=run_id, audit=after)
        return DispatchResult(status="saved", reason="answer_saved", run_id=run_id, audit=after)
    except BaseException as exc:
        logger.error("单槽位调度失败 type=%s", type(exc).__name__)
        stopped = _fault(session, "dispatch_execution_or_storage_failed", run_id)
        if not isinstance(exc, Exception):
            raise
        return stopped
