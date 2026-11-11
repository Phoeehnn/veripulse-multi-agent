"""事实 V2 离线配对调度，复用E2账本而不接通模型网关。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from experiments.archive.e2_checkpoint import GroupId, PAIR_GROUPS, THREE_ARM_GROUPS
from experiments.archive.e2_dispatch import DispatchResult, SlotExecutor, dispatch_next
from experiments.archive.e2_journal import AnswerJournal
from experiments.archive.e2_trace import read_and_audit_trace_events
from experiments.archive.e2_workspace import E2Workspace, open_e2_workspace
from src.yuqing.eval.experiment_session import ExperimentSession, build_experiment_manifest
from src.yuqing.settings import (PROJECT_ROOT, get_cost_config, get_evaluation_config)
from src.yuqing.eval.fact_v2_experiment import (FactV2RunConditions,
    assert_fact_v2_conditions_unchanged, collect_fact_v2_conditions, enable_fact_v2_three_arm,
    fact_v2_budget_ceiling, preflight_fact_v2)


@dataclass
class FactV2OfflineRun:
    """一次明确标记为Mock/离线的V2工作区，不包含模型执行器。"""

    conditions: FactV2RunConditions
    workspace: E2Workspace
    session: ExperimentSession
    journal: AnswerJournal

    def close(self) -> None:
        """先关闭共享账本，再释放工作区锁，保留所有原始文件。"""
        try:
            self.session.close()
        finally:
            self.workspace.close()


_PROVIDER_PERMIT_SEAL = object()


@dataclass(frozen=True, init=False)
class _ProviderPermit:
    """只由成功的许可门卫签发，绑定本次V2条件指纹。"""

    conditions_sha256: str
    _seal: object

    def __init__(self, conditions_sha256: str, seal: object) -> None:
        """只接受模块许可门卫持有的内部封印。"""
        if seal is not _PROVIDER_PERMIT_SEAL:
            raise ValueError("fact_v2_provider_permit_invalid")
        object.__setattr__(self, "conditions_sha256", conditions_sha256)
        object.__setattr__(self, "_seal", seal)


@dataclass(frozen=True)
class FactV2ProviderRun:
    """通过正式提示词许可门卫后创建的运行及其不可伪造许可票据。"""

    run: FactV2OfflineRun
    permit: _ProviderPermit

    def close(self) -> None:
        """关闭正式运行的共享钱包和工作区。"""
        self.run.close()


@dataclass(frozen=True)
class FactV2Progress:
    """固定分母的终态摘要，不包含准确率或正式成绩。"""

    planned_slots: int
    finished_slots: int
    completed_pairs: int
    half_pairs: tuple[str, ...]
    invalid_terminal_slots: int
    unexecuted_slots: int
    consistent: bool
    halted: bool
    halt_reason: str | None


def open_fact_v2_offline_run(
    folder: Path, *, dataset: Literal["dev90", "dev_val_120_v2"],
    experiment_id: str, mode: Literal["full", "pilot"], max_run_cost_cny: Decimal,
    operation: Literal["new", "resume"] = "new", root: Path = PROJECT_ROOT,
    round_label: Literal["R2"] | None = None,
    confirmed_owner_sha256: str | None = None,
    groups: tuple[GroupId, ...] = PAIR_GROUPS,
    _provider_permit: _ProviderPermit | None = None,
) -> FactV2OfflineRun:
    """先校验V2条件和预算，再建固定成对计划/钱包，仅供离线Mock调度。"""
    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("fact_v2_workspace_requires_project_root")
    conditions = (collect_fact_v2_conditions(root, dataset=dataset) if round_label is None else
        collect_fact_v2_conditions(root, dataset=dataset, round_label=round_label))
    conditions = _conditions_for_groups(conditions, groups)
    if _provider_permit is None:
        preflight_fact_v2(conditions, root=root, require_provider_license=False)
    else:
        if not provider_permit_matches(_provider_permit, conditions):
            raise ValueError("fact_v2_provider_permit_invalid")
        preflight_fact_v2(conditions, root=root, require_provider_license=True)
    ceiling = fact_v2_budget_ceiling(mode, max_run_cost_cny)
    config, prices = get_evaluation_config(), get_cost_config()
    manifest = build_experiment_manifest(experiment_id, conditions.conditions, prices, config,
        mode=mode, max_run_cost_cny=ceiling)
    workspace = open_e2_workspace(folder, mode=operation, manifest=manifest,
        conditions=conditions.conditions, sample_ids=[item.sample_id for item in conditions.questions],
        confirmed_owner_sha256=confirmed_owner_sha256)
    session: ExperimentSession | None = None
    try:
        session_factory = ExperimentSession.create if operation == "new" else ExperimentSession.resume
        session = session_factory(folder / "wallet.sqlite", experiment_id, conditions.conditions,
            prices, config, mode=mode, max_run_cost_cny=ceiling)
        journal = AnswerJournal.open(workspace, mode=operation)
        return FactV2OfflineRun(conditions, workspace, session, journal)
    except BaseException:
        if session is not None:
            session.close()
        workspace.close()
        raise


def _dispatch_next(run: FactV2OfflineRun, executor: SlotExecutor) -> DispatchResult:
    """推进一槽；整场预算被拒时保存本题后立即停场。"""
    def check_conditions() -> None:
        """在槽位前后重读V2条件，漂移阻止继续。"""
        assert_fact_v2_conditions_unchanged(run.conditions)

    result = dispatch_next(run.workspace, run.journal, run.session,
        check_conditions=check_conditions, execute_slot=executor)
    if result.status == "saved" and result.run_id is not None:
        row = next((item for item in run.journal.read_answers()
                    if item.run.run_id == result.run_id), None)
        if row is not None:
            try:
                denied = _has_experiment_cost_denial(row)
            except (OSError, ValueError, TypeError, AttributeError):
                run.session.halt("fact_v2_trace_audit_failed")
                return result.model_copy(update={"status": "blocked", "reason": "fact_v2_trace_audit_failed"})
        else:
            run.session.halt("fact_v2_answer_audit_failed")
            return result.model_copy(update={"status": "blocked", "reason": "fact_v2_answer_audit_failed"})
        if denied:
            run.session.halt("experiment_cost_limit")
            return result.model_copy(update={"status": "blocked", "reason": "experiment_cost_limit"})
    return result


def _has_experiment_cost_denial(row: object) -> bool:
    """只从答卷绑定且通过严格身份、指纹审计的Trace识别整场预算拒绝。"""
    payload = getattr(row, "payload", None)
    checked_run = getattr(row, "run", None)
    trace = payload.get("trace") if isinstance(payload, dict) else None
    if not isinstance(trace, dict) or checked_run is None:
        raise ValueError("fact_v2_budget_trace_missing")
    raw_path, digest = trace.get("path"), trace.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str):
        raise ValueError("fact_v2_budget_trace_identity_missing")
    relative = Path(raw_path)
    target = (PROJECT_ROOT / relative).resolve()
    traces_root = (PROJECT_ROOT / "traces").resolve()
    if relative.is_absolute() or target == traces_root or not target.is_relative_to(traces_root):
        raise ValueError("fact_v2_budget_trace_path_invalid")
    single = checked_run.group_id in {"single_shot", "single_v3", "single_v4"}
    reason = payload.get("stop_reason")
    audit, events = read_and_audit_trace_events(target, expected_sha256=digest,
        run=checked_run, expected_agent="fact" if single else "fact_investigator",
        expected_outcome="submitted_judgment" if payload.get("success") else "no_valid_result",
        expected_reason=reason, single_shot=single)
    if not audit.consistent:
        raise ValueError("fact_v2_budget_trace_audit_failed")
    budget_denied = any(event.type == "budget" and event.data.get("allowed") is False
        and event.data.get("reason") == "experiment_cost_limit" for event in events)
    stop_recorded = any(event.type == "stop"
        and event.data.get("reason") == "experiment_cost_limit" for event in events)
    return budget_denied and stop_recorded


def dispatch_fact_v2_offline_next(run: FactV2OfflineRun, executor: object) -> DispatchResult:
    """离线运行只接受显式构造的Mock执行器，不接受任意回调。"""
    from src.yuqing.eval.fact_v2_executor import FactV2MockExecutor

    if not isinstance(executor, FactV2MockExecutor):
        raise ValueError("fact_v2_mock_executor_required")
    return _dispatch_next(run, executor._execute)


def dispatch_fact_v2_provider_next(
    provider_run: FactV2ProviderRun, executor: object,
) -> DispatchResult:
    """正式运行仅接受绑定本次许可票据的Provider执行器。"""
    from src.yuqing.eval.fact_v2_executor import FactV2ProviderExecutor

    if not isinstance(provider_run, FactV2ProviderRun):
        raise ValueError("fact_v2_provider_run_required")
    if (not isinstance(executor, FactV2ProviderExecutor)
            or executor.provider_run is not provider_run
            or not provider_permit_matches(provider_run.permit, provider_run.run.conditions)):
        raise ValueError("fact_v2_provider_executor_required")
    return _dispatch_next(provider_run.run, executor._execute)


def fact_v2_progress(run: FactV2OfflineRun) -> FactV2Progress:
    """按已核验答卷、Trace与钱包返回完整固定分母进度。"""
    rows = run.journal.read_answers()
    audit = run.journal.audit_saved_answers(run.session.load_history())
    status = run.session.snapshot()
    halted_only = status.halted and tuple(issue.code for issue in audit.issues) == ("ledger_halted",)
    if (status.halted and not halted_only) or (not status.halted and not audit.consistent):
        raise ValueError("fact_v2_resume_audit_blocked")
    finished = len(rows)
    invalid = sum(row.payload.get("success") is False for row in rows)
    return FactV2Progress(planned_slots=len(run.workspace.plan), finished_slots=finished,
        completed_pairs=audit.completed_pairs, half_pairs=audit.half_pairs,
        invalid_terminal_slots=invalid, unexecuted_slots=len(run.workspace.plan) - finished,
        consistent=audit.consistent or halted_only, halted=status.halted,
        halt_reason=status.halt_reason)


def _condition_fingerprint(conditions: FactV2RunConditions) -> str:
    """计算模型许可票据绑定的完整条件摘要。"""
    encoded = json.dumps(dict(conditions.conditions), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _provider_startup_gate(conditions: FactV2RunConditions, *, root: Path = PROJECT_ROOT) -> _ProviderPermit:
    """先核完整条件和提示词许可，再签发绑定条件的正式入口票据。"""
    assert_fact_v2_conditions_unchanged(conditions, root=root)
    preflight_fact_v2(conditions, root=root, require_provider_license=True)
    return _ProviderPermit(_condition_fingerprint(conditions), _PROVIDER_PERMIT_SEAL)


def provider_permit_matches(permit: object, conditions: FactV2RunConditions) -> bool:
    """检查票据确由本模块签发且仍绑定当前条件快照。"""
    return (isinstance(permit, _ProviderPermit) and permit._seal is _PROVIDER_PERMIT_SEAL
        and permit.conditions_sha256 == _condition_fingerprint(conditions))


def open_fact_v2_provider_run(
    folder: Path, *, dataset: Literal["dev90", "dev_val_120_v2"], experiment_id: str,
    mode: Literal["full", "pilot"], max_run_cost_cny: Decimal,
    operation: Literal["new", "resume"] = "new", root: Path = PROJECT_ROOT,
    round_label: Literal["R2"] | None = None,
    confirmed_owner_sha256: str | None = None,
    groups: tuple[GroupId, ...] = PAIR_GROUPS,
    expected_conditions: FactV2RunConditions | None = None,
) -> FactV2ProviderRun:
    """提示词和容量许可通过后才创建正式工作区、认领槽位及共享钱包。"""
    conditions = (collect_fact_v2_conditions(root, dataset=dataset) if round_label is None else
        collect_fact_v2_conditions(root, dataset=dataset, round_label=round_label))
    conditions = _conditions_for_groups(conditions, groups)
    _assert_expected_conditions(conditions, expected_conditions)
    permit = _provider_startup_gate(conditions, root=root)
    run = open_fact_v2_offline_run(folder, dataset=dataset, experiment_id=experiment_id,
        mode=mode, max_run_cost_cny=max_run_cost_cny, operation=operation, root=root,
        round_label=round_label,
        confirmed_owner_sha256=confirmed_owner_sha256, groups=groups,
        _provider_permit=permit)
    if _condition_fingerprint(run.conditions) != permit.conditions_sha256:
        run.close()
        raise ValueError("fact_v2_conditions_changed_after_provider_preflight")
    return FactV2ProviderRun(run, permit)


def _assert_expected_conditions(actual: FactV2RunConditions,
                                expected: FactV2RunConditions | None) -> None:
    """在工作区和钱包创建前，闭合实际条件与作者预检快照。"""
    if expected is None:
        return
    if (actual.dataset != expected.dataset or actual.round_label != expected.round_label
            or actual.conditions != expected.conditions):
        raise ValueError("fact_v2_preflight_snapshot_mismatch")


def validate_explicit_budget(mode: Literal["full", "pilot"], override: Decimal) -> Decimal:
    """将CLI输入交给现有硬上限政策，只能收紧已配置整场限额。"""
    if not isinstance(override, Decimal) or not override.is_finite() or override <= 0:
        raise ValueError("fact_v2_budget_override_invalid")
    return fact_v2_budget_ceiling(mode, override)


def _conditions_for_groups(conditions: FactV2RunConditions,
                           groups: tuple[GroupId, ...]) -> FactV2RunConditions:
    """Keep the old pair plan unchanged and gate the exact validation triple."""
    if groups == PAIR_GROUPS:
        return conditions
    if groups == THREE_ARM_GROUPS:
        return enable_fact_v2_three_arm(conditions)
    raise ValueError("fact_v2_group_layout_invalid")
