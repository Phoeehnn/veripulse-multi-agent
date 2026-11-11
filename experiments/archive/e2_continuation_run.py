"""受控续评接线：一个父场只认领一个子场，不解锁父钱包或重复旧题。"""

from decimal import Decimal
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from src.yuqing.eval.dataset import load_questions
from experiments.archive.e2_continuation import ContinuationParent, inspect_continuation_parent
from experiments.archive.e2_report import _detail, _report_rows, _write_result, _json, summarize_e2
from experiments.archive.e2_workspace import E2Workspace, _acquire
from experiments.archive.e2_journal import AnswerJournal
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _parent(conditions: dict[str, str]) -> ContinuationParent:
    """从冻结全题顺序重查父场及原字节，不凭调用方任意传题许可。"""
    ids = [q.sample_id for name in ("questions_30.jsonl", "dev_extra_60.jsonl")
           for q in load_questions(PROJECT_ROOT / "eval" / name)]
    parent = inspect_continuation_parent(PROJECT_ROOT / conditions["continuation:parent"], ids)
    if parent.files != json.loads(conditions["continuation:parent_files"]):
        raise ValueError("continuation_parent_changed")
    return parent


def continuation_ceiling(conditions: dict[str, str], requested: Decimal) -> Decimal:
    """两段估算预算相加不超过父上限，未知预留仍占额度且不变成实际费用。"""
    if "continuation:parent" not in conditions:
        return requested
    parent = _parent(conditions)
    logger.info("续评承接父场：已知=%s 未知预留=%s 可用上限=%s；总实际费用仍未知",
        parent.known_cost, parent.unknown_reserved, parent.remaining_cap)
    return min(requested, parent.remaining_cap)


def _claim_data(workspace: E2Workspace) -> bytes:
    """认领绑定子目录、完整条件指纹和额度，不用活动心跳替代身份。"""
    return _json(dict(child=workspace.folder.relative_to(PROJECT_ROOT).as_posix(),
        condition_sha256=workspace.manifest["conditions_fingerprint"],
        max_cost_cny=str(workspace.manifest["max_cost_cny"]))).encode()


def claim_continuation(workspace: E2Workspace, *, resume: bool) -> None:
    """父锁内只认领一次；创建失败保留文件，恢复必须使用原子场和原额度。"""
    if "continuation:parent" not in workspace.conditions:
        return
    parent = _parent(workspace.conditions)
    if Decimal(workspace.manifest["max_cost_cny"]) > parent.remaining_cap:
        raise ValueError("continuation_budget_exceeded")
    expected = [(p.sample_id, p.group_id) for p in parent.state.plan[parent.finished_pairs * 2:]]
    if [(p.sample_id, p.group_id) for p in workspace.plan] != expected:
        raise ValueError("continuation_child_plan_mismatch")
    with (parent.folder / "workspace.lock").open("r+b") as handle:
        _acquire(handle)
        claim = parent.folder / "continuation.json"
        if resume:
            if claim.read_bytes() != _claim_data(workspace):
                raise ValueError("continuation_claim_mismatch")
        else:
            with claim.open("xb") as stream:
                stream.write(_claim_data(workspace))
                stream.flush()
                os.fsync(stream.fileno())


def check_continuation_claim(workspace: E2Workspace) -> None:
    """每次请求调度前后核对独占认领；缺失、变化均阻塞，不能新领额度。"""
    if "continuation:parent" in workspace.conditions:
        parent = _parent(workspace.conditions)
        if (parent.folder / "continuation.json").read_bytes() != _claim_data(workspace):
            raise ValueError("continuation_claim_changed")


def publish_continuation_summary(workspace: E2Workspace, journal: AnswerJournal,
                                 session: ExperimentSession, files: dict[str, str]) -> str | None:
    """新建分段汇总，保留父场失败题；跨版本、未知费用和未执行显式展示。"""
    if "continuation:parent" not in workspace.conditions or not {"single_shot", "iterative"} <= files.keys():
        return None
    check_continuation_claim(workspace)
    parent = _parent(workspace.conditions)
    requests = replay_history(parent.history, parent.state.manifest).requests
    source = SimpleNamespace(folder=parent.folder, conditions=parent.state.conditions)
    prefix = [_detail(row, source, requests) for row in parent.rows]
    audit, suffix = _report_rows(workspace, journal, session)
    if not suffix:
        raise ValueError("continuation_child_report_blocked")
    wallet = session.snapshot()
    report = dict(kind="audit", scope="segmented_dev90", parent_experiment=parent.state.manifest["experiment_id"],
        child_experiment=workspace.manifest["experiment_id"], summary=summarize_e2(prefix + suffix),
        code_revisions=[parent.state.conditions.get("code_revision"), workspace.conditions.get("code_revision")],
        cost_cny=None, known_cost_subtotal_cny=str(parent.known_cost + wallet.known_cost_subtotal_cny),
        unknown_reserved_cny=str(parent.unknown_reserved + wallet.unknown_reserved_cny),
        parent_unknown_retained=True, child_audit=audit.model_dump(mode="json"),
        code_changes=json.loads(workspace.conditions["continuation:code_changes"]),
        source_files={key: {"path": path, "sha256": sha256(Path(path).read_bytes()).hexdigest()}
                      for key, path in files.items()}, parent_files=parent.files,
        interpretation="分段dev记录；包含停场失败题和未执行项，非同一实验、非独立test成绩；总实际费用未知。")
    return str(_write_result(workspace, "audit", "jsonl", (_json(report) + "\n").encode(),
        "E2分段dev90汇总；父场失败保留，跨版本与未知费用明示"))
