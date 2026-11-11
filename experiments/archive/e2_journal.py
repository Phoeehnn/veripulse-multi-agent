"""持锁答卷日志只追加已结束运行，存储检查不替代恢复和收费许可。"""

from hashlib import sha256
import os
from pathlib import Path
from typing import Literal, Mapping

from experiments.archive.e2_artifacts import E2CheckpointRow, validate_checkpoint_artifacts
from experiments.archive.e2_resume import ResumeAudit, audit_e2_resume
from experiments.archive.e2_workspace import E2Workspace
from src.yuqing.eval.experiment_budget_store import LedgerHistory


class AnswerJournal:
    """顺序使用的答卷日志，任何不确定写入后本对象停止访问，不自动重试。"""

    def __init__(self, workspace: E2Workspace) -> None:
        """绑定持锁工作区，保存当前字节指纹以发现运行期间的外部改动。"""
        self._workspace, self._folder = workspace, workspace.folder.resolve()
        self._expected: str | None = None
        self._failed = False

    def _path(self) -> Path:
        """复检锁状态和日志实际路径，拒绝链接到其他文件。"""
        self._workspace.require_open()
        if self._failed:
            raise ValueError("answer_journal_write_uncertain")
        path = self._folder / "answers.jsonl"
        if path.resolve() != path:
            raise ValueError("answer_journal_path_escape")
        return path

    def require_workspace(self, workspace: E2Workspace) -> None:
        """只允许绑定的同一持锁工作区，防止跨实验混接日志。"""
        self._path()
        if self._workspace is not workspace:
            raise ValueError("answer_journal_workspace_mismatch")

    @classmethod
    def open(cls, workspace: E2Workspace, *, mode: Literal["new", "resume"]) -> "AnswerJournal":
        """新建独占且刷新空日志，恢复缺失不创建；成功前检查全部既有行。"""
        if mode not in {"new", "resume"}:
            raise ValueError("invalid_answer_journal_mode")
        journal = cls(workspace)
        path = journal._path()
        if mode == "new":
            try:
                with path.open("xb") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                journal._failed = True
                raise
        journal.read_answers()
        return journal

    def _validate(self, rows: list[Mapping[str, object] | E2CheckpointRow | str]) -> tuple[E2CheckpointRow, ...]:
        """复用指纹规则并要求完成记录恰为固定计划前缀，不保存未执行后缀。"""
        manifest, plan = self._workspace.manifest, self._workspace.plan
        checked = validate_checkpoint_artifacts(rows, experiment_id=manifest["experiment_id"],
            condition_sha256=manifest["conditions_fingerprint"])
        if len(checked) > len(plan):
            raise ValueError("answer_journal_plan_overflow")
        for item, slot in zip(checked, plan):
            if item.run.execution_status != "finished" or (
                item.run.sample_id, item.run.group_id, item.run.run_id) != (slot.sample_id, slot.group_id, slot.run_id):
                raise ValueError("answer_journal_order_mismatch")
        return checked

    def read_answers(self) -> tuple[E2CheckpointRow, ...]:
        """整批校验字节和记录，不跳空行、不截断末行、不修复指纹。"""
        data = self._path().read_bytes()
        digest = sha256(data).hexdigest()
        if self._expected is not None and digest != self._expected:
            raise ValueError("answer_journal_changed")
        if data and not data.endswith(b"\n"):
            raise ValueError("answer_journal_incomplete_line")
        rows = [line.decode("utf-8") for line in data.split(b"\n")[:-1]] if data else []
        checked = self._validate(rows)
        self._expected = digest
        return checked

    def _write(self, path: Path, data: bytes) -> None:
        """已有文件中核对原字节后追加，缺文件不重建，不回滚可能写入的内容。"""
        with path.open("r+b") as stream:
            before = stream.read()
            if sha256(before).hexdigest() != self._expected:
                raise ValueError("answer_journal_changed_before_write")
            if stream.write(data) != len(data):
                raise OSError("answer_journal_short_write")
            stream.flush()
            os.fsync(stream.fileno())
        self._expected = sha256(before + data).hexdigest()

    def append_answer(self, row: Mapping[str, object] | E2CheckpointRow | str) -> None:
        """只追加下一槽位；费用未知失败也可保存，写入异常立即封闭本对象。"""
        current = self.read_answers()
        checked = self._validate([*current, row])
        data = checked[-1].model_dump_json().encode("utf-8") + b"\n"
        path = self._path()
        try:
            self._write(path, data)
        except BaseException:
            self._failed = True
            raise

    def audit_saved_answers(self, history: LedgerHistory) -> ResumeAudit:
        """调用既有三层恢复检查，只返回报告，不执行下一槽位或解锁钱包。"""
        conditions = self._workspace.conditions
        if conditions.get("recovery:final_segment") == "true":
            return audit_e2_resume(self._workspace.plan, self.read_answers(), history, self._workspace.manifest,
                recovery_source_sha256=conditions.get("recovery:source_sha256"),
                recovery_plan_sha256=conditions.get("recovery:plan_sha256"))
        return audit_e2_resume(self._workspace.plan, self.read_answers(), history, self._workspace.manifest)
