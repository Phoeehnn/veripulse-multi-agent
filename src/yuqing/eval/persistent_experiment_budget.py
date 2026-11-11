"""整场持久费用账本：确认保存后发布内存状态，写入不确定时停止。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Iterator, Mapping
from types import MappingProxyType

from src.yuqing.eval.experiment_budget import ExperimentBudget, ExperimentBudgetStatus, _money
from src.yuqing.eval.experiment_budget_recovery import RecoveredState, apply_event, replay_history
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore, ExperimentBudgetStoreError, LedgerHistory, StaleRevisionError, _manifest_data
from src.yuqing.harness.budget import BudgetDecision
from src.yuqing.settings import EvaluationConfig


class PersistentExperimentBudget:
    """同一实验显式共享此对象；未接模型网关，不保证实际服务账单上限。"""

    def __init__(self, store: ExperimentBudgetStore, state: RecoveredState) -> None:
        """内部工厂交付已校验状态，故障快照与已确认状态分开。"""
        self._store, self._state = store, state
        self._fault_view: RecoveredState | None = None
        self._storage_failed, self._closed, self._lock = False, False, RLock()

    @property
    def storage_path(self) -> Path:
        """Expose the canonical persistent ledger location for runtime isolation checks."""
        return self._store.path

    @classmethod
    def create(cls, path: str | Path, manifest: Mapping[str, object], config: EvaluationConfig) -> PersistentExperimentBudget:
        """按当前模式配置校验额度后新建持久账本，不覆盖既有文件。"""
        return cls._open(path, manifest, config, new=True)

    @classmethod
    def resume(cls, path: str | Path, manifest: Mapping[str, object], config: EvaluationConfig) -> PersistentExperimentBudget:
        """严格恢复既有账本，遗留请求保持停止，不回退到空内存账本。"""
        return cls._open(path, manifest, config, new=False)

    @classmethod
    def _open(cls, path: str | Path, manifest: Mapping[str, object], config: EvaluationConfig,
              *, new: bool) -> PersistentExperimentBudget:
        """复用唯一额度配置与语义恢复检查，失败时释放数据库连接。"""
        clean = _manifest_data(manifest)
        ExperimentBudget(str(clean["experiment_id"]), config, mode=clean["mode"],
            max_run_cost_cny=Decimal(clean["max_cost_cny"]))
        factory = ExperimentBudgetStore.create if new else ExperimentBudgetStore.open_existing
        store = factory(path, clean)
        try:
            history = store.load()
            state = replay_history(history, clean)
            budget = cls(store, state)
            if state.halt_reason == "recovery_incomplete_request" and not any(
                e["event_type"] == "halted" and e.get("reason") == state.halt_reason for e in history.events):
                budget._halt("recovery_incomplete_request")
            return budget
        except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError):
            store.close()
            raise

    def prepare(self, request_id: str, amount: Decimal, identity: Mapping[str, str]) -> None:
        """保存准备及关联身份，计划金额不占额度；身份不能覆盖事件字段。"""
        if not set(identity).issubset({"sample_id", "group_id", "run_id", "variant"}):
            raise ValueError("invalid_request_identity_fields")
        self._write(dict(identity) | dict(event_type="prepared", request_id=request_id, amount_cny=_money(amount)))

    def reserve(self, request_id: str, amount: Decimal, *, identity: Mapping[str, str] | None = None) -> BudgetDecision:
        """预占落盘成功才放行；额度不足不创建事件，写入异常直接抛出。"""
        amount = _money(amount)
        identity = {} if identity is None else dict(identity)
        if not set(identity).issubset({"sample_id", "group_id", "run_id", "variant"}):
            raise ValueError("invalid_request_identity_fields")
        with self._lock:
            self._ensure_open()
            if self.snapshot().halted:
                return BudgetDecision(allowed=False, reason="experiment_halted")
            try:
                self._write(identity | dict(event_type="reserved", request_id=request_id, amount_cny=amount))
            except ValueError as exc:
                if str(exc) == "experiment_cost_limit":
                    return BudgetDecision(allowed=False, reason="experiment_cost_limit")
                raise
            return BudgetDecision(allowed=True, reason="reserved")

    def mark_dispatched(self, request_id: str) -> None:
        """确认可能发出的状态落盘后才返回，失败不能许可调用方发送。"""
        self._write(dict(event_type="dispatched", request_id=request_id))

    def settle(self, request_id: str, actual_cost: Decimal | None) -> None:
        """保存可信回执或未知事件；保存失败保留本次观察到的可信消费。"""
        with self._lock:
            self._ensure_open()
            if actual_cost is None:
                row = dict(event_type="unknown", request_id=request_id)
            else:
                try:
                    row = dict(event_type="settled", request_id=request_id, actual_cost_cny=_money(actual_cost))
                except ValueError:
                    self._halt("invalid_cost_receipt")
                    raise
            self._write(row)

    def cancel_unsent(self, request_id: str) -> None:
        """调用方确认未发送，取消保存成功后才释放预占。"""
        self._write(dict(event_type="cancelled", request_id=request_id, reason="confirmed_unsent"))

    def _write(self, row: Mapping[str, object]) -> None:
        """锁内计算候选、持久化并发布；非法迁移记停用，容量不足正常拒绝。"""
        with self._lock:
            self._ensure_open()
            if self._storage_failed:
                raise ExperimentBudgetStoreError("persistent_ledger_unavailable")
            try:
                candidate = apply_event(self._state, row)
            except ValueError as exc:
                if str(exc) not in {"experiment_cost_limit", "experiment_halted"}:
                    self._halt("ledger_conflict")
                raise
            self._commit(row, candidate)

    def _halt(self, reason: str) -> None:
        """把停止事件持久化，保留首个原因，已经存储故障则不重试写入。"""
        if self._storage_failed:
            raise ExperimentBudgetStoreError("persistent_ledger_unavailable")
        row = dict(event_type="halted", reason=reason)
        self._commit(row, apply_event(self._state, row))

    def _commit(self, row: Mapping[str, object], candidate: RecoveredState) -> None:
        """只有确认revision才发布；失败视图不能用于继续操作或伪称已落盘。"""
        try:
            revision = self._store.append(row, self._state.revision)
            if revision != candidate.revision:
                raise ExperimentBudgetStoreError("unexpected_persisted_revision")
        except (ExperimentBudgetStoreError, sqlite3.Error, OSError) as exc:
            self._storage_failed = True
            reason = "ledger_revision_conflict" if isinstance(exc, StaleRevisionError) else "ledger_write_uncertain"
            view = candidate if row["event_type"] in {"reserved", "settled", "unknown"} else self._state
            self._fault_view = replace(view, halt_reason=self._state.halt_reason or reason)
            raise
        self._state = candidate

    def snapshot(self) -> ExperimentBudgetStatus:
        """写入故障时展示保守预占或本次可信回执，完整费用保持未知。"""
        with self._lock:
            return (self._fault_view or self._state).snapshot()

    def group_committed_cost(self, group_id: str) -> Decimal:
        """在共享协调锁下读取某组已结算、未结算和未知保留总额。"""
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("invalid_group_id")
        with self._lock:
            self._ensure_open()
            return sum((row.actual if row.state == "settled" else row.reserved
                        for row in self._state.requests.values()
                        if row.identity.get("group_id") == group_id and row.state in
                        {"settled", "reserved", "dispatched", "unknown"}), Decimal("0"))

    @contextmanager
    def coordination(self) -> Iterator[None]:
        """同一实验的协调器共用现有锁，关闭或存储故障时拒绝进入。"""
        with self._lock:
            self._ensure_open()
            if self._storage_failed:
                raise ExperimentBudgetStoreError("persistent_ledger_unavailable")
            yield

    def manifest_snapshot(self) -> Mapping[str, object]:
        """返回不可修改的独立实验清单，不暴露可变内部状态。"""
        with self._lock:
            return MappingProxyType(dict(self._state.manifest))

    def load_history(self) -> LedgerHistory:
        """在现有协调锁内核验磁盘历史，拒绝存储故障或外部revision漂移。"""
        with self.coordination():
            history = self._store.load()
            if history.revision != self._state.revision or history.manifest != self._state.manifest:
                raise StaleRevisionError("ledger_history_changed")
            return history

    def halt(self, reason: str) -> None:
        """通过既有写入流程持久化停止，保留首个停止原因。"""
        self._write(dict(event_type="halted", reason=reason))

    def close(self) -> None:
        """关闭连接，不取消预占；关闭后不得继续记账或发起新请求。"""
        with self._lock:
            if not self._closed:
                self._store.close()
                self._closed = True

    def _ensure_open(self) -> None:
        """关闭不是免费重置，所有变更操作在关闭后明确拒绝。"""
        if self._closed:
            raise ExperimentBudgetStoreError("persistent_ledger_closed")
