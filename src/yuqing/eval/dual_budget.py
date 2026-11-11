"""两层预算生命周期；发送标记和结算已实现，尚未接模型网关。"""

import sqlite3
from decimal import Decimal

from pydantic import ValidationError
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStoreError
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.harness.budget import BudgetDecision, BudgetTracker
from src.yuqing.llm import TokenUsage, compute_cost_cny
from src.yuqing.settings import CostConfig


def price_fingerprint(prices: CostConfig) -> str:
    """规范化计费配置后计算指纹，拒绝未确认价格。"""
    clean = CostConfig.model_validate(prices.model_dump())
    if clean.input_per_1m is None:
        raise ValueError("unconfirmed_prices")
    return sha256(clean.model_dump_json().encode("utf-8")).hexdigest()


class DualBudgetCoordinator:
    """单题顺序使用此对象，同一实验共享一个持久钱包及其协调锁。"""

    def __init__(self, task: BudgetTracker, experiment: PersistentExperimentBudget,
                 group_caps: Mapping[str, Decimal] | None = None) -> None:
        """绑定两个账本，价格不一致时在任何预占之前拒绝构造。"""
        self.task, self.experiment = task, experiment
        caps = {} if group_caps is None else dict(group_caps)
        if any(not isinstance(group, str) or not group or not isinstance(cap, Decimal)
               or not cap.is_finite() or cap <= 0 for group, cap in caps.items()):
            raise ValueError("invalid_group_caps")
        if sum(caps.values(), Decimal("0")) > experiment.snapshot().max_cost_cny:
            raise ValueError("group_caps_exceed_experiment_cap")
        self._group_caps = MappingProxyType(caps)
        self._active: str | None = None
        self._dispatched = False
        self._prices: CostConfig | None = None
        if price_fingerprint(task.prices) != experiment.manifest_snapshot()["price_fingerprint"]:
            raise ValueError("price_fingerprint_mismatch")

    def reserve(self, request_id: str, input_tokens: int, output_tokens: int, *,
                identity: Mapping[str, str], final: bool = False,
                correction: bool = False) -> BudgetDecision:
        """两层都确认预占才放行；正常额度拒绝完整撤回，异常保留并停用。"""
        try:
            with self.experiment.coordination():
                try:
                    if price_fingerprint(self.task.prices) != self.experiment.manifest_snapshot()["price_fingerprint"]:
                        raise ValueError("price_fingerprint_mismatch")
                    if self.experiment.snapshot().halted:
                        self.task.halt("experiment_halted")
                        return BudgetDecision(allowed=False, reason="experiment_halted")
                    decision = self.task.reserve_model_call(input_tokens, output_tokens, final=final,
                        correction=correction, request_id=request_id)
                    if not decision.allowed:
                        return decision
                    if self._group_caps and not self._reserve_group(identity):
                        self.task.cancel_model_reservation(request_id, rollback_attempt=True)
                        return BudgetDecision(allowed=False, reason="group_cost_limit")
                    self._active = request_id
                    self._dispatched = False
                    self._prices = CostConfig.model_validate(self.task.prices.model_dump())
                    decision = self.experiment.reserve(request_id, self.task.pending_cost_cny(), identity=identity)
                    if not decision.allowed:
                        self.task.cancel_model_reservation(request_id, rollback_attempt=True)
                        self._active = None
                    return decision
                except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError) as exc:
                    self._stop(exc)
                    raise
        except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError):
            self.task.halt("dual_budget_failed")
            raise

    def _reserve_group(self, identity: Mapping[str, str]) -> bool:
        """在同一持久钱包锁内核对分组累计额，不建立旁路钱包。"""
        group = identity.get("group_id")
        cap = self._group_caps.get(group or "")
        if cap is None:
            return False
        committed = self.experiment.group_committed_cost(group)
        return committed + self.task.pending_cost_cny() <= cap

    def cancel_unsent(self, request_id: str, *, rollback_attempt: bool) -> None:
        """先可靠保存整场取消，再撤回当前单题；任何不一致停止两层账本。"""
        if type(rollback_attempt) is not bool:
            raise ValueError("invalid_rollback_attempt")
        if self._active is None or request_id != self._active:
            raise ValueError("reservation_identity_mismatch")
        try:
            with self.experiment.coordination():
                try:
                    self.experiment.cancel_unsent(request_id)
                    self.task.cancel_model_reservation(request_id, rollback_attempt=rollback_attempt)
                    self._active = None
                except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError) as exc:
                    self._stop(exc)
                    raise
        except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError):
            self.task.halt("dual_budget_failed")
            raise

    def mark_dispatched(self, request_id: str) -> None:
        """先保存整场发送标记，再标记单题；任何失败都不能许可SDK调用。"""
        try:
            with self.experiment.coordination():
                try:
                    if self._active is None or request_id != self._active or self._dispatched:
                        raise ValueError("invalid_dispatch_request")
                    if self.task.status().halted or self.experiment.snapshot().halted:
                        self.cancel_unsent(request_id, rollback_attempt=False)
                        raise ValueError("dispatch_budget_halted")
                    if price_fingerprint(self.task.prices) != price_fingerprint(self._prices):
                        raise ValueError("price_fingerprint_mismatch")
                    self.experiment.mark_dispatched(request_id)
                    self.task.mark_model_dispatched(request_id)
                    self._dispatched = True
                except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError) as exc:
                    self._stop(exc)
                    raise
        except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError):
            self.task.halt("dual_budget_failed")
            raise

    def settle(self, request_id: str, input_tokens: int | None, output_tokens: int | None) -> BudgetDecision:
        """先保留本题可信用量，再持久结算；在途回执不因已有停止而丢弃。"""
        if request_id != self._active or not self._dispatched:
            self.halt("invalid_settlement_request")
            raise ValueError("invalid_settlement_request")
        try:
            usage = TokenUsage.model_validate(dict(input_tokens=input_tokens, output_tokens=output_tokens), strict=True)
        except ValidationError:
            usage = None
        actual = Decimal(str(compute_cost_cny(usage, self._prices))) if usage is not None else None
        try:
            decision = self.task.settle_model_call(input_tokens, output_tokens)
            self._active, self._dispatched = None, False
            with self.experiment.coordination():
                try:
                    self.experiment.settle(request_id, actual)
                    if self.task.status().halted:
                        self.experiment.halt(self.task.status().halt_reason or decision.reason)
                    if self.experiment.snapshot().halted:
                        reason = self.experiment.snapshot().halt_reason or "experiment_halted"
                        self.task.halt(reason)
                        return BudgetDecision(allowed=False, reason=reason)
                    return decision
                except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError) as exc:
                    self._stop(exc)
                    raise
        except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError):
            self.task.halt("dual_budget_failed")
            raise

    def halt(self, reason: str) -> None:
        """先停本题再可靠记录整场停止，保留首个原因且不释放在途预占。"""
        self.task.halt(reason)
        with self.experiment.coordination():
            self.experiment.halt(reason)

    def _stop(self, cause: Exception) -> None:
        """先停止单题；已有整场故障不重试写盘，否则明确持久化协调停止。"""
        self.task.halt("dual_budget_failed")
        if not self.experiment.snapshot().halted:
            try:
                self.experiment.halt("dual_budget_failed")
            except (ValueError, ExperimentBudgetStoreError, sqlite3.Error, OSError):
                raise ExperimentBudgetStoreError("dual_budget_halt_unconfirmed") from cause
