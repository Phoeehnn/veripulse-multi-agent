"""整场共享的内存费用账本；未持久化或接入网关，不表示真实评测已受保护。"""

from dataclasses import dataclass, replace
from decimal import Decimal
from threading import RLock
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.yuqing.harness.budget import BudgetDecision
from src.yuqing.settings import EvaluationConfig


class ExperimentBudgetStatus(BaseModel):
    """独立只读费用快照，保留额与可信消费分开，未知总费用留空。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    experiment_id: str
    max_cost_cny: Decimal
    known_cost_subtotal_cny: Decimal
    pending_reserved_cny: Decimal
    unknown_reserved_cny: Decimal
    committed_cost_cny: Decimal
    cost_cny: Decimal | None
    pending_requests: int
    unknown_requests: int
    halted: bool
    halt_reason: str | None


@dataclass(frozen=True)
class _RequestCost:
    """请求状态不可原地修改，账本在锁内替换一份完整记录。"""

    reserved: Decimal
    state: Literal["reserved", "dispatched", "settled", "cancelled", "unknown"] = "reserved"
    actual: Decimal | None = None


def _money(value: Decimal) -> Decimal:
    """只接受有限非负Decimal，拒绝浮点、字符串、布尔值和特殊数值。"""
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("invalid_money")
    return value


class ExperimentBudget:
    """同一实验的两组显式共用此对象；本卡仅保证内存请求账目一致。"""

    def __init__(self, experiment_id: str, config: EvaluationConfig, *, mode: Literal["full", "pilot"],
                 max_run_cost_cny: Decimal | None = None) -> None:
        """从唯一配置选择模式上限，运行参数只可进一步收紧。"""
        if not isinstance(experiment_id, str) or not experiment_id.strip() or mode not in {"full", "pilot"}:
            raise ValueError("invalid_experiment_identity_or_mode")
        config = EvaluationConfig.model_validate(config.model_dump())
        ceiling = Decimal(str(config.max_run_cost_cny if mode == "full" else config.pilot_max_run_cost_cny))
        limit = ceiling if max_run_cost_cny is None else _money(max_run_cost_cny)
        if limit <= 0 or limit > ceiling:
            raise ValueError("experiment_limit_cannot_increase")
        self._experiment_id, self._limit = experiment_id, limit
        self._requests: dict[str, _RequestCost] = {}
        self._halt_reason: str | None = None
        self._lock = RLock()

    def reserve(self, request_id: str, amount: Decimal) -> BudgetDecision:
        """按请求独占预占，共用额度不足不创建记录，拒绝重复编号。"""
        amount = _money(amount)
        if not isinstance(request_id, str) or UUID(request_id).hex != request_id:
            raise ValueError("invalid_request_id")
        with self._lock:
            if self._halt_reason is not None:
                return BudgetDecision(allowed=False, reason="experiment_halted")
            if request_id in self._requests:
                self._halt_reason = "ledger_conflict"
                raise ValueError("duplicate_experiment_request")
            if self.snapshot().committed_cost_cny + amount > self._limit:
                return BudgetDecision(allowed=False, reason="experiment_cost_limit")
            self._requests[request_id] = _RequestCost(amount)
            return BudgetDecision(allowed=True, reason="reserved")

    def _expect(self, request_id: str, state: str) -> _RequestCost:
        """只允许正确的前置状态；冲突停用账本且不改写已有金额。"""
        record = self._requests.get(request_id)
        if record is None or record.state != state:
            self._halt_reason = self._halt_reason or "ledger_conflict"
            raise ValueError("invalid_experiment_transition")
        return record

    def mark_dispatched(self, request_id: str) -> None:
        """标为可能已经发出，不证明服务端收到；此后禁止普通取消。"""
        with self._lock:
            if self._halt_reason is not None:
                raise ValueError("experiment_halted")
            record = self._expect(request_id, "reserved")
            self._requests[request_id] = replace(record, state="dispatched")

    def cancel_unsent(self, request_id: str) -> None:
        """仅释放尚未标记发出的预占，保留编号防止重复释放或重用。"""
        with self._lock:
            record = self._expect(request_id, "reserved")
            self._requests[request_id] = replace(record, state="cancelled")

    def settle(self, request_id: str, actual_cost: Decimal | None) -> None:
        """可信消费替换预占；未知保留、超预占如实记账，两者都停止后续预占。"""
        with self._lock:
            record = self._expect(request_id, "dispatched")
            if actual_cost is not None:
                try:
                    actual_cost = _money(actual_cost)
                except ValueError:
                    self._halt_reason = self._halt_reason or "invalid_cost_receipt"
                    raise
            if actual_cost is None:
                self._requests[request_id] = replace(record, state="unknown")
                self._halt_reason = self._halt_reason or "usage_unknown"
            else:
                self._requests[request_id] = replace(record, state="settled", actual=actual_cost)
                if actual_cost > record.reserved:
                    self._halt_reason = self._halt_reason or "reservation_exceeded"

    def snapshot(self) -> ExperimentBudgetStatus:
        """锁内汇总一次，读取不改变额度，零保留额的未知请求也不会冒充已知。"""
        with self._lock:
            records = list(self._requests.values())
            known = sum((r.actual for r in records if r.state == "settled"), Decimal(0))
            pending = [r for r in records if r.state in {"reserved", "dispatched"}]
            unknown = [r for r in records if r.state == "unknown"]
            pending_cost = sum((r.reserved for r in pending), Decimal(0))
            unknown_cost = sum((r.reserved for r in unknown), Decimal(0))
            complete = not pending and not unknown and self._halt_reason != "ledger_conflict"
            return ExperimentBudgetStatus(experiment_id=self._experiment_id, max_cost_cny=self._limit,
                known_cost_subtotal_cny=known, pending_reserved_cny=pending_cost,
                unknown_reserved_cny=unknown_cost, committed_cost_cny=known + pending_cost + unknown_cost,
                cost_cny=known if complete else None, pending_requests=len(pending), unknown_requests=len(unknown),
                halted=self._halt_reason is not None, halt_reason=self._halt_reason)
