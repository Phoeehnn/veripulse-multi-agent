"""严格重放费用事件；只计算状态，不写数据库或发起模型请求。"""

from dataclasses import dataclass, replace
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from src.yuqing.eval.experiment_budget_store import LedgerHistory, _event_data, _manifest_data

if TYPE_CHECKING:
    from src.yuqing.eval.experiment_budget import ExperimentBudgetStatus


@dataclass(frozen=True)
class RecoveredRequest:
    """不可变请求状态；prepared中的计划金额不会占用额度。"""

    state: str
    reserved: Decimal
    actual: Decimal | None
    identity: Mapping[str, object]


@dataclass(frozen=True)
class RecoveredState:
    """正常重放与重启恢复共用的不可变费用状态。"""

    manifest: Mapping[str, object]
    requests: Mapping[str, RecoveredRequest]
    revision: int = 0
    halt_reason: str | None = None

    def snapshot(self) -> "ExperimentBudgetStatus":
        """区分可信消费与保留额，未完成准备也不能宣称费用完整。"""
        from src.yuqing.eval.experiment_budget import ExperimentBudgetStatus

        records = tuple(self.requests.values())
        known = sum((r.actual for r in records if r.state == "settled"), Decimal(0))
        pending = [r for r in records if r.state in {"reserved", "dispatched"}]
        unknown = [r for r in records if r.state == "unknown"]
        pending_cost = sum((r.reserved for r in pending), Decimal(0))
        unknown_cost = sum((r.reserved for r in unknown), Decimal(0))
        complete = not pending and not unknown and not any(r.state == "prepared" for r in records)
        return ExperimentBudgetStatus(experiment_id=str(self.manifest["experiment_id"]),
            max_cost_cny=Decimal(self.manifest["max_cost_cny"]), known_cost_subtotal_cny=known,
            pending_reserved_cny=pending_cost, unknown_reserved_cny=unknown_cost,
            committed_cost_cny=known + pending_cost + unknown_cost,
            cost_cny=known if complete and self.halt_reason is None else None,
            pending_requests=len(pending), unknown_requests=len(unknown),
            halted=self.halt_reason is not None, halt_reason=self.halt_reason)


def _transition(old: RecoveredRequest | None, event: Mapping[str, object]) -> RecoveredRequest:
    """检查单个请求状态迁移，拒绝重复编号、金额变化和身份变化。"""
    kind = event["event_type"]
    identity = {k: event[k] for k in ("sample_id", "group_id", "run_id", "variant") if k in event}
    if old is not None:
        if any(k in old.identity and old.identity[k] != v for k, v in identity.items()):
            raise ValueError("request_identity_conflict")
        identity = dict(old.identity) | identity
    if kind in {"prepared", "reserved"}:
        if old is not None and not (kind == "reserved" and old.state == "prepared"):
            raise ValueError("duplicate_request")
        if "amount_cny" not in event:
            raise ValueError("reservation_amount_required")
        amount = Decimal(event.get("amount_cny", "0"))
        if old is not None and old.reserved != amount:
            raise ValueError("reservation_amount_conflict")
        return RecoveredRequest(str(kind), amount, None, MappingProxyType(identity))
    required = {"dispatched": "reserved", "cancelled": "reserved", "settled": "dispatched", "unknown": "dispatched"}
    if old is None or old.state != required.get(kind):
        raise ValueError("invalid_request_transition")
    if "amount_cny" in event and Decimal(event["amount_cny"]) != old.reserved:
        raise ValueError("reservation_amount_conflict")
    if kind == "cancelled" and event.get("reason") != "confirmed_unsent":
        raise ValueError("cancellation_requires_unsent_confirmation")
    if kind == "settled" and "actual_cost_cny" not in event:
        raise ValueError("settled_cost_required")
    actual = Decimal(event["actual_cost_cny"]) if kind == "settled" else None
    return replace(old, state=str(kind), actual=actual, identity=MappingProxyType(identity))


def apply_event(state: RecoveredState, event: Mapping[str, object]) -> RecoveredState:
    """正常运行时纯计算一个候选状态，不因未结算请求自动判成重启故障。"""
    clean = _event_data(event)
    kind = clean["event_type"]
    if "actual_cost_cny" in clean and kind != "settled":
        raise ValueError("unexpected_actual_cost")
    if kind == "halted":
        if not clean.get("reason") or "amount_cny" in clean:
            raise ValueError("invalid_halt_event")
        return replace(state, revision=state.revision + 1, halt_reason=state.halt_reason or str(clean["reason"]))
    if state.halt_reason is not None and kind in {"prepared", "reserved", "dispatched"}:
        raise ValueError("experiment_halted")
    request_id = str(clean["request_id"])
    record = _transition(state.requests.get(request_id), clean)
    requests = dict(state.requests) | {request_id: record}
    reason = "usage_unknown" if kind == "unknown" else None
    if kind == "settled" and record.actual > record.reserved:
        reason = "reservation_exceeded"
    candidate = replace(state, requests=MappingProxyType(requests), revision=state.revision + 1,
        halt_reason=state.halt_reason or reason)
    if kind == "reserved" and candidate.snapshot().committed_cost_cny > Decimal(state.manifest["max_cost_cny"]):
        raise ValueError("experiment_cost_limit")
    return candidate


def replay_history(history: LedgerHistory, expected_manifest: Mapping[str, object]) -> RecoveredState:
    """重放存储层已校验的历史，重启遗留请求保留额度并停止新付费请求。"""
    manifest = _manifest_data(expected_manifest)
    if _manifest_data(history.manifest) != manifest:
        raise ValueError("ledger_manifest_mismatch")
    if history.revision != len(history.events) or len(history.event_hashes) != history.revision:
        raise ValueError("invalid_history_revision")
    state = RecoveredState(MappingProxyType(manifest), MappingProxyType({}))
    for event in history.events:
        state = apply_event(state, event)
    if any(r.state in {"prepared", "reserved", "dispatched"} for r in state.requests.values()):
        state = replace(state, halt_reason=state.halt_reason or "recovery_incomplete_request")
    return state
