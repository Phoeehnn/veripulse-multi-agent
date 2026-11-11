"""恢复校验使用临时数据库，不调用模型。"""

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from src.yuqing.eval.experiment_budget_recovery import apply_event, replay_history
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore, LedgerHistory


def manifest(**changes: object) -> dict[str, object]:
    """生成显式实验条件，不读取真实模型或题目。"""
    return dict(experiment_id="recovery", mode="pilot", max_cost_cny="2",
        price_fingerprint="a" * 64, conditions_fingerprint="b" * 64, format_version=1) | changes


def event(kind: str, request_id: str, **fields: object) -> dict[str, object]:
    """生成一个最小事件，金额和原因由场景明确提供。"""
    return dict(event_type=kind, request_id=request_id) | fields


def history(tmp_path: Path, events: list[dict[str, object]]) -> LedgerHistory:
    """从真实存储接口取得已核验历史。"""
    store = ExperimentBudgetStore.create(tmp_path / (uuid4().hex + ".sqlite"), manifest())
    for revision, row in enumerate(events):
        store.append(row, revision)
    result = store.load()
    store.close()
    return result


def settled(request_id: str, amount: str = "0.10") -> list[dict[str, object]]:
    """构造一次预占、发出、可信结算。"""
    return [event("reserved", request_id, amount_cny="1"), event("dispatched", request_id),
        event("settled", request_id, actual_cost_cny=amount)]


def test_replay_settled_cost_counted_once(tmp_path: Path) -> None:
    """重复恢复只重建同一消费，不追加或叠加。"""
    saved = history(tmp_path, settled(uuid4().hex))
    first = replay_history(saved, manifest())
    assert first == replay_history(saved, manifest())
    assert first.snapshot().cost_cny == Decimal("0.10")
    assert first.snapshot().pending_reserved_cny == 0
    with pytest.raises(TypeError):
        first.requests[uuid4().hex] = None


@pytest.mark.parametrize("kind,held", [("prepared", "0"), ("reserved", "1"), ("dispatched", "1")])
def test_replay_orphan_reservation_is_held_and_halted(tmp_path: Path, kind: str, held: str) -> None:
    """孤立准备不虚构消费；遗留预占与发送全部保留额度。"""
    request_id = uuid4().hex
    rows = [event("prepared" if kind == "prepared" else "reserved", request_id, amount_cny="1")]
    if kind == "dispatched":
        rows.append(event(kind, request_id))
    state = replay_history(history(tmp_path, rows), manifest())
    status = state.snapshot()
    assert status.halted and status.halt_reason == "recovery_incomplete_request"
    assert status.committed_cost_cny == Decimal(held) and status.cost_cny is None
    with pytest.raises(ValueError, match="experiment_halted"):
        apply_event(state, event("reserved", uuid4().hex, amount_cny="0.1"))


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_id", "changed_amount", "changed_identity", "no_cost", "no_amount", "bad_cancel", "unexpected_actual", "over_cap"])
def test_replay_rejects_illegal_transition(tmp_path: Path, failure: str) -> None:
    """格式与哈希合法的事件也必须经过账目语义检查。"""
    request_id = uuid4().hex
    rows = settled(request_id)
    if failure == "missing": rows = [rows[-1]]
    elif failure == "duplicate": rows.append(rows[-1])
    elif failure == "wrong_id": rows[1]["request_id"] = uuid4().hex
    elif failure == "changed_amount": rows[1]["amount_cny"] = "0.9"
    elif failure == "changed_identity":
        rows[0]["group_id"], rows[1]["group_id"] = "v3", "suggested"
    elif failure == "no_cost": rows[-1].pop("actual_cost_cny")
    elif failure == "no_amount": rows[0].pop("amount_cny")
    elif failure == "bad_cancel": rows = [rows[0], event("cancelled", request_id)]
    elif failure == "unexpected_actual": rows[1]["actual_cost_cny"] = "0"
    elif failure == "over_cap": rows[0]["amount_cny"] = "3"
    with pytest.raises(ValueError):
        replay_history(history(tmp_path, rows), manifest())


@pytest.mark.parametrize("kind,reason", [("unknown", "usage_unknown"), ("settled", "reservation_exceeded")])
def test_replay_preserves_unknown_and_overreservation_halt(tmp_path: Path, kind: str, reason: str) -> None:
    """缺少额外halted事件也能推导停止，已知部分消费仍保留。"""
    request_id = uuid4().hex
    rows = settled(uuid4().hex) + settled(request_id)[:2]
    rows.append(event(kind, request_id, **({"actual_cost_cny": "1.1"} if kind == "settled" else {})))
    state = replay_history(history(tmp_path, rows), manifest())
    status = state.snapshot()
    assert status.halted and status.halt_reason == reason
    assert status.known_cost_subtotal_cny == Decimal("1.2" if kind == "settled" else "0.1")
    assert status.unknown_reserved_cny == (0 if kind == "settled" else 1)
    assert status.cost_cny is None


def test_cancelled_request_remains_tombstone(tmp_path: Path) -> None:
    """明确未发送取消释放额度，编号依然不能复用。"""
    request_id = uuid4().hex
    rows = [event("prepared", request_id, amount_cny="1"), event("reserved", request_id, amount_cny="1"),
        event("cancelled", request_id, reason="confirmed_unsent")]
    state = replay_history(history(tmp_path, rows), manifest())
    assert state.snapshot().cost_cny == 0 and not state.snapshot().halted
    with pytest.raises(ValueError, match="duplicate_request"):
        apply_event(state, event("reserved", request_id, amount_cny="1"))


def test_live_pending_does_not_trigger_restart_halt(tmp_path: Path) -> None:
    """正常事件应用与冷启动恢复明确区分。"""
    state = replay_history(history(tmp_path, []), manifest())
    updated = apply_event(state, event("reserved", uuid4().hex, amount_cny="1"))
    assert not updated.snapshot().halted and updated.snapshot().pending_requests == 1
    assert state.revision == 0 and not state.requests


def test_halt_preserves_first_reason_and_allows_existing_receipt(tmp_path: Path) -> None:
    """停用后仍能记已发出请求的可信消费，首个原因不覆盖。"""
    request_id = uuid4().hex
    rows = settled(request_id)[:2] + [dict(event_type="halted", reason="ledger_conflict"),
        dict(event_type="halted", reason="later_error"), settled(request_id)[-1]]
    state = replay_history(history(tmp_path, rows), manifest())
    assert state.halt_reason == "ledger_conflict"
    assert state.snapshot().known_cost_subtotal_cny == Decimal("0.10")


@pytest.mark.parametrize("field,value", [("conditions_fingerprint", "c" * 64), ("price_fingerprint", "d" * 64), ("max_cost_cny", "1"), ("mode", "full")])
def test_replay_rejects_changed_manifest(tmp_path: Path, field: str, value: str) -> None:
    """换条件、价格、模式或有效额度均拒绝混用实验。"""
    with pytest.raises(ValueError, match="ledger_manifest_mismatch"):
        replay_history(history(tmp_path, []), manifest(**{field: value}))


def test_zero_reservation_unknown_is_not_free(tmp_path: Path) -> None:
    """零预占但回执未知，仍然是未知消费而不是已确认免费。"""
    request_id = uuid4().hex
    rows = [event("reserved", request_id, amount_cny="0"), event("dispatched", request_id), event("unknown", request_id)]
    status = replay_history(history(tmp_path, rows), manifest()).snapshot()
    assert status.halted and status.unknown_requests == 1 and status.cost_cny is None


@pytest.mark.parametrize("failure", ["halt_without_reason", "reserve_after_unknown", "prepared_amount_changed"])
def test_replay_rejects_unsafe_continuation(tmp_path: Path, failure: str) -> None:
    """不能用缺原因停止、未知后新增请求或变更计划金额绕过状态检查。"""
    request_id = uuid4().hex
    if failure == "halt_without_reason":
        rows = [dict(event_type="halted")]
    elif failure == "reserve_after_unknown":
        rows = settled(request_id)[:2] + [event("unknown", request_id), event("reserved", uuid4().hex, amount_cny="0.1")]
    else:
        rows = [event("prepared", request_id, amount_cny="1"), event("reserved", request_id, amount_cny="0.9")]
    with pytest.raises(ValueError):
        replay_history(history(tmp_path, rows), manifest())
