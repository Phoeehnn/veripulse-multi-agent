"""整场持久账本的离线故障注入和真实SQLite重开测试。"""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import pytest

from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore, ExperimentBudgetStoreError, LedgerWriteError, StaleRevisionError
from src.yuqing.settings import get_evaluation_config


def manifest(**changes: object) -> dict[str, object]:
    """金额显式给出，仅使用当前已批准模式和配置。"""
    return dict(experiment_id="persistent-test", mode="pilot", max_cost_cny="2",
        price_fingerprint="a" * 64, conditions_fingerprint="b" * 64, format_version=1) | changes


def create(tmp_path: Path, **changes: object) -> PersistentExperimentBudget:
    """所有数据库位于pytest专属临时目录。"""
    return PersistentExperimentBudget.create(tmp_path / "ledger.sqlite", manifest(**changes), get_evaluation_config())


def dispatch(budget: PersistentExperimentBudget, amount: str = "1") -> str:
    """预占并保存可能发出状态，供结算场景使用。"""
    request_id = uuid4().hex
    assert budget.reserve(request_id, Decimal(amount)).allowed
    budget.mark_dispatched(request_id)
    return request_id


def fail_write(*args: object, **kwargs: object) -> int:
    """模拟数据库写入失败，不执行真实写入。"""
    raise LedgerWriteError("simulated_failure")


def test_public_history_rejects_closed_and_external_revision(tmp_path: Path) -> None:
    """公共历史读取校验磁盘并拒绝外部写入造成的内存漂移。"""
    budget = create(tmp_path)
    assert budget.load_history().revision == 0
    other = ExperimentBudgetStore.open_existing(tmp_path / "ledger.sqlite", manifest())
    try:
        other.append(dict(event_type="halted", reason="external_stop"), 0)
    finally:
        other.close()
    with pytest.raises(StaleRevisionError, match="history_changed"):
        budget.load_history()
    budget.close()
    with pytest.raises(ExperimentBudgetStoreError, match="closed"):
        budget.load_history()


def test_persistent_roundtrip_matches_live_state(tmp_path: Path) -> None:
    """合法准备、预占、发出、结算关闭重开后金额和身份一致。"""
    budget = create(tmp_path)
    request_id = uuid4().hex
    budget.prepare(request_id, Decimal("1"), dict(sample_id="341", group_id="v3", run_id=uuid4().hex))
    assert budget.snapshot().committed_cost_cny == 0
    assert budget.reserve(request_id, Decimal("1")).allowed
    budget.mark_dispatched(request_id)
    budget.settle(request_id, Decimal("0.10"))
    before = budget.snapshot()
    revision = budget._state.revision
    budget.close()
    reopened = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert reopened.snapshot() == before and reopened._state.revision == revision
    assert reopened._state.requests[request_id].identity["group_id"] == "v3"
    reopened.close()


def test_failed_reservation_never_returns_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """失败预占不放行、 confirmed状态不变，故障视图保守保留可能额度。"""
    budget = create(tmp_path)
    monkeypatch.setattr(budget._store, "append", fail_write)
    with pytest.raises(LedgerWriteError):
        budget.reserve(uuid4().hex, Decimal("1"))
    assert budget._state.revision == 0 and budget._store.load().revision == 0
    assert budget.snapshot().halted and budget.snapshot().pending_reserved_cny == 1
    assert not budget.reserve(uuid4().hex, Decimal("0")).allowed
    budget.close()


def test_failed_cancel_keeps_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """取消写失败不能释放原预占。"""
    budget = create(tmp_path)
    request_id = uuid4().hex
    budget.reserve(request_id, Decimal("1"))
    monkeypatch.setattr(budget._store, "append", fail_write)
    with pytest.raises(LedgerWriteError):
        budget.cancel_unsent(request_id)
    assert budget.snapshot().halted and budget.snapshot().pending_reserved_cny == 1
    assert budget._state.requests[request_id].state == "reserved"
    budget.close()


def test_committed_but_unacknowledged_write_blocks_operations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟数据库已提交但返回失败，不发布成功、不重复预占。"""
    budget = create(tmp_path)
    original = budget._store.append
    attempts: list[int] = []

    def commit_then_fail(row: Mapping[str, object], revision: int) -> int:
        """先真实写入，随后模拟丢失确认。"""
        attempts.append(revision)
        original(row, revision)
        raise LedgerWriteError("acknowledgement_lost")

    monkeypatch.setattr(budget._store, "append", commit_then_fail)
    with pytest.raises(LedgerWriteError):
        budget.reserve(uuid4().hex, Decimal("1"))
    assert budget._state.revision == 0 and budget._store.load().revision == 1
    assert not budget.reserve(uuid4().hex, Decimal("1")).allowed and attempts == [0]
    budget.close()
    resumed = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert resumed.snapshot().halted and resumed.snapshot().pending_reserved_cny == 1
    assert len(resumed._state.requests) == 1
    resumed.close()


def test_failed_settlement_preserves_observed_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """可信消费保存失败仍显示本次观察小计，不宣称已经可靠落盘。"""
    budget = create(tmp_path)
    request_id = dispatch(budget)
    monkeypatch.setattr(budget._store, "append", fail_write)
    with pytest.raises(LedgerWriteError):
        budget.settle(request_id, Decimal("0.1"))
    assert budget.snapshot().known_cost_subtotal_cny == Decimal("0.1")
    assert budget.snapshot().cost_cny is None and budget.snapshot().halt_reason == "ledger_write_uncertain"
    assert budget._state.requests[request_id].state == "dispatched"
    with pytest.raises(ExperimentBudgetStoreError, match="persistent_ledger_unavailable"):
        budget.mark_dispatched(request_id)
    budget.close()
    resumed = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert resumed.snapshot().pending_reserved_cny == 1 and resumed.snapshot().halted
    assert resumed.snapshot().known_cost_subtotal_cny == 0
    resumed.close()


def test_stale_revision_halts_without_blind_retry(tmp_path: Path) -> None:
    """其他连接推进revision后当前对象拒绝继续，不自动增号重试。"""
    budget = create(tmp_path)
    writer = ExperimentBudgetStore.open_existing(tmp_path / "ledger.sqlite", manifest())
    writer.append(dict(event_type="reserved", request_id=uuid4().hex, amount_cny="1"), 0)
    with pytest.raises(StaleRevisionError):
        budget.reserve(uuid4().hex, Decimal("0.1"))
    assert budget.snapshot().halt_reason == "ledger_revision_conflict"
    assert not budget.reserve(uuid4().hex, Decimal("0.1")).allowed
    assert writer.load().revision == 1
    writer.close()
    budget.close()


def test_resume_pending_request_remains_halted(tmp_path: Path) -> None:
    """恢复停止写入历史，补齐回执并再次重开也不能意外解除。"""
    budget = create(tmp_path)
    request_id = dispatch(budget)
    budget.close()
    for iteration in range(2):
        budget = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
        assert budget.snapshot().halt_reason == "recovery_incomplete_request"
        if iteration == 0:
            budget.settle(request_id, Decimal("0.1"))
        assert not budget.reserve(uuid4().hex, Decimal("0.1")).allowed
        revision = budget._state.revision
        budget.close()
    budget = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert budget._state.revision == revision
    assert budget.snapshot().known_cost_subtotal_cny == Decimal("0.1")
    budget.close()


def test_unknown_cost_survives_restart(tmp_path: Path) -> None:
    """保留已知消费与未知预占，重启不把未知费用当免费。"""
    budget = create(tmp_path)
    budget.settle(dispatch(budget, "0.5"), Decimal("0.1"))
    budget.settle(dispatch(budget), None)
    before = budget.snapshot()
    budget.close()
    budget = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert budget.snapshot() == before and before.unknown_reserved_cny == 1
    assert before.known_cost_subtotal_cny == Decimal("0.1") and before.cost_cny is None
    budget.close()


@pytest.mark.parametrize("mode,limit", [("pilot", "2.01"), ("full", "10.01")])
def test_runtime_limit_cannot_exceed_config(tmp_path: Path, mode: str, limit: str) -> None:
    """拒绝创建超额账本，校验失败不先创建文件。"""
    with pytest.raises(ValueError, match="experiment_limit_cannot_increase"):
        create(tmp_path, mode=mode, max_cost_cny=limit)
    assert not (tmp_path / "ledger.sqlite").exists()


def test_two_groups_share_one_persistent_budget(tmp_path: Path) -> None:
    """两组并发请求同一对象，合计只有一份共享额度。"""
    budget = create(tmp_path)

    def request(group: str) -> bool:
        """各组准备1元请求并竞争同一2元上限。"""
        request_id = uuid4().hex
        budget.prepare(request_id, Decimal("1"), dict(group_id=group))
        return budget.reserve(request_id, Decimal("1")).allowed

    with ThreadPoolExecutor(max_workers=4) as pool:
        allowed = list(pool.map(request, ["v3", "suggested"] * 4))
    assert sum(allowed) == 2 and budget.snapshot().committed_cost_cny == 2
    budget.close()


def test_duplicate_settlement_persists_halt(tmp_path: Path) -> None:
    """重复回执不重复计费，并持久保留账本冲突停止状态。"""
    budget = create(tmp_path)
    request_id = dispatch(budget)
    budget.settle(request_id, Decimal("0.1"))
    with pytest.raises(ValueError):
        budget.settle(request_id, Decimal("0.1"))
    budget.close()
    resumed = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert resumed.snapshot().halt_reason == "ledger_conflict"
    assert resumed.snapshot().known_cost_subtotal_cny == Decimal("0.1")
    resumed.close()


def test_missing_or_changed_database_never_falls_back(tmp_path: Path) -> None:
    """缺文件或条件改变直接失败，不创建空的内存钱包。"""
    path = tmp_path / "ledger.sqlite"
    with pytest.raises(FileNotFoundError):
        PersistentExperimentBudget.resume(path, manifest(), get_evaluation_config())
    assert not path.exists()
    create(tmp_path).close()
    with pytest.raises(ValueError, match="ledger_manifest_mismatch"):
        PersistentExperimentBudget.resume(path, manifest(conditions_fingerprint="c" * 64), get_evaluation_config())


def test_failed_dispatch_never_authorizes_sending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """标记发送写失败必须抛出，已确认状态仍停留预占。"""
    budget = create(tmp_path)
    request_id = uuid4().hex
    budget.reserve(request_id, Decimal("1"))
    monkeypatch.setattr(budget._store, "append", fail_write)
    with pytest.raises(LedgerWriteError):
        budget.mark_dispatched(request_id)
    assert budget._state.requests[request_id].state == "reserved"
    assert budget.snapshot().halted and budget.snapshot().pending_reserved_cny == 1
    budget.close()


def test_invalid_receipt_persists_stop_without_releasing_hold(tmp_path: Path) -> None:
    """非法费用回执不能算免费，停止原因与原预占持久保留。"""
    budget = create(tmp_path)
    request_id = dispatch(budget)
    with pytest.raises(ValueError):
        budget.settle(request_id, Decimal("NaN"))
    budget.close()
    budget = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert budget.snapshot().halt_reason == "invalid_cost_receipt"
    assert budget.snapshot().pending_reserved_cny == 1 and budget.snapshot().cost_cny is None
    budget.close()


@pytest.mark.parametrize("receipt,reason", [(None, "usage_unknown"), (Decimal("1.1"), "reservation_exceeded")])
def test_stopped_budget_accepts_other_dispatched_receipt(tmp_path: Path, receipt: Decimal | None, reason: str) -> None:
    """停用后仍记录其他已发出请求的可信消费，首个原因重启后保留。"""
    budget = create(tmp_path)
    first, second = dispatch(budget), dispatch(budget)
    budget.settle(first, receipt)
    budget.settle(second, Decimal("0.2"))
    status = budget.snapshot()
    assert status.halt_reason == reason
    assert status.known_cost_subtotal_cny == Decimal("0.2" if receipt is None else "1.3")
    budget.close()
    budget = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    assert budget.snapshot() == status
    budget.close()


@pytest.mark.parametrize("operation", ["prepare", "reserve", "dispatch", "settle", "cancel"])
def test_closed_budget_rejects_mutation(tmp_path: Path, operation: str) -> None:
    """关闭连接不重置额度，所有变更入口明确拒绝。"""
    budget = create(tmp_path)
    request_id = dispatch(budget)
    budget.close()
    budget.close()
    with pytest.raises(ExperimentBudgetStoreError, match="persistent_ledger_closed"):
        if operation == "prepare": budget.prepare(uuid4().hex, Decimal("1"), {})
        elif operation == "reserve": budget.reserve(uuid4().hex, Decimal("1"))
        elif operation == "dispatch": budget.mark_dispatched(request_id)
        elif operation == "settle": budget.settle(request_id, None)
        else: budget.cancel_unsent(request_id)


def test_reserve_identity_is_atomic(tmp_path: Path) -> None:
    """身份与预占在同一事件保存，拒绝时没有prepared遗留。"""
    budget = create(tmp_path)
    try:
        tags = dict(sample_id="341", group_id="v3", run_id=uuid4().hex)
        assert budget.reserve(uuid4().hex, Decimal("1"), identity=tags).allowed
        history = budget._store.load()
        assert len(history.events) == 1 and history.events[0]["event_type"] == "reserved"
        assert all(history.events[0][key] == value for key, value in tags.items())
        assert not budget.reserve(uuid4().hex, Decimal("2"), identity=tags).allowed
        assert len(budget._store.load().events) == 1
    finally:
        budget.close()


@pytest.mark.parametrize("field", ["event_type", "request_id", "amount_cny", "prompt"])
def test_reserve_identity_cannot_override_event(tmp_path: Path, field: str) -> None:
    """身份白名单不能覆盖控制字段或夹带提示词。"""
    budget = create(tmp_path)
    try:
        with pytest.raises(ValueError, match="invalid_request_identity_fields"):
            budget.reserve(uuid4().hex, Decimal("1"), identity={field: "bad"})
        assert budget._store.load().events == ()
        assert not budget.snapshot().halted
    finally:
        budget.close()


def test_public_manifest_is_immutable_copy(tmp_path: Path) -> None:
    """公开清单不能改写账本身份。"""
    budget = create(tmp_path)
    try:
        view = budget.manifest_snapshot()
        with pytest.raises(TypeError):
            view["max_cost_cny"] = "100"
        assert budget.manifest_snapshot()["max_cost_cny"] == "2"
    finally:
        budget.close()


def test_public_halt_is_persistent_and_preserves_first_reason(tmp_path: Path) -> None:
    """公开停止能恢复，后续原因不覆盖首个原因。"""
    budget = create(tmp_path)
    budget.halt("first_stop")
    budget.halt("second_stop")
    budget.close()
    reopened = PersistentExperimentBudget.resume(tmp_path / "ledger.sqlite", manifest(), get_evaluation_config())
    try:
        assert reopened.snapshot().halt_reason == "first_stop"
        assert not reopened.reserve(uuid4().hex, Decimal("1")).allowed
    finally:
        reopened.close()


def test_public_coordination_supports_nested_operations(tmp_path: Path) -> None:
    """协调锁复用RLock，锁内预占和取消不会自锁。"""
    budget = create(tmp_path)
    request_id = uuid4().hex
    with budget.coordination():
        assert budget.reserve(request_id, Decimal("1")).allowed
        budget.cancel_unsent(request_id)
    assert budget.snapshot().committed_cost_cny == 0
    budget.close()
    with pytest.raises(ExperimentBudgetStoreError, match="persistent_ledger_closed"):
        with budget.coordination():
            pytest.fail("closed ledger entered coordination")
