"""整场内存账本离线测试，不联网，不接入真实评测或持久化。"""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from uuid import uuid4

from pydantic import ValidationError
import pytest

from src.yuqing.eval.experiment_budget import ExperimentBudget
from src.yuqing.settings import get_evaluation_config


def ledger(mode: str = "full", limit: Decimal | None = None) -> ExperimentBudget:
    """用当前已批准配置新建独立账本，测试限额只收紧。"""
    return ExperimentBudget(uuid4().hex, get_evaluation_config(), mode=mode, max_run_cost_cny=limit)


def dispatched(budget: ExperimentBudget, amount: str) -> str:
    """预占并标记可能发出，供结算测试使用。"""
    request_id=uuid4().hex
    assert budget.reserve(request_id,Decimal(amount)).allowed
    budget.mark_dispatched(request_id)
    return request_id


def test_formal_and_pilot_limits_and_restriction() -> None:
    """模式分别选10/2，运行参数只收紧，不能凭新建两组账本获得隐含额度。"""
    assert ledger().snapshot().max_cost_cny==Decimal('10')
    assert ledger('pilot').snapshot().max_cost_cny==Decimal('2')
    assert ledger('pilot',Decimal('1')).snapshot().max_cost_cny==Decimal('1')
    with pytest.raises(ValueError):
        ledger('pilot',Decimal('3'))
    with pytest.raises(ValueError):
        ledger('full',Decimal('11'))


def test_exact_boundary_and_denial_leave_no_reservation() -> None:
    """等于上限可获准，超过时请求和费用状态保持原样。"""
    budget=ledger(limit=Decimal('1'))
    request_id=dispatched(budget,'1')
    before=budget.snapshot()
    decision=budget.reserve(uuid4().hex,Decimal('0.01'))
    assert not decision.allowed and decision.reason=='experiment_cost_limit'
    assert budget.snapshot()==before
    budget.settle(request_id,Decimal('0.1'))
    assert budget.reserve(uuid4().hex,Decimal('0.9')).allowed
    assert budget.snapshot().committed_cost_cny==Decimal('1.0')


def test_actual_replaces_reservation_without_double_counting() -> None:
    """实际费用替代预占，已知费用与未结算额分别汇总。"""
    budget=ledger()
    first=dispatched(budget,'1')
    second=dispatched(budget,'2')
    budget.settle(first,Decimal('0.1'))
    status=budget.snapshot()
    assert status.known_cost_subtotal_cny==Decimal('0.1')
    assert status.pending_reserved_cny==Decimal('2') and status.cost_cny is None
    assert status.committed_cost_cny==Decimal('2.1')
    budget.settle(second,Decimal('0.2'))
    assert budget.snapshot().cost_cny==Decimal('0.3')


def test_unknown_keeps_known_subtotal_and_stops_new_requests() -> None:
    """已经确认的消费不丢，未知请求保留预占且阻止后续付费。"""
    budget=ledger()
    first=dispatched(budget,'1')
    second=dispatched(budget,'2')
    budget.settle(first,Decimal('0.1'))
    budget.settle(second,None)
    status=budget.snapshot()
    assert status.known_cost_subtotal_cny==Decimal('0.1')
    assert status.unknown_reserved_cny==Decimal('2') and status.pending_reserved_cny==0
    assert status.committed_cost_cny==Decimal('2.1') and status.cost_cny is None
    assert status.halted and status.halt_reason=='usage_unknown'
    assert not budget.reserve(uuid4().hex,Decimal('0')).allowed


def test_zero_amount_unknown_is_not_free_known_usage() -> None:
    """即使保留额为零，未知状态也不能被金额求和伪装成已知。"""
    budget=ledger()
    budget.settle(dispatched(budget,'0'),None)
    status=budget.snapshot()
    assert status.unknown_reserved_cny==0 and status.unknown_requests==1
    assert status.cost_cny is None and status.halted


def test_only_unsent_reservation_can_be_cancelled() -> None:
    """取消未发出的请求不产生消费，已发出不能因缺回执当作没花钱。"""
    budget=ledger()
    unsent=uuid4().hex
    assert budget.reserve(unsent,Decimal('1')).allowed
    budget.cancel_unsent(unsent)
    assert budget.snapshot().cost_cny==0 and budget.snapshot().committed_cost_cny==0
    request_id=dispatched(budget,'2')
    with pytest.raises(ValueError):
        budget.cancel_unsent(request_id)
    assert budget.snapshot().pending_reserved_cny==Decimal('2')
    assert budget.snapshot().halt_reason=='ledger_conflict'


@pytest.mark.parametrize('repeat',['reserve','dispatch','settle','cancel','reuse_cancelled'])
def test_duplicate_transition_cannot_spend_or_release_twice(repeat: str) -> None:
    """重复或冲突操作停用账本，绝不重复累加或再次退款。"""
    budget=ledger()
    request_id=uuid4().hex
    assert budget.reserve(request_id,Decimal('1')).allowed
    if repeat in {'dispatch','settle'}:
        budget.mark_dispatched(request_id)
    if repeat=='settle':
        budget.settle(request_id,Decimal('0.1'))
    elif repeat in {'cancel','reuse_cancelled'}:
        budget.cancel_unsent(request_id)
    before=budget.snapshot()
    with pytest.raises(ValueError):
        if repeat in {'reserve','reuse_cancelled'}:
            budget.reserve(request_id,Decimal('2'))
        elif repeat=='dispatch':
            budget.mark_dispatched(request_id)
        elif repeat=='settle':
            budget.settle(request_id,Decimal('0.2'))
        else:
            budget.cancel_unsent(request_id)
    status=budget.snapshot()
    assert status.committed_cost_cny==before.committed_cost_cny
    assert status.halt_reason=='ledger_conflict' and status.cost_cny is None


def test_over_reservation_keeps_actual_even_beyond_cap() -> None:
    """实际超过预占如实记账，不能裁剪到预占或总上限。"""
    budget=ledger(limit=Decimal('1'))
    budget.settle(dispatched(budget,'1'),Decimal('1.5'))
    status=budget.snapshot()
    assert status.cost_cny==status.committed_cost_cny==Decimal('1.5')
    assert status.halt_reason=='reservation_exceeded'
    assert not budget.reserve(uuid4().hex,Decimal('0.01')).allowed


@pytest.mark.parametrize('value',[True,'1',0.1,Decimal('-1'),Decimal('NaN'),Decimal('Infinity')])
def test_invalid_reservation_money_does_not_create_state(value: object) -> None:
    """控制金额必须为有限非负Decimal，非法输入不占钱。"""
    budget=ledger()
    before=budget.snapshot()
    with pytest.raises(ValueError):
        budget.reserve(uuid4().hex,value)
    assert budget.snapshot()==before


@pytest.mark.parametrize('value',[True,'1',0.1,Decimal('-1'),Decimal('NaN'),Decimal('Infinity')])
def test_invalid_cost_receipt_retains_reservation_and_halts(value: object) -> None:
    """发出后的非法回执不能释放钱或继续付费。"""
    budget=ledger()
    request_id=dispatched(budget,'1')
    with pytest.raises(ValueError):
        budget.settle(request_id,value)
    status=budget.snapshot()
    assert status.pending_reserved_cny==Decimal('1') and status.cost_cny is None
    assert status.halt_reason=='invalid_cost_receipt'
    assert not budget.reserve(uuid4().hex,Decimal('1')).allowed


def test_two_groups_share_same_wallet() -> None:
    """单次组和循环组显式持有同一实例，后一组不能重新领预算。"""
    shared=ledger('pilot')
    group_wallets={'single':shared,'loop':shared}
    first=dispatched(group_wallets['single'],'1.5')
    group_wallets['single'].settle(first,Decimal('1.5'))
    assert not group_wallets['loop'].reserve(uuid4().hex,Decimal('0.6')).allowed
    assert group_wallets['loop'].snapshot().known_cost_subtotal_cny==Decimal('1.5')


def test_concurrent_reservations_do_not_overbook() -> None:
    """同一进程多线程竞争共享账本仍只能合计预占2元，不声称跨进程安全。"""
    budget=ledger('pilot')
    def reserve_one(_: int) -> bool:
        """独立请求竞争同一个账本锁。"""
        return budget.reserve(uuid4().hex,Decimal('1')).allowed
    with ThreadPoolExecutor(max_workers=8) as pool:
        allowed=list(pool.map(reserve_one,range(20)))
    assert sum(allowed)==2 and budget.snapshot().committed_cost_cny==Decimal('2')


def test_snapshot_is_independent_and_readonly() -> None:
    """调用方不能用修改快照改变额度，旧快照也不会跟着新请求变化。"""
    budget=ledger()
    before=budget.snapshot()
    with pytest.raises(ValidationError):
        before.max_cost_cny=Decimal('100')
    dispatched(budget,'1')
    assert before.committed_cost_cny==0 and budget.snapshot().committed_cost_cny==1


@pytest.mark.parametrize('request_id',['','not-uuid','A'*32,True])
def test_invalid_request_identity_is_rejected(request_id: object) -> None:
    """请求必须使用程序生成的32位小写UUID标识，非法编号不创建账目。"""
    budget=ledger()
    with pytest.raises(ValueError):
        budget.reserve(request_id,Decimal('1'))
    assert budget.snapshot().committed_cost_cny==0


def test_dispatched_receipts_can_settle_after_other_request_halts() -> None:
    """一份未知回执停用新请求，但不能阻止另一份已发出请求如实记账。"""
    budget=ledger()
    first=dispatched(budget,'1')
    other=dispatched(budget,'2')
    budget.settle(first,None)
    budget.settle(other,Decimal('0.2'))
    status=budget.snapshot()
    assert status.halt_reason=='usage_unknown' and status.known_cost_subtotal_cny==Decimal('0.2')
    assert status.unknown_reserved_cny==Decimal('1') and status.pending_requests==0



@pytest.mark.parametrize('value',[Decimal('0'),Decimal('-1'),True,'2',Decimal('NaN'),Decimal('Infinity')])
def test_runtime_limit_rejects_invalid_values(value: object) -> None:
    """运行参数不能用零、负数或非法类型绕过模式上限。"""
    with pytest.raises(ValueError):
        ledger('pilot',value)


@pytest.mark.parametrize('operation',['dispatch','settle','cancel'])
def test_unknown_request_transition_fails_closed(operation: str) -> None:
    """不存在的请求不能凭空结算或退款，冲突后拒绝新预占。"""
    budget=ledger()
    request_id=uuid4().hex
    with pytest.raises(ValueError):
        if operation=='dispatch':
            budget.mark_dispatched(request_id)
        elif operation=='settle':
            budget.settle(request_id,Decimal('0.1'))
        else:
            budget.cancel_unsent(request_id)
    assert budget.snapshot().halt_reason=='ledger_conflict'
    assert budget.snapshot().committed_cost_cny==0
    assert not budget.reserve(uuid4().hex,Decimal('1')).allowed
