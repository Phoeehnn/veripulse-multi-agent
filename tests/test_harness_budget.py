"""Selected original budget reservation tests; no datasets or provider calls."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import pytest
from pydantic import ValidationError
from src.yuqing import settings
from src.yuqing.harness.budget import BudgetTracker

@pytest.fixture
def config() -> settings.HarnessConfig:
    """复制配置确定的配置，测试不改实际文件。"""
    raw = settings.yaml.safe_load((settings.PROJECT_ROOT / "config/weights.yaml").read_text(encoding="utf-8"))
    return settings.HarnessConfig.model_validate({**raw["harness"],
        "token_budget_per_call": 6000, "final_token_reserve": 8000})


@pytest.fixture
def prices() -> settings.CostConfig:
    """使用已确认单价，仅算账，不请求模型。"""
    return settings.CostConfig(input_per_1m=2, output_per_1m=8)


def test_reservation_is_replaced_by_actual_usage_without_double_settlement(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """预占6000+2000后按1000+500结算，只计一次实际用量。"""
    tracker = BudgetTracker(config, prices)
    assert tracker.reserve_model_call(6000, 2000).allowed
    assert tracker.tokens == 8000 and tracker.cost_cny == Decimal("0.028")
    assert tracker.reserve_model_call(100, 100).reason == "call_pending"
    assert tracker.reserve_tool_call("search_evidence").reason == "budget_unavailable"
    assert tracker.settle_model_call(1000, 500).allowed
    assert tracker.tokens == 1500 and tracker.cost_cny == Decimal("0.006")
    with pytest.raises(ValueError, match="重复结算"):
        tracker.settle_model_call(0, 0)
    assert tracker.tokens == 1500 and tracker.steps == 1


def test_normal_calls_preserve_tokens_for_one_final_call(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """普通调用不能吃掉8000预留，收尾仍在50000总预算以内。"""
    tracker = BudgetTracker(config, prices)
    for _ in range(5):
        assert tracker.reserve_model_call(6000, 2000).allowed
        assert tracker.settle_model_call(6000, 2000).allowed
    assert tracker.reserve_model_call(6000, 2000).reason == "token_limit"
    assert tracker.tokens == 40000 and tracker.steps == 5
    assert tracker.reserve_model_call(6000, 2000, final=True).allowed
    assert tracker.settle_model_call(6000, 2000).allowed
    assert tracker.tokens == 48000 and tracker.cost_cny == Decimal("0.168")
    assert not tracker.reserve_model_call(1, 1, final=True).allowed
    assert not tracker.reserve_model_call(1, 1).allowed
    assert tracker.reserve_tool_call("read_evidence").reason == "final_only"
    assert tracker.reserve_tool_call("submit_verdict").allowed
    assert not tracker.reserve_tool_call("submit_verdict").allowed


def test_retries_consume_step_quota_and_final_is_separate_but_bounded(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """六次实际请求含格式重试占六步，收尾只有一次且占相同总预算。"""
    tracker = BudgetTracker(config, prices)
    for _ in range(config.max_steps):
        assert tracker.reserve_model_call(100, 100).allowed
        assert tracker.settle_model_call(100, 100).allowed
    assert tracker.steps == 6
    assert tracker.reserve_model_call(100, 100).reason == "step_limit"
    assert tracker.reserve_model_call(100, 100, final=True).allowed
    assert tracker.settle_model_call(100, 100).allowed
    assert tracker.tokens == 1400 and not tracker.reserve_model_call(100, 100, final=True).allowed


def test_correction_preset_reserves_two_closures_and_allows_one_submission_only() -> None:
    """C档首个普通请求须留下两次最大收尾的额度，纠错后即封闭。"""
    prices = settings.CostConfig(input_per_1m=2, output_per_1m=8)
    base = settings.get_propagation_harness_config()
    config = settings.PropagationHarnessConfig.model_validate(
        base.model_dump() | {"max_cost_cny": 0.12})
    tracker = BudgetTracker(config, prices, correction_enabled=True)
    bound, output = config.token_budget_per_call, config.max_output_tokens
    for final, correction in ((False, False), (True, False), (True, True)):
        assert tracker.reserve_model_call(bound, output, final=final, correction=correction).allowed
        assert tracker.settle_model_call(bound, output).allowed
    assert tracker.cost_cny == Decimal("0.108")
    assert tracker.final_used and tracker.correction_used and tracker.steps == 1
    assert tracker.reserve_model_call(bound, output, final=True, correction=True).reason == "correction_unavailable"
    assert tracker.reserve_model_call(100, 100, correction=True).reason == "correction_unavailable"
    assert tracker.reserve_tool_call("search_comments").reason == "final_only"


@pytest.mark.parametrize("tool,limit", [("search_evidence", 3), ("read_evidence", 6), ("load_skill", 2)])
def test_tool_attempts_use_quota_before_execution(
    config: settings.HarnessConfig, prices: settings.CostConfig, tool: str, limit: int,
) -> None:
    """额度在工具运行前消耗，执行失败不会让次数免费恢复。"""
    tracker = BudgetTracker(config, prices)
    for _ in range(limit):
        assert tracker.reserve_tool_call(tool).allowed
    assert not tracker.reserve_tool_call(tool).allowed
    assert tracker.tool_calls[tool] == limit
    assert not tracker.reserve_tool_call("shell").allowed
    assert tracker.tool_calls["shell"] == 0


def test_verdict_submission_is_once_per_model_response(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """提交次数由已预算的响应轮次约束，无模型响应不能凭空提交。"""
    tracker = BudgetTracker(config, prices)
    assert not tracker.reserve_tool_call("submit_verdict").allowed
    for _ in range(2):
        assert tracker.reserve_model_call(100, 100).allowed
        assert tracker.settle_model_call(100, 100).allowed
        assert tracker.reserve_tool_call("submit_verdict").allowed
        assert not tracker.reserve_tool_call("submit_verdict").allowed
    assert tracker.tool_calls["submit_verdict"] == 2


@pytest.mark.parametrize("actual", [(None, 500), (500, None), (-1, 500), (True, 500), (1.5, 500)])
def test_unknown_or_invalid_usage_keeps_reservation_and_stops_paid_calls(
    config: settings.HarnessConfig, prices: settings.CostConfig, actual: tuple,
) -> None:
    """未知用量不等于零；保留预占并拒绝普通和收尾付费请求。"""
    tracker = BudgetTracker(config, prices)
    assert tracker.reserve_model_call(6000, 2000).allowed
    assert tracker.settle_model_call(*actual).reason == "usage_unknown"
    assert tracker.tokens == 8000 and tracker.cost_cny == Decimal("0.028")
    assert not tracker.reserve_model_call(100, 100).allowed
    assert not tracker.reserve_model_call(100, 100, final=True).allowed
    assert not tracker.reserve_tool_call("search_evidence").allowed


@pytest.mark.parametrize("actual", [(101, 100), (100, 101)])
def test_actual_usage_exceeding_reservation_is_recorded_and_stops(
    config: settings.HarnessConfig, prices: settings.CostConfig, actual: tuple[int, int],
) -> None:
    """服务端超预占不能抹掉实际消耗，也不能继续假装预算可靠。"""
    tracker = BudgetTracker(config, prices)
    assert tracker.reserve_model_call(100, 100).allowed
    assert tracker.settle_model_call(*actual).reason == "reservation_exceeded"
    assert tracker.tokens == sum(actual)
    assert not tracker.reserve_model_call(1, 1, final=True).allowed


def test_cost_boundary_is_inclusive_and_final_cost_is_reserved(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """恰好等于上限可用，超出时在模型请求前拒绝且不改变计数。"""
    small = settings.HarnessConfig.model_validate({**config.model_dump(), "max_cost_cny": 0.056})
    tracker = BudgetTracker(small, prices)
    assert tracker.reserve_model_call(6000, 2000).allowed
    assert tracker.settle_model_call(6000, 2000).allowed
    assert tracker.reserve_model_call(1, 1).reason == "cost_limit"
    assert tracker.tokens == 8000 and tracker.steps == 1
    assert tracker.reserve_model_call(6000, 2000, final=True).allowed
    assert tracker.settle_model_call(6000, 2000).allowed
    assert tracker.cost_cny == Decimal("0.056")


def test_exact_total_token_boundary_is_allowed(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """整次预算刚好等于普通加收尾时两次都可用。"""
    small = settings.HarnessConfig.model_validate({**config.model_dump(), "max_total_tokens": 16000})
    tracker = BudgetTracker(small, prices)
    assert tracker.reserve_model_call(6000, 2000).allowed
    assert tracker.settle_model_call(6000, 2000).allowed
    assert not tracker.reserve_model_call(1, 1).allowed
    assert tracker.reserve_model_call(6000, 2000, final=True).allowed
    assert tracker.settle_model_call(6000, 2000).allowed
    assert tracker.tokens == 16000


@pytest.mark.parametrize("planned_usage", [(6001, 100), (100, 2001), (-1, 100), (True, 100), (1.5, 100)])
def test_invalid_or_oversized_request_does_not_spend_budget(
    config: settings.HarnessConfig, prices: settings.CostConfig, planned_usage: tuple,
) -> None:
    """非整数、负数或单次超限请求在预占前拒绝。"""
    tracker = BudgetTracker(config, prices)
    assert not tracker.reserve_model_call(*planned_usage).allowed
    assert tracker.steps == tracker.tokens == 0 and tracker.cost_cny == 0


def test_parallel_tasks_have_independent_budgets(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """两个任务并行分别记账，不共享计数器或工具次数。"""
    first, second = BudgetTracker(config, prices), BudgetTracker(config, prices)
    def run(tracker: BudgetTracker, count: int) -> None:
        for _ in range(count):
            assert tracker.reserve_model_call(100, 100).allowed
            assert tracker.settle_model_call(100, 100).allowed
            assert tracker.reserve_tool_call("search_evidence").allowed
    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.submit(run, first, 1), pool.submit(run, second, 3)
        left.result()
        right.result()
    assert (first.steps, first.tokens, first.tool_calls["search_evidence"]) == (1, 200, 1)
    assert (second.steps, second.tokens, second.tool_calls["search_evidence"]) == (3, 600, 3)


def test_running_budget_keeps_its_config_and_prices_snapshot(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """外部配置对象变化不能悄悄改掉已开始任务的权限额度或单价。"""
    tracker = BudgetTracker(config, prices)
    config.max_tool_calls["search_evidence"] = 1
    prices.input_per_1m = 100
    assert tracker.reserve_model_call(6000, 2000).allowed
    assert tracker.settle_model_call(6000, 2000).allowed
    assert tracker.cost_cny == Decimal("0.028")
    for _ in range(3):
        assert tracker.reserve_tool_call("search_evidence").allowed
    assert not tracker.reserve_tool_call("search_evidence").allowed


@pytest.mark.parametrize("changes", [
    {"max_steps": 0}, {"max_steps": True}, {"max_tool_calls": {"search_evidence": -1}},
    {"max_total_tokens": 8000}, {"final_token_reserve": 7999}, {"max_cost_cny": float("inf")},
    {"tool_timeout_s": {}}, {"unknown_key": 1},
])
def test_invalid_budget_config_is_rejected(config: settings.HarnessConfig, changes: dict) -> None:
    """坏上限、预留不一致、未知键或遗漏超时都不能悄悄启动。"""
    with pytest.raises(ValidationError):
        settings.HarnessConfig.model_validate({**config.model_dump(), **changes})


def test_unknown_prices_or_insufficient_final_cost_fail_before_reservation(
    config: settings.HarnessConfig, prices: settings.CostConfig,
) -> None:
    """有费用上限时未知价格不能按免费算，最小收尾费用必须可用。"""
    with pytest.raises(ValueError, match="单价尚未确认"):
        BudgetTracker(config, settings.CostConfig())
    low = settings.HarnessConfig.model_validate({**config.model_dump(), "max_cost_cny": 0.001})
    with pytest.raises(ValueError, match="不足以预留"):
        BudgetTracker(low, prices)

