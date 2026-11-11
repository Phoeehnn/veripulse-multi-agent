"""固定核查的受预算终局包装，不启动整场评测或恢复旧运行。"""

import logging
from pathlib import Path
from typing import Callable, Mapping

from src.yuqing.agents.fact_check import verify
from src.yuqing.eval.dual_budget import DualBudgetCoordinator
from experiments.archive.e2_checkpoint import PlannedRun
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.trace import RunContext, record_event, use_run_context
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.schemas import EvidenceChunk, ExpertOpinion, FactVerdict

logger = logging.getLogger(__name__)
SingleResult = tuple[FactVerdict, ExpertOpinion, list[EvidenceChunk]]


def _halt_after_audit_failure(coordinator: DualBudgetCoordinator | None) -> None:
    """终局未确认时停止已绑定的两层钱包，不退款也不重试审计。"""
    if coordinator is None:
        return
    coordinator.task.halt("run_outcome_audit_failed")
    try:
        if not coordinator.experiment.snapshot().halted:
            coordinator.halt("run_outcome_audit_failed")
    except Exception as exc:
        logger.error("单次终局失败后整场停用未确认 type=%s", type(exc).__name__)


def _validate_startup(context: RunContext, identity: Mapping[str, str],
                      factory: Callable[[BudgetTracker], DualBudgetCoordinator]) -> dict[str, str]:
    """强制单次组完整身份和共享工厂，不允许降级到旧请求路径。"""
    if not callable(factory) or not isinstance(identity, Mapping):
        raise ValueError("missing_shared_budget")
    tags = dict(identity)
    slot = PlannedRun.model_validate({key: tags.get(key) for key in ("sample_id", "group_id", "run_id")})
    if context.agent != "fact" or slot.group_id != "single_shot" or (
        slot.sample_id, slot.run_id) != (context.sample_id, context.run_id):
        raise ValueError("single_run_identity_mismatch")
    return tags


def run_single_with_terminal(claim: str, index: BM25Index, system_prompt: str, context: RunContext, *,
                              store_path: Path, coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator],
                              experiment_identity: Mapping[str, str]) -> SingleResult | None:
    """调用原verify一次并写唯一终局；已有Trace拒绝执行，审计失败不交付成功。"""
    coordinator: DualBudgetCoordinator | None = None
    result: SingleResult | None = None
    reason = "no_valid_result"
    try:
        ctx = RunContext.model_validate(context.model_dump())
        if ctx.trace_path.exists():
            logger.error("单次运行已有Trace，拒绝重复执行")
            return None
        with use_run_context(ctx):
            try:
                tags = _validate_startup(ctx, experiment_identity, coordinator_factory)
                def bind(budget: BudgetTracker) -> DualBudgetCoordinator:
                    """复用作者提供的共享工厂，同时保存绑定以处理终局审计失败。"""
                    nonlocal coordinator
                    bound = coordinator_factory(budget)
                    if not isinstance(bound, DualBudgetCoordinator) or bound.task is not budget:
                        raise ValueError("coordinator_budget_mismatch")
                    coordinator = bound
                    return bound
                result = verify(claim, index, system_prompt, store_path=store_path,
                    coordinator_factory=bind, experiment_identity=tags)
                reason = "completed" if result is not None else "no_valid_result"
            except Exception as exc:
                reason = "startup_or_execution_error"
                logger.error("单次运行启动或执行失败 type=%s", type(exc).__name__)
            try:
                record_event("run_outcome", {"outcome_class": "submitted_judgment" if result is not None else "no_valid_result",
                    "reason": reason})
            except Exception as exc:
                _halt_after_audit_failure(coordinator)
                logger.error("单次终局审计未确认 type=%s", type(exc).__name__)
                return None
            return result
    except Exception as exc:
        logger.error("单次运行上下文建立或恢复失败 type=%s", type(exc).__name__)
        return None
