"""评测会话明确新建或恢复，复用唯一持久钱包，不自行执行模型。"""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from typing import Callable, Literal, Mapping

from src.yuqing.eval.dual_budget import DualBudgetCoordinator, price_fingerprint
from src.yuqing.eval.experiment_budget import ExperimentBudget, ExperimentBudgetStatus
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStoreError, LedgerHistory, _manifest_data
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.settings import CostConfig, EvaluationConfig, PROJECT_ROOT


def build_experiment_manifest(experiment_id: str, conditions: Mapping[str, str], prices: CostConfig,
                              config: EvaluationConfig, *, mode: Literal["full", "pilot"],
                              max_run_cost_cny: Decimal | None = None) -> dict[str, object]:
    """规范条件与价格指纹，复用批准额度检查，不负责收集文件内容。"""
    if not isinstance(conditions, Mapping) or not conditions or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str)
        for key, value in conditions.items()):
        raise ValueError("invalid_experiment_conditions")
    frozen = json.dumps(dict(conditions), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    limit = ExperimentBudget(experiment_id, config, mode=mode, max_run_cost_cny=max_run_cost_cny).snapshot().max_cost_cny
    return _manifest_data(dict(experiment_id=experiment_id, mode=mode, max_cost_cny=limit,
        price_fingerprint=price_fingerprint(prices), conditions_fingerprint=sha256(frozen.encode("utf-8")).hexdigest(),
        format_version=1))


def _ledger_path(path: str | Path) -> Path:
    """解析实际路径，账本只能位于项目traces内部，不能把目录本身当文件。"""
    project = PROJECT_ROOT.resolve()
    target, root = Path(path).resolve(), (project / "traces").resolve()
    if not root.is_relative_to(project) or target == root or not target.is_relative_to(root):
        raise ValueError("experiment_ledger_outside_project_traces")
    return target


class ExperimentSession:
    """一场实验持有一个钱包，关闭仅关闭连接，不清空消费或保留额。"""

    def __init__(self, budget: PersistentExperimentBudget) -> None:
        """接收已校验持久钱包，不创建替代内存账本。"""
        self._budget, self._closed, self._lock = budget, False, RLock()

    @property
    def persistent_identity(self) -> tuple[str, str]:
        """Bind one session to both its immutable experiment ID and SQLite file."""
        manifest = self._budget.manifest_snapshot()
        return str(manifest["experiment_id"]), str(self._budget.storage_path)

    @classmethod
    def create(cls, path: str | Path, experiment_id: str, conditions: Mapping[str, str], prices: CostConfig,
               config: EvaluationConfig, *, mode: Literal["full", "pilot"],
               max_run_cost_cny: Decimal | None = None) -> ExperimentSession:
        """新建不覆盖；额度和条件校验失败不生成可用钱包。"""
        return cls._open(path, experiment_id, conditions, prices, config, mode=mode,
                         max_run_cost_cny=max_run_cost_cny, new=True)

    @classmethod
    def resume(cls, path: str | Path, experiment_id: str, conditions: Mapping[str, str], prices: CostConfig,
               config: EvaluationConfig, *, mode: Literal["full", "pilot"],
               max_run_cost_cny: Decimal | None = None) -> ExperimentSession:
        """按完全一致的条件恢复，失败不新建空钱包、不自动解除停止。"""
        return cls._open(path, experiment_id, conditions, prices, config, mode=mode,
                         max_run_cost_cny=max_run_cost_cny, new=False)

    @classmethod
    def _open(cls, path: str | Path, experiment_id: str, conditions: Mapping[str, str], prices: CostConfig,
              config: EvaluationConfig, *, mode: Literal["full", "pilot"], max_run_cost_cny: Decimal | None,
              new: bool) -> ExperimentSession:
        """固定清单与D盘路径后复用既有创建/恢复，不设置自动降级分支。"""
        manifest = build_experiment_manifest(experiment_id, conditions, prices, config, mode=mode,
                                             max_run_cost_cny=max_run_cost_cny)
        factory = PersistentExperimentBudget.create if new else PersistentExperimentBudget.resume
        return cls(factory(_ledger_path(path), manifest, config))

    def coordinator_factory(self, *, group_caps: Mapping[str, Decimal] | None = None
                            ) -> Callable[[BudgetTracker], DualBudgetCoordinator]:
        """提供复用本会话钱包及可选原子分组额度的工厂。"""
        caps = None if group_caps is None else dict(group_caps)
        with self._lock:
            self._ensure_open()
        def bind(task: BudgetTracker) -> DualBudgetCoordinator:
            """仅组装两层钱包，不发请求、不新建或重新打开数据库。"""
            with self._lock:
                self._ensure_open()
                with self._budget.coordination():
                    return DualBudgetCoordinator(task, self._budget, group_caps=caps)
        return bind

    def _ensure_open(self) -> None:
        """会话关闭不可当成免费重置。"""
        if self._closed:
            raise ExperimentBudgetStoreError("experiment_session_closed")

    def snapshot(self) -> ExperimentBudgetStatus:
        """读取独立状态，关闭后仍可核对最后记录，不解除停止。"""
        return self._budget.snapshot()

    def load_history(self) -> LedgerHistory:
        """公开读取本会话校验历史，不暴露私有存储或重建钱包。"""
        with self._lock:
            self._ensure_open()
            return self._budget.load_history()

    def halt(self, reason: str) -> None:
        """复用持久钱包停账，失败继续抛出供调度器明确报告未确认。"""
        with self._lock:
            self._ensure_open()
            self._budget.halt(reason)

    def close(self) -> None:
        """幂等关闭连接，不取消预占，不删除或重写账本。"""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._budget.close()
