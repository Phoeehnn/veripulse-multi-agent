"""仅离线回放的24槽两阶段Runner，复用原模型网关和持久预算。"""

from dataclasses import dataclass
from contextlib import nullcontext
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.yuqing import llm
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStoreError
from src.yuqing.eval.fact_v2_experiment import build_fact_v2_node_config
from experiments.archive.scope_pair_audit import audit_slots
from experiments.archive.scope_pair_inputs import ScopeInputs, ScopePacket, assert_unchanged, initial_messages, review_messages
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.fact_v2_contract import DeliveredEvidenceV2, build_fact_run_context_v2, validate_fact_v2_core
from src.yuqing.harness.tools import ToolRegistry, register_v2_submission_tool
from src.yuqing.harness.trace import RunContext, record_event, use_run_context, write_archive
from src.yuqing.settings import CostConfig, PROJECT_ROOT, get_cost_config, get_evaluation_config, get_fact_v2_evaluation_config, get_harness_config


@dataclass(frozen=True)
class ScopeCaps:
    """实验专用额度；不覆盖任何默认配置，收紧值也进入条件指纹。"""

    phase: Decimal
    case: Decimal
    total: Decimal


def proposed_caps() -> ScopeCaps:
    """从已公布题单读取拟定值；本模块没有付费许可或Provider入口。"""
    raw = json.loads((PROJECT_ROOT / "docs/cards/scope-pair-v1-selection.json").read_bytes())["proposed_budget"]
    return ScopeCaps(*(Decimal(raw[key]) for key in ("per_stage_cny", "per_arm_case_cny", "total_cny")))


def slot_order(inputs: ScopeInputs) -> tuple[tuple[str, str], ...]:
    """固定全部12对，并交替S/M次序。"""
    return tuple((p.sample_id, arm) for i, p in enumerate(inputs.packets)
                 for arm in (("S", "M") if i % 2 == 0 else ("M", "S")))


class ScopePairRunner:
    """仅消费显式SDK回放对象，无真实客户端回退或自动重跑。"""

    def __init__(self, inputs: ScopeInputs, caps: ScopeCaps, prices: CostConfig) -> None:
        """复制阶段配置并冻结价格，禁止离线请求配置超过已公布上限。"""
        ceiling = proposed_caps()
        if any(not value.is_finite() or value <= 0 or value > limit for value, limit in
               zip((caps.phase, caps.case, caps.total), (ceiling.phase, ceiling.case, ceiling.total), strict=True)):
            raise ValueError("invalid_scope_caps")
        self.inputs, self.caps = inputs, caps
        self.prices = CostConfig.model_validate(prices.model_dump())
        base = build_fact_v2_node_config(get_harness_config(), get_fact_v2_evaluation_config())
        self.config = type(base).model_validate(base.model_dump() | {"max_cost_cny": float(caps.phase)})
        BudgetTracker(self.config, self.prices)
        self.contract = build_fact_run_context_v2("initial", initial_manifest_path=Path("offline-packet.json"))
        registry = ToolRegistry()
        register_v2_submission_tool(registry, self.contract.schema, self.contract.response_schema)
        self.tools = [registry.get("submit_verdict")]
        self._frozen = self._runtime_fingerprint()

    def _runtime_fingerprint(self) -> str:
        """校验运行时对象未被调用方改写，不包含密钥。"""
        return json.dumps([self.config.model_dump(), self.prices.model_dump(), self.inputs.system,
            self.inputs.reviews, [p.model_dump(mode="json") for p in self.inputs.packets],
            [(t.name, t.risk, t.description, t.args_model.model_json_schema(),
              t.response_model.model_json_schema()) for t in self.tools],
            self.contract.schema_sha256, self.contract.limits.model_dump(), self.caps.__dict__], sort_keys=True, default=str)

    def run_offline(self, root: Path, replay: llm.OfflineCompletionReplay) -> dict[str, object]:
        """强制只读回放，异常后仍发布全部槽位终态并保留未知占款。"""
        if type(replay) is not llm.OfflineCompletionReplay:
            raise TypeError("scope_runner_requires_offline_replay")
        return self._run(root, replay)

    def _run(self, root: Path, replay: llm.OfflineCompletionReplay | None) -> dict[str, object]:
        """共用执行核心；真实入口由scope_pair_launch核验许可并认领。"""
        if not root.resolve().is_relative_to((PROJECT_ROOT / "traces").resolve()):
            raise ValueError("scope_run_outside_project_traces")
        self._check()
        root.mkdir(parents=True, exist_ok=False)
        expected = slot_order(self.inputs)
        rows = [{"sample_id": sid, "arm": arm, "status": "stopped", "reason": "not_started",
                 "stages": [], "report": None} for sid, arm in expected]
        mode = "offline_only" if replay is not None else "live"
        conditions = dict(self.inputs.fingerprints) | {"runtime": self._frozen, "mode": mode}
        session = ExperimentSession.create(root / "wallet.sqlite", uuid4().hex, conditions,
            self.prices, get_evaluation_config(), mode="full", max_run_cost_cny=self.caps.total)
        try:
            error = None
            try:
                with llm.use_offline_completion_replay(replay) if replay is not None else nullcontext():
                    self._execute(rows, root, session)
            except Exception as exc:
                error = type(exc).__name__
            issues = self._audit(rows, session, root)
            if issues and error is None:
                try:
                    session.halt("scope_slot_audit_failed")
                except Exception as exc:
                    error = type(exc).__name__
            output = {"mode": mode, "slots": rows, "audit_issues": issues,
                      "wallet": session.snapshot().model_dump(mode="json"), "conditions": conditions,
                      "execution_error": error}
            if error:
                output["wallet"]["cost_cny"] = None
            with (root / "summary.json").open("x", encoding="utf-8") as stream:
                json.dump(output, stream, ensure_ascii=False, indent=2)
            return output
        finally:
            session.close()

    def _check(self) -> None:
        """在每次请求前及场次结束检查配置与材料冻结。"""
        assert_unchanged(self.inputs)
        if hasattr(self, "launch_check"):
            self.launch_check()
        if self._runtime_fingerprint() != self._frozen:
            raise ValueError("scope_runtime_configuration_drift")

    def _audit(self, rows: list[dict[str, object]], session: ExperimentSession, root: Path) -> list[str]:
        """审计I/O失败也形成终态记录，不能当作干净场次继续。"""
        try:
            return audit_slots(rows, slot_order(self.inputs), session, root,
                (self.caps.phase, self.caps.case, self.caps.total))
        except (OSError, ValueError, KeyError, TypeError, ExperimentBudgetStoreError) as exc:
            return [f"slot_audit_exception:{type(exc).__name__}"]

    def _execute(self, rows: list[dict[str, object]], root: Path, session: ExperimentSession) -> None:
        """严格按槽执行，未知费用/审计异常立即终止后续请求。"""
        packets = {p.sample_id: p for p in self.inputs.packets}
        for row in rows:
            if session.snapshot().halted:
                row["reason"] = "experiment_halted"
                continue
            row["status"], row["reason"] = "failed", "execution_error"
            try:
                self._slot(row, packets[row["sample_id"]], root, session)
                self._check()
                if self._audit(rows, session, root):
                    session.halt("scope_slot_audit_failed")
            except Exception as exc:
                row["status"] = "fallback" if row["report"] is not None else "failed"
                row["reason"] = f"offline_exception:{type(exc).__name__}"
                session.halt("scope_execution_or_freeze_error")

    def _slot(self, row: dict[str, object], packet: ScopePacket, root: Path, session: ExperimentSession) -> None:
        """两阶段同校验，第二阶段退卷回退首稿；不按Gold挑选。"""
        messages = initial_messages(self.inputs, packet)
        for stage in (1, 2):
            self._check()
            result = self._stage(row, packet, messages, root, session, stage)
            if not row["stages"][-1]["accepted"]:
                row["status"] = "fallback" if row["report"] is not None else "failed"
                row["reason"] = row["stages"][-1]["reason"]
                return
            row["report"] = row["stages"][-1]["report"]
            if stage == 1:
                messages = review_messages(self.inputs, messages, row["arm"], result.assistant_message)
        row["status"], row["reason"] = "completed", "both_stages_accepted"

    def _stage(self, row: dict[str, object], packet: ScopePacket, messages: list[dict[str, object]],
               root: Path, session: ExperimentSession, stage: int) -> object:
        """通过真实网关预占、结算一次；只允许提交，无额外搜索阅读。"""
        ctx = RunContext(run_id=uuid4().hex, sample_id=packet.sample_id, agent="fact_investigator", trace_root=root)
        item = {"stage": stage, "run_id": ctx.run_id, "trace_path": str(ctx.trace_path),
                "status": "startup_error", "reason": "interrupted", "request_id": None,
                "accepted": False, "report": None, "cost_cny": "0", "elapsed_s": None}
        row["stages"].append(item)
        budget = BudgetTracker(self.config, self.prices)
        caps = {f"{arm}/{p.sample_id}": self.caps.case for p in self.inputs.packets for arm in ("S", "M")}
        coordinator = session.coordinator_factory(group_caps=caps)(budget)
        identity = dict(sample_id=packet.sample_id, group_id=f'{row["arm"]}/{packet.sample_id}',
                        run_id=ctx.run_id, variant=f"stage{stage}")
        started = perf_counter()
        with use_run_context(ctx):
            outcome = llm.call_with_tools_outcome(messages, self.tools, budget, final=True,
                coordinator=coordinator, experiment_identity=identity)
            item.update(status=outcome.status, reason=outcome.reason, request_id=outcome.request_id,
                cost_cny=None if budget.status().halt_reason == "usage_unknown" else str(budget.cost_cny),
                elapsed_s=perf_counter() - started)
            if outcome.result is not None:
                checked = validate_fact_v2_core(outcome.result.tool_calls[0].arguments,
                    delivered={f"E{i}": DeliveredEvidenceV2(text=e.text, read_end=e.read_end)
                        for i, e in enumerate(packet.entries, start=1)}, limits=self.contract.limits,
                    plan_max_text_chars=self.contract.plan_max_text_chars, mode="initial")
                item.update(accepted=checked.validation.accepted,
                    report=checked.report.model_dump(mode="json") if checked.validation.accepted else None,
                    reason="accepted" if checked.validation.accepted else "citation_or_core_rejected")
                archive = write_archive("submission", {"arguments": outcome.result.tool_calls[0].arguments,
                    "validation": checked.validation.model_dump(mode="json")})
                record_event("scope_submission", {"archive": archive, "accepted": item["accepted"]})
            if budget.status().halted or outcome.status in {"startup_error", "audit_error", "reservation_exceeded", "usage_unknown"}:
                session.halt(outcome.reason)
        return outcome.result
