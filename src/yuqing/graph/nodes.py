"""构造有界案件图节点；具体Agent与台账操作由可信服务适配器注入。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.errors import GraphBubbleUp

from src.yuqing.graph.state import CaseState
from src.yuqing.report.brief import render_case_brief


NodeHandler = Callable[[CaseState], dict[str, Any]]
_PARALLEL_WRITE_KEYS = {
    "fact_agent": {"fact", "fact_rounds", "usage", "failures"},
    "propagation_agent": {"propagation", "prop_rounds", "leads", "usage", "failures"},
}


@dataclass(frozen=True)
class CaseNodeServices:
    """由运行组合根提供已绑定预算、权限、台账与Trace的节点服务。"""

    prepare: NodeHandler
    fact_round1: NodeHandler
    propagation_round1: NodeHandler
    merge_round: NodeHandler
    arbitrate: NodeHandler
    validate_action: NodeHandler
    fact_followup: NodeHandler
    apply_result: NodeHandler
    score: NodeHandler
    human_gate: NodeHandler
    brief: NodeHandler | None = None
    alert_gate: NodeHandler | None = None
    propagation_followup: NodeHandler | None = None
    propagation_review: NodeHandler | None = None


def _safe_call(name: str, handler: NodeHandler, state: CaseState,
               *, recoverable: bool = False) -> dict[str, Any]:
    """把适配器异常变成带节点身份的失败记录，避免伪装成成功结果。"""
    try:
        result = handler(state)
        allowed = _PARALLEL_WRITE_KEYS.get(name)
        if allowed is not None and not set(result).issubset(allowed):
            raise ValueError("parallel_agent_state_write_not_allowed")
        return result
    except GraphBubbleUp:
        raise
    except Exception as exc:
        patch: dict[str, Any] = {"failures": [{"node": name, "type": type(exc).__name__}]}
        reason_code = _controlled_m6b_reason(exc)
        if reason_code is not None:
            patch["failures"][0]["reason_code"] = reason_code
        if recoverable:
            patch.update(_recoverable_failure_patch(name, state))
        else:
            patch["stop_reason"] = f"{name}_failed"
        return patch


def _controlled_m6b_reason(exc: Exception) -> str | None:
    """Expose only fixed diagnostic categories for expected M6b boundary failures."""
    raw = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else ""
    if raw.startswith(("propagation_source_", "propagation_archive_", "propagation_inheritance_")):
        return "source_material_mismatch"
    if raw in {"propagation_final_report_mismatch", "propagation_report_not_qualified"}:
        return "citation_not_qualified"
    if raw == "propagation_followup_claim_mismatch":
        return "claim_mismatch"
    if (raw.startswith("m6b_claim_") or raw == "m6b_pending_task_mismatch"
            or raw.startswith("propagation_followup_")):
        return "action_claim_mismatch"
    return None


def _recoverable_failure_patch(name: str, state: CaseState) -> dict[str, Any]:
    """Record an attempted expert round without erasing previously trusted rounds."""
    if name == "fact_agent":
        return {"fact": None, "fact_rounds": [{"round": 1, "status": "failed",
            "stop_reason": "no_valid_result"}]}
    if name == "propagation_agent":
        return {"propagation": None, "prop_rounds": [{"status": "failed"}]}
    if name == "fact_followup":
        rows = state.get("fact_rounds", [])
        return {"fact_rounds": [{"round": len(rows) + 1, "status": "failed",
            "stop_reason": "no_valid_result"}]}
    if name == "propagation_followup":
        return {"prop_rounds": [{"round": len(state.get("prop_rounds", [])) + 1,
            "status": "failed", "stop_reason": "no_valid_result"}]}
    return {}


def build_nodes(services: CaseNodeServices) -> dict[str, NodeHandler]:
    """生成节点映射；Agent分支只写各自结果，串行节点负责业务状态。"""
    handlers = {
        "prepare": services.prepare,
        "route_round1": _mark_unavailable_fact,
        "fact_agent": services.fact_round1,
        "propagation_agent": services.propagation_round1,
        "merge_round": services.merge_round,
        "arbiter": services.arbitrate,
        "validate_action": services.validate_action,
        "fact_followup": services.fact_followup,
        "propagation_followup": services.propagation_followup or _missing_propagation_followup,
        "propagation_review": services.propagation_review or _missing_propagation_review,
        "apply_result": services.apply_result,
        "score": services.score,
        "human_gate": services.human_gate,
        "alert_gate": services.alert_gate or _deny_alert,
        "brief": services.brief or _brief,
    }
    return {name: _node(name, handler) for name, handler in handlers.items()}


def _deny_alert(_state: CaseState) -> dict[str, Any]:
    """缺少显式告警门卫时保持关闭。"""
    return {"alert": {"status": "blocked", "reason": "alert_gate_not_configured"}}


def _missing_propagation_followup(_state: CaseState) -> dict[str, Any]:
    """缺少显式补查处理器时停止，不能把接受派单冒充已执行。"""
    return {"stop_reason": "propagation_followup_not_configured"}


def _missing_propagation_review(_state: CaseState) -> dict[str, Any]:
    """缺少显式复核处理器时停止，不把仲裁提案冒充已执行。"""
    return {"stop_reason": "propagation_review_not_configured"}


def _mark_unavailable_fact(state: CaseState) -> dict[str, Any]:
    """证据覆盖为none时明确记录事实专家未派发的原因。"""
    if state.get("coverage") != "none":
        return {}
    return {"fact_rounds": [{"status": "not_dispatched", "reason": "evidence_unavailable"}]}


def _brief(state: CaseState) -> dict[str, Any]:
    """Render the deterministic minimum brief from final structured state."""
    return {"brief": render_case_brief(state)}


def _node(name: str, handler: NodeHandler) -> NodeHandler:
    """给单个业务处理器绑定故障策略并保留原有State输入输出合同。"""
    recoverable = name in {"fact_agent", "propagation_agent", "fact_followup", "propagation_followup",
                           "propagation_review"}

    def invoke(state: CaseState) -> dict[str, Any]:
        """运行节点并将可预期基础异常写入失败列表。"""
        return _safe_call(name, handler, state, recoverable=recoverable)

    return invoke
