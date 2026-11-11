"""把已校验的案件节点接成有界LangGraph并隔离每次运行的检查点。"""

from __future__ import annotations

from uuid import uuid4
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.yuqing.graph.nodes import CaseNodeServices, build_nodes
from src.yuqing.graph.state import CaseState
from src.yuqing.graph.fact_adapter import FactNodeAdapter
from src.yuqing.settings import GraphConfig, get_graph_config, get_m5_run_budget_config, get_m6c_graph_config


def build_case_graph(config: GraphConfig, services: CaseNodeServices, *, checkpointer: BaseCheckpointSaver | None = None,
                     m6c: bool = False
                     ) -> CompiledStateGraph:
    """编译M5基础图；只接受已实现的冻结事实V2路径和独立M5预算档。"""
    config = GraphConfig.model_validate(config.model_dump())
    if config.fact_node_version != "v2":
        raise ValueError("selected_fact_node_version_not_implemented")
    if config.checkpointer == "sqlite" and checkpointer is None:
        raise ValueError("sqlite_checkpointer_required")
    expected = get_m6c_graph_config() if m6c else get_graph_config()
    if config.model_dump() != expected.model_dump():
        raise ValueError("graph_config_drift")
    FactNodeAdapter(config.fact_node_version)
    get_m5_run_budget_config()
    nodes = build_nodes(services)
    graph = StateGraph(CaseState)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", _after_prepare,
        {"round1": "route_round1", "score": "score", "resume_fixed": "validate_action"})
    graph.add_conditional_edges("route_round1", _round1_destinations,
        {"fact_agent": "fact_agent", "propagation_agent": "propagation_agent"})
    graph.add_edge("fact_agent", "merge_round")
    graph.add_edge("propagation_agent", "merge_round")
    graph.add_conditional_edges("merge_round", _after_merge,
        {"arbiter": "arbiter", "score": "score", "capture": END})
    graph.add_conditional_edges("arbiter", _after_arbiter,
        {"validate_action": "validate_action", "score": "score", "capture": END})
    graph.add_conditional_edges("validate_action", _after_validation,
        {"fact_followup": "fact_followup", "propagation_followup": "propagation_followup",
         "propagation_review": "propagation_review", "score": "score"})
    graph.add_edge("fact_followup", "apply_result")
    graph.add_conditional_edges("propagation_followup", _after_propagation_followup,
        {"apply_result": "apply_result", "score": "score"})
    graph.add_conditional_edges("propagation_review", _after_propagation_followup,
        {"apply_result": "apply_result", "score": "score"})
    graph.add_conditional_edges("apply_result", _after_apply,
        {"arbiter": "arbiter", "score": "score"})
    graph.add_edge("score", "human_gate")
    graph.add_conditional_edges("human_gate", route_after_review,
        {"alert": "alert_gate", "brief": "brief"})
    graph.add_edge("alert_gate", "brief")
    graph.add_edge("brief", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def invoke_case_graph(graph: CompiledStateGraph, state: CaseState, *, thread_id: str | None = None) -> dict[str, Any]:
    """Invoke a compiled graph under a unique checkpoint thread and safe recursion cap."""
    config = get_m6c_graph_config() if state.get("m6c_enabled") is True else get_graph_config()
    thread_id = thread_id or uuid4().hex
    initial = dict(state)
    initial["thread_id"] = thread_id
    return graph.invoke(initial, config={"configurable": {"thread_id": thread_id},
                                         "recursion_limit": config.recursion_limit})


def resume_case_graph(graph: CompiledStateGraph, thread_id: str,
                      review: dict[str, Any]) -> dict[str, Any]:
    """Resume one paused case on its original checkpoint thread."""
    config = get_graph_config()
    return graph.invoke(Command(resume=review), config={"configurable": {"thread_id": thread_id},
        "recursion_limit": config.recursion_limit})


def route_after_review(state: CaseState) -> str:
    """Only a validated approval may reach the alert gate."""
    review = state.get("review") or {}
    if state.get("approval") == "approved" and review.get("decision") in {"approve", "modify"}:
        return "alert"
    return "brief"


def _after_prepare(state: CaseState) -> str:
    """M8受信分支可从共享仲裁快照恢复；其他运行保持原路由。"""
    if state.get("_m8_capture_shared_decision") is True:
        return "round1"
    if state.get("stop_reason"):
        return "score"
    if state.get("_m8_resume_group") == "B4-nf":
        return "score"
    if state.get("_m8_resume_group") == "B4-fixed":
        return "resume_fixed"
    return "round1"


def _after_merge(state: CaseState) -> str:
    """Do not ask the arbiter to act on a failed ledger merge."""
    if state.get("stop_reason"):
        return "capture" if state.get("_m8_capture_shared_decision") is True else "score"
    return "arbiter"


def _after_apply(state: CaseState) -> str:
    """A broken state application closes safely; recoverable follow-up failures still reach arbiter."""
    return "score" if state.get("stop_reason") else "arbiter"


def _round1_destinations(state: CaseState) -> list[str]:
    """无事实证据覆盖时只调度传播分支，不等待未启动的事实分支。"""
    if state.get("coverage") == "none":
        return ["propagation_agent"]
    return ["fact_agent", "propagation_agent"]


def _after_arbiter(state: CaseState) -> str:
    """首次决策交校验；任何已执行动作后不再请求仲裁模型。"""
    if state.get("_m8_capture_shared_decision") is True:
        return "capture"
    decisions = state.get("decisions", [])
    if state.get("stop_reason") or not decisions:
        return "score"
    latest = decisions[-1]
    if state.get("m6c_enabled") is True:
        maximum = state["collab_config"]["max_actions"]
        if len(state.get("actions", [])) >= maximum:
            return "score"
        if not latest.get("action") or latest.get("next_action") == "finalize":
            return "score"
        return "validate_action"
    if state.get("actions"):
        return "score"
    return "score" if latest.get("next_action") == "finalize" else "validate_action"


def _after_validation(state: CaseState) -> str:
    """只有程序明确接受事实补查才执行；拒绝、预算停账或未知状态直接定稿。"""
    result = state.get("action_validation") or {}
    action = result.get("action") or {}
    if (result.get("accepted") and action.get("kind") == "investigate"
            and action.get("target") == "propagation" and not state.get("stop_reason")):
        return "propagation_followup"
    if (result.get("accepted") and action.get("kind") == "review"
            and action.get("target") == "propagation" and state.get("m6c_enabled") is True
            and not state.get("stop_reason")):
        return "propagation_review"
    if (result.get("accepted") and action.get("target") == "fact"
            and state.get("coverage") != "none" and not state.get("stop_reason")):
        return "fact_followup"
    return "score"


def _after_propagation_followup(state: CaseState) -> str:
    """只有已装配的传播补查可继续回写，缺失处理器受控定稿。"""
    return "score" if state.get("stop_reason") else "apply_result"
