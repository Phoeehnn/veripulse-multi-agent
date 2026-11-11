"""案件级LangGraph状态合同；运行时对象只通过节点依赖注入。"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class CaseState(TypedDict, total=False):
    """跨节点传递可序列化案件材料，专家并行分支只写各自输出键。"""

    case_id: str
    thread_id: str
    ticket: dict[str, Any]
    sample: dict[str, Any]
    is_demo: bool
    scenario: str
    as_of: str | None
    coverage: str
    collab_config: dict[str, Any]
    budget_plan: dict[str, Any]
    compliance: dict[str, Any]
    fact_rounds: Annotated[list[dict[str, Any]], add]
    prop_rounds: Annotated[list[dict[str, Any]], add]
    leads: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    task_results: list[dict[str, Any]]
    reviews: list[dict[str, Any]]
    fact: dict[str, Any] | None
    fact_selection: dict[str, Any]
    propagation: dict[str, Any] | None
    risk: dict[str, Any] | None
    approval: str | None
    review: dict[str, Any]
    decision_id: str
    alert: dict[str, Any] | None
    brief: str | None
    failures: Annotated[list[dict[str, Any]], add]
    usage: Annotated[list[dict[str, Any]], add]
    action_validation: dict[str, Any]
    pending_task: dict[str, Any] | None
    stop_reason: str | None
    review_recorded: bool
    m6c_enabled: bool
    m6c_force_review: bool
    _m8_capture_shared_decision: bool
    _m8_resume_group: str
