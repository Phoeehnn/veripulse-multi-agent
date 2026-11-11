"""用正式动作门卫逐个计算仲裁材料中的可派问题提示。"""

from __future__ import annotations

from typing import Any, Literal

from src.yuqing.collab.validate import (CaseActionSnapshot, CaseReferenceSnapshot,
                                        IssueSnapshot, validate_action)
from src.yuqing.schemas import CollabAction


def build_issue_dispatchability_cards(
    snapshot: CaseActionSnapshot, references: CaseReferenceSnapshot,
) -> tuple[dict[str, Any], ...]:
    """对每个目标的问题用原门卫试校验，生成不许可动作的可派编号和原因。"""
    cards = []
    for target in ("fact", "propagation"):
        rows = [_issue_row(issue, target, snapshot, references) for issue in snapshot.issues
                if issue.owner == target]
        cards.append({"target": target,
            "dispatchable_issue_ids": [row["issue_id"] for row in rows if row["dispatchable"]],
            "unavailable": [{"issue_id": row["issue_id"], "reason": row["reason"]}
                            for row in rows if not row["dispatchable"]],
            "lead_ids_by_issue": [{"issue_id": row["issue_id"], "lead_ids": row["lead_ids"]}
                                  for row in rows if row["lead_ids"]]})
    return tuple(cards)


def _issue_row(issue: IssueSnapshot, target: Literal["fact", "propagation"],
               snapshot: CaseActionSnapshot, references: CaseReferenceSnapshot) -> dict[str, Any]:
    """Evaluate one synthetic single-issue proposal using the unchanged production gate."""
    action = CollabAction(kind="investigate", target=target, issue_ids=[issue.issue_id],
        question=issue.issue_id, done_when="确认或否定关键事实" if target == "fact" else "补充传播证据",
        lead_ids=_lead_ids(issue))
    result = validate_action(action=action, case_id=snapshot.case_id,
        decision_round=snapshot.decision_round, snapshot=snapshot, references=references)
    return {"issue_id": issue.issue_id, "dispatchable": result.allowed,
            "reason": result.reason, "lead_ids": _lead_ids(issue)}


def _lead_ids(issue: IssueSnapshot) -> list[str]:
    """Keep program lead IDs separate from evidence references such as fact:F1:E1."""
    return [ref.removeprefix("lead:") for ref in issue.origin_refs if ref.startswith("lead:")]
