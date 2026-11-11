"""交叉复核结果的纯程序校验，供后续M6c执行路径复用。"""

from src.yuqing.schemas import PropagationReport, ReviewResult


def validate_review_result(
    result: ReviewResult,
    *,
    action_id: str,
    allowed_fields: frozenset[str],
    current_report: PropagationReport,
    returned_comment_ids: frozenset[str],
    available_stats_fields: frozenset[str],
    allowed_peer_refs: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """按设计合同校验复核任务、当前值、修改值和复核者自己的依据。"""
    issues: list[str] = []
    if result.action_id != action_id:
        issues.append("review_action_id_mismatch")
    fields = [item.field for item in result.items]
    if len(fields) != len(set(fields)):
        issues.append("review_fields_duplicate")
    if set(fields) < set(allowed_fields):
        issues.append("review_fields_incomplete")
    own_refs = returned_comment_ids | available_stats_fields
    allowed_values = {
        "narrative": frozenset({"当前事件", "旧事重提", "无法判断"}),
        "emotion_level": frozenset({"低", "中", "高"}),
    }
    for index, item in enumerate(result.items):
        prefix = f"items[{index}]"
        if item.field not in allowed_fields:
            issues.append(f"{prefix}.field:not_assigned")
        if item.old_value != getattr(current_report, item.field):
            issues.append(f"{prefix}.old_value:stale")
        if item.decision == "revised":
            if item.new_value is None or item.new_value not in allowed_values[item.field]:
                issues.append(f"{prefix}.new_value:invalid")
            elif item.new_value == item.old_value:
                issues.append(f"{prefix}.new_value:unchanged")
        elif item.new_value is not None:
            issues.append(f"{prefix}.new_value:not_allowed")
        if item.decision in {"revised", "disagree"} and not item.basis_refs:
            issues.append(f"{prefix}.basis_refs:required")
        elif any(ref not in own_refs for ref in item.basis_refs):
            issues.append(f"{prefix}.basis_refs:not_own_reference")
        if allowed_peer_refs is not None and any(ref not in allowed_peer_refs for ref in item.peer_refs):
            issues.append(f"{prefix}.peer_refs:not_peer_reference")
    return tuple(issues)
