"""Render a deterministic Markdown brief from the serializable case state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_case_brief(state: Mapping[str, Any]) -> str:
    """Render the M5 minimum brief without model calls or external side effects."""
    risk = _mapping(state.get("risk"))
    review = _mapping(state.get("review"))
    modified_disposition = review.get("final_disposition") if review.get("decision") == "modify" else None
    lines = ["# 案件研判简报"]
    if state.get("is_demo"):
        lines.append("> 演示案例")
    lines.extend([
        f"- 案件：{_value(state.get('case_id'))}",
        f"- 风险等级：{_value(risk.get('level'))}",
        f"- 证据充分性：{_value(risk.get('evidence_sufficiency'))}",
        f"- 事实判断：{_value(risk.get('fact_status'))}",
        f"- 程序建议动作：{_value(risk.get('disposition'))}" if modified_disposition else
            f"- 建议动作：{_value(risk.get('disposition'))}",
        *([f"- 人工确定的最终建议动作：{_value(modified_disposition)}"] if modified_disposition else []),
        f"- 审批状态：{_value(state.get('approval'))}",
        f"- 复核记录：{_review_summary(state)}",
        "",
        "## 风险维度",
        _bullet_mapping(risk.get("dimensions") or risk.get("used_dims")),
        f"- 缺席维度：{_value(risk.get('absent_dims') or '无')}",
        f"- 计分权重：{_value(risk.get('weights_used') or '未提供')}",
        "",
        "## 问题台账",
        _issue_table(state.get("issues")),
        "",
        "## 证据链编号",
        _evidence_lines(state),
        "",
        "## 协作动作",
        _action_lines(state.get("actions")),
        "",
        "## 失败与缺席说明",
        _failure_lines(state),
        "",
        "## 用量记录",
        _usage_lines(state.get("usage"), state.get("usage_summary"),
                     is_demo=bool(state.get("is_demo"))),
    ])
    return "\n".join(lines).strip() + "\n"


def _review_summary(state: Mapping[str, Any]) -> str:
    """Render the human decision and its stated reason without inferring approval."""
    review = _mapping(state.get("review"))
    decision = review.get("decision")
    reason = _value(review.get("reason"))
    if decision == "reject":
        return f"人工驳回；理由：{reason}"
    if decision == "modify":
        return f"人工修改建议动作；理由：{reason}"
    if decision == "approve":
        return f"人工批准；理由：{reason}"
    return "无人工意见"


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a read-only mapping view or an empty mapping for absent data."""
    return value if isinstance(value, Mapping) else {}


def _value(value: Any) -> str:
    """Render missing or nested values safely as plain text."""
    return "未提供" if value is None else str(value).replace("\n", " ")


def _bullet_mapping(value: Any) -> str:
    """Render structured risk dimensions without interpreting their meaning."""
    if isinstance(value, Mapping):
        return "\n".join(f"- {key}：{_value(item)}" for key, item in value.items()) or "- 无"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(f"- {item}" for item in value) or "- 无"
    return "- 未提供"


def _issue_table(value: Any) -> str:
    """Render issue identifiers, text, epistemic status, and trusted refs."""
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    header = "| 编号 | 问题 | 证据状态 | 依据编号 |\n|---|---|---|---|"
    body = [
        "| {id} | {text} | {status} | {refs} |".format(
            id=_value(row.get("issue_id")), text=_value(row.get("text")),
            status=_value(row.get("epistemic")),
            refs="、".join(map(str, row.get("evidence_refs", ()))),
        )
        for row in rows if isinstance(row, Mapping)
    ]
    return "\n".join([header, *body]) if body else "无问题记录。"


def _evidence_lines(state: Mapping[str, Any]) -> str:
    """List evidence references by round without inferring their validity."""
    rounds = []
    for key, label in (("fact_rounds", "事实轮次"), ("prop_rounds", "传播轮次")):
        for index, item in enumerate(state.get(key, ()), start=1):
            if isinstance(item, Mapping):
                refs = item.get("delivery_refs") or item.get("ledger_refs") or item.get("returned_cids", ())
                rounds.append(f"- {label} {index}：{_value(item.get('status'))}；编号：{'、'.join(map(str, refs)) or '无'}")
    return "\n".join(rounds) or "无证据交付记录。"


def _action_lines(value: Any) -> str:
    """List recorded collaboration actions and their program outcomes."""
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    lines = [f"- {_value(row.get('kind'))}：{_value(row.get('status') or row.get('reason'))}"
             for row in rows if isinstance(row, Mapping)]
    return "\n".join(lines) or "无协作动作记录。"


def _failure_lines(state: Mapping[str, Any]) -> str:
    """Expose node failures and absent dimensions as explicit records."""
    lines = [f"- 节点 {item.get('node', '未知')}：{item.get('type', '未知')}"
             for item in state.get("failures", ()) if isinstance(item, Mapping)]
    absent = _mapping(_mapping(state.get("risk")).get("absent_dims"))
    lines.extend(f"- 缺席维度 {key}：{value}" for key, value in absent.items())
    return "\n".join(lines) or "无失败或缺席记录。"


def _usage_lines(value: Any, summary: Any = None, *, is_demo: bool = False) -> str:
    """Render case wallet and Trace totals without turning unknown consumption into zero."""
    if is_demo and (not isinstance(summary, Mapping) or summary.get("request_count", 0) == 0):
        return "- 无模型请求（Mock 演示）"
    if isinstance(summary, Mapping):
        cost = summary.get("estimated_cost_cny")
        cost_text = f"¥{cost}" if cost is not None else (
            f"未知（已知小计 ¥{summary.get('known_cost_subtotal_cny', '0.000000')}）")
        lines = [f"- 请求数：{_value(summary.get('request_count'))}",
            f"- 输入 token：{_value(summary.get('input_tokens'))}",
            f"- 输出 token：{_value(summary.get('output_tokens'))}",
            f"- 估算费用：{cost_text}",
            f"- 未知/未结算请求：{_value(summary.get('unknown_or_pending_requests'))}"]
        if summary.get("issues"):
            lines.append(f"- 用量核对：需检查（{', '.join(map(str, summary['issues']))}）")
        return "\n".join(lines)
    rows = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()
    lines = [f"- {_value(row.get('agent') or row.get('node'))}：费用 {_value(row.get('cost_cny'))}；用量 {_value(row.get('input_output_tokens'))}"
             for row in rows if isinstance(row, Mapping)]
    return "\n".join(lines) or "无用量记录。"
