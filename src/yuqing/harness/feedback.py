"""把程序校验结果转换成不回显模型内容的有限反馈。"""

from __future__ import annotations

from typing import Iterable, Mapping

from pydantic import BaseModel, ValidationError

from src.yuqing.schemas import KeyFactV2, ToolValidationIssue
from src.yuqing.settings import get_fact_v2_limits_config

_FACT_PATHS = {
    "key_facts[].text", "key_facts[].status", "key_facts[].evidence_ids", "key_facts[].quotes[]",
    "reasons[].text", "reasons[].evidence_ids", "label", "confidence",
}
_PROPAGATION_FIELDS = {
    "reposts", "comments", "likes", "root_followers", "root_verified",
    "nodes_1h", "nodes_6h", "nodes_24h", "peak_nodes_per_hour", "max_depth",
    "direct_reposts", "text_repost_ratio", "visible_comments",
}
_FORMAT_REASONS = {
    "text_without_tools": "请按当前任务调用已提供的工具，不要用普通文字代替提交。",
    "invalid_tool_arguments": "工具参数未符合当前工具 Schema；按下方字段和约束修正。",
    "invalid_json": "JSON 输出未符合当前 Schema；按下方字段和约束修正。",
    "mixed_tool_batch": "提交工具必须单独调用；不要把提交与其他工具放在同一批。",
}
_ERROR_TYPES = {
    "missing": "缺少必填内容", "extra_forbidden": "含有 Schema 未定义字段",
    "string_too_short": "文本短于最小长度", "string_too_long": "文本超过最大长度",
    "greater_than": "数值未达到下限", "greater_than_equal": "数值低于下限",
    "less_than": "数值超过上限", "less_than_equal": "数值超过上限",
    "literal_error": "枚举值不合法", "enum": "枚举值不合法",
    "string_pattern_mismatch": "文本格式不符合要求", "int_type": "应填写整数",
    "float_type": "应填写数字", "string_type": "应填写文本",
    "list_type": "应填写列表", "model_type": "对象结构不符合要求",
}


def format_retry_feedback(reason: str, issues: tuple[ToolValidationIssue, ...],
                          tool_schemas: Mapping[str, type[BaseModel]], max_chars: int) -> str:
    """用受信任工具 Schema 解释有限的结构错误，不回显未知名称或原始内容。"""
    message = _FORMAT_REASONS.get(reason, "本次响应未通过工具协议检查；请遵守当前工具 Schema。")
    bundles = [issue for issue in issues if issue.trusted_feedback]
    bundle_locs = {issue.loc for issue in bundles}
    details = [_describe_validation_issue(issue, tool_schemas) for issue in issues
               if not (issue.reason_code in {"fact_quote_count_invalid", "fact_evidence_quote_set_mismatch"}
                       and any(_same_fact(issue.loc, loc) for loc in bundle_locs))]
    details = [item for item in details if item]
    details.extend(issue.trusted_feedback for issue in bundles if issue.trusted_feedback)
    text = "程序格式反馈（不是证据）：" + message
    if details:
        text += "\n" + "\n".join(dict.fromkeys(details))
    text += "\n请按现有额度继续；不要编造字段、材料或编号。"
    return text[:max_chars]


def fact_submission_feedback(issues: tuple[str, ...], *, parsed_report: object = None) -> tuple[str, ...]:
    """将固定原因码与可信解析后的事实摘要合并为安全反馈。"""
    bundles = _fact_bundle_feedback(parsed_report) if issues else ()
    merged = tuple(message for _, message in bundles)
    bundled_indices = {index for index, _ in bundles}
    messages = []
    for issue in issues:
        path, _, detail = issue.partition(":")
        code = detail.strip().split(maxsplit=1)[0] if detail.strip() else issue
        if code in {"fact_quote_count_invalid", "fact_evidence_quote_set_mismatch"} and _fact_index(path) in bundled_indices:
            continue
        canonical = _canonical_fact_path(path.strip())
        message = _FACT_ISSUES.get(code) or _fact_schema_message(code)
        if canonical and message:
            messages.append(f"{canonical}：{message}")
        elif message:
            messages.append(message)
        else:
            messages.append("提交内容：有一项格式或证据检查未通过，请按 Schema 和已读证据修正。")
    return tuple(dict.fromkeys((*merged, *messages)))


def _fact_index(path: str) -> int | None:
    """读取校验路径中的关键事实索引，不读取路径正文内容。"""
    parts = path.replace("[", ".").replace("]", "").split(".")
    try:
        position = parts.index("key_facts")
        return int(parts[position + 1])
    except (ValueError, IndexError):
        return None


def _fact_bundle_feedback(parsed_report: object) -> tuple[tuple[int, str], ...]:
    """按可信单事实投影生成B反馈，不合并不同事实的问题。"""
    if not isinstance(parsed_report, BaseModel):
        return ()
    facts = getattr(parsed_report, "key_facts", None)
    if not isinstance(facts, list):
        return ()
    limit = get_fact_v2_limits_config().max_quotes_per_direction
    messages: list[tuple[int, str]] = []
    for index, candidate in enumerate(facts):
        try:
            raw = candidate.model_dump(mode="python") if isinstance(candidate, BaseModel) else candidate
            fact = KeyFactV2.model_validate(raw, strict=True)
        except (ValidationError, TypeError, ValueError):
            continue
        status = _field(fact, "status")
        evidence_ids = _field(fact, "evidence_ids")
        quotes = _field(fact, "quotes")
        if not isinstance(evidence_ids, list) or not isinstance(quotes, list):
            continue
        if len(evidence_ids) != len(set(evidence_ids)):
            continue
        count_invalid, mismatch = _quote_diagnostics(status, evidence_ids, quotes, limit)
        overflow_count = _fact_excess_count(status, evidence_ids, quotes, limit)
        if not ((count_invalid and mismatch) or overflow_count is not None):
            continue
        if status == "冲突":
            messages.append((index, f"这条冲突事实一侧有 {overflow_count or len(evidence_ids)} 个编号、共 {len(quotes)} 条摘句；支持侧和反驳侧各自最多 {limit} 个编号。每个编号须有对应摘句，evidence_ids 与两侧摘句编号必须完全一致。"))
        else:
            messages.append((index, f"这条事实有 {len(evidence_ids)} 个编号、{len(quotes)} 条摘句；支持或反驳事实应有 1 到 {limit} 条摘句，最多 {limit} 个编号。每个编号须有一条摘句，evidence_ids 与摘句编号必须完全一致。"))
    return tuple(messages)


def fact_bundle_feedback_from_json(raw: str) -> tuple[tuple[int, str], ...]:
    """以严格Pydantic外壳和单事实模型投影生成不回显字符串的反馈。"""
    from src.yuqing.schemas import FactVerdictV2Envelope
    try:
        parsed = FactVerdictV2Envelope.model_validate_json(raw, strict=True)
    except ValidationError:
        return ()
    return _fact_bundle_feedback(parsed)


def _same_fact(left: tuple[str | int, ...], right: tuple[str | int, ...]) -> bool:
    """检查两个校验位置是否落在同一条关键事实。"""
    return len(left) > 1 and len(right) > 1 and left[0] == right[0] and left[1] == right[1]


def _quote_diagnostics(status: object, evidence_ids: list[object], quotes: list[object],
                       limit: int) -> tuple[bool, bool]:
    """从单条可信事实投影独立重算支持/反驳的数量与集合诊断。"""
    quote_ids = [_field(quote, "evidence_id") for quote in quotes]
    mismatch = set(evidence_ids) != set(quote_ids)
    if status in {"支持", "反驳"}:
        return not 1 <= len(quotes) <= limit, mismatch
    return False, mismatch


def _field(value: object, name: str) -> object:
    """读取Pydantic对象或其可信封套内的字典字段。"""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _fact_excess_count(status: object, evidence_ids: list[object], quotes: list[object],
                       limit: int) -> int | None:
    """返回超限的可信编号数；冲突按单侧而非两侧合计判断。"""
    if status in {"支持", "反驳"}:
        return len(evidence_ids) if len(evidence_ids) > limit else None
    if status != "冲突":
        return None
    sides: dict[str, set[object]] = {"支持": set(), "反驳": set()}
    for quote in quotes:
        side = _field(quote, "side")
        evidence_id = _field(quote, "evidence_id")
        if side in sides and evidence_id is not None:
            sides[side].add(evidence_id)
    return next((len(ids) for ids in sides.values() if len(ids) > limit), None)


def propagation_submission_feedback(issues: tuple[str, ...], *, available_fields: Iterable[str],
                                    returned_comment_ids: Iterable[str]) -> tuple[str, ...]:
    """只给出本次真实统计字段和已返回评论编号所允许的修正提示。"""
    fields = sorted(set(available_fields) & _PROPAGATION_FIELDS)
    comments = sorted(item for item in set(returned_comment_ids) if _is_comment_id(item))
    messages = []
    for issue in issues:
        message = _PROPAGATION_ISSUES.get(issue)
        if issue == "spread_basis_invalid":
            allowed = "、".join(fields) if fields else "无"
            message = f"spread_basis：只能填写本次统计回执中非空且可用的字段；当前可用字段：{allowed}。"
        elif issue in {"emotion_quote_invalid", "narrative_basis_invalid", "narrative_basis_required",
                       "lead_quote_invalid"}:
            allowed = "、".join(["claim", *comments]) if issue == "emotion_quote_invalid" else "、".join(comments)
            message = f"{_PROPAGATION_PATHS[issue]}：引用须为连续原文；仅可用本次返回的来源编号：{allowed or '无'}。"
        messages.append(message or "报告：有一项来源或格式检查未通过，请按当前工具回执和 Schema 修正。")
    return tuple(dict.fromkeys(messages))


def _describe_validation_issue(issue: ToolValidationIssue,
                               tool_schemas: Mapping[str, type[BaseModel]]) -> str | None:
    """只描述能在本轮工具 Schema 中定位的字段。"""
    schema_model = tool_schemas.get(issue.tool_name or "")
    if schema_model is None:
        return None
    schema = schema_model.model_json_schema()
    path = schema_path(schema, issue.loc)
    specific = _fact_schema_message(issue.reason_code)
    if specific:
        return f"{issue.tool_name}.{path or '事实报告'}：{specific}"
    category = _ERROR_TYPES.get(issue.error_type, "参数不符合 Schema 约束")
    if path is None:
        return f"{issue.tool_name} 参数：{category}。请勿添加 Schema 中没有的字段。"
    hint = _constraint_hint(schema, issue.loc)
    suffix = f"；合法约束：{hint}" if hint else ""
    return f"{issue.tool_name}.{path}：{category}{suffix}。"


def _fact_schema_message(reason_code: str | None) -> str | None:
    """仅把固定事实V2原因码翻译成受配置约束的指导文字。"""
    messages = {
        "fact_evidence_quote_set_mismatch": "同一关键事实的 evidence_ids 必须与 quotes 中 evidence_id 的编号集合完全一致；请自行核对，不要让程序代删或代补编号。",
        "conflict_quote_sides_incomplete": "冲突事实必须分别提供支持和反驳方向的原文摘句。",
        "conflict_quote_side_required": "冲突事实的每条摘句都必须标明支持或反驳方向。",
        "fact_quote_side_mismatch": "摘句方向必须与该关键事实的状态一致。",
        "fact_duplicate_evidence_ids": "同一关键事实的 evidence_ids 不能重复。",
        "fact_duplicate_quotes": "同一关键事实不能重复提交相同摘句。",
    }
    if reason_code == "fact_quote_count_invalid":
        limit = get_fact_v2_limits_config().max_quotes_per_direction
        return f"支持或反驳事实需有1到{limit}条对应方向的原文摘句。"
    if reason_code == "report_quote_count_exceeded":
        limit = get_fact_v2_limits_config().max_quotes_per_report
        return f"整份答卷的原文摘句总数不能超过{limit}条；请自行精简，不要删除程序未指出的内容。"
    return messages.get(reason_code)


def schema_path(root: dict[str, object], location: tuple[str | int, ...]) -> str | None:
    """沿可信 JSON Schema 展开字段路径，遇到未知字段即隐藏路径。"""
    node: dict[str, object] = root
    parts = []
    for part in location:
        node = _resolve_nullable_union(root, _resolve_ref(root, node))
        if isinstance(part, int):
            child = node.get("items")
            if not isinstance(child, dict):
                return None
            node = child
            if parts:
                parts[-1] += "[]"
            continue
        properties = node.get("properties")
        if not isinstance(properties, dict) or part not in properties:
            return None
        child = properties[part]
        if not isinstance(child, dict):
            return None
        node = child
        parts.append(part)
    return ".".join(parts) if parts else None


def _constraint_hint(root: dict[str, object], location: tuple[str | int, ...]) -> str:
    """从 Schema 读取枚举或长度/数值边界，不使用模型提供的值。"""
    node = _schema_node(root, location)
    if node is None:
        return ""
    if isinstance(node.get("enum"), list):
        return " / ".join(str(item) for item in node["enum"])
    bounds = [f"{key}={node[key]}" for key in (
        "minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "pattern")
        if key in node]
    return ", ".join(bounds)


def _schema_node(root: dict[str, object], location: tuple[str | int, ...]) -> dict[str, object] | None:
    """返回已在 Schema 中逐层确认的字段节点。"""
    node = root
    for part in location:
        node = _resolve_nullable_union(root, _resolve_ref(root, node))
        if isinstance(part, int):
            child = node.get("items")
        else:
            properties = node.get("properties")
            child = properties.get(part) if isinstance(properties, dict) else None
        if not isinstance(child, dict):
            return None
        node = child
    return _resolve_nullable_union(root, _resolve_ref(root, node))


def _resolve_nullable_union(root: dict[str, object], node: dict[str, object]) -> dict[str, object]:
    """只展开“单一类型或 null”的二分支 Schema，不猜测复杂联合类型。"""
    branches = node.get("anyOf")
    if not isinstance(branches, list):
        return node
    objects = [branch for branch in branches if isinstance(branch, dict)]
    if len(objects) != len(branches):
        return node
    null_branches = [branch for branch in objects if branch.get("type") == "null"]
    value_branches = [branch for branch in objects if branch.get("type") != "null"]
    if len(null_branches) != 1 or len(value_branches) != 1:
        return node
    return _resolve_ref(root, value_branches[0])


def _resolve_ref(root: dict[str, object], node: dict[str, object]) -> dict[str, object]:
    """仅解析本地 JSON Schema 定义引用。"""
    reference = node.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        return node
    definitions = root.get("$defs")
    resolved = definitions.get(reference.removeprefix("#/$defs/")) if isinstance(definitions, dict) else None
    return resolved if isinstance(resolved, dict) else node


def _canonical_fact_path(path: str) -> str | None:
    """归一化列表下标，并只保留事实输出 Schema 中的已知路径。"""
    import re

    normalized = re.sub(r"\[\d+\]", "[]", path)
    normalized = re.sub(r"\.\d+\.", "[].", normalized)
    for candidate in _FACT_PATHS:
        if normalized == candidate:
            return normalized
    return None


def _is_comment_id(value: str) -> bool:
    """确认动态评论编号符合项目内部编号形式。"""
    return value.startswith("C") and value[1:].isdigit()


_FACT_ISSUES = {
    "empty_text": "必须提供非空理由或关键事实。",
    "empty_citations": "至少引用一个本次已阅读且正文已交付的证据编号。",
    "unknown_evidence": "只能引用当前证据台账中的编号；先搜索并阅读资料。",
    "unread_evidence": "引用前必须先阅读并收到对应证据正文。",
    "undelivered_evidence": "该证据正文没有在本次模型上下文中交付，不能引用。",
    "undelivered": "该编号对应的正文没有在本次模型上下文中交付，不能引用。",
    "exceeds_delivered_set": "编号数量超过本次实际交付的证据集合；请只引用本轮收到正文的材料。",
    "not_found": "摘句未能在本次已交付正文中定位；请按正文原句重新摘取，不能改写后冒充原文。",
    "offset_unresolved": "程序无法确认摘句在已交付正文中的位置；请重新选择可定位的连续原文。",
    "evidence_not_listed": "摘句编号必须同时列在同一关键事实的 evidence_ids 中。",
    "normalized_length_invalid": "摘句长度不符合已登记的引文长度限制。",
    "supported_with_refuting_fact": "结论为支持，但关键事实中出现反驳方向；先核对并统一事实状态和结论。",
    "refuted_without_refuting_fact": "结论为反驳时，至少要有关键事实明确处于反驳状态。",
    "supported_without_supporting_fact": "结论为支持时，至少要有关键事实明确处于支持状态。",
    "supported_with_conflicting_fact": "结论为支持，但关键事实中出现冲突状态；先核对并统一结论。",
    "all_facts_unmentioned": "所有关键事实均为未提及时，最终标签必须是信息不足。",
    "contradicts_refuted_key_fact": "结论标签与关键事实状态冲突；先统一方向，再提交。",
    "all_key_facts_unmentioned": "所有关键事实均为未提及时，最终标签必须是“信息不足”。",
}
_PROPAGATION_PATHS = {
    "emotion_quote_invalid": "emotion_signals[].quote/source",
    "narrative_basis_invalid": "narrative_basis",
    "narrative_basis_required": "narrative_basis",
    "lead_quote_invalid": "leads[].quote/source",
}
_PROPAGATION_ISSUES = {
    "emotion_signal_required": "emotion_signals：情绪为中或高时须至少提供一条合格信号。",
    "spread_must_be_unknown": "spread_level：缺少有效统计时只能填写“未知”。",
    "spread_basis_required": "spread_basis：扩散等级不是“未知”时，须引用可用统计字段。",
    "narrative_requires_propagation_data": "narrative：没有传播数据时只能填写“无法判断”，依据列表留空。",
}
