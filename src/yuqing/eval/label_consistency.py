"""离线分析事实标签与关键事实状态是否符合 V-1 规则。"""

import json
from collections import Counter

from src.yuqing.schemas import FactVerdict


RULESET_VERSION = "V-1"


def check_label_consistency(verdict: FactVerdict) -> tuple[str, ...]:
    """返回 V-1 明确规定的标签与关键事实矛盾代码。"""
    statuses = {fact.status for fact in verdict.key_facts}
    issues: list[str] = []
    if verdict.label == "反驳" and "反驳" not in statuses:
        issues.append("refuted_label_without_refuted_fact")
    if verdict.label == "支持":
        if "支持" not in statuses:
            issues.append("supported_label_without_supported_fact")
        if "冲突" in statuses:
            issues.append("supported_label_with_conflicting_fact")
    return tuple(issues)


def evaluate_answer(row: dict[str, str], *, source_csv: str,
                    source_sha256: str, audit_sha256: str) -> dict[str, str]:
    """校验一份已执行答卷并生成不改变原结论的 V-1 审计行。"""
    sample_id, group = row["sample_id"], row["variant"]
    if group not in {"single_shot", "iterative"}:
        raise ValueError("invalid_result_group")
    base = {"row_type": "sample", "sample_id": sample_id, "variant": group,
            "gold_label": row["gold_label"], "predicted_label": row["predicted_label"],
            "stop_reason": row["stop_reason"], "source_csv": source_csv,
            "source_sha256": source_sha256, "source_audit_sha256": audit_sha256,
            "ruleset_version": RULESET_VERSION}
    if not row["predicted_label"]:
        if row["result"]:
            raise ValueError("failed_row_contains_verdict")
        return base | {"v1_status": "not_evaluable", "violations_json": "[]",
                       "key_facts_json": "[]", "original_correct": ""}
    try:
        payload = json.loads(row["result"])
        verdict = FactVerdict.model_validate(payload["verdict"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("invalid_saved_verdict") from exc
    if verdict.label != row["predicted_label"] or row["gold_label"] not in {"支持", "反驳", "信息不足"}:
        raise ValueError("saved_label_mismatch")
    correct = verdict.label == row["gold_label"]
    if row["correct"].lower() != str(correct).lower():
        raise ValueError("saved_correctness_mismatch")
    issues = check_label_consistency(verdict)
    facts = [{"text": fact.text, "status": fact.status, "evidence_ids": fact.evidence_ids}
             for fact in verdict.key_facts]
    return base | {"v1_status": "rejected" if issues else "passed",
                   "violations_json": json.dumps(issues, ensure_ascii=False),
                   "key_facts_json": json.dumps(facts, ensure_ascii=False),
                   "original_correct": str(correct).lower()}


def summarize_answers(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """按单次与循环组统计 V-1 结果及其对原正确答案的影响。"""
    summaries: dict[str, dict[str, int]] = {}
    for group in ("single_shot", "iterative"):
        subset = [row for row in rows if row["variant"] == group]
        rejected = [row for row in subset if row["v1_status"] == "rejected"]
        summaries[group] = {
            "total": len(subset),
            "evaluable": sum(row["v1_status"] != "not_evaluable" for row in subset),
            "passed": sum(row["v1_status"] == "passed" for row in subset),
            "rejected": len(rejected),
            "rejected_originally_correct": sum(row["original_correct"] == "true" for row in rejected),
            "rejected_originally_wrong": sum(row["original_correct"] == "false" for row in rejected),
            "not_evaluable": sum(row["v1_status"] == "not_evaluable" for row in subset)}
    return summaries
