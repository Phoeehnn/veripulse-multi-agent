"""独立检查台账编号、真实阅读记录与可确定的标签矛盾。"""

from pydantic import ValidationError

from src.yuqing.harness.ledger import EvidenceLedger
from src.yuqing.schemas import FactVerdict, SubmissionReview, ToolRequest


def check(verdict: FactVerdict, ledger: EvidenceLedger) -> list[str]:
    """返回可定位的问题；空列表表示格式、读取与指定一致性检查通过。"""
    problems: list[str] = []
    for field in ("reasons", "key_facts"):
        for index, item in enumerate(getattr(verdict, field)):
            location = f"{field}[{index}].evidence_ids"
            if not item.text.strip():
                problems.append(f"{field}[{index}].text: empty_text")
            needs_citation = field == "reasons" or item.status != "未提及"
            if needs_citation and not item.evidence_ids:
                problems.append(f"{location}: empty_citations")
            for evidence_id in dict.fromkeys(item.evidence_ids):
                entry = ledger.get(evidence_id)
                if entry is None:
                    problems.append(f"{location}: unknown_evidence {evidence_id}")
                elif entry.read_end == 0:
                    problems.append(f"{location}: unread_evidence {evidence_id}")
    statuses = {fact.status for fact in verdict.key_facts}
    if verdict.label == "支持" and "反驳" in statuses:
        problems.append("label: contradicts_refuted_key_fact")
    if statuses == {"未提及"} and verdict.label != "信息不足":
        problems.append("label: all_key_facts_unmentioned")
    return problems


def check_submission_batch(requests: list[ToolRequest], *, submission_tool: str = "submit_verdict") -> list[str]:
    """提交必须独占一批，禁止同轮边读边提交或多次提交。"""
    if any(request.name == submission_tool for request in requests) and len(requests) != 1:
        return ["submission_must_be_alone"]
    return []


def review_submission(verdict: FactVerdict, ledger: EvidenceLedger,
                      *, delivered_evidence_ids: frozenset[str]) -> SubmissionReview:
    """复检判断并保存快照，只有此前由可信程序确认交付的已读引用才合格。"""
    snapshot = verdict.model_dump_json()
    try:
        candidate = FactVerdict.model_validate_json(snapshot)
    except ValidationError as exc:
        issues = tuple(f"{'.'.join(map(str, item['loc']))}: {item['type']}" for item in exc.errors())
        return SubmissionReview(verdict_json=snapshot, issues=issues)
    problems = check(candidate, ledger)
    for field in ("reasons", "key_facts"):
        for index, item in enumerate(getattr(candidate, field)):
            for evidence_id in dict.fromkeys(item.evidence_ids):
                if evidence_id not in delivered_evidence_ids:
                    problems.append(f"{field}[{index}].evidence_ids: undelivered_evidence {evidence_id}")
    return SubmissionReview(verdict_json=snapshot, issues=tuple(problems))
