"""Select the scoring conclusion from trusted fact-round artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.yuqing.graph.fact_adapter import FactRuntimeArtifact


@dataclass(frozen=True)
class FactRoundChoice:
    """Serializable conclusion and review metadata derived from validated rounds."""

    report: dict[str, Any] | None
    round_no: int | None
    reason: str
    force_review: bool = False
    conflicting_issue_ids: tuple[str, ...] = ()


def select_fact_rounds(artifacts: tuple[FactRuntimeArtifact, ...]) -> FactRoundChoice:
    """Apply design §7.8 using only completed, archive-validated reports."""
    ordered = tuple(sorted(artifacts, key=lambda item: item.archive_input.scope.fact_round_no))
    first = next((item for item in ordered if item.archive_input.scope.fact_round_no == 1), None)
    second = next((item for item in ordered if item.archive_input.scope.fact_round_no > 1), None)
    if first is None or not _valid(first):
        return FactRoundChoice(None, None, "first_round_unavailable")
    if second is None or not _valid(second):
        return FactRoundChoice(first.summary.report, 1, "second_round_failed_or_missing")
    first_label, second_label = first.summary.report["label"], second.summary.report["label"]
    reverse_ids = _reverse_evidence_issue_ids(first_label, second)
    opposite = {first_label, second_label} == {"支持", "反驳"}
    if reverse_ids or opposite:
        issue_ids = reverse_ids or _qualified_issue_ids(second)
        report = dict(second.summary.report)
        report["label"] = "信息不足"
        return FactRoundChoice(report, second.archive_input.scope.fact_round_no,
                               "rounds_conflict", True, issue_ids)
    if (second_label == "信息不足" and first_label in {"支持", "反驳"}
            and _has_qualified_fact_citations(first)):
        return FactRoundChoice(first.summary.report, 1, "insufficient_followup_keeps_decisive_first")
    return FactRoundChoice(second.summary.report, second.archive_input.scope.fact_round_no,
                           "second_round_selected")


def _valid(artifact: FactRuntimeArtifact) -> bool:
    """Require a completed report with accepted program validation and archive deliveries."""
    return (artifact.summary.status == "completed" and artifact.summary.report is not None
            and artifact.snapshot.validation.accepted)


def _reverse_evidence_issue_ids(first_label: str, second: FactRuntimeArtifact) -> tuple[str, ...]:
    """Return follow-up issue IDs whose qualified status reverses a decisive first label."""
    if first_label not in {"支持", "反驳"}:
        return ()
    delivered = {item.evidence_id for item in second.snapshot.delivered}
    issue_ids = []
    for fact in second.summary.report["key_facts"]:
        status = fact.get("status")
        reverses = ((first_label == "支持" and status in {"反驳", "冲突"})
                    or (first_label == "反驳" and status == "冲突"))
        refs = fact.get("evidence_ids", [])
        if reverses and refs and set(refs).issubset(delivered):
            issue_id = fact.get("issue_id")
            if isinstance(issue_id, str) and issue_id not in issue_ids:
                issue_ids.append(issue_id)
    return tuple(issue_ids)


def _qualified_issue_ids(artifact: FactRuntimeArtifact) -> tuple[str, ...]:
    """Collect follow-up issues whose facts cite evidence in the verified delivery snapshot."""
    delivered = {item.evidence_id for item in artifact.snapshot.delivered}
    issue_ids = []
    for fact in artifact.summary.report["key_facts"]:
        refs, issue_id = fact.get("evidence_ids", []), fact.get("issue_id")
        if refs and set(refs).issubset(delivered) and isinstance(issue_id, str) and issue_id not in issue_ids:
            issue_ids.append(issue_id)
    return tuple(issue_ids)


def _has_qualified_fact_citations(artifact: FactRuntimeArtifact) -> bool:
    """Require at least one key fact to cite only evidence proven delivered and read."""
    delivered = {item.evidence_id for item in artifact.snapshot.delivered}
    return any((refs := fact.get("evidence_ids", [])) and set(refs).issubset(delivered)
               for fact in artifact.summary.report["key_facts"])
