"""Validate fact-agent proposal appendices and trusted issue-text sources."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import unicodedata
from typing import Literal, Mapping, Sequence

from src.yuqing.schemas import (FactScopeV2, FollowupScopeV2, Issue, IssueTextSourceV2,
    ProposedIssueAuditV2, ProposedIssueOriginV2, ProposedIssueRejectionV2, ProposedIssueV2,
    ValidatedFactSnapshotV2)
from src.yuqing.settings import FactV2LimitsConfig


@dataclass(frozen=True)
class ProposalCandidateV2:
    """One valid appendix item with its original index and qualified evidence."""

    source_index: int
    proposal: ProposedIssueV2
    evidence_refs: tuple[str, ...]
    normalized_text: str
    origin_ref: str


@dataclass(frozen=True)
class ProposalIntakeV2:
    """Validated appendix candidates ready for board-owned ID and link assignment."""

    audit: ProposedIssueAuditV2
    candidates: tuple[ProposalCandidateV2, ...]
    existing_issue_texts: tuple[tuple[str, str], ...] = ()
    diagnostic: str | None = None


@dataclass(frozen=True)
class ParsedProposalAppendixV2:
    """Keep eligible items, item rejections, and whole-appendix rejection separate."""

    present: bool
    items: tuple[tuple[int, ProposedIssueV2], ...] = ()
    rejected: tuple[ProposedIssueRejectionV2, ...] = ()
    total_reason: str | None = None


@dataclass(frozen=True)
class TrustedIssueSourceV2:
    """One issue text and citation set recovered from a trusted source snapshot."""

    snapshot: ValidatedFactSnapshotV2
    report_identity: str
    source_kind: Literal["key_fact", "agent_proposed"]
    source_index: int
    origin_ref: str
    full_text: str
    evidence_refs: tuple[str, ...]


def normalize_issue_text(text: str) -> str:
    """Normalize text for deduplication without removing punctuation or negation."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).strip().casefold()


def _digest(text: str) -> str:
    """Fingerprint exact UTF-8 source text."""
    return sha256(text.encode("utf-8")).hexdigest()


def fact_snapshot_identity(snapshot: ValidatedFactSnapshotV2) -> str:
    """Bind report, complete scope, validation, delivery, and archive fingerprints."""
    canonical = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
    return _digest(canonical)


def issue_origin_ref(kind: str, report_sha256: str, identity: str, index: int) -> str:
    """Build a stable origin reference that includes report and execution identity."""
    if kind == "key_fact":
        return f"fact:{report_sha256}:{identity}:key:{index}"
    return f"agent-proposed:{report_sha256}:{identity}:{index}"


def _trusted_issue_texts(issues: Sequence[Issue], sources: Sequence[IssueTextSourceV2],
                         scope: FactScopeV2 | FollowupScopeV2,
                         trusted_sources: Mapping[str, TrustedIssueSourceV2]) -> dict[str, str]:
    """Recover full issue text only from checked program-owned source records."""
    by_issue: dict[str, IssueTextSourceV2] = {}
    for source in sources:
        issue = next((item for item in issues if item.issue_id == source.issue_id), None)
        trusted = trusted_sources.get(source.origin_ref)
        if (issue is None or trusted is None or source.case_id != scope.case_id or source.sample_id != scope.sample_id
                or source.claim_sha256 != scope.claim_sha256 or source.origin_ref not in issue.origin_refs
                or source.full_text_sha256 != _digest(source.full_text)
                or issue.text != source.full_text[:60]
                or not _source_matches_record(source, trusted)):
            raise ValueError("existing_issue_source_invalid")
        prior = by_issue.get(source.issue_id)
        if prior is not None and prior.full_text != source.full_text:
            raise ValueError("existing_issue_source_invalid")
        by_issue[source.issue_id] = source
    texts: dict[str, str] = {}
    for issue in issues:
        source = by_issue.get(issue.issue_id)
        if source is not None:
            texts[issue.issue_id] = source.full_text
        elif len(issue.text) < 60:
            texts[issue.issue_id] = issue.text
        else:
            raise ValueError("existing_issue_text_incomplete")
    return texts


def _source_origin_matches(source: IssueTextSourceV2) -> bool:
    """Bind an indexed full text to the accepted report and its original item index."""
    parts = source.origin_ref.split(":")
    if source.source_kind == "key_fact":
        return (len(parts) == 5 and parts[0] == "fact" and parts[1] == source.report_sha256
                and len(parts[2]) == 64 and parts[3:] == ["key", str(source.source_index)])
    if source.source_kind == "agent_proposed":
        return (len(parts) == 4 and parts[0] == "agent-proposed" and parts[1] == source.report_sha256
                and len(parts[2]) == 64 and parts[3] == str(source.source_index))
    return True


def _source_matches_record(source: IssueTextSourceV2, trusted: TrustedIssueSourceV2) -> bool:
    """Compare a stored sidecar with the exact item recovered from its source report."""
    scope = trusted.snapshot.scope
    return (source.source_kind == trusted.source_kind and source.source_index == trusted.source_index
            and source.origin_ref == trusted.origin_ref and source.full_text == trusted.full_text
            and source.run_id == scope.run_id and source.fact_round_no == scope.fact_round_no
            and source.report_sha256 == trusted.snapshot.report_sha256
            and source.case_id == scope.case_id and source.sample_id == scope.sample_id
            and source.claim_sha256 == scope.claim_sha256
            and source.full_text_sha256 == _digest(trusted.full_text)
            and _source_origin_matches(source))


def _rejection(index: int, reason: str) -> ProposedIssueRejectionV2:
    """Build a narrow, machine-readable rejection record."""
    return ProposedIssueRejectionV2(source_index=index, reason=reason)


def _invalid_item(item: object, index: int, delivered_ids: set[str],
                  max_chars: int) -> ProposedIssueRejectionV2 | ProposedIssueV2:
    """Validate one raw appendix item without affecting the accepted core report."""
    if not isinstance(item, dict) or any(not isinstance(key, str) for key in item):
        return _rejection(index, "invalid_type")
    if set(item) != {"question", "evidence_ids"}:
        return _rejection(index, "unknown_field")
    question, evidence = item["question"], item["evidence_ids"]
    if not isinstance(question, str) or not question.strip():
        return _rejection(index, "blank_question")
    if len(question) > max_chars:
        return _rejection(index, "question_too_long")
    if not isinstance(evidence, list) or not evidence or any(
            not isinstance(value, str) or value not in delivered_ids for value in evidence):
        return _rejection(index, "invalid_evidence")
    if len(evidence) != len(set(evidence)):
        return _rejection(index, "duplicate_evidence")
    return ProposedIssueV2(question=question, evidence_ids=evidence)


def parse_proposal_appendix(
    appendix: object, appendix_present: bool, delivered_ids: set[str], limits: FactV2LimitsConfig,
) -> ParsedProposalAppendixV2:
    """Apply the shared whole-list and per-item eligibility rules without board effects."""
    if not appendix_present or appendix == []:
        return ParsedProposalAppendixV2(present=False)
    if appendix is None:
        return ParsedProposalAppendixV2(True, total_reason="null_not_allowed")
    if not isinstance(appendix, list):
        return ParsedProposalAppendixV2(True, total_reason="invalid_type")
    if len(appendix) > limits.max_proposed_issues:
        return ParsedProposalAppendixV2(True, total_reason="too_many_items")
    accepted: list[tuple[int, ProposedIssueV2]] = []
    rejected: list[ProposedIssueRejectionV2] = []
    for index, raw in enumerate(appendix):
        checked = _invalid_item(raw, index, delivered_ids, limits.max_proposed_question_chars)
        if isinstance(checked, ProposedIssueRejectionV2):
            rejected.append(checked)
        else:
            accepted.append((index, checked))
    return ParsedProposalAppendixV2(True, tuple(accepted), tuple(rejected))


def validate_proposal_appendix(
    *, appendix: object, appendix_present: bool, report_sha256: str, report_identity: str,
    scope: FactScopeV2 | FollowupScopeV2,
    fact_round_no: int, delivered_ids: set[str],
    issues: Sequence[Issue], text_sources: Sequence[IssueTextSourceV2],
    origins: Sequence[ProposedIssueOriginV2], trusted_sources: Mapping[str, TrustedIssueSourceV2],
    limits: FactV2LimitsConfig,
) -> ProposalIntakeV2:
    """Validate proposals and return trusted issue text for board-owned deduplication."""
    parsed = parse_proposal_appendix(appendix, appendix_present, delivered_ids, limits)
    if not parsed.present:
        empty_audit = ProposedIssueAuditV2(report_sha256=report_sha256, appendix_status="absent")
        return ProposalIntakeV2(audit=empty_audit, candidates=())
    if parsed.total_reason is not None:
        audit = ProposedIssueAuditV2(report_sha256=report_sha256, appendix_status="rejected_total",
                                     appendix_rejection_reason=parsed.total_reason)
        return ProposalIntakeV2(audit=audit, candidates=(), diagnostic=parsed.total_reason)
    try:
        existing_texts = _trusted_issue_texts(issues, text_sources, scope, trusted_sources)
        _validate_origins(origins, issues, text_sources, trusted_sources, scope.case_id)
    except ValueError as error:
        reason = str(error)
        audit = ProposedIssueAuditV2(report_sha256=report_sha256, appendix_status="rejected_total",
                                     appendix_rejection_reason=reason)
        return ProposalIntakeV2(audit=audit, candidates=(), diagnostic=reason)
    intake = _materialize_intake(parsed, report_sha256, report_identity, fact_round_no)
    return ProposalIntakeV2(intake.audit, intake.candidates, tuple(existing_texts.items()),
                            intake.diagnostic)


def _validate_origins(origins: Sequence[ProposedIssueOriginV2], issues: Sequence[Issue],
                      sources: Sequence[IssueTextSourceV2],
                      trusted_sources: Mapping[str, TrustedIssueSourceV2], case_id: str) -> None:
    """Require every replay key to resolve to a matching trusted issue and source record."""
    by_issue = {item.issue_id: item for item in issues}
    by_source = {(item.issue_id, item.origin_ref): item for item in sources}
    origin_refs = {item.origin_ref for item in origins}
    source_refs = {item.origin_ref for item in sources if item.source_kind == "agent_proposed"}
    issue_refs = {ref for issue in issues for ref in issue.origin_refs
                  if ref.startswith("agent-proposed:")}
    if origin_refs != source_refs or issue_refs != origin_refs:
        raise ValueError("existing_issue_source_invalid")
    refs: set[str] = set()
    for origin in origins:
        issue = by_issue.get(origin.issue_id)
        source = by_source.get((origin.issue_id, origin.origin_ref))
        trusted = trusted_sources.get(origin.origin_ref)
        created = [event for event in issue.history if event.field == "created"] if issue is not None else []
        if (origin.origin_ref in refs or origin.case_id != case_id or issue is None
                or origin.origin_ref not in issue.origin_refs or source is None or trusted is None
                or source.source_kind != "agent_proposed" or source.case_id != case_id
                or (source.run_id, source.fact_round_no, source.report_sha256, source.source_index)
                != (origin.run_id, origin.fact_round_no, origin.report_sha256, origin.source_index)
                or source.full_text_sha256 != _digest(source.full_text)
                or issue.text != source.full_text[:60] or not _source_matches_record(source, trusted)
                or tuple(origin.evidence_refs) != trusted.evidence_refs or len(created) != 1
                or origin.created_round_no != created[0].round):
            raise ValueError("existing_issue_source_invalid")
        refs.add(origin.origin_ref)


def _materialize_intake(
    parsed: ParsedProposalAppendixV2, report_sha256: str, report_identity: str,
    fact_round_no: int,
) -> ProposalIntakeV2:
    """Turn parsed items into report-bound candidates without assigning board links."""
    candidates: list[ProposalCandidateV2] = []
    for index, checked in parsed.items:
        normalized = normalize_issue_text(checked.question)
        ref = issue_origin_ref("agent_proposed", report_sha256, report_identity, index)
        refs = tuple(f"F{fact_round_no}:{key}" for key in checked.evidence_ids)
        candidates.append(ProposalCandidateV2(index, checked, refs, normalized, ref))
    status = ("mixed" if parsed.items and parsed.rejected else "accepted" if parsed.items
              else "rejected_items")
    audit = ProposedIssueAuditV2(report_sha256=report_sha256, appendix_status=status,
        accepted=[proposal for _, proposal in parsed.items], rejected=list(parsed.rejected))
    return ProposalIntakeV2(audit=audit, candidates=tuple(candidates))
