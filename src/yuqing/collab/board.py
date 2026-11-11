"""Build and update a deterministic, evidence-scoped case issue board."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Literal, Mapping

from src.yuqing.eval.label_consistency import check_label_consistency
from src.yuqing.schemas import (
    FactScopeV2,
    FollowupScopeV2,
    FactVerdict,
    Issue,
    IssueEvent,
    IssueTextSourceV2,
    KeyFact,
    Lead,
    ProposedIssueAuditV2,
    ProposedIssueLinkV2,
    ProposedIssueOriginV2,
    PropagationReport,
    ValidatedFactSnapshotV2,
)
from src.yuqing.collab.proposals import (ProposalCandidateV2, ProposalIntakeV2,
    TrustedIssueSourceV2, fact_snapshot_identity, issue_origin_ref, normalize_issue_text,
    parse_proposal_appendix, validate_proposal_appendix)
from src.yuqing.harness.fact_v2_contract import DeliveredEvidenceV2
from src.yuqing.settings import get_fact_v2_limits_config

FACT_RUNTIME_AGENT = "fact_investigator"


@dataclass(frozen=True)
class FactReferenceSnapshot:
    """Trusted references already read, delivered, and accepted for one case round."""

    case_id: str
    round_no: int
    eligible_ids: frozenset[str]
    report_sha256: str


@dataclass(frozen=True)
class PropagationReferenceSnapshot:
    """Validated comments returned to the propagation agent for one case."""

    case_id: str
    round_no: int
    has_data: bool
    report_valid: bool
    returned_comment_ids: frozenset[str]
    report_sha256: str


@dataclass(frozen=True)
class FactIssueUpdate:
    """One verified key-fact update supplied by the caller after submission checks."""

    issue_id: str
    key_index: int
    fact: KeyFact


@dataclass(frozen=True)
class BoardBuildResult:
    """A valid issue board or diagnostics that prevent the board from being used."""

    issues: tuple[Issue, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class FactBoardV2Result:
    """Atomic V2 board publication, including auditable proposal sources."""

    issues: tuple[Issue, ...]
    text_sources: tuple[IssueTextSourceV2, ...]
    proposal_origins: tuple[ProposedIssueOriginV2, ...]
    fact_reports: tuple[ValidatedFactSnapshotV2, ...]
    proposal_audit: ProposedIssueAuditV2
    pending_human_issue_ids: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()
    proposal_diagnostic: str | None = None


def _qualified_fact_refs(
    case_id: str, round_no: int, snapshot: FactReferenceSnapshot | None, evidence_ids: list[str]
) -> list[str]:
    """Scope eligible E identifiers to the verified case round."""
    if snapshot is None or snapshot.case_id != case_id or snapshot.round_no != round_no:
        return []
    prefix = f"F{snapshot.round_no}:"
    return [f"{prefix}{item}" for item in dict.fromkeys(evidence_ids)
            if item in snapshot.eligible_ids and item.startswith("E") and item[1:].isdigit()]


def _report_sha256(report: FactVerdict | PropagationReport) -> str:
    """Fingerprint a validated report using stable, sorted JSON serialization."""
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _fact_snapshot_matches(
    case_id: str, round_no: int, report: FactVerdict, snapshot: FactReferenceSnapshot | None,
) -> bool:
    """Check the report identity and scope attached to a fact-reference snapshot."""
    return bool(snapshot and snapshot.case_id == case_id and snapshot.round_no == round_no
                and snapshot.report_sha256 == _report_sha256(report))


def _fact_state(fact: KeyFact, refs: list[str]) -> str:
    """Map a key fact to a board state, requiring at least one qualified reference."""
    if not refs or fact.status == "未提及":
        return "insufficient"
    return {"支持": "supported", "反驳": "refuted", "冲突": "conflicting"}[fact.status]


def map_fact_state(
    fact: KeyFact, *, case_id: str, round_no: int,
    trusted_refs: FactReferenceSnapshot | None,
) -> tuple[str, tuple[str, ...]]:
    """Map one fact status using only qualified references in its trusted snapshot."""
    refs = _qualified_fact_refs(case_id, round_no, trusted_refs, fact.evidence_ids)
    return _fact_state(fact, refs), tuple(refs)


def _created_event(round_no: int, state: str, refs: list[str]) -> list[IssueEvent]:
    """Create the required initial history item for an issue."""
    return [IssueEvent(round=round_no, field="created", old=None, new=state,
                       by="program", refs=list(refs))]


def _issue(
    issue_id: str, text: str, kind: Literal["关键事实", "线索", "传播性质", "冲突"],
    origin: str, owner: Literal["fact", "propagation"], key: bool, state: str,
    round_no: int, refs: list[str], workflow: str = "open",
) -> Issue:
    """Create a validated issue with its initial program history."""
    return Issue(issue_id=issue_id, text=text[:60], kind=kind, origin_refs=[origin],
                 owner=owner, key=key, workflow=workflow, epistemic=state,
                 evidence_refs=list(refs), history=_created_event(round_no, state, refs))


def _fact_issues(
    case_id: str, round_no: int, verdict: FactVerdict | None,
    snapshot: FactReferenceSnapshot | None, fact_status: Literal["completed", "failed", "not_dispatched"],
) -> list[Issue]:
    """Build ordered fact issues, separating absent coverage from execution failure."""
    if verdict is not None:
        issues = []
        for index, fact in enumerate(verdict.key_facts, start=1):
            state, qualified = map_fact_state(
                fact, case_id=case_id, round_no=round_no, trusted_refs=snapshot)
            refs = list(qualified)
            issues.append(_issue(f"I{index}", fact.text, "关键事实", f"fact:F{round_no}:K{index}",
                                 "fact", True, state, round_no, refs))
        return issues
    state = "unavailable" if fact_status == "not_dispatched" else "unknown"
    workflow = "open" if fact_status == "not_dispatched" else "failed"
    return [_issue("I1", "声明的主要事实是否属实", "关键事实", f"fact:F{round_no}",
                   "fact", True, state, round_no, [], workflow)]


def _propagation_diagnostic(
    case_id: str, round_no: int, report: PropagationReport | None,
    snapshot: PropagationReferenceSnapshot | None,
) -> str | None:
    """Reject propagation data unless report and every cited comment are verified."""
    if snapshot is not None and (snapshot.case_id != case_id or snapshot.round_no != round_no):
        return "propagation_snapshot_scope_mismatch"
    if report is None:
        return None
    if (snapshot is None or not snapshot.report_valid or not snapshot.has_data
            or snapshot.report_sha256 != _report_sha256(report)):
        return "propagation_report_not_verified"
    cited = set(report.narrative_basis) | {lead.source for lead in report.leads}
    if not cited.issubset(snapshot.returned_comment_ids):
        return "propagation_reference_not_returned"
    return None


def _append_leads(
    issues: list[Issue], report: PropagationReport,
    snapshot: PropagationReferenceSnapshot, round_no: int,
) -> None:
    """Append validated, returned-comment leads in their stable report order."""
    for lead in report.leads:
        if lead.source not in snapshot.returned_comment_ids:
            continue
        number = sum(issue.kind == "线索" for issue in issues) + 1
        issues.append(_issue(f"I{len(issues) + 1}", lead.claim_to_verify, "线索",
                             f"lead:L{number}", "fact", False, "unknown", round_no, []))


def build_board(
    *, case_id: str, round_no: int, fact_verdict: FactVerdict | None,
    fact_status: Literal["completed", "failed", "not_dispatched"],
    fact_refs: FactReferenceSnapshot | None,
    propagation_report: PropagationReport | None,
    propagation_refs: PropagationReferenceSnapshot | None,
) -> BoardBuildResult:
    """Build a stable first-round issue board from verified snapshots only."""
    if (fact_verdict is None) == (fact_status == "completed"):
        return BoardBuildResult(issues=(), diagnostics=("fact_status_verdict_mismatch",))
    if fact_verdict is not None:
        diagnostics = check_label_consistency(fact_verdict)
        if diagnostics:
            return BoardBuildResult(issues=(), diagnostics=tuple(diagnostics))
    if fact_verdict is not None and not _fact_snapshot_matches(case_id, round_no, fact_verdict, fact_refs):
        return BoardBuildResult(issues=(), diagnostics=("fact_report_not_verified",))
    prop_issue = _propagation_diagnostic(case_id, round_no, propagation_report, propagation_refs)
    if prop_issue:
        return BoardBuildResult(issues=(), diagnostics=(prop_issue,))
    issues = _fact_issues(case_id, round_no, fact_verdict, fact_refs, fact_status)
    if propagation_report is not None and propagation_refs is not None:
        _append_leads(issues, propagation_report, propagation_refs, round_no)
        basis = [f"prop:{cid}" for cid in dict.fromkeys(propagation_report.narrative_basis)
                 if cid in propagation_refs.returned_comment_ids]
        state = "supported" if propagation_report.narrative in {"当前事件", "旧事重提"} and basis else "insufficient"
        issues.append(_issue(f"I{len(issues) + 1}", "传播内容是当前事件还是旧事重提",
                             "传播性质", "prop:narrative", "propagation", True,
                             state, round_no, basis))
    elif propagation_refs is not None and propagation_refs.case_id == case_id and propagation_refs.has_data:
        issues.append(_issue(f"I{len(issues) + 1}", "传播内容是当前事件还是旧事重提",
                             "传播性质", "prop:narrative", "propagation", True,
                             "unknown", round_no, [], "failed"))
    return BoardBuildResult(issues=tuple(issues), diagnostics=())


def build_fact_board_v2(
    *, snapshot: ValidatedFactSnapshotV2, expected_scope: FactScopeV2 | FollowupScopeV2,
    issues: tuple[Issue, ...] = (),
    text_sources: tuple[IssueTextSourceV2, ...] = (),
    proposal_origins: tuple[ProposedIssueOriginV2, ...] = (),
    fact_reports: tuple[ValidatedFactSnapshotV2, ...] = (), action_round: int,
) -> FactBoardV2Result:
    """Revalidate a trusted V2 snapshot and atomically return board, sources, and C1 audit."""
    if type(action_round) is not int or action_round < 1:
        raise ValueError("fact_board_action_round_invalid")
    if snapshot.scope != expected_scope:
        raise ValueError("fact_board_scope_mismatch")
    if isinstance(snapshot.scope, FollowupScopeV2) and action_round != snapshot.scope.action_round:
        raise ValueError("fact_board_action_round_mismatch")
    report = _checked_v2_report(snapshot)
    scope = snapshot.scope
    if scope.agent != FACT_RUNTIME_AGENT:
        raise ValueError("fact_board_agent_scope_mismatch")
    report_identity, core_applied = _check_fact_report_history(snapshot, fact_reports)
    current = tuple(item.model_copy(deep=True) for item in issues)
    if len({item.issue_id for item in current}) != len(current):
        raise ValueError("duplicate_issue_id")
    delivered = {item.evidence_id: DeliveredEvidenceV2(item.text, item.read_end)
                 for item in snapshot.delivered}
    new_sources: list[IssueTextSourceV2] = []
    if not core_applied:
        if isinstance(scope, FollowupScopeV2):
            current = _apply_v2_followup(current, report["key_facts"], delivered, scope, action_round)
        else:
            current, new_sources = _create_v2_initial_issues(
                current, report["key_facts"], delivered, scope, snapshot.report_sha256,
                report_identity, action_round)
    report_sources = fact_reports if core_applied else fact_reports + (snapshot,)
    base_sources = list(text_sources) + new_sources
    trusted_sources = _build_proposal_source_index(report_sources)
    intake = validate_proposal_appendix(
        appendix=report.get("proposed_issues"), appendix_present="proposed_issues" in report,
        report_sha256=snapshot.report_sha256, report_identity=report_identity, scope=scope,
        fact_round_no=scope.fact_round_no,
        delivered_ids=set(delivered), issues=current, text_sources=base_sources,
        origins=proposal_origins, trusted_sources=trusted_sources,
        limits=get_fact_v2_limits_config())
    final_issues, final_sources, final_origins, audit, pending = _append_v2_proposals(
        current, base_sources, proposal_origins, intake, scope, action_round)
    return FactBoardV2Result(tuple(final_issues), tuple(final_sources), tuple(final_origins),
        report_sources, audit, tuple(pending), proposal_diagnostic=intake.diagnostic)


def _checked_v2_report(snapshot: ValidatedFactSnapshotV2) -> dict[str, object]:
    """Check the trusted material scope, report fingerprint, and accepted status."""
    from hashlib import sha256

    report_raw = snapshot.report_json.encode("utf-8")
    if sha256(report_raw).hexdigest() != snapshot.report_sha256:
        raise ValueError("fact_board_report_sha_mismatch")
    if not snapshot.validation.accepted:
        raise ValueError("fact_board_report_not_accepted")
    report = json.loads(snapshot.report_json)
    if not isinstance(report, dict):
        raise ValueError("fact_board_report_shape_invalid")
    return report


def _check_fact_report_history(
    snapshot: ValidatedFactSnapshotV2, history: tuple[ValidatedFactSnapshotV2, ...],
) -> tuple[str, bool]:
    """Reject report identity collisions and identify an already-applied exact core."""
    identity = fact_snapshot_identity(snapshot)
    seen: set[str] = set()
    run_identities: dict[str, str] = {}
    round_identities: dict[int, str] = {}
    applied = False
    current_scope = snapshot.scope
    for prior in history:
        _checked_v2_report(prior)
        prior_scope = prior.scope
        prior_identity = fact_snapshot_identity(prior)
        if prior_identity in seen:
            raise ValueError("duplicate_fact_report_identity")
        seen.add(prior_identity)
        if prior_scope.agent != FACT_RUNTIME_AGENT or (prior_scope.case_id, prior_scope.sample_id,
                prior_scope.claim_sha256) != (
                current_scope.case_id, current_scope.sample_id, current_scope.claim_sha256):
            raise ValueError("fact_report_source_scope_mismatch")
        if (prior_scope.run_id in run_identities
                and run_identities[prior_scope.run_id] != prior_identity):
            raise ValueError("fact_report_identity_conflict")
        if (prior_scope.fact_round_no in round_identities
                and round_identities[prior_scope.fact_round_no] != prior_identity):
            raise ValueError("fact_report_identity_conflict")
        run_identities[prior_scope.run_id] = prior_identity
        round_identities[prior_scope.fact_round_no] = prior_identity
        same_round = prior_scope.fact_round_no == current_scope.fact_round_no
        same_run = prior_scope.run_id == current_scope.run_id
        if prior_identity == identity:
            applied = True
        elif same_round or same_run:
            raise ValueError("fact_report_identity_conflict")
    return identity, applied


def _build_proposal_source_index(
    snapshots: tuple[ValidatedFactSnapshotV2, ...],
) -> dict[str, TrustedIssueSourceV2]:
    """Recover source text and citations only from accepted snapshots in this board state."""
    index: dict[str, TrustedIssueSourceV2] = {}
    seen: set[str] = set()
    for snapshot in snapshots:
        report = _checked_v2_report(snapshot)
        scope = snapshot.scope
        identity = fact_snapshot_identity(snapshot)
        if identity in seen:
            continue
        seen.add(identity)
        delivered = {item.evidence_id for item in snapshot.delivered}
        for source_index, item in enumerate(report.get("key_facts", [])):
            source = _trusted_source_entry(snapshot, identity, source_index, item, delivered)
            if source is not None:
                _insert_trusted_source(index, source)
        parsed = parse_proposal_appendix(report.get("proposed_issues"),
            "proposed_issues" in report, delivered, get_fact_v2_limits_config())
        if parsed.total_reason is None:
            for source_index, proposal in parsed.items:
                source = _trusted_proposal_source_entry(
                    snapshot, identity, source_index, proposal.question,
                    proposal.evidence_ids)
                _insert_trusted_source(index, source)
    return index


def _trusted_source_entry(
    snapshot: ValidatedFactSnapshotV2, identity: str, source_index: int,
    item: object, delivered: set[str],
) -> TrustedIssueSourceV2 | None:
    """Extract only an original key fact with delivered citations."""
    if not isinstance(item, dict):
        return None
    scope = snapshot.scope
    if not isinstance(item.get("text"), str):
        return None
    text = item["text"]
    evidence = item.get("evidence_ids", [])
    if not isinstance(evidence, list) or any(not isinstance(value, str) for value in evidence):
        return None
    if any(value not in delivered for value in evidence):
        return None
    refs = tuple(f"F{scope.fact_round_no}:{value}" for value in dict.fromkeys(evidence))
    origin_ref = issue_origin_ref("key_fact", snapshot.report_sha256, identity, source_index)
    return TrustedIssueSourceV2(snapshot, identity, "key_fact", source_index, origin_ref, text, refs)


def _trusted_proposal_source_entry(
    snapshot: ValidatedFactSnapshotV2, identity: str, source_index: int,
    text: str, evidence_ids: list[str],
) -> TrustedIssueSourceV2:
    """Bind a parsed eligible proposal to its exact delivered evidence and source report."""
    scope = snapshot.scope
    refs = tuple(f"F{scope.fact_round_no}:{item}" for item in evidence_ids)
    origin_ref = issue_origin_ref("agent_proposed", snapshot.report_sha256, identity, source_index)
    return TrustedIssueSourceV2(snapshot, identity, "agent_proposed", source_index,
                                 origin_ref, text, refs)


def _insert_trusted_source(
    index: dict[str, TrustedIssueSourceV2], source: TrustedIssueSourceV2,
) -> None:
    """Reject a source-reference collision instead of choosing one report silently."""
    previous = index.get(source.origin_ref)
    if previous is not None and previous != source:
        raise ValueError("fact_report_identity_conflict")
    index[source.origin_ref] = source


def _qualified_v2_refs(evidence_ids: list[str], delivered: Mapping[str, DeliveredEvidenceV2],
                        fact_round_no: int) -> list[str]:
    """Scope only confirmed evidence identifiers to their fact-agent round."""
    return [f"F{fact_round_no}:{item}" for item in dict.fromkeys(evidence_ids) if item in delivered]


def _create_v2_initial_issues(
    existing: tuple[Issue, ...], facts: list[dict[str, object]],
    delivered: Mapping[str, DeliveredEvidenceV2], scope: FactScopeV2, report_sha: str,
    report_identity: str, action_round: int,
) -> tuple[tuple[Issue, ...], list[IssueTextSourceV2]]:
    """Create first-round key-fact issues and full-text source records in stable order."""
    result = list(existing)
    sources: list[IssueTextSourceV2] = []
    next_number = _next_issue_number(result)
    for index, fact in enumerate(facts):
        text = str(fact["text"])
        refs = _qualified_v2_refs(fact["evidence_ids"], delivered, scope.fact_round_no)
        state = _v2_state([fact["status"]], bool(refs))
        origin_ref = issue_origin_ref("key_fact", report_sha, report_identity, index)
        existing_issue = next((item for item in result if origin_ref in item.origin_refs), None)
        if existing_issue is not None:
            continue
        issue_id = f"I{next_number}"
        next_number += 1
        result.append(_issue(issue_id, text, "关键事实", origin_ref, "fact", True,
                             state, action_round, refs))
        sources.append(_text_source(scope, issue_id, origin_ref, "key_fact", report_sha, index, text))
    return tuple(result), sources


def _next_issue_number(issues: list[Issue]) -> int:
    """Choose the next numeric I identifier without colliding with existing board IDs."""
    numbers = [int(match.group(1)) for issue in issues
               if (match := re.fullmatch(r"I(\d+)", issue.issue_id))]
    return max(numbers, default=0) + 1


def _v2_state(statuses: list[str], has_refs: bool) -> str:
    """Aggregate one issue's facts without treating missing subfacts as support."""
    values = set(statuses)
    if "冲突" in values or {"支持", "反驳"}.issubset(values):
        return "conflicting"
    if "未提及" in values or not has_refs:
        return "insufficient"
    if "支持" in values:
        return "supported"
    if "反驳" in values:
        return "refuted"
    return "insufficient"


def _apply_v2_followup(
    issues: tuple[Issue, ...], facts: list[dict[str, object]],
    delivered: Mapping[str, DeliveredEvidenceV2], scope: FollowupScopeV2, action_round: int,
) -> tuple[Issue, ...]:
    """Apply only answered targets, grouping all report facts by their trusted issue id."""
    by_id = {item.issue_id: item for item in issues}
    if set(scope.issue_ids) - by_id.keys() or any(by_id[item].owner != "fact" for item in scope.issue_ids):
        raise ValueError("fact_board_followup_target_invalid")
    groups: dict[str, list[dict[str, object]]] = {item: [] for item in scope.issue_ids}
    for fact in facts:
        groups[fact["issue_id"]].append(fact)
    for issue_id, group in groups.items():
        if not group:
            continue
        refs = list(dict.fromkeys(ref for fact in group for ref in _qualified_v2_refs(
            fact["evidence_ids"], delivered, scope.fact_round_no)))
        state = _v2_state([fact["status"] for fact in group], bool(refs))
        by_id[issue_id] = _update_v2_issue(by_id[issue_id], state, refs, scope.action_id, action_round)
    return tuple(by_id[item.issue_id] for item in issues)


def _update_v2_issue(issue: Issue, state: str, refs: list[str], action_id: str,
                     action_round: int) -> Issue:
    """Copy a answered issue, retaining old state on conflict and adding action-round events."""
    if (issue.epistemic, state) in {("supported", "refuted"), ("refuted", "supported")}:
        state = "conflicting"
    all_refs = list(dict.fromkeys(issue.evidence_refs + refs))
    history = list(issue.history)
    if issue.epistemic != state:
        history.append(_event(action_round, action_id, "epistemic", issue.epistemic, state, all_refs))
    if issue.evidence_refs != all_refs:
        history.append(_event(action_round, action_id, "evidence_refs", ",".join(issue.evidence_refs),
                              ",".join(all_refs), all_refs))
    if issue.workflow != "done":
        history.append(_event(action_round, action_id, "workflow", issue.workflow, "done", all_refs))
    return issue.model_copy(update={"epistemic": state, "evidence_refs": all_refs,
                                   "workflow": "done", "history": history}, deep=True)


def _text_source(scope: FactScopeV2 | FollowupScopeV2, issue_id: str, origin_ref: str,
                 kind: Literal["key_fact", "agent_proposed"], report_sha: str,
                 index: int, text: str) -> IssueTextSourceV2:
    """Build a full-text sidecar tied to the accepted report and displayed issue."""
    return IssueTextSourceV2(case_id=scope.case_id, issue_id=issue_id, origin_ref=origin_ref,
        sample_id=scope.sample_id, claim_sha256=scope.claim_sha256,
        source_kind=kind, run_id=scope.run_id, fact_round_no=scope.fact_round_no,
        report_sha256=report_sha, source_index=index, full_text=text,
        full_text_sha256=sha256(text.encode("utf-8")).hexdigest())


def _append_v2_proposals(
    issues: tuple[Issue, ...], sources: list[IssueTextSourceV2],
    origins: tuple[ProposedIssueOriginV2, ...], intake: ProposalIntakeV2,
    scope: FactScopeV2 | FollowupScopeV2, action_round: int,
) -> tuple[list[Issue], list[IssueTextSourceV2], list[ProposedIssueOriginV2],
           ProposedIssueAuditV2, list[str]]:
    """Materialize valid C1 proposals and all audit links as one returned transaction."""
    candidates: tuple[ProposalCandidateV2, ...] = intake.candidates
    current, source_rows, origin_rows = list(issues), list(sources), list(origins)
    text_to_id: dict[str, str] = {}
    for issue_id, text in intake.existing_issue_texts:
        text_to_id.setdefault(normalize_issue_text(text), issue_id)
    origin_by_ref = {item.origin_ref: item for item in origins}
    links: list[ProposedIssueLinkV2] = []
    pending: list[str] = []
    next_number = _next_issue_number(current)
    for candidate in candidates:
        prior = origin_by_ref.get(candidate.origin_ref)
        issue_id = prior.issue_id if prior is not None else text_to_id.get(candidate.normalized_text)
        if issue_id is not None:
            disposition = "replay" if prior is not None else "duplicate"
            links.append(ProposedIssueLinkV2(source_index=candidate.source_index,
                issue_id=issue_id, disposition=disposition))
            old_issue = next((item for item in current if item.issue_id == issue_id), None)
            if (prior is not None and isinstance(scope, FollowupScopeV2) and old_issue is not None
                    and old_issue.workflow == "open" and old_issue.epistemic == "unknown"):
                pending.append(old_issue.issue_id)
            continue
        issue_id = f"I{next_number}"
        next_number += 1
        current.append(_issue(issue_id, candidate.proposal.question, "线索", candidate.origin_ref,
            "fact", False, "unknown", action_round, list(candidate.evidence_refs)))
        source_rows.append(_text_source(scope, issue_id, candidate.origin_ref, "agent_proposed",
            intake.audit.report_sha256, candidate.source_index, candidate.proposal.question))
        origin_rows.append(ProposedIssueOriginV2(case_id=scope.case_id, run_id=scope.run_id,
            fact_round_no=scope.fact_round_no, created_round_no=action_round,
            report_sha256=intake.audit.report_sha256, source_index=candidate.source_index,
            evidence_refs=list(candidate.evidence_refs), issue_id=issue_id, origin_ref=candidate.origin_ref))
        text_to_id[candidate.normalized_text] = issue_id
        pending.append(issue_id) if isinstance(scope, FollowupScopeV2) else None
        links.append(ProposedIssueLinkV2(source_index=candidate.source_index,
            issue_id=issue_id, disposition="created"))
    audit = ProposedIssueAuditV2(report_sha256=intake.audit.report_sha256,
        appendix_status=intake.audit.appendix_status,
        appendix_rejection_reason=intake.audit.appendix_rejection_reason,
        accepted=intake.audit.accepted, rejected=intake.audit.rejected, links=links)
    return current, source_rows, origin_rows, audit, pending


def _event(round_no: int, action_id: str, field: str, old: str, new: str,
           refs: list[str]) -> IssueEvent:
    """Create one program-generated update event."""
    return IssueEvent(round=round_no, action_id=action_id, field=field,
                      old=old, new=new, by="program", refs=list(refs))


def _mark_failed(issue: Issue, action_id: str, round_no: int) -> Issue:
    """Mark work as failed while retaining its existing evidence state and references."""
    history = list(issue.history)
    if issue.workflow != "failed":
        history.append(_event(round_no, action_id, "workflow", issue.workflow,
                              "failed", list(issue.evidence_refs)))
    return issue.model_copy(update={"workflow": "failed", "history": history}, deep=True)


def _updated_issue(
    issue: Issue, fact: KeyFact, refs: list[str], action_id: str, round_no: int,
) -> Issue:
    """Return a copied issue with a qualified update and preserved history."""
    if not refs:
        state = "insufficient"
    else:
        state = _fact_state(fact, refs)
    if (issue.epistemic, state) in {("supported", "refuted"), ("refuted", "supported")}:
        state = "conflicting"
    new_refs = list(dict.fromkeys(issue.evidence_refs + refs)) if state == "conflicting" else refs
    history = list(issue.history)
    if issue.epistemic != state:
        history.append(_event(round_no, action_id, "epistemic", issue.epistemic, state, new_refs))
    if issue.evidence_refs != new_refs:
        history.append(_event(round_no, action_id, "evidence_refs", ",".join(issue.evidence_refs),
                              ",".join(new_refs), new_refs))
    if issue.workflow != "done":
        history.append(_event(round_no, action_id, "workflow", issue.workflow, "done", new_refs))
    return issue.model_copy(update={"epistemic": state, "evidence_refs": new_refs,
                                   "workflow": "done", "history": history}, deep=True)


def apply_issue_updates(
    *, issues: tuple[Issue, ...], updates: tuple[FactIssueUpdate, ...], case_id: str,
    round_no: int, action_id: str, target_issue_ids: tuple[str, ...],
    fact_verdict: FactVerdict | None = None, fact_refs: FactReferenceSnapshot | None = None,
    failed_issue_ids: tuple[str, ...] = (),
) -> tuple[Issue, ...]:
    """Atomically apply only approved fact-issue updates tied to the accepted report."""
    _validate_update_batch(issues, updates, failed_issue_ids, target_issue_ids)
    if updates:
        if fact_verdict is None or not _fact_snapshot_matches(case_id, round_no, fact_verdict, fact_refs):
            raise ValueError("fact_report_snapshot_mismatch")
        if check_label_consistency(fact_verdict):
            raise ValueError("fact_report_label_inconsistent")
        _validate_report_updates(updates, fact_verdict)
    by_id = {issue.issue_id: issue.model_copy(deep=True) for issue in issues}
    for issue_id in failed_issue_ids:
        by_id[issue_id] = _mark_failed(by_id[issue_id], action_id, round_no)
    for update in updates:
        refs = _qualified_fact_refs(case_id, round_no, fact_refs, update.fact.evidence_ids)
        by_id[update.issue_id] = _updated_issue(by_id[update.issue_id], update.fact, refs, action_id, round_no)
    return tuple(by_id[issue.issue_id] for issue in issues)


def _validate_update_batch(
    issues: tuple[Issue, ...], updates: tuple[FactIssueUpdate, ...],
    failed_ids: tuple[str, ...], target_ids: tuple[str, ...],
) -> None:
    """Reject malformed or unauthorized batches before any issue is copied or changed."""
    all_ids = [issue.issue_id for issue in issues]
    update_ids = [update.issue_id for update in updates]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate_issue_id")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("duplicate_target_issue_id")
    if len(update_ids) != len(set(update_ids)) or len(failed_ids) != len(set(failed_ids)):
        raise ValueError("duplicate_update_issue_id")
    if set(update_ids) & set(failed_ids):
        raise ValueError("success_failure_overlap")
    by_id = {issue.issue_id: issue for issue in issues}
    if set(target_ids) - by_id.keys():
        raise ValueError("unknown_target_issue_id")
    if (set(update_ids) | set(failed_ids)) - set(target_ids):
        raise ValueError("non_target_issue_id")
    if any(by_id[item].owner != "fact" for item in target_ids):
        raise ValueError("wrong_issue_owner")


def _validate_report_updates(
    updates: tuple[FactIssueUpdate, ...], report: FactVerdict,
) -> None:
    """Require every submitted item to equal its indexed item in the accepted report."""
    for update in updates:
        index = update.key_index - 1
        if update.key_index < 1 or index >= len(report.key_facts):
            raise ValueError("key_index_out_of_range")
        if update.fact != report.key_facts[index]:
            raise ValueError("fact_report_item_mismatch")
