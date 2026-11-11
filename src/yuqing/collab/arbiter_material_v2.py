"""Build an arbiter-facing fact packet only from re-read V2 archives and board sources."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.yuqing.collab.board import (FactBoardV2Result, _build_proposal_source_index,
    _check_fact_report_history)
from src.yuqing.collab.fact_material import FactArchiveReaderV2
from src.yuqing.collab.fact_import_contract import M4FactImportBindingV1, M4FactImportProvenanceV1
from src.yuqing.collab.proposals import (_trusted_issue_texts, _validate_origins,
    TrustedIssueSourceV2, fact_snapshot_identity)
from src.yuqing.harness.fact_v2_contract import FactRunContextV2
from src.yuqing.schemas import (AgentResultV2, FactScopeV2, FollowupScopeV2,
    Issue, IssueTextSourceV2, ProposedIssueOriginV2, TaskInput,
    ValidatedFactSnapshotV2, VerifiedQuoteV2)


class FactArchiveInputV2(BaseModel):
    """程序指定的一份事实运行及其归档读取上下文。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    reader: FactArchiveReaderV2
    result: AgentResultV2
    task: TaskInput
    contract: FactRunContextV2
    scope: FactScopeV2 | FollowupScopeV2
    historical_import: M4FactImportBindingV1 | None = None


class ArbiterQuoteV2(BaseModel):
    """已由程序定位的引文及其有限上下文，不含文件路径。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_ref: str
    text: str
    side: Literal["支持", "反驳"] | None = None
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    left_context: str
    right_context: str
    ambiguous: bool


class ArbiterKeyFactMaterialV2(BaseModel):
    """一个事实点及其程序生成的同轮引用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    status: Literal["支持", "反驳", "未提及", "冲突"]
    evidence_refs: tuple[str, ...]
    quotes: tuple[ArbiterQuoteV2, ...] = ()
    issue_id: str | None = None


class ArbiterReasonMaterialV2(BaseModel):
    """一个事实报告理由及其程序生成的同轮引用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    evidence_refs: tuple[str, ...]


class ArbiterFactReportV2(BaseModel):
    """经过归档重验的事实报告摘要，不暴露原始归档位置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fact_round_no: int = Field(gt=0, strict=True)
    mode: Literal["initial", "followup"]
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    key_facts: tuple[ArbiterKeyFactMaterialV2, ...]
    reasons: tuple[ArbiterReasonMaterialV2, ...]


class ArbiterEvidenceV2(BaseModel):
    """模型实际可见的已读证据前缀，不包含来源元数据或未读尾部。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_ref: str
    text: str = Field(min_length=1)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    delivered_chars: int = Field(gt=0, strict=True)


class ArbiterMaterialV2(BaseModel):
    """严格绑定案件、决策轮、声明、问题台账和重验事实材料的输入包。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1, pattern=r"\S")
    sample_id: str = Field(min_length=1, pattern=r"\S")
    decision_round: int = Field(gt=0, strict=True)
    claim: str = Field(min_length=1, pattern=r"\S")
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issues: tuple[Issue, ...]
    fact_reports: tuple[ArbiterFactReportV2, ...]
    evidence: tuple[ArbiterEvidenceV2, ...]
    available_refs: tuple[str, ...]
    # Program-only provenance sidecars: omitted from model messages, retained for C1 gating.
    proposal_origins: tuple[ProposedIssueOriginV2, ...] = ()
    text_sources: tuple[IssueTextSourceV2, ...] = ()
    fact_absence_reason: Literal["no_coverage", "agent_failed", "not_dispatched", "unknown"] | None = None
    fact_import_provenance: tuple[M4FactImportProvenanceV1, ...] = ()


@dataclass(frozen=True)
class _ReadFact:
    """One snapshot read from the public archive verifier and its parsed report."""

    snapshot: ValidatedFactSnapshotV2
    report: BaseModel
    mode: Literal["initial", "followup"]
    import_provenance: M4FactImportProvenanceV1 | None = None


def build_arbiter_material_v2(
    *, case_id: str, sample_id: str, decision_round: int, claim: str,
    board: FactBoardV2Result, fact_runs: tuple[FactArchiveInputV2, ...],
    fact_absence_reason: Literal["no_coverage", "agent_failed", "not_dispatched", "unknown"] | None = None,
) -> ArbiterMaterialV2:
    """Re-read all fact archives and bind their exact history to one same-case board."""
    claim_sha = _sha256(claim)
    if not case_id.strip() or not sample_id.strip() or not claim.strip() or decision_round < 1:
        raise ValueError("arbiter_material_identity_invalid")
    if (fact_runs and fact_absence_reason is not None) or (not fact_runs and fact_absence_reason is None):
        raise ValueError("arbiter_material_fact_presence_mismatch")
    _validate_run_inputs(fact_runs)
    reads = tuple(_read_fact(item, case_id, sample_id, claim, claim_sha) for item in fact_runs)
    snapshots = tuple(item.snapshot for item in reads)
    _validate_unique_snapshots(snapshots)
    if tuple(fact_snapshot_identity(item) for item in snapshots) != tuple(
            fact_snapshot_identity(item) for item in board.fact_reports):
        raise ValueError("arbiter_material_board_report_mismatch")
    _validate_report_history(snapshots)
    _validate_board_sources(board, snapshots, case_id, sample_id, claim_sha)
    evidence = _evidence_rows(snapshots)
    report_rows = tuple(_report_row(item) for item in reads)
    refs = tuple(sorted({f"issue:{item.issue_id}" for item in board.issues}
                        | {item.evidence_ref for item in evidence}))
    return ArbiterMaterialV2(case_id=case_id, sample_id=sample_id, decision_round=decision_round,
        claim=claim, claim_sha256=claim_sha, issues=_model_issue_rows(board.issues), fact_reports=report_rows,
        evidence=evidence, available_refs=refs, proposal_origins=board.proposal_origins,
        text_sources=board.text_sources, fact_absence_reason=fact_absence_reason,
        fact_import_provenance=tuple(item.import_provenance for item in reads
                                     if item.import_provenance is not None))


def _validate_run_inputs(fact_runs: tuple[FactArchiveInputV2, ...]) -> None:
    """Reject repeated run or fact-round identities before opening any archive."""
    run_ids = [item.scope.run_id for item in fact_runs]
    rounds = [item.scope.fact_round_no for item in fact_runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("arbiter_material_duplicate_run")
    if len(rounds) != len(set(rounds)):
        raise ValueError("arbiter_material_duplicate_fact_round")


def _validate_unique_snapshots(snapshots: tuple[ValidatedFactSnapshotV2, ...]) -> None:
    """Reject repeated complete reports while preserving board replay behavior."""
    identities = [fact_snapshot_identity(item) for item in snapshots]
    if len(identities) != len(set(identities)):
        raise ValueError("arbiter_material_duplicate_report")


def _read_fact(item: FactArchiveInputV2, case_id: str, sample_id: str, claim: str,
               claim_sha: str) -> _ReadFact:
    """Use the public reader and require its input contract to match the expected case."""
    scope = item.scope
    if (item.task.sample_id != sample_id or item.task.claim != claim or scope != item.contract.scope
            or scope.case_id != case_id or scope.sample_id != sample_id
            or scope.claim_sha256 != claim_sha or scope.agent != "fact_investigator"):
        raise ValueError("arbiter_material_fact_scope_mismatch")
    mode: Literal["initial", "followup"] = "followup" if isinstance(scope, FollowupScopeV2) else "initial"
    if item.contract.mode != mode:
        raise ValueError("arbiter_material_fact_mode_mismatch")
    provenance = None
    if item.historical_import is None:
        snapshot = item.reader.read(item.result, item.task, item.contract, scope)
    else:
        from src.yuqing.collab.m4_fact_history_import import read_m4_historical_fact

        snapshot, provenance = read_m4_historical_fact(item)
    raw_report = json.loads(snapshot.report_json)
    if not isinstance(raw_report, dict):
        raise ValueError("arbiter_material_report_shape_invalid")
    core_report = {key: value for key, value in raw_report.items() if key != "proposed_issues"}
    report = item.contract.schema.model_validate(core_report)
    return _ReadFact(snapshot=snapshot, report=report, mode=mode, import_provenance=provenance)


def _validate_report_history(snapshots: tuple[ValidatedFactSnapshotV2, ...]) -> None:
    """Reuse board history identity rules and require oldest-to-newest unique rounds."""
    ordered = sorted(snapshots, key=lambda item: item.scope.fact_round_no)
    if ordered != list(snapshots):
        raise ValueError("arbiter_material_fact_history_order_invalid")
    history: tuple[ValidatedFactSnapshotV2, ...] = ()
    for snapshot in snapshots:
        _, already_applied = _check_fact_report_history(snapshot, history)
        if already_applied:
            raise ValueError("arbiter_material_duplicate_report")
        history += (snapshot,)


def _validate_board_sources(board: FactBoardV2Result, snapshots: tuple[ValidatedFactSnapshotV2, ...],
                            case_id: str, sample_id: str, claim_sha: str) -> None:
    """Reuse the board's source-index and full-text checks without replaying board intake."""
    issue_ids = [item.issue_id for item in board.issues]
    if len(issue_ids) != len(set(issue_ids)):
        raise ValueError("arbiter_material_duplicate_issue_id")
    if not snapshots:
        if board.fact_reports or board.text_sources or board.proposal_origins:
            raise ValueError("arbiter_material_fact_source_without_report")
        if any(_is_v2_origin(ref) for issue in board.issues for ref in issue.origin_refs):
            raise ValueError("arbiter_material_issue_source_unknown")
        _validate_issue_fact_references(board.issues, frozenset())
        return
    scope = snapshots[0].scope
    if (scope.case_id, scope.sample_id, scope.claim_sha256) != (case_id, sample_id, claim_sha):
        raise ValueError("arbiter_material_fact_scope_mismatch")
    available = {f"F{item.scope.fact_round_no}:{evidence.evidence_id}"
                 for item in snapshots for evidence in item.delivered}
    _validate_issue_fact_references(board.issues, frozenset(available))
    source_index = _build_proposal_source_index(snapshots)
    _trusted_issue_texts(board.issues, board.text_sources, scope, source_index)
    _validate_origins(board.proposal_origins, board.issues, board.text_sources, source_index, case_id)
    _validate_issue_sources(board.issues, board.text_sources, source_index)


def _validate_issue_fact_references(issues: tuple[Issue, ...], available: frozenset[str]) -> None:
    """Validate internal fact references before namespacing them for model input."""
    for issue in issues:
        for reference in _issue_fact_refs(issue):
            if reference.startswith("fact:F"):
                raise ValueError("arbiter_material_board_fact_reference_namespace_invalid")
            if not reference.startswith("F"):
                continue
            if re.fullmatch(r"F[1-9][0-9]*:E[1-9][0-9]*", reference) is None:
                raise ValueError("arbiter_material_board_fact_reference_format_invalid")
            if reference not in available:
                raise ValueError("arbiter_material_board_fact_reference_unknown")


def _is_v2_origin(origin_ref: str) -> bool:
    """Identify qualified V2 source references without treating legacy seeds as V2."""
    parts = origin_ref.split(":")
    if origin_ref.startswith("agent-proposed:"):
        return True
    return parts[0] == "fact" and len(parts) >= 4


def _model_issue_rows(issues: tuple[Issue, ...]) -> tuple[Issue, ...]:
    """Copy board issues and translate their internal fact references for the model."""
    rows = deepcopy(issues)
    for issue in rows:
        issue.evidence_refs = [_model_fact_ref(ref) for ref in issue.evidence_refs]
        for event in issue.history:
            event.refs = [_model_fact_ref(ref) for ref in event.refs]
    return rows


def _model_fact_ref(reference: str) -> str:
    """Add the required namespace to an internal fact evidence reference."""
    return f"fact:{reference}" if reference.startswith("F") else reference


def _issue_fact_refs(issue: Issue) -> tuple[str, ...]:
    """Collect current and historical fact refs before exposing an issue to the model."""
    return tuple(issue.evidence_refs) + tuple(
        reference for event in issue.history for reference in event.refs)


def _validate_issue_sources(issues: tuple[Issue, ...],
                            text_sources: tuple[IssueTextSourceV2, ...],
                            trusted_sources: dict[str, TrustedIssueSourceV2]) -> None:
    """Require every qualified V2 key-fact origin to retain its exact source sidecar."""
    by_origin = {}
    for source in text_sources:
        by_origin.setdefault(source.origin_ref, []).append(source)
    for issue in issues:
        for origin_ref in issue.origin_refs:
            if not _is_v2_origin(origin_ref):
                continue
            trusted = trusted_sources.get(origin_ref)
            matches = by_origin.get(origin_ref, [])
            if trusted is None or len(matches) != 1 or matches[0].issue_id != issue.issue_id:
                raise ValueError("arbiter_material_issue_source_unknown")
            if not set(trusted.evidence_refs).issubset(issue.evidence_refs):
                raise ValueError("arbiter_material_issue_source_mismatch")


def _evidence_rows(snapshots: tuple[ValidatedFactSnapshotV2, ...]) -> tuple[ArbiterEvidenceV2, ...]:
    """Expose each delivered read prefix under a fact-round-scoped reference."""
    rows: dict[str, ArbiterEvidenceV2] = {}
    for snapshot in snapshots:
        for evidence in snapshot.delivered:
            ref = _fact_ref(snapshot.scope.fact_round_no, evidence.evidence_id)
            text = evidence.text[:evidence.read_end]
            row = ArbiterEvidenceV2(evidence_ref=ref, text=text, body_sha256=evidence.body_sha256,
                                    delivered_chars=len(text))
            if ref in rows and rows[ref] != row:
                raise ValueError("arbiter_material_evidence_ref_conflict")
            rows[ref] = row
    return tuple(rows[key] for key in sorted(rows))


def _report_row(read: _ReadFact) -> ArbiterFactReportV2:
    """Transform a strictly parsed report and verified quote records to scoped references."""
    snapshot, report = read.snapshot, read.report
    round_no = snapshot.scope.fact_round_no
    quote_rows = {(item.fact_index, item.quote_index): item for item in snapshot.validation.quote_records}
    facts = tuple(_key_fact_row(fact, index, round_no, quote_rows)
                  for index, fact in enumerate(report.key_facts))
    reasons = tuple(ArbiterReasonMaterialV2(text=item.text,
        evidence_refs=tuple(_fact_ref(round_no, ref) for ref in item.evidence_ids))
        for item in report.reasons)
    return ArbiterFactReportV2(fact_round_no=round_no, mode=read.mode,
        snapshot_sha256=fact_snapshot_identity(snapshot), report_sha256=snapshot.report_sha256,
        trace_sha256=snapshot.trace_sha256,
        label=report.label, confidence=report.confidence, key_facts=facts, reasons=reasons)


def _key_fact_row(fact: BaseModel, index: int, round_no: int,
                  quote_rows: dict[tuple[int, int], VerifiedQuoteV2]) -> ArbiterKeyFactMaterialV2:
    """Map a fact's evidence IDs and quotes without exposing source metadata."""
    quotes = []
    for quote_index, quote in enumerate(fact.quotes):
        record = quote_rows.get((index, quote_index))
        if record is None or record.evidence_id != quote.evidence_id:
            raise ValueError("arbiter_material_verified_quote_missing")
        quotes.append(ArbiterQuoteV2(evidence_ref=_fact_ref(round_no, quote.evidence_id),
            text=record.original_quote, side=quote.side, start=record.start, end=record.end,
            left_context=record.left_context, right_context=record.right_context,
            ambiguous=record.ambiguous))
    return ArbiterKeyFactMaterialV2(text=fact.text, status=fact.status,
        evidence_refs=tuple(_fact_ref(round_no, ref) for ref in fact.evidence_ids),
        quotes=tuple(quotes), issue_id=getattr(fact, "issue_id", None))


def _fact_ref(round_no: int, evidence_id: str) -> str:
    """Build the canonical model-visible fact reference for one report round."""
    return f"fact:F{round_no}:{evidence_id}"


def _sha256(text: str) -> str:
    """Return a stable SHA-256 over UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


def arbiter_material_sha256(material: ArbiterMaterialV2) -> str:
    """Fingerprint the exact arbiter-facing material, independent of archive paths."""
    payload = json.dumps(material.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return _sha256(payload)
