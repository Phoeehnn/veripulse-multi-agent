"""Purely evaluate whether one collaboration action made substantive progress."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from types import MappingProxyType

from src.yuqing.schemas import Issue, Progress


_GAINED_STATES = {"supported", "refuted", "conflicting"}


@dataclass(frozen=True)
class DeliveredEvidence:
    """Identity of a delivered evidence body and the actual prefix endpoint sent."""

    body_sha256: str
    read_end: int


@dataclass(frozen=True)
class EvidenceDeliverySnapshot:
    """Trusted cumulative or per-action delivery metadata for one case round."""

    case_id: str
    round_no: int
    action_id: str | None
    deliveries: Mapping[str, DeliveredEvidence]

    def __post_init__(self) -> None:
        """Freeze a defensive copy so later caller mutations cannot alter the snapshot."""
        object.__setattr__(self, "deliveries", MappingProxyType(dict(self.deliveries)))


def _snapshot_valid(snapshot: EvidenceDeliverySnapshot) -> bool:
    """Check that a delivery snapshot carries complete body identity metadata."""
    return all(re.fullmatch(r"[0-9a-f]{64}", item.body_sha256) and item.read_end > 0
               for item in snapshot.deliveries.values())


def _has_new_reference(
    issue_id: str, after: Issue, new_valid_refs: Mapping[str, frozenset[str]],
    before_delivery: EvidenceDeliverySnapshot, action_delivery: EvidenceDeliverySnapshot,
) -> bool:
    """Require an eligible citation delivered in this action with new body or range."""
    reported = new_valid_refs.get(issue_id, frozenset())
    candidates = reported & set(action_delivery.deliveries) & set(after.evidence_refs)
    prior_ranges: dict[str, int] = {}
    for delivery in before_delivery.deliveries.values():
        prior_ranges[delivery.body_sha256] = max(
            prior_ranges.get(delivery.body_sha256, 0), delivery.read_end)
    return any(
        delivery.body_sha256 not in prior_ranges
        or delivery.read_end > prior_ranges[delivery.body_sha256]
        for ref in candidates
        for delivery in (action_delivery.deliveries[ref],)
    )


def _allowed_transition(before: str, after: str) -> bool:
    """Return whether the evidence-state change is substantive under design_v3 §7.6."""
    if before in {"unknown", "insufficient"}:
        return after in _GAINED_STATES
    return before == "conflicting" and after in {"supported", "refuted"}


def evaluate_progress(
    *, before: Sequence[Issue], after: Sequence[Issue], target_issue_ids: Sequence[str],
    new_valid_refs: Mapping[str, frozenset[str]],
    case_id: str, round_no: int, action_id: str,
    before_delivery: EvidenceDeliverySnapshot,
    action_delivery: EvidenceDeliverySnapshot,
) -> Progress:
    """Report only evidence-backed transitions of issues that existed before the action."""
    _validate_delivery_snapshots(case_id, round_no, action_id, before_delivery, action_delivery)
    old_by_id = {issue.issue_id: issue for issue in before}
    new_by_id = {issue.issue_id: issue for issue in after}
    _validate_candidate_refs(new_valid_refs, new_by_id, action_delivery)
    resolved: list[str] = []
    reasons: list[str] = []
    for issue_id in dict.fromkeys(target_issue_ids):
        old, new = old_by_id.get(issue_id), new_by_id.get(issue_id)
        if old is None:
            reasons.append("target_not_present_before_action")
            continue
        if new is None:
            reasons.append("target_missing_after_action")
            continue
        if old.epistemic == new.epistemic:
            reasons.append("evidence_state_unchanged")
            continue
        if (old.epistemic, new.epistemic) in {("supported", "refuted"), ("refuted", "supported")}:
            reasons.append("reversal_must_be_recorded_as_conflicting")
            continue
        if not _allowed_transition(old.epistemic, new.epistemic):
            reasons.append("evidence_state_transition_not_substantive")
            continue
        if not _has_new_reference(issue_id, new, new_valid_refs, before_delivery, action_delivery):
            reasons.append("no_new_qualified_reference")
            continue
        resolved.append(issue_id)
        reasons.append("new_reference_changed_evidence_state")
    return Progress(substantive=bool(resolved), resolved_issue_ids=resolved, reasons=reasons)


def _validate_delivery_snapshots(
    case_id: str, round_no: int, action_id: str,
    before: EvidenceDeliverySnapshot, action: EvidenceDeliverySnapshot,
) -> None:
    """Reject missing or mismatched trusted delivery identity before judging progress."""
    if not _snapshot_valid(before) or not _snapshot_valid(action):
        raise ValueError("delivery_identity_missing")
    if (before.case_id != case_id or action.case_id != case_id
            or before.round_no != round_no or action.round_no != round_no):
        raise ValueError("delivery_snapshot_scope_mismatch")
    if before.action_id != action_id or action.action_id != action_id:
        raise ValueError("delivery_action_mismatch")


def _validate_candidate_refs(
    candidates: Mapping[str, frozenset[str]], after: Mapping[str, Issue],
    action: EvidenceDeliverySnapshot,
) -> None:
    """Require every proposed new reference to be retained and delivered in this action."""
    for issue_id, refs in candidates.items():
        issue = after.get(issue_id)
        if (issue is None or not refs.issubset(action.deliveries.keys())
                or not refs.issubset(issue.evidence_refs)):
            raise ValueError("candidate_reference_not_delivered")
