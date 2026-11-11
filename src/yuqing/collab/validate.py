"""Validate collaboration actions against scoped, program-built snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from src.yuqing.schemas import CollabAction, Issue, Progress


RejectReason = Literal[
    "action_limit", "disabled", "bad_issue", "evidence_unavailable",
    "target_unavailable", "bad_refs", "duplicate", "no_progress",
    "nothing_left", "budget",
]
ActionTarget = Literal["fact", "propagation"]
InputErrorCode = Literal[
    "case_scope_mismatch", "snapshot_shape_invalid", "action_shape_invalid",
]


class InputContractError(ValueError):
    """Identify invalid trusted snapshots separately from action-rule denials."""

    def __init__(self, code: InputErrorCode) -> None:
        """Store a stable diagnostic code suitable for a caller trace."""
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ActionConfig:
    """Read-only collaboration switches supplied by the trusted caller."""

    max_actions: int
    targets: frozenset[ActionTarget]
    review_enabled: bool

    def __post_init__(self) -> None:
        """Freeze a defensive copy of enabled targets."""
        object.__setattr__(self, "targets", frozenset(self.targets))


@dataclass(frozen=True)
class TargetSnapshot:
    """Trusted availability facts for evidence and propagation targets."""

    fact_coverage: Literal["full", "demo", "none", "mismatched"]
    propagation_has_data: bool
    visible_comment_ids: frozenset[str]
    returned_comment_ids: frozenset[str]
    propagation_report_exists: bool

    def __post_init__(self) -> None:
        """Freeze comment identifiers so callers cannot change availability later."""
        object.__setattr__(self, "visible_comment_ids", frozenset(self.visible_comment_ids))
        object.__setattr__(self, "returned_comment_ids", frozenset(self.returned_comment_ids))


@dataclass(frozen=True)
class BudgetSnapshot:
    """Exact case-level money values; absent values are treated as unknown."""

    settled: Decimal | None
    reserved: Decimal | None
    action_cap: Decimal | None
    final_decision_reserve: Decimal | None
    case_limit: Decimal | None


@dataclass(frozen=True)
class IssueSnapshot:
    """Immutable issue fields needed to decide whether an action is useful."""

    issue_id: str
    kind: str
    owner: str
    key: bool
    workflow: str
    epistemic: str
    origin_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze origin references copied from the mutable board model."""
        object.__setattr__(self, "origin_refs", tuple(self.origin_refs))


@dataclass(frozen=True)
class ActionSnapshot:
    """Immutable summary of an action already attempted in this case."""

    kind: str
    target: str
    issue_ids: tuple[str, ...]
    question: str

    def __post_init__(self) -> None:
        """Freeze target issue IDs copied from the action model."""
        object.__setattr__(self, "issue_ids", tuple(self.issue_ids))


@dataclass(frozen=True)
class CaseActionSnapshot:
    """Program-built view of one case's current action and board state."""

    case_id: str
    decision_round: int
    issues: tuple[IssueSnapshot, ...]
    executed_actions: tuple[ActionSnapshot, ...]
    previous_progress: bool | None
    config: ActionConfig
    target: TargetSnapshot
    budget: BudgetSnapshot
    human_paused: bool = False

    def __post_init__(self) -> None:
        """Freeze board and action collections at the trusted-input boundary."""
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "executed_actions", tuple(self.executed_actions))


@dataclass(frozen=True)
class CaseReferenceSnapshot:
    """Verified references and lead-to-comment links for one case decision round."""

    case_id: str
    decision_round: int
    leads: Mapping[str, str]
    returned_comment_ids: frozenset[str]
    trusted_refs: frozenset[str]
    forwarded_refs: frozenset[str]

    def __post_init__(self) -> None:
        """Copy and freeze lead links and evidence-reference collections."""
        object.__setattr__(self, "leads", MappingProxyType(dict(self.leads)))
        object.__setattr__(self, "returned_comment_ids", frozenset(self.returned_comment_ids))
        object.__setattr__(self, "trusted_refs", frozenset(self.trusted_refs))
        object.__setattr__(self, "forwarded_refs", frozenset(self.forwarded_refs))


@dataclass(frozen=True)
class ActionValidation:
    """Machine-readable result of the first applicable action rule."""

    allowed: bool
    reason: RejectReason | None
    rule_no: int | None
    promote_issue_ids: tuple[str, ...] = ()


def issue_snapshot(issue: Issue) -> IssueSnapshot:
    """Copy only the immutable issue fields needed by the action validator."""
    return IssueSnapshot(issue.issue_id, issue.kind, issue.owner, issue.key,
                         issue.workflow, issue.epistemic, tuple(issue.origin_refs))


def action_snapshot(action: CollabAction) -> ActionSnapshot:
    """Copy an attempted action into a small immutable history record."""
    return ActionSnapshot(action.kind, action.target, tuple(action.issue_ids), action.question)


def _reject(reason: RejectReason, rule_no: int) -> ActionValidation:
    """Create a denial result with its stable rule number."""
    return ActionValidation(allowed=False, reason=reason, rule_no=rule_no)


def _check_input_contract(
    case_id: str, decision_round: int, snapshot: CaseActionSnapshot,
    refs: CaseReferenceSnapshot,
) -> None:
    """Reject stale, cross-case, or malformed trusted snapshots before rule checks."""
    if (not isinstance(snapshot, CaseActionSnapshot)
            or not isinstance(refs, CaseReferenceSnapshot)):
        raise InputContractError("snapshot_shape_invalid")
    if (not isinstance(case_id, str) or not case_id or type(decision_round) is not int
            or decision_round < 1 or not isinstance(snapshot.case_id, str)
            or type(snapshot.decision_round) is not int
            or not isinstance(refs.case_id, str) or type(refs.decision_round) is not int
            or snapshot.case_id != case_id or refs.case_id != case_id
            or snapshot.decision_round != decision_round
            or refs.decision_round != decision_round):
        raise InputContractError("case_scope_mismatch")
    if (not isinstance(snapshot.config, ActionConfig)
            or not isinstance(snapshot.target, TargetSnapshot)
            or not isinstance(snapshot.budget, BudgetSnapshot)
            or not all(isinstance(item, IssueSnapshot) for item in snapshot.issues)
            or not all(isinstance(item, ActionSnapshot) for item in snapshot.executed_actions)):
        raise InputContractError("snapshot_shape_invalid")
    if not _valid_snapshot_shape(snapshot, refs):
        raise InputContractError("snapshot_shape_invalid")


def _valid_snapshot_shape(snapshot: CaseActionSnapshot, refs: CaseReferenceSnapshot) -> bool:
    """Check collection, configuration, target, and reference snapshot shapes."""
    config = snapshot.config
    valid_targets = {"fact", "propagation"}
    if (type(snapshot.issues) is not tuple or type(snapshot.executed_actions) is not tuple
            or not all(_valid_issue_snapshot(item) for item in snapshot.issues)
            or not all(_valid_action_snapshot(item) for item in snapshot.executed_actions)):
        return False
    if len({item.issue_id for item in snapshot.issues}) != len(snapshot.issues):
        return False
    return _valid_scalar_snapshot_shape(snapshot, refs, config, valid_targets)


def _valid_scalar_snapshot_shape(
    snapshot: CaseActionSnapshot, refs: CaseReferenceSnapshot,
    config: ActionConfig, valid_targets: set[str],
) -> bool:
    """Validate scalar, comment-set, lead-map, and reference collection fields."""
    target = snapshot.target
    if (type(config.max_actions) is not int or config.max_actions < 0
            or type(config.targets) is not frozenset or not config.targets.issubset(valid_targets)
            or type(config.review_enabled) is not bool or type(snapshot.human_paused) is not bool
            or (snapshot.previous_progress is not None
                and type(snapshot.previous_progress) is not bool)):
        return False
    if not _valid_comment_contract(target, refs):
        return False
    if not all(isinstance(key, str) and isinstance(value, str)
               for key, value in refs.leads.items()):
        return False
    return (set(refs.leads.values()).issubset(refs.returned_comment_ids)
            and type(refs.trusted_refs) is frozenset
            and type(refs.forwarded_refs) is frozenset
            and all(isinstance(item, str)
                    for item in refs.trusted_refs | refs.forwarded_refs))


def _valid_comment_contract(target: TargetSnapshot, refs: CaseReferenceSnapshot) -> bool:
    """Require identical returned-comment IDs within the target's visible scope."""
    return (type(target.propagation_has_data) is bool
            and type(target.propagation_report_exists) is bool
            and isinstance(target.fact_coverage, str)
            and target.fact_coverage in {"full", "demo", "none", "mismatched"}
            and type(target.visible_comment_ids) is frozenset
            and type(target.returned_comment_ids) is frozenset
            and type(refs.returned_comment_ids) is frozenset
            and all(isinstance(item, str) for item in target.visible_comment_ids)
            and all(isinstance(item, str) for item in target.returned_comment_ids)
            and all(isinstance(item, str) for item in refs.returned_comment_ids)
            and target.returned_comment_ids == refs.returned_comment_ids
            and target.returned_comment_ids.issubset(target.visible_comment_ids))


def _valid_issue_snapshot(issue: IssueSnapshot) -> bool:
    """Check the typed scalar and immutable-reference shape of one issue view."""
    return (isinstance(issue.issue_id, str) and bool(issue.issue_id)
            and isinstance(issue.kind, str) and issue.kind in {"关键事实", "线索", "传播性质", "冲突"}
            and isinstance(issue.owner, str) and issue.owner in {"fact", "propagation"}
            and type(issue.key) is bool and isinstance(issue.workflow, str)
            and issue.workflow in {"open", "assigned", "done", "failed", "skipped"}
            and isinstance(issue.epistemic, str)
            and issue.epistemic in {"unknown", "supported", "refuted", "insufficient",
                                    "conflicting", "unavailable"}
            and isinstance(issue.origin_refs, tuple)
            and all(isinstance(item, str) for item in issue.origin_refs))


def _valid_action_snapshot(action: ActionSnapshot) -> bool:
    """Check the immutable shape of one previously executed action."""
    return (isinstance(action.kind, str) and action.kind in {"investigate", "review"}
            and isinstance(action.target, str) and action.target in {"fact", "propagation"}
            and isinstance(action.question, str) and isinstance(action.issue_ids, tuple)
            and all(isinstance(item, str) for item in action.issue_ids))


def _enabled(action: CollabAction, config: ActionConfig) -> bool:
    """Check the configured action-kind and target combination."""
    if action.kind == "investigate":
        return action.target in config.targets
    return (action.kind == "review" and action.target == "propagation"
            and action.target in config.targets and config.review_enabled)


def _valid_issue(issue: IssueSnapshot, action: CollabAction) -> bool:
    """Check issue ownership, state, and the narrower review issue kinds."""
    if issue.workflow == "assigned" or issue.epistemic not in {
        "unknown", "insufficient", "conflicting",
    } or issue.owner != action.target:
        return False
    return action.kind != "review" or issue.kind in {"传播性质", "冲突"}


def _target_available(action: CollabAction, snapshot: TargetSnapshot) -> str | None:
    """Return the rule-four denial, if the selected target cannot do useful work."""
    if action.kind == "review":
        return None if snapshot.propagation_report_exists else "target_unavailable"
    if action.target == "fact":
        return "evidence_unavailable" if snapshot.fact_coverage == "none" else None
    unseen = snapshot.visible_comment_ids - snapshot.returned_comment_ids
    return None if snapshot.propagation_has_data and unseen else "target_unavailable"


def _normalized_question(question: str) -> str:
    """Normalize whitespace and case for duplicate-action detection."""
    return " ".join(question.split()).casefold()


def _duplicate(action: CollabAction, previous: Sequence[ActionSnapshot]) -> bool:
    """Reject a repeated target tuple or repeated normalized task question."""
    issue_ids = tuple(sorted(action.issue_ids))
    question = _normalized_question(action.question)
    return any(
        (old.kind, old.target, tuple(sorted(old.issue_ids)))
        == (action.kind, action.target, issue_ids)
        or _normalized_question(old.question) == question
        for old in previous
    )


def _promotable_lead_ids(
    action: CollabAction, issues: Mapping[str, IssueSnapshot], refs: CaseReferenceSnapshot,
) -> tuple[str, ...]:
    """Find selected non-key lead issues whose source comments were returned in this case."""
    if action.kind != "investigate" or action.target != "fact":
        return ()
    selected_leads = set(action.lead_ids)
    return tuple(
        issue.issue_id for issue_id in action.issue_ids
        if (issue := issues[issue_id]).kind == "线索" and not issue.key
        and issue.owner == "fact" and issue.workflow != "assigned"
        and issue.epistemic in {"unknown", "insufficient", "conflicting"}
        and any(origin == f"lead:{lead_id}" and refs.leads.get(lead_id) in refs.returned_comment_ids
                for origin in issue.origin_refs for lead_id in selected_leads)
    )


def _has_selected_open_target(
    action: CollabAction, issues: Mapping[str, IssueSnapshot], promotable: tuple[str, ...],
) -> bool:
    """Require the second action to name its own open key issue or valid lead issue."""
    allowed_kinds = {"传播性质", "冲突"} if action.kind == "review" else None
    return any(
        (issue := issues[issue_id]).key and issue.owner == action.target
        and issue.workflow != "assigned"
        and issue.epistemic in {"unknown", "insufficient", "conflicting"}
        and (allowed_kinds is None or issue.kind in allowed_kinds)
        or issue_id in promotable
        for issue_id in action.issue_ids
    )


def _valid_budget(snapshot: BudgetSnapshot) -> bool:
    """Require complete finite nonnegative Decimal inputs for exact comparison."""
    amounts = (snapshot.settled, snapshot.reserved, snapshot.action_cap,
               snapshot.final_decision_reserve, snapshot.case_limit)
    return all(isinstance(value, Decimal) and value.is_finite() and value >= 0
               for value in amounts)


def _budget_fits(snapshot: BudgetSnapshot) -> bool:
    """Check settled cost, reservations, this action, and final reserve together."""
    if not _valid_budget(snapshot):
        return False
    total = (snapshot.settled + snapshot.reserved + snapshot.action_cap
             + snapshot.final_decision_reserve)
    return total <= snapshot.case_limit


def validate_action(
    *, action: CollabAction, case_id: str, decision_round: int,
    snapshot: CaseActionSnapshot, references: CaseReferenceSnapshot,
) -> ActionValidation:
    """Apply design_v3 §7.4 in order using snapshots scoped to the expected case."""
    _check_input_contract(case_id, decision_round, snapshot, references)
    config = snapshot.config
    issues = {issue.issue_id: issue for issue in snapshot.issues}
    previous = snapshot.executed_actions
    if snapshot.human_paused or config.max_actions <= len(previous):
        return _reject("action_limit", 1)
    if not _enabled(action, config):
        return _reject("disabled", 2)
    if not _valid_action_issue_ids(action):
        raise InputContractError("action_shape_invalid")
    if (len(set(action.issue_ids)) != len(action.issue_ids)
            or any(issue_id not in issues or not _valid_issue(issues[issue_id], action)
                   for issue_id in action.issue_ids)):
        return _reject("bad_issue", 3)
    unavailable = _target_available(action, snapshot.target)
    if unavailable is not None:
        return _reject(unavailable, 4)
    if not _valid_refs(action, references):
        return _reject("bad_refs", 5)
    if _duplicate(action, previous):
        return _reject("duplicate", 6)
    promotable = _promotable_lead_ids(action, issues, references)
    if previous and not snapshot.previous_progress:
        return _reject("no_progress", 7)
    if previous and not _has_selected_open_target(action, issues, promotable):
        return _reject("nothing_left", 7)
    if not _budget_fits(snapshot.budget):
        return _reject("budget", 8)
    return ActionValidation(True, None, None, promotable)


def _valid_action_issue_ids(action: CollabAction) -> bool:
    """Check issue identifiers before deduplication or dictionary membership."""
    return (isinstance(action, CollabAction) and type(action.issue_ids) is list
            and bool(action.issue_ids)
            and all(isinstance(issue_id, str) and bool(issue_id)
                    for issue_id in action.issue_ids))


def _valid_refs(action: CollabAction, snapshot: CaseReferenceSnapshot) -> bool:
    """Verify lead/comment links and forwarded evidence against one scoped snapshot."""
    if not set(action.lead_ids).issubset(snapshot.leads):
        return False
    if any(snapshot.leads[item] not in snapshot.returned_comment_ids for item in action.lead_ids):
        return False
    if not snapshot.forwarded_refs.issubset(snapshot.trusted_refs):
        return False
    if action.kind == "review":
        return bool(action.review_fields) and not action.lead_ids
    return not action.review_fields and (action.target == "fact" or not action.lead_ids)
