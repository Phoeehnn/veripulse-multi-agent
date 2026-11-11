"""Offline checks for scoped collaboration-action validation."""

from decimal import Decimal
from typing import Literal

import pytest

from src.yuqing.collab.validate import (
    ActionConfig, ActionValidation, BudgetSnapshot, CaseActionSnapshot,
    CaseReferenceSnapshot, InputContractError, IssueSnapshot, TargetSnapshot,
    action_snapshot, issue_snapshot, validate_action,
)
from src.yuqing.schemas import CollabAction, Issue, IssueEvent, Progress


def _issue(
    issue_id: str = "I1", *, owner: str = "fact", state: str = "unknown",
    workflow: str = "open", key: bool = True, kind: str = "关键事实",
    origin: str | None = None,
) -> Issue:
    """Build a compact issue for one validator scenario."""
    return Issue(issue_id=issue_id, text=f"核实{issue_id}", kind=kind,
                 origin_refs=[origin or "fact:F1:K1"], owner=owner, key=key,
                 workflow=workflow, epistemic=state,
                 history=[IssueEvent(round=1, field="created", old=None,
                                     new=state, by="program")])


def _action(
    issue_ids: list[str] | None = None, *, kind: str = "investigate",
    target: str = "fact", question: str = "核实主要事实", leads: list[str] | None = None,
    review_fields: list[str] | None = None,
) -> CollabAction:
    """Build an action accepted by the collaboration message schema."""
    return CollabAction(kind=kind, target=target, issue_ids=issue_ids or ["I1"],
                        question=question, done_when="找到一手来源",
                        lead_ids=leads or [], review_fields=review_fields or [])


def _target(
    *, coverage: Literal["full", "demo", "none", "mismatched"] = "full",
    has_data: bool = True,
    visible: frozenset[str] = frozenset({"C1"}),
    returned: frozenset[str] = frozenset(), report: bool = True,
) -> TargetSnapshot:
    """Create trusted availability facts without any external calls."""
    return TargetSnapshot(coverage, has_data, visible, returned, report)


def _budget(
    *, settled: Decimal | None = Decimal("0.1"), reserved: Decimal | None = Decimal("0.1"),
    action: Decimal | None = Decimal("0.2"), final: Decimal | None = Decimal("0.1"),
    limit: Decimal | None = Decimal("0.5"),
) -> BudgetSnapshot:
    """Create exact budget inputs for tests."""
    return BudgetSnapshot(settled, reserved, action, final, limit)


def _snapshot(
    issues: tuple[Issue, ...] | None = None, *, prior: tuple[CollabAction, ...] = (),
    progress: Progress | None = None, config: ActionConfig | None = None,
    target: TargetSnapshot | None = None, budget: BudgetSnapshot | None = None,
    case_id: str = "case-A", round_no: int = 2, paused: bool = False,
) -> CaseActionSnapshot:
    """Build the immutable case-action view from mutable test input models."""
    source_issues = (_issue(),) if issues is None else issues
    return CaseActionSnapshot(
        case_id, round_no, tuple(issue_snapshot(item) for item in source_issues),
        tuple(action_snapshot(item) for item in prior),
        progress.substantive if progress else None,
        config or ActionConfig(1, frozenset({"fact"}), False),
        target or _target(), budget or _budget(), paused,
    )


def _references(
    *, case_id: str = "case-A", round_no: int = 2,
    leads: dict[str, str] | None = None,
    returned: frozenset[str] = frozenset({"C1"}),
    trusted: frozenset[str] = frozenset({"F1:E1"}),
    forwarded: frozenset[str] = frozenset(),
) -> CaseReferenceSnapshot:
    """Build the immutable, case-scoped evidence and comment-link view."""
    return CaseReferenceSnapshot(case_id, round_no, leads or {}, returned, trusted, forwarded)


def _validate(
    action: CollabAction | None = None, issues: tuple[Issue, ...] | None = None,
    *, prior: tuple[CollabAction, ...] = (), progress: Progress | None = None,
    config: ActionConfig | None = None, target: TargetSnapshot | None = None,
    budget: BudgetSnapshot | None = None, leads: dict[str, str] | None = None,
    returned: frozenset[str] | None = None,
    known_refs: frozenset[str] = frozenset(), forwarded_refs: frozenset[str] = frozenset(),
    case_id: str = "case-A", round_no: int = 2, paused: bool = False,
) -> ActionValidation:
    """Call the production validator with safe default snapshots."""
    visible_returned = returned if returned is not None else (
        target.returned_comment_ids if target is not None else frozenset({"C1"})
    )
    return validate_action(
        action=action or _action(), case_id=case_id, decision_round=round_no,
        snapshot=_snapshot(issues, prior=prior, progress=progress, config=config,
                           target=target or _target(returned=visible_returned),
                           budget=budget, case_id=case_id,
                           round_no=round_no, paused=paused),
        references=_references(case_id=case_id, round_no=round_no, leads=leads,
                               returned=visible_returned, trusted=known_refs,
                               forwarded=forwarded_refs),
    )


def test_action_limit_and_human_pause_take_priority() -> None:
    """Rule one denies before inspecting an otherwise invalid action."""
    blocked = _validate(action=_action(kind="review", target="fact"),
                        config=ActionConfig(0, frozenset(), False), paused=True)
    assert (blocked.allowed, blocked.reason, blocked.rule_no) == (False, "action_limit", 1)


@pytest.mark.parametrize(
    ("action", "config"),
    [
        (_action(target="propagation"), ActionConfig(1, frozenset({"fact"}), False)),
        (_action(kind="review", target="propagation"),
         ActionConfig(1, frozenset({"fact", "propagation"}), False)),
        (_action(kind="review", target="fact"), ActionConfig(1, frozenset({"fact"}), True)),
    ],
)
def test_disabled_target_or_kind_is_rejected(
    action: CollabAction, config: ActionConfig,
) -> None:
    """Typed action existence does not enable a disabled collaboration path."""
    result = _validate(action=action, config=config)
    assert (result.reason, result.rule_no) == ("disabled", 2)


@pytest.mark.parametrize(
    "issues",
    [(), (_issue(workflow="assigned"),), (_issue(owner="propagation"),),
     (_issue(state="supported"),)],
)
def test_bad_issue_rejected_but_failed_fact_issue_can_retry(
    issues: tuple[Issue, ...],
) -> None:
    """Missing, assigned, wrong-owner, and resolved issues fail; failed work can retry."""
    result = _validate(issues=issues)
    assert (result.reason, result.rule_no) == ("bad_issue", 3)
    failed = _validate(issues=(_issue(workflow="failed"),))
    assert failed.allowed is True


@pytest.mark.parametrize(
    ("action", "snapshot", "reason"),
    [
        (_action(), _target(coverage="none"), "evidence_unavailable"),
        (_action(target="propagation"), _target(has_data=False), "disabled"),
        (_action(target="propagation"),
         _target(visible=frozenset({"C1"}), returned=frozenset({"C1"})), "disabled"),
    ],
)
def test_target_availability_and_missing_coverage(
    action: CollabAction, snapshot: TargetSnapshot, reason: str,
) -> None:
    """No coverage is distinct from mismatched coverage; unavailable tools are denied."""
    if action.target == "propagation":
        result = _validate(action=action, config=ActionConfig(1, frozenset({"fact"}), False),
                           target=snapshot)
        assert (result.reason, result.rule_no) == ("disabled", 2)
        enabled = ActionConfig(1, frozenset({"fact", "propagation"}), False)
        result = _validate(action=action, config=enabled, target=snapshot,
                           issues=(_issue(owner="propagation", kind="传播性质"),))
        assert (result.reason, result.rule_no) == ("target_unavailable", 4)
    else:
        result = _validate(action=action, target=snapshot)
        assert (result.reason, result.rule_no) == (reason, 4)
    assert _validate(target=_target(coverage="mismatched")).allowed is True
    review = _validate(
        action=_action(kind="review", target="propagation", review_fields=["narrative"]),
        issues=(_issue(owner="propagation", kind="传播性质"),),
        config=ActionConfig(1, frozenset({"propagation"}), True),
        target=_target(report=False),
    )
    assert (review.reason, review.rule_no) == ("target_unavailable", 4)


@pytest.mark.parametrize(
    ("action", "known_refs", "forwarded", "leads"),
    [
        (_action(leads=["L9"]), frozenset(), frozenset(), {}),
        (_action(), frozenset({"F1:E1"}), frozenset({"F9:E9"}), {}),
        (_action(kind="review", target="propagation"), frozenset(), frozenset(), {}),
    ],
)
def test_untrusted_lead_and_reference_payloads_are_rejected(
    action: CollabAction, known_refs: frozenset[str], forwarded: frozenset[str],
    leads: dict[str, str],
) -> None:
    """The validator checks returned comment links and never trusts model-supplied IDs."""
    result = _validate(
        action=action, known_refs=known_refs, forwarded_refs=forwarded, leads=leads,
        issues=(_issue(owner="propagation", kind="传播性质"),) if action.kind == "review" else None,
        config=ActionConfig(1, frozenset({"fact", "propagation"}), True),
    )
    assert (result.reason, result.rule_no) == ("bad_refs", 5)


def test_duplicate_actions_ignore_issue_order_and_normalize_question_text() -> None:
    """Same action tuple or repeated normalized question is rejected, including failed work."""
    prior = _action(["I1", "I2"], question="核实 两个事实")
    candidate = _action(["I2", "I1"], question="完全不同的问题")
    issue_set = (_issue("I1"), _issue("I2"))
    config = ActionConfig(2, frozenset({"fact"}), False)
    result = _validate(candidate, issue_set, prior=(prior,), progress=Progress(substantive=True),
                       config=config)
    assert (result.reason, result.rule_no) == ("duplicate", 6)
    repeated = _validate(
        _action(["I2"], question="  核实   两个事实 "), issue_set, prior=(prior,),
        progress=Progress(substantive=True), config=config,
    )
    assert (repeated.reason, repeated.rule_no) == ("duplicate", 6)


def test_second_action_checks_selected_key_issue_and_allows_linked_lead_promotion() -> None:
    """A global key issue cannot justify another target; a selected verified lead can."""
    first = _action(question="第一次核查")
    config = ActionConfig(2, frozenset({"fact"}), False)
    second = _action(["I2"], question="第二次核查")
    prior = (first,)
    progress = Progress(substantive=True)
    unrelated_key = (_issue("I1", state="supported"), _issue("I2", key=False),
                     _issue("I3", key=True))
    denied = _validate(second, unrelated_key, prior=prior, progress=progress, config=config)
    assert (denied.reason, denied.rule_no) == ("nothing_left", 7)
    open_key = _validate(second, (_issue("I1", state="supported"), _issue("I2")),
                         prior=prior, progress=progress, config=config)
    assert open_key.allowed is True

    lead_issue = _issue("I2", kind="线索", key=False, origin="lead:L1")
    allowed = _validate(_action(["I2"], question="跟进线索", leads=["L1"]),
                        (_issue("I1", state="supported"), lead_issue), prior=prior,
                        progress=progress, config=config, leads={"L1": "C1"})
    assert allowed.allowed is True
    assert allowed.promote_issue_ids == ("I2",)
    no_progress = _validate(_action(["I2"], question="没有进展", leads=["L1"]),
                            (_issue("I1", state="supported"), lead_issue), prior=prior,
                            progress=Progress(substantive=False), config=config,
                            leads={"L1": "C1"})
    assert (no_progress.reason, no_progress.rule_no) == ("no_progress", 7)


def test_second_action_does_not_promote_unlinked_or_unreturned_lead() -> None:
    """A trusted lead ID must match the selected issue origin and returned source comment."""
    prior = (_action(question="第一次核查"),)
    config = ActionConfig(2, frozenset({"fact"}), False)
    progress = Progress(substantive=True)
    issue = _issue("I2", kind="线索", key=False, origin="lead:L2")
    action = _action(["I2"], question="跟进线索", leads=["L1"])
    result = _validate(action, (_issue("I1", state="supported"), issue), prior=prior,
                       progress=progress, config=config, leads={"L1": "C1"})
    assert (result.reason, result.rule_no) == ("nothing_left", 7)


def test_case_contract_rejects_cross_case_and_old_decision_snapshot() -> None:
    """Case identity and decision round are checked independently of evidence round IDs."""
    action = _action()
    snapshot = _snapshot(case_id="case-B")
    refs = _references(case_id="case-A")
    with pytest.raises(InputContractError, match="case_scope_mismatch"):
        validate_action(action=action, case_id="case-A", decision_round=2,
                        snapshot=snapshot, references=refs)
    with pytest.raises(InputContractError, match="case_scope_mismatch"):
        validate_action(action=action, case_id="case-A", decision_round=3,
                        snapshot=_snapshot(), references=_references())
    with pytest.raises(InputContractError, match="case_scope_mismatch"):
        validate_action(action=action, case_id="case-A", decision_round=2,
                        snapshot=_snapshot(), references=_references(case_id="case-B"))
    historical = _validate(known_refs=frozenset({"F1:E1"}),
                          forwarded_refs=frozenset({"F1:E1"}))
    assert historical.allowed is True


def test_input_snapshots_copy_mutable_sources_and_reject_bad_shapes() -> None:
    """Caller edits after construction cannot rewrite a decision snapshot."""
    issue = _issue()
    raw_leads = {"L1": "C1"}
    snapshot = _snapshot((issue,))
    refs = _references(leads=raw_leads)
    issue.key = False
    raw_leads["L1"] = "C9"
    assert snapshot.issues[0].key is True
    assert refs.leads["L1"] == "C1"
    with pytest.raises(TypeError):
        refs.leads["L2"] = "C2"

    malformed = CaseActionSnapshot(
        "case-A", 2,
        (IssueSnapshot("I1", "关键事实", "fact", True, "open", "unknown", ()),
         IssueSnapshot("I1", "关键事实", "fact", True, "open", "unknown", ())),
        (), None, ActionConfig(1, frozenset({"fact"}), False), _target(), _budget(),
    )
    with pytest.raises(InputContractError, match="snapshot_shape_invalid"):
        validate_action(action=_action(), case_id="case-A", decision_round=2,
                        snapshot=malformed, references=_references())
    invalid_kind = CaseActionSnapshot(
        "case-A", 2, (IssueSnapshot("I1", "x", "fact", True, "open", "unknown", ()),),
        (), None, ActionConfig(1, frozenset({"fact"}), False), _target(), _budget(),
    )
    with pytest.raises(InputContractError, match="snapshot_shape_invalid"):
        validate_action(action=_action(), case_id="case-A", decision_round=2,
                        snapshot=invalid_kind, references=_references())
    bad_config = ActionConfig(float("nan"), frozenset({"fact"}), False)
    with pytest.raises(InputContractError, match="snapshot_shape_invalid"):
        validate_action(action=_action(), case_id="case-A", decision_round=2,
                        snapshot=_snapshot(config=bad_config), references=_references())
    bad_target = TargetSnapshot("unknown", True, frozenset(), frozenset(), False)
    with pytest.raises(InputContractError, match="snapshot_shape_invalid"):
        validate_action(action=_action(), case_id="case-A", decision_round=2,
                        snapshot=_snapshot(target=bad_target), references=_references())
    invalid_source = _references(leads={"L1": "C9"})
    with pytest.raises(InputContractError, match="snapshot_shape_invalid"):
        validate_action(action=_action(), case_id="case-A", decision_round=2,
                        snapshot=_snapshot(target=_target(visible=frozenset({"C1"}),
                                                          returned=frozenset({"C1"}))),
                        references=invalid_source)


@pytest.mark.parametrize(
    ("target_returned", "reference_returned", "visible", "expected"),
    [
        (frozenset({"C1"}), frozenset({"C1"}), frozenset({"C1"}), True),
        (frozenset({"C1"}), frozenset({"C1", "C2"}), frozenset({"C1", "C2"}), False),
        (frozenset({"C1", "C2"}), frozenset({"C1", "C2"}), frozenset({"C1"}), False),
        (frozenset({"C1", "C2"}), frozenset({"C1"}), frozenset({"C1", "C2"}), False),
    ],
)
def test_returned_comment_snapshots_must_match_and_be_visible(
    target_returned: frozenset[str], reference_returned: frozenset[str],
    visible: frozenset[str], expected: bool,
) -> None:
    """Only the same returned comment set inside visible scope is trusted."""
    target = _target(visible=visible, returned=target_returned)
    if expected:
        result = _validate(target=target, returned=reference_returned)
        assert result.allowed is True
    else:
        snapshot = _snapshot(target=target)
        with pytest.raises(InputContractError) as exc_info:
            validate_action(action=_action(), case_id="case-A", decision_round=2,
                            snapshot=snapshot,
                            references=_references(returned=reference_returned))
        assert exc_info.value.code == "snapshot_shape_invalid"


@pytest.mark.parametrize("bad_ids", [[ ["I1"] ], [{"I1": "x"}], "I1", {"I1": "x"}])
def test_invalid_issue_id_shapes_raise_controlled_contract_error(bad_ids: object) -> None:
    """Malformed issue identifiers fail before set or mapping operations."""
    action = _action().model_copy(update={"issue_ids": bad_ids})
    with pytest.raises(InputContractError) as exc_info:
        _validate(action=action)
    assert exc_info.value.code == "action_shape_invalid"


def test_unhashable_issue_snapshot_id_is_rejected_as_bad_snapshot_shape() -> None:
    """Malformed trusted issue IDs cannot escape as an incidental TypeError."""
    malformed_issue = IssueSnapshot(["I1"], "关键事实", "fact", True,
                                    "open", "unknown", ())
    snapshot = _snapshot()
    malformed = CaseActionSnapshot(
        snapshot.case_id, snapshot.decision_round, (malformed_issue,),
        snapshot.executed_actions, snapshot.previous_progress, snapshot.config,
        snapshot.target, snapshot.budget,
    )
    with pytest.raises(InputContractError) as exc_info:
        validate_action(action=_action(), case_id="case-A", decision_round=2,
                        snapshot=malformed, references=_references())
    assert exc_info.value.code == "snapshot_shape_invalid"


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (_budget(), True), (_budget(action=Decimal("0.200001")), False),
        (_budget(reserved=Decimal("0.100001")), False), (_budget(reserved=None), False),
        (_budget(action=Decimal("NaN")), False), (_budget(settled=Decimal("-0.01")), False),
    ],
)
def test_budget_uses_exact_complete_snapshots_and_does_not_mutate_inputs(
    budget: BudgetSnapshot, expected: bool,
) -> None:
    """Exact boundary passes; overage or unknown/corrupt amounts fail closed."""
    values = tuple(getattr(budget, name) for name in (
        "settled", "reserved", "action_cap", "final_decision_reserve", "case_limit",
    ))
    result = _validate(budget=budget)
    assert result.allowed is expected
    assert tuple(getattr(budget, name) for name in (
        "settled", "reserved", "action_cap", "final_decision_reserve", "case_limit",
    )) == values
    if not expected:
        assert (result.reason, result.rule_no) == ("budget", 8)


def test_budget_failure_returns_no_lead_promotion_proposal() -> None:
    """Promotion is proposed only after every rule, including the budget, passes."""
    lead_issue = _issue("I2", kind="线索", key=False, origin="lead:L1")
    action = _action(["I2"], question="跟进线索", leads=["L1"])
    result = _validate(action, (_issue("I1", state="supported"), lead_issue),
                       prior=(_action(question="第一次核查"),),
                       progress=Progress(substantive=True),
                       config=ActionConfig(2, frozenset({"fact"}), False),
                       leads={"L1": "C1"}, budget=_budget(action=Decimal("0.200001")))
    assert (result.reason, result.rule_no, result.promote_issue_ids) == ("budget", 8, ())
