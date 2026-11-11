"""Build read-only, case-scoped arbiter budgets and capability cards."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import sqlite3
from typing import Literal
from uuid import UUID

from src.yuqing.eval.experiment_budget_recovery import RecoveredRequest, RecoveredState, apply_event
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStoreError
from src.yuqing.eval.experiment_budget import ExperimentBudgetStatus
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.schemas import CapabilityCardV2
from src.yuqing.settings import CaseBudgetConfig, CollabConfig, get_case_budget_config, get_collab_config


@dataclass(frozen=True)
class RunActionBinding:
    """Link wallet requests to an initial expert, arbiter decision, or follow-up action."""

    run_id: str
    role: Literal["initial", "decision", "action"]
    decision_round: int
    request_ids: tuple[str, ...]
    action_id: str | None = None
    target: Literal["fact", "propagation"] | None = None
    action_kind: Literal["investigate", "review"] | None = None
    action_status: Literal["claimed", "completed", "failed", "cancelled"] | None = None
    shared_prefix: bool = False

    def __post_init__(self) -> None:
        """Freeze request IDs and enforce role-specific binding fields."""
        object.__setattr__(self, "request_ids", tuple(self.request_ids))
        if not isinstance(self.run_id, str) or UUID(self.run_id).hex != self.run_id:
            raise ValueError("invalid_run_binding_identity")
        if type(self.decision_round) is not int or self.decision_round < 1:
            raise ValueError("invalid_run_binding_identity")
        if any(not isinstance(item, str) or UUID(item).hex != item for item in self.request_ids):
            raise ValueError("invalid_binding_request_id")
        if len(self.request_ids) != len(set(self.request_ids)):
            raise ValueError("duplicate_binding_request_id")
        if self.role not in {"initial", "decision", "action"}:
            raise ValueError("invalid_run_binding_role")
        if self.role == "decision" and any(value is not None for value in (
                self.action_id, self.target, self.action_kind, self.action_status)):
            raise ValueError("decision_binding_contains_action")
        if self.role == "initial" and any(value is not None for value in (
                self.action_id, self.action_kind, self.action_status)):
            raise ValueError("initial_binding_contains_action")
        if self.role == "initial" and self.target not in {"fact", "propagation"}:
            raise ValueError("initial_expert_identity_missing")
        if self.role == "initial" and self.decision_round != 1:
            raise ValueError("initial_binding_round_mismatch")
        if self.role == "action" and not all((self.action_id, self.target, self.action_kind, self.action_status)):
            raise ValueError("action_binding_incomplete")
        if (self.role == "action" and (self.target not in {"fact", "propagation"}
                or self.action_kind not in {"investigate", "review"}
                or self.action_status not in {"claimed", "completed", "failed", "cancelled"})):
            raise ValueError("invalid_action_binding_fields")
        if self.role == "action" and (not isinstance(self.action_id, str) or not self.action_id.strip()):
            raise ValueError("invalid_action_binding_id")
        if self.action_kind == "review" and self.target != "propagation":
            raise ValueError("review_target_mismatch")
        if (type(self.shared_prefix) is not bool or (self.shared_prefix
                and self.role not in {"initial", "decision"})
                or (self.shared_prefix and self.role == "decision" and self.decision_round != 1)):
            raise ValueError("invalid_shared_prefix_binding")


@dataclass(frozen=True)
class ArbiterBudgetScope:
    """Trusted identity and action history required to derive one case view."""

    experiment_id: str
    case_id: str
    sample_id: str
    group_id: str
    variant: str
    current_run_id: str
    decision_round: int
    bindings: tuple[RunActionBinding, ...]
    current_run_cost: Decimal
    fact_coverage: Literal["full", "demo", "none", "mismatched"]
    propagation_has_data: bool
    propagation_report_available: bool
    unreturned_comment_count: int
    human_paused: bool = False
    previous_progress: bool | None = None
    completed_actions: int = 0
    propagation_action_cap: Decimal | None = None

    def __post_init__(self) -> None:
        """Freeze the caller's binding list before it is checked against the wallet."""
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if not all(type(item) is RunActionBinding for item in self.bindings):
            raise ValueError("invalid_run_binding_collection")
        required = (self.experiment_id, self.case_id, self.sample_id, self.group_id, self.variant)
        if any(not isinstance(item, str) or not item.strip() for item in required):
            raise ValueError("budget_scope_identity_missing")
        if not isinstance(self.current_run_id, str) or UUID(self.current_run_id).hex != self.current_run_id:
            raise ValueError("invalid_current_run_id")
        if (type(self.decision_round) is not int or self.decision_round < 1
                or type(self.completed_actions) is not int or self.completed_actions < 0
                or type(self.unreturned_comment_count) is not int or self.unreturned_comment_count < 0):
            raise ValueError("invalid_budget_scope_counts")
        if type(self.human_paused) is not bool or type(self.propagation_has_data) is not bool:
            raise ValueError("invalid_budget_scope_flags")
        if type(self.propagation_report_available) is not bool:
            raise ValueError("invalid_budget_scope_flags")
        if self.fact_coverage not in {"full", "demo", "none", "mismatched"}:
            raise ValueError("invalid_fact_coverage")
        if self.previous_progress is not None and type(self.previous_progress) is not bool:
            raise ValueError("invalid_previous_progress")


@dataclass(frozen=True)
class ArbiterBudgetView:
    """Exact case and shared-wallet amounts plus three read-only capability cards."""

    available: bool
    reason: str | None
    wallet_state: Literal["healthy", "unknown", "halted", "unverifiable"]
    wallet_stop_reason: str | None
    settled: Decimal | None
    pending: Decimal | None
    final_reserve: Decimal | None
    action_pool: Decimal | None
    case_balance: Decimal | None
    shared_balance: Decimal | None
    cards: tuple[CapabilityCardV2, ...]


def _blocked(scope: ArbiterBudgetScope, reason: str, reserve: Decimal,
             *, wallet_state: Literal["healthy", "unknown", "halted", "unverifiable"],
             wallet_stop_reason: str | None, collab_config: CollabConfig) -> ArbiterBudgetView:
    """Return three unavailable cards while keeping unknown money explicitly unknown."""
    remaining = max(0, collab_config.max_actions - scope.completed_actions)
    definitions = (("fact", "investigate", scope.fact_coverage),
                   ("propagation", "investigate", None), ("propagation", "review", None))
    cards = tuple(CapabilityCardV2(
        expert=expert, action_kind=kind, case_id=scope.case_id, decision_round=scope.decision_round,
        fact_coverage=coverage, propagation_has_data=scope.propagation_has_data if expert == "propagation" else None,
        propagation_report_available=scope.propagation_report_available if expert == "propagation" else None,
        unreturned_comment_count=scope.unreturned_comment_count if expert == "propagation" else None,
        question_types=_question_types(expert, kind), available=False, remaining_actions=remaining,
        action_limit_cny=Decimal(0),
        available_balance_cny=None, final_reserve_cny=reserve, unavailable_reason=reason)
        for expert, kind, coverage in definitions)
    return ArbiterBudgetView(False, reason, wallet_state, wallet_stop_reason,
                             None, None, reserve, None, None, None, cards)


def _scope_blocked(scope: ArbiterBudgetScope, reason: str, reserve: Decimal,
                   wallet_state: Literal["healthy", "unknown", "halted", "unverifiable"],
                   wallet_stop_reason: str | None, collab_config: CollabConfig) -> ArbiterBudgetView:
    """Create a blocked view without losing the independently verified wallet state."""
    return _blocked(scope, reason, reserve, wallet_state=wallet_state,
                    wallet_stop_reason=wallet_stop_reason, collab_config=collab_config)


def _wallet_status(snapshot: ExperimentBudgetStatus, state: RecoveredState) -> tuple[
        Literal["healthy", "unknown", "halted"], str | None]:
    """Keep global wallet stop state independent from case-scope validation."""
    if snapshot.unknown_requests:
        return "unknown", "wallet_usage_unknown"
    if snapshot.halted or state.halt_reason:
        return "halted", snapshot.halt_reason or state.halt_reason or "wallet_halted"
    return "healthy", None


def _request_states(wallet: PersistentExperimentBudget) -> tuple[RecoveredState, ExperimentBudgetStatus]:
    """Read a hash-verified wallet snapshot and replay its events without writing."""
    with wallet.coordination():
        history = wallet.load_history()
        snapshot = wallet.snapshot()
        state = RecoveredState(history.manifest, {})
        for event in history.events:
            state = apply_event(state, event)
        if state.revision != history.revision or snapshot.experiment_id != history.manifest["experiment_id"]:
            raise ValueError("wallet_snapshot_identity_mismatch")
        return state, snapshot


def _validate_scope(scope: ArbiterBudgetScope, state: RecoveredState) -> dict[str, RunActionBinding]:
    """Require one current decision run and unique, correctly scoped action bindings."""
    if state.manifest["experiment_id"] != scope.experiment_id:
        raise ValueError("experiment_scope_mismatch")
    bindings = {item.run_id: item for item in scope.bindings}
    if len(bindings) != len(scope.bindings) or scope.current_run_id not in bindings:
        raise ValueError("run_binding_set_invalid")
    if any(item.shared_prefix for item in scope.bindings):
        if (scope.group_id not in {"B4-nf", "B4-fixed"} or scope.variant != scope.group_id):
            raise ValueError("shared_prefix_scope_invalid")
        if any(item.shared_prefix and item.role == "decision" and item.decision_round != 1
               for item in scope.bindings):
            raise ValueError("shared_prefix_round_invalid")
    current = bindings[scope.current_run_id]
    if current.role != "decision" or current.decision_round != scope.decision_round:
        raise ValueError("current_decision_binding_mismatch")
    initial_targets = [item.target for item in scope.bindings if item.role == "initial"]
    if len(initial_targets) != len(set(initial_targets)):
        raise ValueError("duplicate_initial_expert_binding")
    if sum(item.role == "decision" and item.decision_round == scope.decision_round
           for item in scope.bindings) != 1:
        raise ValueError("decision_round_binding_ambiguous")
    action_ids = [item.action_id for item in scope.bindings if item.role == "action"]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("duplicate_action_binding")
    if len(action_ids) != scope.completed_actions:
        raise ValueError("action_history_binding_mismatch")
    decision_rounds = [item.decision_round for item in scope.bindings if item.role == "decision"]
    expected_rounds = set(range(1, scope.decision_round + 1))
    if len(decision_rounds) != len(set(decision_rounds)) or set(decision_rounds) != expected_rounds:
        raise ValueError("decision_round_binding_ambiguous")
    return bindings


def _case_requests(scope: ArbiterBudgetScope, state: RecoveredState,
                   bindings: dict[str, RunActionBinding]) -> dict[str, RecoveredRequest]:
    """Match every scoped wallet request to exactly one trusted run binding."""
    by_id: dict[str, RecoveredRequest] = {}
    bound_ids = {request_id for item in bindings.values() for request_id in item.request_ids}
    if len(bound_ids) != sum(len(item.request_ids) for item in bindings.values()):
        raise ValueError("request_bound_to_multiple_runs")
    for request_id, request in state.requests.items():
        identity = request.identity
        run_id = str(identity.get("run_id", ""))
        binding = bindings.get(run_id)
        if binding is not None:
            shared_identity = (binding.shared_prefix
                and identity.get("group_id") == "shared-first-round"
                and identity.get("variant") == "shared-first-round")
            if (identity.get("sample_id") != scope.sample_id
                    or not (shared_identity or (identity.get("group_id") == scope.group_id
                    and identity.get("variant") == scope.variant))):
                raise ValueError("bound_request_scope_mismatch")
            if request_id not in binding.request_ids:
                raise ValueError("case_request_run_unmapped")
            by_id[request_id] = request
            continue
        if identity.get("sample_id") != scope.sample_id:
            continue
        if identity.get("group_id") is None or identity.get("variant") is None:
            raise ValueError("case_request_identity_incomplete")
        if identity.get("group_id") != scope.group_id or identity.get("variant") != scope.variant:
            continue
        raise ValueError("case_request_run_unmapped")
    if set(by_id) != bound_ids:
        raise ValueError("bound_request_missing_from_wallet")
    for binding in bindings.values():
        if binding.decision_round > scope.decision_round:
            raise ValueError("future_run_binding")
    return by_id


def _action_spend(bindings: dict[str, RunActionBinding], requests: dict[str, RecoveredRequest],
                  caps: dict[str, Decimal]) -> Decimal:
    """Count actual action spend once and retain allocations for claimed actions."""
    amount = Decimal(0)
    for binding in bindings.values():
        if binding.role != "action":
            continue
        cap_key = ("fact_investigate" if binding.target == "fact" else
                   "propagation_review" if binding.action_kind == "review" else "propagation_investigate")
        if binding.action_status == "claimed":
            if binding.request_ids:
                raise ValueError("claimed_action_already_has_run_requests")
            continue
        if binding.action_status != "cancelled" and not binding.request_ids:
            raise ValueError("action_run_requests_missing")
        action_total = Decimal(0)
        for request_id in binding.request_ids:
            request = requests[request_id]
            if request.state == "settled":
                action_total += request.actual or Decimal(0)
            elif request.state not in {"cancelled"}:
                raise ValueError("action_request_not_final")
        if action_total > caps[cap_key]:
            raise ValueError("action_spend_exceeds_cap")
        amount += action_total
    return amount


def _question_types(expert: str, kind: str) -> list[str]:
    """Describe supported question categories without granting permission."""
    if expert == "fact":
        return ["关键事实", "来源核验"]
    return ["传播叙事", "情绪等级"] if kind == "review" else ["传播性质", "传播评论"]


def _physical_reason(scope: ArbiterBudgetScope, expert: str, kind: str,
                     config: CollabConfig, remaining: int) -> str | None:
    """Apply stable action-switch, coverage, progress, and data availability gates."""
    if scope.human_paused:
        return "human_paused"
    if remaining <= 0:
        return "action_limit"
    if scope.completed_actions and scope.previous_progress is not True:
        return "no_progress"
    if expert not in config.targets:
        return "disabled"
    if expert == "fact" and scope.fact_coverage == "none":
        return "evidence_unavailable"
    if kind == "review" and (not config.review_enabled or not scope.propagation_report_available):
        return "review_unavailable"
    if expert == "propagation" and kind == "investigate" and not scope.propagation_has_data:
        return "target_unavailable"
    if expert == "propagation" and kind == "investigate" and scope.unreturned_comment_count <= 0:
        return "target_unavailable"
    return None


def _make_cards(scope: ArbiterBudgetScope, *, config: CollabConfig, coverage: str,
                remaining: int, reserve: Decimal, case_balance: Decimal,
                action_pool: Decimal, shared_balance: Decimal, caps: dict[str, Decimal]) -> tuple[CapabilityCardV2, ...]:
    """Create the fixed three cards without reserving or changing any wallet amount."""
    rows = (("fact", "investigate", "fact_investigate"),
            ("propagation", "investigate", "propagation_investigate"),
            ("propagation", "review", "propagation_review"))
    cards = []
    for expert, kind, cap_name in rows:
        fixed = caps[cap_name]
        action_balance = min(fixed, action_pool, case_balance)
        reason = _physical_reason(scope, expert, kind, config, remaining)
        if reason is None and (action_balance < fixed or fixed + reserve > shared_balance):
            reason = "budget"
        cards.append(CapabilityCardV2(
            expert=expert, action_kind=kind, case_id=scope.case_id, decision_round=scope.decision_round,
            fact_coverage=coverage if expert == "fact" else None,
            propagation_has_data=scope.propagation_has_data if expert == "propagation" else None,
            propagation_report_available=scope.propagation_report_available if expert == "propagation" else None,
            unreturned_comment_count=scope.unreturned_comment_count if expert == "propagation" else None,
            question_types=_question_types(expert, kind),
            available=reason is None, remaining_actions=remaining,
            action_limit_cny=action_balance, available_balance_cny=case_balance,
            final_reserve_cny=reserve, unavailable_reason=reason))
    return tuple(cards)


def _remaining_action_pool(bindings: dict[str, RunActionBinding], requests: dict[str, RecoveredRequest],
                           budget_config: object) -> tuple[Decimal, dict[str, Decimal]]:
    """Return the unspent pool after actual action costs and claimed fixed allocations."""
    caps = {name: Decimal(str(value)) for name, value in budget_config.action_caps.items()}
    claimed = sum((caps["fact_investigate" if item.target == "fact" else
        "propagation_review" if item.action_kind == "review" else "propagation_investigate"]
        for item in bindings.values() if item.role == "action" and item.action_status == "claimed"), Decimal(0))
    spent = _action_spend(bindings, requests, caps)
    return Decimal(str(budget_config.plan["action_pool"])) - spent - claimed, caps


def _calculate_view(scope: ArbiterBudgetScope, snapshot: ExperimentBudgetStatus,
                    requests: dict[str, RecoveredRequest], bindings: dict[str, RunActionBinding],
                    budget_config: object, collab_config: CollabConfig,
                    wallet_state: Literal["healthy", "unknown", "halted", "unverifiable"],
                    wallet_stop_reason: str | None) -> ArbiterBudgetView:
    """Calculate a healthy wallet's case amounts and fixed capability cards."""
    reserve = Decimal(str(budget_config.final_decision_reserve))
    current = [requests[item] for item in bindings[scope.current_run_id].request_ids]
    if any(item.state not in {"settled", "cancelled"} for item in current):
        return _scope_blocked(scope, "current_run_not_settled", reserve, wallet_state,
                              wallet_stop_reason, collab_config)
    current_spend = sum((item.actual or Decimal(0) for item in current if item.state == "settled"), Decimal(0))
    if current_spend != scope.current_run_cost:
        return _scope_blocked(scope, "current_run_cost_mismatch", reserve, wallet_state,
                              wallet_stop_reason, collab_config)
    if any(item.state in {"prepared", "reserved", "dispatched"} for item in requests.values()):
        return _scope_blocked(scope, "case_request_unsettled", reserve, wallet_state,
                              wallet_stop_reason, collab_config)
    settled = sum((item.actual or Decimal(0) for item in requests.values() if item.state == "settled"), Decimal(0))
    pending = sum((item.reserved for item in requests.values() if item.state in {"reserved", "dispatched"}), Decimal(0))
    try:
        action_pool, caps = _remaining_action_pool(bindings, requests, budget_config)
    except (KeyError, ValueError):
        return _scope_blocked(scope, "action_spend_unverifiable", reserve, wallet_state,
                              wallet_stop_reason, collab_config)
    if action_pool < 0:
        return _scope_blocked(scope, "action_pool_exceeded", reserve, wallet_state,
                              wallet_stop_reason, collab_config)
    case_balance = max(Decimal(0), Decimal(str(budget_config.max_cost_cny)) - settled - pending - reserve)
    shared_balance = max(Decimal(0), snapshot.max_cost_cny - snapshot.committed_cost_cny)
    cards = _make_cards(scope, config=collab_config, coverage=scope.fact_coverage,
        remaining=max(0, collab_config.max_actions - scope.completed_actions), reserve=reserve,
        case_balance=case_balance, action_pool=action_pool, shared_balance=shared_balance, caps=caps)
    return ArbiterBudgetView(True, None, wallet_state, wallet_stop_reason, settled, pending,
                             reserve, action_pool, case_balance, shared_balance, cards)


def build_arbiter_budget_view(wallet: PersistentExperimentBudget,
                              scope: ArbiterBudgetScope,
                              budget_config: CaseBudgetConfig | None = None,
                              collab_config: CollabConfig | None = None) -> ArbiterBudgetView:
    """Derive S/P/R/Q/W and capability cards from a read-only verified ledger snapshot."""
    budget_config = budget_config or get_case_budget_config()
    collab_config = collab_config or get_collab_config()
    if scope.propagation_action_cap is not None:
        from src.yuqing.settings import get_propagation_harness_config
        cap = scope.propagation_action_cap
        if not cap.is_finite() or cap <= 0 or cap > Decimal(str(get_propagation_harness_config().max_cost_cny)):
            raise ValueError("invalid_propagation_action_cap")
        collab_config = collab_config.model_copy(update={"targets": list(dict.fromkeys([*collab_config.targets, "propagation"]))})
        budget_config = budget_config.model_copy(update={"action_caps": {**budget_config.action_caps,
            "propagation_investigate": min(float(cap), budget_config.action_caps["propagation_investigate"])}})
    reserve_config = Decimal(str(budget_config.final_decision_reserve))
    if (type(scope.current_run_cost) is not Decimal or not scope.current_run_cost.is_finite()
            or scope.current_run_cost < 0):
        raise ValueError("invalid_current_run_cost")
    try:
        state, snapshot = _request_states(wallet)
    except (ExperimentBudgetStoreError, sqlite3.Error, OSError, ValueError, TypeError):
        return _blocked(scope, "wallet_history_unavailable", reserve_config, wallet_state="unverifiable",
                        wallet_stop_reason="wallet_history_unavailable", collab_config=collab_config)
    wallet_state, wallet_stop_reason = _wallet_status(snapshot, state)
    try:
        bindings = _validate_scope(scope, state)
        requests = _case_requests(scope, state, bindings)
    except (ValueError, TypeError) as exc:
        return _scope_blocked(scope, str(exc) or "budget_scope_unverifiable", reserve_config,
                              wallet_state, wallet_stop_reason, collab_config)
    if wallet_state != "healthy":
        return _scope_blocked(scope, wallet_stop_reason or "wallet_unavailable", reserve_config,
                              wallet_state, wallet_stop_reason, collab_config)
    return _calculate_view(scope, snapshot, requests, bindings, budget_config, collab_config,
                           wallet_state, wallet_stop_reason)
