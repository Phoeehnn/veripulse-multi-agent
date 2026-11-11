"""Hold non-serializable node artifacts for one isolated graph invocation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from typing import Any, Callable, Literal
from uuid import uuid4

from langgraph.types import interrupt
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict

from src.yuqing.graph.fact_adapter import (
    FactNodeAdapter, FactNodeCall, FactNodeFollowup, FactRuntimeArtifact,
)
from src.yuqing.collab.arbiter_material_v2 import FactArchiveInputV2
from src.yuqing.collab.arbiter_material_v2 import ArbiterMaterialV2, build_arbiter_material_v2
from src.yuqing.collab.board import (FactBoardV2Result, PropagationReferenceSnapshot,
    build_board, build_fact_board_v2, _report_sha256, _next_issue_number)
from src.yuqing.collab.progress import DeliveredEvidence, EvidenceDeliverySnapshot, evaluate_progress
from src.yuqing.schemas import (CollabAction, FactScopeV2, FollowupScopeV2, Issue,
    IssueEvent, Progress, TaskResult, ValidatedFactSnapshotV2)
from src.yuqing.graph.fact_selection import FactRoundChoice, select_fact_rounds
from src.yuqing.graph.nodes import CaseNodeServices, NodeHandler
from src.yuqing.agents.arbiter import score_case
from src.yuqing.schemas import ComplianceResult
from src.yuqing.settings import GraphConfig, get_disposition_profiles, get_policy_config
from src.yuqing.agents.compliance import check_compliance
from src.yuqing.agents.arbiter_loop import run_arbiter_agent_v2
from src.yuqing.agents.arbiter_loop import ArbiterMaterial
from src.yuqing.agents.propagation import PropagationResult, run_propagation_agent
from src.yuqing.eval.dual_budget import DualBudgetCoordinator
from src.yuqing.collab.arbiter_budget_v2 import ArbiterBudgetScope
from src.yuqing.harness.policy import PolicyConfig
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.harness.trace import RunContext, record_event, use_run_context
from src.yuqing.collab.validate import (CaseActionSnapshot, CaseReferenceSnapshot,
    ActionValidation, issue_snapshot, validate_action)
from src.yuqing.schemas import AgentSpec, ArbiterResultV2, ArbiterResultM6c, CollabAction, Sample
from src.yuqing.settings import ArbiterHarnessConfig, PropagationHarnessConfig


ScoreInputs = Callable[[dict[str, Any]], dict[str, Any]]
FactCallFactory = Callable[[dict[str, Any]], FactNodeCall]
FollowupFactory = Callable[[dict[str, Any], FactNodeCall], FactNodeFollowup]
WalletFactory = Callable[[], object | None]
ResourceFactory = Callable[["CaseRuntimeContext"], object]
ActionSnapshotFactory = Callable[[dict[str, Any]], CaseActionSnapshot]
ReferenceSnapshotFactory = Callable[[dict[str, Any]], CaseReferenceSnapshot]
ArbiterCall = Callable[[dict[str, Any], ArbiterMaterialV2], dict[str, Any]]
PropagationCallFactory = Callable[[dict[str, Any]], "PropagationNodeCall"]
ArbiterCallFactory = Callable[[dict[str, Any], ArbiterMaterialV2], "ArbiterNodeCall"]


@dataclass(frozen=True)
class ForkedCaseRuntime:
    """State and fresh invocation resources for one isolated experiment branch."""

    state: dict[str, Any]
    context: "CaseRuntimeContext"
    wallet: object
    trace: object


@dataclass(frozen=True)
class PendingFollowupIntent:
    """程序在补查准备前固定的动作身份，用于记录构建阶段失败。"""

    action: CollabAction
    action_id: str
    action_round: int
    decision_round: int


@dataclass(frozen=True)
class CaseRuntimeAssembly:
    """One compiled case graph, immutable seed, and invocation-bound sidecar."""

    graph: CompiledStateGraph
    state: dict[str, Any]
    context: "CaseRuntimeContext"
    request_factory: "CaseRequestFactory"

    def invoke(self) -> dict[str, Any]:
        """Run the preflighted graph once using this assembly's isolated identity."""
        return invoke_case_runtime(self.graph, self.state, self.context)


@dataclass(frozen=True)
class PropagationNodeCall:
    """Trusted objects needed by the existing propagation loop, held outside State."""

    spec: AgentSpec
    sample: Sample
    context: RunContext
    node_config: PropagationHarnessConfig
    worker_scope: EvidenceWorkerScope
    as_of: datetime | None = None
    policy_config: PolicyConfig | None = None
    coordinator_factory: Callable[[object], DualBudgetCoordinator] | None = None
    experiment_identity: dict[str, str] | None = None
    contract_version: Literal["v1.0", "v1.1"] = "v1.0"


@dataclass(frozen=True)
class ArbiterNodeCall:
    """Per-decision trusted snapshots and budget scope for the existing V2 arbiter loop."""

    spec: AgentSpec
    supplemental: ArbiterMaterial
    context: RunContext
    node_config: ArbiterHarnessConfig
    snapshot: CaseActionSnapshot
    references: CaseReferenceSnapshot
    budget_scope: ArbiterBudgetScope
    coordinator_factory: Callable[[object], DualBudgetCoordinator] | None = None
    experiment_identity: dict[str, str] | None = None
    case_leads: dict[str, Any] | None = None
    budget_profile: Literal["m5", "m6c"] = "m5"


@dataclass(frozen=True)
class CaseRequestFactory:
    """Build capacity requests through the exact message builders used by live Agents."""

    def fact_initial(self, sample_id: str, claim: str, entries: tuple[object, ...],
                     single_prompt: str, loop_prompt: str, capacity: object) -> object:
        """Build both fact requests using production V2 delivery and tool schemas."""
        from src.yuqing.eval.fact_v2_conditions import build_fact_v2_initial_requests

        return build_fact_v2_initial_requests(sample_id, claim, entries, single_prompt,
                                               loop_prompt, capacity)

    def fact_history(self, spec: AgentSpec, contract: object, initial_message: str,
                     program_messages: tuple[str, ...]) -> list[dict[str, object]]:
        """Rebuild an archived follow-up history with the production loop assembler."""
        from src.yuqing.harness.loop import _initial_message_history

        return _initial_message_history(spec, contract, initial_message, program_messages)

    def arbiter_v2(self, spec: AgentSpec, material: ArbiterMaterialV2,
                   supplemental: ArbiterMaterial, cards: tuple[object, ...],
                   dispatchability: tuple[dict[str, object], ...], *,
                   compact: bool = True) -> list[dict[str, object]]:
        """Build the exact compact V2 request body consumed by the production arbiter."""
        from src.yuqing.agents.arbiter_loop import _v2_messages

        return _v2_messages(spec, material, supplemental, cards, dispatchability,
                            compact_material=compact)

    def with_feedback(self, messages: list[dict[str, object]], feedback: str
                       ) -> list[dict[str, object]]:
        """Append the standard correction message to a captured production history."""
        return [*messages, {"role": "user", "content": feedback}]


def bind_propagation_node_handler(factory: PropagationCallFactory,
                                  context: "CaseRuntimeContext | None" = None) -> NodeHandler:
    """Connect a per-case input factory to the existing budgeted propagation Agent loop."""
    def call(state: dict[str, Any]) -> dict[str, Any]:
        """Run one propagation request and return only validated serializable outputs."""
        request = factory(state)
        if context is not None and isinstance(request.context, RunContext):
            context.register_usage_context(request.context)
        result = run_propagation_agent(request.spec, request.sample, request.context,
            node_config=request.node_config, worker_scope=request.worker_scope,
            as_of=request.as_of, policy_config=request.policy_config,
            coordinator_factory=request.coordinator_factory,
            experiment_identity=request.experiment_identity,
            **({"contract_version": request.contract_version}
               if request.contract_version != "v1.0" else {}))
        if context is not None and result is not None:
            context.store_propagation(result)
        return _propagation_patch(result, state)
    return call


def _propagation_patch(result: PropagationResult | None,
                       state: dict[str, Any]) -> dict[str, Any]:
    """Serialize a trusted propagation result without exposing runtime objects in State."""
    if result is None:
        return {"propagation": None, "prop_rounds": [{"status": "failed"}]}
    return {"propagation": result.report.model_dump(mode="json"),
        "leads": [item.model_dump(mode="json") for item in result.leads_by_id.values()],
        "prop_rounds": [{"status": result.stop_reason, "trace_path": result.trace_path,
            "run_id": result.run_id, "input_output_tokens": result.input_output_tokens,
            "cost_cny": result.cost_cny}]}


def bind_arbiter_v2_handler(factory: ArbiterCallFactory,
                             context: "CaseRuntimeContext | None" = None) -> ArbiterCall:
    """Connect built V2 material to the existing, budgeted V2 arbiter implementation."""
    def call(state: dict[str, Any], material: ArbiterMaterialV2) -> dict[str, Any]:
        """Run one arbiter decision and preserve previous decisions as graph history."""
        request = factory(state, material)
        if context is not None and isinstance(request.context, RunContext):
            context.register_usage_context(request.context)
        result = run_arbiter_agent_v2(request.spec, material, request.supplemental,
            request.context, node_config=request.node_config, snapshot=request.snapshot,
            references=request.references, budget_scope=request.budget_scope,
            coordinator_factory=request.coordinator_factory,
            experiment_identity=request.experiment_identity, budget_profile=request.budget_profile,
            case_leads=request.case_leads)
        patch = _arbiter_patch(result, state)
        if isinstance(result, ArbiterResultM6c) and result.decision.register_conflict is not None:
            issues = _register_m6c_conflict(context, result)
            patch.update(issues=[item.model_dump(mode="json") for item in issues],
                         m6c_force_review=True)
            with use_run_context(request.context):
                record_event("m6c_conflict_registered", {"decision_round": result.decision_round,
                    "issue_ids": [item.issue_id for item in issues if item.kind == "冲突"],
                    "fact_refs": result.decision.register_conflict.fact_refs,
                    "prop_refs": result.decision.register_conflict.prop_refs})
        return patch
    return call


def _arbiter_patch(result: ArbiterResultV2 | None, state: dict[str, Any]) -> dict[str, Any]:
    """Append a validated arbiter decision, or stop with a failure reason."""
    if result is None:
        return {"stop_reason": "arbiter_no_valid_decision",
            "failures": [{"node": "arbiter", "type": "no_valid_decision"}]}
    decisions = [*state.get("decisions", []), result.decision.model_dump(mode="json")]
    return {"decisions": decisions}


def _register_m6c_conflict(context: "CaseRuntimeContext | None", result: ArbiterResultM6c
                           ) -> tuple[Issue, ...]:
    """Turn only an accepted M6c conflict note into a program-owned conflicting issue."""
    if context is None or context.case_id != result.case_id:
        raise ValueError("m6c_conflict_context_missing")
    note = result.decision.register_conflict
    if note is None:
        return context.current_issues()
    origin = f"arbiter:{result.decision_round}:{result.run_id}"
    current = context.current_issues()
    if any(origin in item.origin_refs for item in current):
        return current
    refs = list(dict.fromkeys([ref.removeprefix("fact:") if ref.startswith("fact:F") else ref
                              for ref in [*note.fact_refs, *note.prop_refs]]))
    issue = Issue(issue_id=f"I{_next_issue_number(current)}", text=note.text,
        kind="冲突", origin_refs=[origin], owner="propagation", key=True, workflow="open",
        epistemic="conflicting", evidence_refs=refs,
        history=[IssueEvent(round=result.decision_round, field="created", old=None,
            new="conflicting", by="program", refs=refs)])
    updated = (*current, issue)
    if context._board is not None:
        context._board = replace(context._board, issues=updated)
    else:
        context._no_fact_issues = updated
    return updated


def assemble_case_runtime(state: dict[str, Any], handlers: CaseBusinessHandlers,
                           score_inputs: ScoreInputs, *, config: GraphConfig,
                           eval_mode: bool, context: "CaseRuntimeContext | None" = None,
                           external_calls_authorized: bool = False,
                           request_factory: CaseRequestFactory | None = None,
                           checkpointer: BaseCheckpointSaver | None = None) -> CaseRuntimeAssembly:
    """Build one invocation; live services fail closed before any wallet-bearing handler runs."""
    if not external_calls_authorized and not _handlers_are_offline(handlers):
        raise ValueError("case_external_calls_not_authorized")
    seed = dict(state)
    case_id, sample = seed.get("case_id"), seed.get("sample") or {}
    sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
    if not isinstance(case_id, str) or not isinstance(sample_id, str):
        raise ValueError("case_assembly_identity_missing")
    invocation = context or CaseRuntimeContext(case_id, sample_id)
    if (invocation.case_id != case_id or invocation.sample_id != sample_id
            or seed.get("thread_id") not in (None, invocation.thread_id)):
        raise ValueError("case_runtime_invocation_identity_mismatch")
    services = build_case_runtime_services(handlers, score_inputs, eval_mode=eval_mode,
        runtime_context=invocation)
    from src.yuqing.graph.build import build_case_graph

    return CaseRuntimeAssembly(build_case_graph(config, services, checkpointer=checkpointer,
        m6c=seed.get("m6c_enabled") is True), seed, invocation,
        request_factory or CaseRequestFactory())


def _handlers_are_offline(handlers: CaseBusinessHandlers) -> bool:
    """Treat only the explicitly constructed demo handlers as safe without call approval."""
    return getattr(handlers, "offline_mock", False) is True


def fork_case_runtime(snapshot: dict[str, Any], source_context: "CaseRuntimeContext",
                      source_wallet: object, source_trace: object,
                      wallet_factory: ResourceFactory,
                      trace_factory: ResourceFactory) -> ForkedCaseRuntime:
    """Fork serializable state while requiring new context, wallet, and trace identities."""
    from src.yuqing.graph.fork import fork_case_state

    state = fork_case_state(snapshot)
    sample = state.get("sample")
    case_id = state.get("case_id")
    sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
    if (case_id != source_context.case_id or not isinstance(sample_id, str)
            or sample_id != source_context.sample_id):
        raise ValueError("case_fork_source_identity_mismatch")
    if set(source_context._fact_artifacts) - {1}:
        raise ValueError("case_fork_requires_first_round_snapshot")
    context = CaseRuntimeContext(source_context.case_id, sample_id,
                                 thread_id=state["thread_id"])
    initial = source_context._fact_artifacts.get(1)
    if initial is not None:
        # Verified artifacts are immutable; share their source reader so a fork can
        # re-audit original spend without copying locks or SQLite handles.
        context.store_fact(1, initial)
    if source_context._board is not None:
        context._board = deepcopy(source_context._board)
    wallet, trace = None, None
    try:
        wallet, trace = wallet_factory(context), trace_factory(context)
        context.attach_usage_wallet(wallet)
        if (_wallet_identity(wallet) == _wallet_identity(source_wallet)
                or _trace_identity_overlaps(trace, source_trace)):
            raise ValueError("case_fork_resources_not_isolated")
    except Exception:
        _close_new_resource(trace, source_trace)
        _close_new_resource(wallet, source_wallet)
        raise
    return ForkedCaseRuntime(state, context, wallet, trace)


def _wallet_identity(wallet: object) -> tuple[object, object]:
    """Return persistent wallet identity, rejecting objects without a stable ledger identity."""
    identity = getattr(wallet, "persistent_identity", None)
    if callable(identity):
        identity = identity()
    if identity is None:
        identity = getattr(wallet, "experiment_id", None), getattr(wallet, "ledger_path", None)
    try:
        experiment_id, ledger_path = identity
    except (TypeError, ValueError) as exc:
        raise ValueError("case_fork_wallet_identity_missing") from exc
    if not experiment_id or not ledger_path:
        raise ValueError("case_fork_wallet_identity_missing")
    return experiment_id, ledger_path


def _trace_identity_overlaps(new: object, source: object) -> bool:
    """Require both trace run ID and path to differ independently across a fork."""
    new_run, new_path = getattr(new, "run_id", None), getattr(new, "trace_path", None)
    old_run, old_path = getattr(source, "run_id", None), getattr(source, "trace_path", None)
    return not all((new_run, new_path, old_run, old_path)) or new_run == old_run or new_path == old_path


def _close_new_resource(resource: object | None, source: object) -> None:
    """Release a rejected fork resource without closing its source counterpart."""
    close = getattr(resource, "close", None)
    if resource is not None and resource is not source and callable(close):
        close()


@dataclass(frozen=True)
class CaseBusinessHandlers:
    """Case-bound adapters for the existing fact, propagation, board, and arbiter modules."""

    prepare: NodeHandler
    fact_round1: NodeHandler
    propagation_round1: NodeHandler
    merge_round: NodeHandler
    arbitrate: NodeHandler
    validate_action: NodeHandler
    fact_followup: NodeHandler
    apply_result: NodeHandler
    offline_mock: bool = False
    propagation_followup: NodeHandler | None = None
    propagation_review: NodeHandler | None = None


@dataclass(frozen=True)
class CaseServiceBindings:
    """Per-case trusted inputs and replaceable model-call edges for production composition."""

    claim: str
    comments: tuple[tuple[str, str], ...]
    coverage: str
    fact_round1: NodeHandler
    propagation_round1: NodeHandler
    arbitrate: ArbiterCall
    fact_followup: NodeHandler
    action_snapshot: ActionSnapshotFactory
    references: ReferenceSnapshotFactory
    score_inputs: ScoreInputs
    offline_mock: bool = False
    request_factory: CaseRequestFactory = field(default_factory=CaseRequestFactory)
    propagation_followup: NodeHandler | None = None
    propagation_review: NodeHandler | None = None
    claim_propagation: Callable | None = None
    apply_result: NodeHandler | None = None
    merge_round: NodeHandler | None = None


def build_bound_case_handlers(context: CaseRuntimeContext,
                              bindings: CaseServiceBindings) -> CaseBusinessHandlers:
    """Bind existing compliance, V2 board, action gate, archive progress, and injected model edges."""
    return CaseBusinessHandlers(
        prepare=lambda state: _bound_prepare(state, bindings),
        fact_round1=bindings.fact_round1,
        propagation_round1=bindings.propagation_round1,
        merge_round=bindings.merge_round or (lambda state: _bound_merge(state, context, bindings)),
        arbitrate=lambda state: _bound_arbitrate(state, context, bindings),
        validate_action=lambda state: _bound_gate(state, context, bindings),
        fact_followup=bindings.fact_followup,
        apply_result=bindings.apply_result or (lambda state: _bound_apply(state, context)),
        offline_mock=bindings.offline_mock, propagation_followup=bindings.propagation_followup,
        propagation_review=bindings.propagation_review)


def assemble_bound_case_runtime(state: dict[str, Any], bindings: CaseServiceBindings, *,
                                config: GraphConfig, eval_mode: bool,
                                context: CaseRuntimeContext | None = None,
                                external_calls_authorized: bool = False) -> CaseRuntimeAssembly:
    """Build one graph assembly with real deterministic business services and injected call edges."""
    case_id, sample = state.get("case_id"), state.get("sample") or {}
    sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
    if not isinstance(case_id, str) or not isinstance(sample_id, str):
        raise ValueError("case_assembly_identity_missing")
    invocation = context or CaseRuntimeContext(case_id, sample_id)
    handlers = build_bound_case_handlers(invocation, bindings)
    return assemble_case_runtime(state, handlers, bindings.score_inputs, config=config,
        eval_mode=eval_mode, context=invocation,
        external_calls_authorized=external_calls_authorized,
        request_factory=bindings.request_factory)


def _bound_prepare(_state: dict[str, Any], bindings: CaseServiceBindings) -> dict[str, Any]:
    """Run the existing compliance rules over the immutable case claim and comments."""
    result = check_compliance(bindings.claim, bindings.comments)
    return {"coverage": bindings.coverage, "compliance": result.model_dump(mode="json")}


def _bound_merge(state: dict[str, Any], context: CaseRuntimeContext,
                 bindings: CaseServiceBindings, propagation_source: object | None = None
                 ) -> dict[str, Any]:
    """Build an atomic fact/propagation board, including explicit absent-fact cases."""
    artifacts = context.fact_snapshots()
    source_comments = () if propagation_source is None else propagation_source.comments
    source_stats = None if propagation_source is None else propagation_source.stats
    source_has_data = bool(source_comments) or bool(source_stats and source_stats.has_data)
    propagation, prop_refs = context.propagation_view(state, bool(bindings.comments) or source_has_data)
    if propagation is None and source_has_data:
        prop_refs = PropagationReferenceSnapshot(context.case_id, propagation_source.round_no,
            True, False, frozenset(item.comment_id for item in source_comments), "")
    if not artifacts:
        built = build_board(case_id=context.case_id, round_no=1, fact_verdict=None,
            fact_status=_fact_absence_status(state, bindings.coverage), fact_refs=None,
            propagation_report=propagation, propagation_refs=prop_refs)
        if built.diagnostics:
            raise ValueError("case_merge_absent_fact_board_invalid")
        context.publish_no_fact_issues(built.issues)
        issues = built.issues
    else:
        snapshot = artifacts[-1]
        board = context.build_fact_snapshot(snapshot, snapshot.scope,
            state.get("action_round", snapshot.scope.fact_round_no))
        prop_board = build_board(case_id=context.case_id, round_no=1, fact_verdict=None,
            fact_status="not_dispatched", fact_refs=None, propagation_report=propagation,
            propagation_refs=prop_refs)
        if prop_board.diagnostics:
            raise ValueError("case_merge_propagation_board_invalid")
        prop_issues = tuple(item for item in prop_board.issues
            if not any(ref.startswith("fact:") for ref in item.origin_refs))
        issues = (*board.issues, *_renumber_issues(prop_issues, len(board.issues)))
        context.publish_board(replace(board, issues=issues))
    comments = dict(bindings.comments)
    comments.update((item.comment_id, item.text) for item in source_comments)
    if context._propagation_result is not None:
        comments.update((item.comment_id, item.text) for item in context._propagation_result.comments)
    compliance = check_compliance(bindings.claim, tuple(comments.items()))
    return {"issues": [item.model_dump(mode="json") for item in issues],
        "compliance": compliance.model_dump(mode="json")}


def _fact_absence_status(state: dict[str, Any], coverage: str) -> str:
    """Distinguish unavailable evidence coverage from a failed dispatched fact agent."""
    if coverage == "none":
        return "not_dispatched"
    rounds = state.get("fact_rounds", [])
    if any(isinstance(row, dict) and row.get("status") == "not_dispatched" for row in rounds):
        return "not_dispatched"
    return "failed"


def _renumber_issues(issues: tuple[Issue, ...], offset: int) -> tuple[Issue, ...]:
    """Append propagated issues after fact issues without colliding with their IDs."""
    return tuple(issue.model_copy(update={"issue_id": f"I{offset + index}"})
        for index, issue in enumerate(issues, start=1))


def _bound_gate(state: dict[str, Any], context: CaseRuntimeContext,
                bindings: CaseServiceBindings) -> dict[str, Any]:
    """Apply the existing eight-rule gate to the latest submitted action."""
    decisions = state.get("decisions", [])
    decision = decisions[-1] if decisions else {}
    raw_action = decision.get("action") if isinstance(decision, dict) else None
    if raw_action is None:
        context._pending_intent = None
        context._pending_followup = None
        return {"action_validation": {"accepted": False, "reason": "no_action"},
            "pending_task": None}
    action = CollabAction.model_validate(raw_action)
    base = bindings.action_snapshot(state)
    issues = tuple(issue_snapshot(item) for item in context.current_issues())
    snapshot, refs = replace(base, issues=issues), bindings.references(state)
    if action.target == "propagation" and bindings.claim_propagation is None:
        snapshot = replace(snapshot, target=replace(snapshot.target, propagation_has_data=False))
    checked = validate_action(action=action, case_id=snapshot.case_id,
        decision_round=snapshot.decision_round, snapshot=snapshot, references=refs)
    context.reserve_followup(action, snapshot, checked.allowed)
    if checked.allowed and action.target == "propagation" and bindings.claim_propagation is not None:
        bindings.claim_propagation(action, snapshot, refs, checked)
    patch = _validation_patch(checked, action)
    intent = context._pending_intent
    patch["pending_task"] = None if intent is None else {
        "action_id": intent.action_id, "action_round": intent.action_round,
        "decision_round": intent.decision_round, "kind": action.kind,
        "target": action.target, "issue_ids": list(action.issue_ids)}
    return patch


def _bound_arbitrate(state: dict[str, Any], context: CaseRuntimeContext,
                     bindings: CaseServiceBindings) -> dict[str, Any]:
    """Re-read archives into the production V2 arbiter material before the replaceable call edge."""
    decisions = state.get("decisions", [])
    if context._board is not None:
        material = build_arbiter_material_v2(case_id=context.case_id,
            sample_id=context.sample_id, decision_round=len(decisions) + 1,
            claim=bindings.claim, board=context._board, fact_runs=context.fact_runs())
    else:
        material = _build_absent_fact_material(state, context, bindings,
            len(decisions) + 1)
    return bindings.arbitrate(state, material)


def _build_absent_fact_material(state: dict[str, Any], context: CaseRuntimeContext,
                                bindings: CaseServiceBindings,
                                decision_round: int) -> ArbiterMaterialV2:
    """Build explicit no-coverage or failed-fact material without inventing an archive hash."""
    issues = context.current_issues()
    refs = {f"issue:{item.issue_id}" for item in issues}
    from hashlib import sha256

    claim_hash = sha256(bindings.claim.encode("utf-8")).hexdigest()
    return ArbiterMaterialV2(case_id=context.case_id, sample_id=context.sample_id,
        decision_round=decision_round, claim=bindings.claim, claim_sha256=claim_hash,
        issues=issues, fact_reports=(), evidence=(), available_refs=tuple(sorted(refs)),
        fact_absence_reason=_fact_absence_reason(state, bindings.coverage))


def _fact_absence_reason(state: dict[str, Any], coverage: str) -> str:
    """Use the recorded dispatch state to distinguish no source from an agent failure."""
    if coverage == "none":
        return "no_coverage"
    rounds = state.get("fact_rounds", [])
    if any(isinstance(row, dict) and row.get("status") == "not_dispatched" for row in rounds):
        return "not_dispatched"
    if any(isinstance(row, dict) and row.get("status") == "failed" for row in rounds):
        return "agent_failed"
    return "unknown"


def _bound_apply(state: dict[str, Any], context: CaseRuntimeContext) -> dict[str, Any]:
    """Apply only an archive-verified followup and preserve its TaskResult and progress."""
    artifacts = context.fact_runs()
    intent = context._pending_intent
    if (artifacts and isinstance(artifacts[-1].scope, FollowupScopeV2)
            and intent is not None and artifacts[-1].scope.action_id == intent.action_id):
        artifact = context._fact_artifacts[artifacts[-1].scope.fact_round_no]
        board, result = context.apply_verified_followup(artifact)
        issue_ids = list(artifacts[-1].scope.issue_ids)
    elif context._pending_followup is not None or context._pending_intent is not None:
        pending = context._pending_followup or context._pending_intent
        if pending is None:
            raise ValueError("case_followup_intent_missing")
        board, result = context.apply_failed_followup(pending)
        issue_ids = list(pending.scope.issue_ids) if isinstance(pending, FactNodeFollowup) \
            else list(pending.action.issue_ids)
    else:
        raise ValueError("case_apply_followup_archive_missing")
    context._pending_followup = None
    context._pending_intent = None
    return {"issues": [item.model_dump(mode="json") for item in context.current_issues()],
        "task_results": [*state.get("task_results", []), result.model_dump(mode="json")],
        "pending_task": None,
        "actions": [*state.get("actions", []), {"action_id": result.action_id,
            "kind": result.kind, "target": result.target,
            "question": intent.action.question if intent is not None else "",
            "issue_ids": issue_ids, "status": result.status}]}


def _validation_patch(result: ActionValidation, action: CollabAction) -> dict[str, Any]:
    """Convert the trusted action-gate result to a serializable graph patch."""
    return {"action_validation": {"accepted": result.allowed,
        "reason": result.reason, "rule_no": result.rule_no,
        "promote_issue_ids": list(result.promote_issue_ids),
        "action": action.model_dump(mode="json") if result.allowed else None}}


def bind_fact_node_handlers(adapter: FactNodeAdapter, context: "CaseRuntimeContext",
                            call_factory: FactCallFactory,
                            followup_factory: FollowupFactory,
                            wallet_factory: WalletFactory) -> tuple[NodeHandler, NodeHandler]:
    """Bind the frozen fact adapter to one invocation sidecar without putting artifacts in State."""
    def first_round(state: dict[str, Any]) -> dict[str, Any]:
        """Run and independently verify the first fact archive for this case invocation."""
        call = call_factory(state)
        run_context = getattr(call, "trace", None)
        if isinstance(run_context, RunContext):
            context.register_usage_context(run_context)
        artifact = adapter.run_round1_runtime(call, wallet=wallet_factory())
        if artifact is None:
            return {"fact": None, "fact_rounds": [_summary_row(None)]}
        context.store_fact(1, artifact)
        return {"fact": artifact.summary.report, "fact_rounds": [_summary_row(artifact)]}

    def followup(state: dict[str, Any]) -> dict[str, Any]:
        """Run only a caller-built, gate-approved followup and retain its verified archive."""
        intent = context._pending_intent
        if intent is None:
            raise ValueError("case_followup_intent_missing")
        pending = state.get("pending_task")
        if not isinstance(pending, dict) or pending.get("action_id") != intent.action_id:
            raise ValueError("case_followup_state_intent_mismatch")
        factory_state = {**state, "_program_action_id": intent.action_id,
            "_program_action_round": intent.action_round,
            "_program_decision_round": intent.decision_round}
        call = call_factory(factory_state)
        request = followup_factory(factory_state, call)
        if request.scope.action_id != intent.action_id:
            raise ValueError("case_followup_action_identity_mismatch")
        run_context = getattr(call, "trace", None)
        if isinstance(run_context, RunContext):
            context.register_usage_context(run_context)
        context._pending_followup = request
        artifact = adapter.run_followup_runtime(request, wallet=wallet_factory())
        if artifact is None:
            return {"fact": None, "fact_rounds": [_summary_row(None,
                round_no=context.next_fact_round())]}
        round_no = artifact.archive_input.scope.fact_round_no
        context.store_fact(round_no, artifact)
        return {"fact": artifact.summary.report, "fact_rounds": [_summary_row(artifact)]}

    return first_round, followup


def _summary_row(artifact: FactRuntimeArtifact | None, *, round_no: int = 1) -> dict[str, Any]:
    """Convert an adapter result to JSON-safe facts while omitting rich archive objects."""
    if artifact is None:
        return {"round": round_no, "status": "failed", "stop_reason": "no_valid_result"}
    result = artifact.summary
    return {"round": artifact.archive_input.scope.fact_round_no, "status": result.status,
        "stop_reason": result.stop_reason, "ledger_refs": list(result.ledger_refs),
        "delivery_refs": list(result.delivery_refs), "archive_path": result.archive_path,
        "archive_sha256": result.archive_sha256, "usage": result.usage}


def build_case_runtime_services(handlers: CaseBusinessHandlers, score_inputs: ScoreInputs,
                                *, eval_mode: bool,
                                runtime_context: "CaseRuntimeContext | None" = None) -> CaseNodeServices:
    """Compose case handlers with the existing deterministic score, eval gate, and brief."""
    def brief(state: dict[str, Any]) -> dict[str, Any]:
        """Render the final deterministic Markdown report."""
        from src.yuqing.report.brief import render_case_brief

        report_state = dict(state)
        if runtime_context is not None:
            report_state["usage_summary"] = runtime_context.usage_summary()
        return {"brief": render_case_brief(report_state)}

    score = _build_score_handler(score_inputs, runtime_context)
    human_gate = _build_human_gate(eval_mode)
    alert_gate = _build_alert_gate()
    return CaseNodeServices(handlers.prepare, handlers.fact_round1, handlers.propagation_round1,
        handlers.merge_round, handlers.arbitrate, handlers.validate_action,
        handlers.fact_followup, handlers.apply_result, score, human_gate, brief, alert_gate,
        handlers.propagation_followup, handlers.propagation_review)


def _build_score_handler(score_inputs: ScoreInputs,
                         runtime_context: "CaseRuntimeContext | None") -> NodeHandler:
    """Bind the existing deterministic score logic to the current case context."""
    def score(state: dict[str, Any]) -> dict[str, Any]:
        """Run existing rule-based scoring on a trusted, program-built argument set."""
        selected_state = dict(state)
        patch: dict[str, Any] = {}
        choice = runtime_context.select_fact_conclusion() if runtime_context is not None else None
        if choice is not None:
            selected_state["fact"] = choice.report
        inputs = score_inputs(selected_state)
        if choice is not None:
            if choice.force_review:
                inputs.update(fact_status="信息不足", unresolved_key_conflict=True,
                              arbiter_requests_review=True)
            patch["fact_selection"] = {"round": choice.round_no, "reason": choice.reason,
                "force_review": choice.force_review,
                "conflicting_issue_ids": list(choice.conflicting_issue_ids)}
            patch["fact"] = choice.report
        risk = score_case(**inputs)
        if choice is not None and choice.conflicting_issue_ids:
            changed = runtime_context.mark_conflicting(choice.conflicting_issue_ids,
                choice.round_no or 1)
            if changed:
                patch["issues"] = list(changed)
        patch["risk"] = risk.model_dump(mode="json")
        return patch
    return score


def _build_human_gate(eval_mode: bool) -> NodeHandler:
    """Build the record-only evaluation gate or the resumable human review gate."""
    def human_gate(state: dict[str, Any]) -> dict[str, Any]:
        """Pause only when a non-evaluation case requires a human decision."""
        risk = state.get("risk") or {}
        if eval_mode:
            approval = "review_recorded" if risk.get("needs_human_review") else "not_needed"
            return {"approval": approval, "review_recorded": approval == "review_recorded"}
        if not risk.get("needs_human_review"):
            return {"approval": "not_needed"}
        decision_id = _case_decision_id(state)
        choices = _review_choices(risk)
        supplied = interrupt({"case_id": state["case_id"], "decision_id": decision_id,
            "risk": risk, "suggested_actions": choices})
        review = _validate_human_review(supplied, state, decision_id, choices)
        approval = "rejected" if review["decision"] == "reject" else "approved"
        return {"approval": approval, "decision_id": decision_id, "review": review}
    return human_gate


def _build_alert_gate() -> NodeHandler:
    """Build a fail-closed R2 gate that records a dry-run once per decision."""
    from src.yuqing.harness.policy import decide
    from src.yuqing.harness.tools import ToolRegistry, ToolSpec

    registry = ToolRegistry()
    registry.register(ToolSpec(name="dispatch_alert", description="记录待发送告警（dry-run）",
        args_model=AlertDispatchArgs, risk="R2"))

    def alert_gate(state: dict[str, Any]) -> dict[str, Any]:
        """Recheck approval and configured R2 permission before recording an alert."""
        review = state.get("review") or {}
        decision_id = _case_decision_id(state)
        if (state.get("approval") != "approved" or review.get("decision") not in {"approve", "modify"}
                or review.get("case_id") != state.get("case_id")
                or review.get("decision_id") != decision_id):
            return {"alert": {"status": "blocked", "reason": "approval_not_verified"}}
        permission = decide(registry, "alert_gate", "dispatch_alert", caller="controller",
                             config=get_policy_config())
        if permission.action != "dry_run":
            return {"alert": {"status": "blocked", "reason": permission.reason}}
        idempotency_key = f"{state['case_id']}:{decision_id}"
        previous = state.get("alert")
        if isinstance(previous, dict) and previous.get("idempotency_key") == idempotency_key:
            return {"alert": previous}
        disposition = review.get("final_disposition") or (state.get("risk") or {}).get("disposition")
        return {"alert": {"case_id": state["case_id"], "decision_id": decision_id,
            "idempotency_key": idempotency_key, "status": "dry_run", "dry_run": True,
            "would_send": True, "disposition": disposition}}
    return alert_gate


class AlertDispatchArgs(BaseModel):
    """声明R2门卫检查所需的最少告警身份字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    decision_id: str
    disposition: str


def _case_decision_id(state: dict[str, Any]) -> str:
    """Bind an approval to the exact case decision and risk snapshot."""
    payload = {"case_id": state.get("case_id"), "decisions": state.get("decisions", []),
        "risk": state.get("risk")}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _review_choices(risk: dict[str, Any]) -> list[str]:
    """Return only configured actions for this scenario and risk level."""
    profiles = get_disposition_profiles()
    scenario = risk.get("scenario")
    row = "system_failure" if risk.get("status") == "system_failure" else str(risk.get("level"))
    table = profiles.get(scenario, {}).get(row)
    if not isinstance(table, dict):
        raise ValueError("review_disposition_profile_missing")
    return list(dict.fromkeys(table.values()))


def _validate_human_review(supplied: Any, state: dict[str, Any], decision_id: str,
                           choices: list[str]) -> dict[str, Any]:
    """Reject a resume value that lacks the current case identity or valid choice."""
    if not isinstance(supplied, dict):
        raise ValueError("human_review_payload_invalid")
    decision, reason, reviewer = supplied.get("decision"), supplied.get("reason"), supplied.get("reviewer")
    if (supplied.get("case_id") != state.get("case_id") or supplied.get("decision_id") != decision_id
            or decision not in {"approve", "reject", "modify"}
            or not isinstance(reason, str) or not reason.strip()
            or not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError("human_review_identity_or_fields_invalid")
    disposition = supplied.get("final_disposition")
    if decision == "modify" and disposition not in choices:
        raise ValueError("human_review_disposition_invalid")
    if decision != "modify" and disposition is not None:
        raise ValueError("human_review_unexpected_disposition")
    return {"case_id": state["case_id"], "decision_id": decision_id, "decision": decision,
        "reason": reason.strip(), "reviewer": reviewer.strip(), "final_disposition": disposition}


def build_case_dry_run_handlers() -> tuple[CaseBusinessHandlers, ScoreInputs]:
    """Create explicit model-free handlers while keeping deterministic score logic real."""
    compliance = ComplianceResult(claim_status="checked_no_hit", comment_status="no_comments")
    handlers = CaseBusinessHandlers(
        prepare=lambda _state: {"coverage": "full", "compliance": compliance.model_dump(mode="json")},
        fact_round1=lambda _state: {"fact": {"label": "信息不足", "confidence": 0.0},
            "fact_rounds": [{"status": "mock_completed", "delivery_refs": []}]},
        propagation_round1=lambda _state: {"propagation": {"emotion_level": "低",
            "spread_level": "未知"}, "prop_rounds": [{"status": "mock_completed"}]},
        merge_round=lambda _state: {"issues": []},
        arbitrate=lambda _state: {"decisions": [{"next_action": "finalize", "mock": True}]},
        validate_action=lambda _state: {"action_validation": {"accepted": False,
            "reason": "mock_finalize_no_action"}},
        fact_followup=lambda _state: {"fact": None}, apply_result=lambda _state: {},
        offline_mock=True)

    def score_inputs(state: dict[str, Any]) -> dict[str, Any]:
        """Map mock reports into the same deterministic production scoring function."""
        fact = state.get("fact") or {"label": "信息不足"}
        return {"fact_status": fact["label"], "fact_risk": 0.0,
            "fact_coverage": "full", "fact_failed": False, "fact_citation_ok": True,
            "emotion_level": state["propagation"]["emotion_level"],
            "spread_level": state["propagation"]["spread_level"], "compliance": compliance,
            "scenario": "general"}

    return handlers, score_inputs


def build_case_dry_run_services() -> CaseNodeServices:
    """Build the model-free graph services used by offline CLI and unit checks."""
    handlers, score_inputs = build_case_dry_run_handlers()
    return build_case_runtime_services(handlers, score_inputs, eval_mode=True)


def build_case_dry_run_assembly(sample_id: str, *, eval_mode: bool = True) -> CaseRuntimeAssembly:
    """Use the shared assembly path for a complete dry-run without wallet or Provider."""
    handlers, score_inputs = build_case_dry_run_handlers()
    state = {"case_id": f"dry-run:{sample_id}", "sample": {"sample_id": sample_id},
        "ticket": {"claim": "离线模拟，不代表真实判断"},
        "collab_config": {"eval_mode": eval_mode}}
    from src.yuqing.settings import get_graph_config

    return assemble_case_runtime(state, handlers, score_inputs, config=get_graph_config(),
                                 eval_mode=eval_mode)
from src.yuqing.graph.build import invoke_case_graph
from langgraph.graph.state import CompiledStateGraph


@dataclass
class CaseRuntimeContext:
    """Per-invocation sidecar for verified objects that cannot enter LangGraph State."""

    case_id: str
    sample_id: str
    thread_id: str = field(default_factory=lambda: uuid4().hex)
    _fact_artifacts: dict[int, FactRuntimeArtifact] = field(default_factory=dict, repr=False)
    _pending_followup: FactNodeFollowup | None = field(default=None, repr=False)
    _pending_intent: PendingFollowupIntent | None = field(default=None, repr=False)
    _board: FactBoardV2Result | None = field(default=None, repr=False)
    _no_fact_issues: tuple[Issue, ...] | None = field(default=None, repr=False)
    _propagation_result: PropagationResult | None = field(default=None, repr=False)
    _propagation_rounds: dict[int, PropagationResult] = field(default_factory=dict, repr=False)
    _usage_contexts: dict[str, RunContext] = field(default_factory=dict, repr=False)
    _usage_wallet: object | None = field(default=None, repr=False)
    _invoked: bool = field(default=False, repr=False)

    def register_usage_context(self, run: RunContext) -> None:
        """Retain each program-created model run for the final case brief."""
        if run.sample_id != self.sample_id:
            raise ValueError("case_usage_run_sample_mismatch")
        existing = self._usage_contexts.get(run.run_id)
        if existing is not None and existing != run:
            raise ValueError("case_usage_run_identity_conflict")
        self._usage_contexts[run.run_id] = run.model_copy(deep=True)

    def attach_usage_wallet(self, wallet: object) -> None:
        """Bind the case-owned ledger used to reconcile model request traces."""
        if self._usage_wallet is not None and self._usage_wallet is not wallet:
            raise ValueError("case_usage_wallet_identity_conflict")
        self._usage_wallet = wallet

    def usage_summary(self) -> dict[str, Any]:
        """Return Trace and wallet totals without placing runtime objects in graph State."""
        from src.yuqing.report.case_usage import summarize_case_usage

        return summarize_case_usage(tuple(self._usage_contexts.values()), self._usage_wallet)

    def store_fact(self, round_no: int, artifact: FactRuntimeArtifact) -> None:
        """Store a verified fact artifact under its round after identity and uniqueness checks."""
        scope = artifact.archive_input.scope
        if (round_no < 1 or round_no in self._fact_artifacts or scope.case_id != self.case_id
                or scope.sample_id != self.sample_id or scope.fact_round_no != round_no
                or artifact.archive_input.result.run_id != scope.run_id
                or artifact.snapshot.scope != scope):
            raise ValueError("case_runtime_fact_artifact_scope_mismatch")
        self._fact_artifacts[round_no] = artifact

    def fact_runs(self) -> tuple[FactArchiveInputV2, ...]:
        """Return the exact archived fact runs in round order for trusted material assembly."""
        return tuple(self._fact_artifacts[index].archive_input
                     for index in sorted(self._fact_artifacts))

    def fact_snapshots(self) -> tuple[ValidatedFactSnapshotV2, ...]:
        """Return only archive-reader verified snapshots in round order."""
        return tuple(self._fact_artifacts[index].snapshot
                     for index in sorted(self._fact_artifacts))

    def next_fact_round(self) -> int:
        """Return the next sequential fact round for a failed result summary."""
        return max(self._fact_artifacts, default=0) + 1

    def publish_fact_snapshot(self, snapshot: ValidatedFactSnapshotV2,
                              expected_scope: FactScopeV2 | FollowupScopeV2,
                              action_round: int) -> FactBoardV2Result:
        """Publish an archive-verified report through the production V2 issue-board builder."""
        board = self.build_fact_snapshot(snapshot, expected_scope, action_round)
        self._board = board
        return board

    def publish_board(self, board: FactBoardV2Result) -> None:
        """Publish one fully constructed board only after all deterministic checks pass."""
        self._board, self._no_fact_issues = board, None

    def publish_no_fact_issues(self, issues: tuple[Issue, ...]) -> None:
        """Keep a validated absent-fact board without inventing a V2 report audit record."""
        self._no_fact_issues, self._board = tuple(issues), None

    def current_issues(self) -> tuple[Issue, ...]:
        """Return the currently published fact-backed or absent-fact issue rows."""
        if self._board is not None:
            return self._board.issues
        return self._no_fact_issues or ()

    def store_propagation(self, result: PropagationResult) -> None:
        """Retain the typed propagation result used to authenticate its comment references."""
        if result.sample_id != self.sample_id or self._propagation_result is not None:
            raise ValueError("case_runtime_propagation_identity_mismatch")
        self._propagation_result = result

    def propagation_view(self, state: dict[str, Any], source_has_data: bool = False
                         ) -> tuple[Any, PropagationReferenceSnapshot | None]:
        """Return propagation material only when State matches its typed runtime result."""
        result = self._propagation_result
        report = state.get("propagation")
        if result is None:
            if report is not None:
                raise ValueError("case_runtime_propagation_archive_missing")
            if source_has_data:
                return None, PropagationReferenceSnapshot(self.case_id, 1, True, False,
                    frozenset(), "")
            return None, None
        if report != result.report.model_dump(mode="json"):
            raise ValueError("case_runtime_propagation_state_mismatch")
        has_data = bool(result.comments or result.stats and result.stats.has_data)
        refs = PropagationReferenceSnapshot(self.case_id, max(self._propagation_rounds, default=1), has_data,
            True, frozenset(item.comment_id for item in result.comments),
            _report_sha256(result.report))
        return result.report if has_data else None, refs

    def reserve_followup(self, action: CollabAction, snapshot: CaseActionSnapshot,
                         allowed: bool) -> None:
        """Bind an approved fact task identity before request or scope construction begins."""
        self._pending_intent = None
        self._pending_followup = None
        if allowed and action.kind == "investigate" and action.target == "fact":
            self._pending_intent = PendingFollowupIntent(action, uuid4().hex,
                len(snapshot.executed_actions) + 1, snapshot.decision_round)

    def build_fact_snapshot(self, snapshot: ValidatedFactSnapshotV2,
                            expected_scope: FactScopeV2 | FollowupScopeV2,
                            action_round: int) -> FactBoardV2Result:
        """Build a candidate board without publishing partially computed changes."""
        old = self._board
        return build_fact_board_v2(snapshot=snapshot, expected_scope=expected_scope,
            issues=() if old is None else old.issues,
            text_sources=() if old is None else old.text_sources,
            proposal_origins=() if old is None else old.proposal_origins,
            fact_reports=() if old is None else old.fact_reports, action_round=action_round)

    def apply_verified_followup(self, artifact: FactRuntimeArtifact) -> tuple[FactBoardV2Result, TaskResult]:
        """Re-read a V2 archive, update the board, and measure only genuinely new progress."""
        scope = artifact.archive_input.scope
        if not isinstance(scope, FollowupScopeV2) or self._board is None:
            raise ValueError("case_runtime_followup_context_missing")
        verified = artifact.archive_input.reader.read(artifact.archive_input.result,
            artifact.archive_input.task, artifact.archive_input.contract, scope)
        if verified != artifact.snapshot:
            raise ValueError("case_runtime_followup_archive_changed")
        before = self._board
        after = self.build_fact_snapshot(verified, scope, scope.action_round)
        before_delivery = self._delivery_snapshot(scope, before.fact_reports, before=True)
        action_delivery = self._delivery_snapshot(scope, (verified,), before=False)
        progress = evaluate_progress(before=before.issues, after=after.issues,
            target_issue_ids=scope.issue_ids, new_valid_refs=_followup_refs(after, scope),
            case_id=self.case_id, round_no=scope.action_round, action_id=scope.action_id,
            before_delivery=before_delivery, action_delivery=action_delivery)
        result = _task_result(artifact, before, after, progress, scope)
        self._board = after
        return after, result

    def apply_failed_followup(self, request: FactNodeFollowup | PendingFollowupIntent
                              ) -> tuple[FactBoardV2Result | None, TaskResult]:
        """Publish a failed task status without changing its evidence conclusion."""
        intent = request if isinstance(request, PendingFollowupIntent) else None
        scope = request.scope if isinstance(request, FactNodeFollowup) else None
        issue_ids = intent.action.issue_ids if intent is not None else scope.issue_ids
        action_id = intent.action_id if intent is not None else scope.action_id
        action_round = intent.action_round if intent is not None else scope.action_round
        board = self._board
        current = board.issues if board is not None else self._no_fact_issues
        if current is None or not issue_ids:
            raise ValueError("case_runtime_failed_followup_scope_mismatch")
        if scope is not None and (scope.case_id != self.case_id or scope.sample_id != self.sample_id):
            raise ValueError("case_runtime_failed_followup_scope_mismatch")
        before = {item.issue_id: item for item in current}
        if any(item not in before or before[item].owner != "fact" for item in issue_ids):
            raise ValueError("case_runtime_failed_followup_target_invalid")
        issues = tuple(_failed_workflow(item, action_id, action_round)
                       if item.issue_id in issue_ids else item for item in current)
        result = TaskResult(action_id=action_id, kind="investigate", target="fact",
            status="failed", stop_reason="no_valid_result",
            issue_updates=[event for item in issues if item.issue_id in issue_ids
                           for event in item.history if event.action_id == action_id],
            progress=Progress(substantive=False, reasons=["task_failed"]), usage={},
            trace_path=str(request.call.trace.trace_path) if isinstance(request, FactNodeFollowup) else None)
        if board is not None:
            self._board = replace(board, issues=issues)
        else:
            self._no_fact_issues = issues
        return self._board, result

    def select_fact_conclusion(self) -> FactRoundChoice:
        """Select the report used for scoring from this invocation's verified round archive."""
        return select_fact_rounds(tuple(self._fact_artifacts[index]
                                        for index in sorted(self._fact_artifacts)))

    def mark_conflicting(self, issue_ids: tuple[str, ...], round_no: int) -> tuple[dict[str, Any], ...]:
        """Add a program history event for selected issues whose two fact rounds conflict."""
        if self._board is None or not issue_ids:
            return ()
        updated = []
        for issue in self._board.issues:
            if issue.issue_id in issue_ids and issue.epistemic != "conflicting":
                event = IssueEvent(round=round_no, field="epistemic", old=issue.epistemic,
                    new="conflicting", by="program")
                issue = issue.model_copy(update={"epistemic": "conflicting",
                    "history": issue.history + [event]})
            updated.append(issue)
        self._board = replace(self._board, issues=tuple(updated))
        return tuple(item.model_dump(mode="json") for item in updated)

    def _delivery_snapshot(self, scope: FollowupScopeV2,
                           snapshots: tuple[ValidatedFactSnapshotV2, ...], *,
                           before: bool) -> EvidenceDeliverySnapshot:
        """Convert verified archive deliveries into the progress evaluator's narrow identity view."""
        selected = tuple(item for item in snapshots
                         if item.scope.fact_round_no < scope.fact_round_no) if before else snapshots
        deliveries = {f"F{item.scope.fact_round_no}:{evidence.evidence_id}":
            DeliveredEvidence(evidence.body_sha256, evidence.read_end)
            for item in selected for evidence in item.delivered}
        return EvidenceDeliverySnapshot(self.case_id, scope.action_round, scope.action_id, deliveries)


def _followup_refs(board: FactBoardV2Result, scope: FollowupScopeV2) -> dict[str, frozenset[str]]:
    """Select only references added in this verified follow-up round for assigned issues."""
    prefix = f"F{scope.fact_round_no}:"
    return {issue.issue_id: frozenset(ref for ref in issue.evidence_refs if ref.startswith(prefix))
            for issue in board.issues if issue.issue_id in scope.issue_ids}


def _task_result(artifact: FactRuntimeArtifact, before: FactBoardV2Result,
                 after: FactBoardV2Result, progress: Progress,
                 scope: FollowupScopeV2) -> TaskResult:
    """Build the structured action result from verified board history and runtime summary."""
    old = {item.issue_id: item for item in before.issues}
    new = {item.issue_id: item for item in after.issues}
    changed = {field for issue_id in scope.issue_ids if issue_id in old and issue_id in new
        for field in ("epistemic", "workflow", "evidence_refs")
        if getattr(old[issue_id], field) != getattr(new[issue_id], field)}
    events = [event for issue_id in scope.issue_ids if issue_id in new
        for event in new[issue_id].history if event.action_id == scope.action_id]
    refs = [ref for issue_id in scope.issue_ids if issue_id in new
        for ref in new[issue_id].evidence_refs if ref.startswith(f"F{scope.fact_round_no}:")]
    return TaskResult(action_id=scope.action_id, kind="investigate", target="fact",
        status="completed", new_refs=list(dict.fromkeys(refs)), changed_fields=sorted(changed),
        issue_updates=events, validation_flags={}, progress=progress,
        usage=artifact.summary.usage, trace_path=artifact.archive_input.result.trace_path)


def _failed_workflow(issue: Any, action_id: str, round_no: int) -> Any:
    """Mark only assigned work as failed while preserving its evidence conclusion."""
    if issue.workflow == "failed":
        return issue
    event = IssueEvent(round=round_no, action_id=action_id, field="workflow",
        old=issue.workflow, new="failed", by="program", refs=list(issue.evidence_refs))
    return issue.model_copy(update={"workflow": "failed", "history": [*issue.history, event]},
                            deep=True)


def invoke_case_runtime(graph: CompiledStateGraph, state: dict[str, object],
                        context: CaseRuntimeContext) -> dict[str, object]:
    """Invoke one graph using the same isolated identity as its non-State sidecar."""
    sample = state.get("sample")
    sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
    if state.get("case_id") != context.case_id or sample_id != context.sample_id:
        raise ValueError("case_runtime_invocation_identity_mismatch")
    if context._invoked:
        raise ValueError("case_runtime_context_already_invoked")
    context._invoked = True
    return invoke_case_graph(graph, state, thread_id=context.thread_id)
