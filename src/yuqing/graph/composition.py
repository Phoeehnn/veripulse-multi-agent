"""Compose existing Agents and trusted case services around one shared wallet."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.yuqing.agents.arbiter_loop import ArbiterMaterial
from src.yuqing.agents.fact_check import compute_fact_risk
from src.yuqing.collab.arbiter_budget_v2 import (
    ArbiterBudgetScope, RunActionBinding, _request_states, build_arbiter_budget_view,
)
from src.yuqing.collab.validate import (
    ActionConfig, ActionSnapshot, BudgetSnapshot, CaseActionSnapshot,
    CaseReferenceSnapshot, TargetSnapshot, issue_snapshot,
)
from src.yuqing.eval.dual_budget import DualBudgetCoordinator
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.graph.fact_adapter import FactNodeAdapter, FactNodeCall, FactNodeFollowup, FactRuntimeArtifact
from src.yuqing.graph.runtime import (
    ArbiterNodeCall, CaseRuntimeAssembly, CaseRuntimeContext, CaseServiceBindings,
    PropagationNodeCall, assemble_bound_case_runtime, assemble_case_runtime,
    build_bound_case_handlers, bind_arbiter_v2_handler,
    bind_fact_node_handlers, bind_propagation_node_handler,
)
from src.yuqing.harness.inherited_evidence import (
    authorize_followup_action_v2, build_followup_scope_v2, prepare_inherited_evidence_v2,
)
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.harness.trace import RunContext
from src.yuqing.harness.initial_evidence import load_initial_evidence_manifest
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.schemas import (
    AgentSpec, CollabAction, ComplianceResult, FactVerdict, FollowupBrief, Sample, TaskInput,
)
from src.yuqing.settings import (
    get_collab_config, get_graph_config, get_m5_arbiter_harness_config,
    get_m5_case_budget_config, get_m6c_arbiter_harness_config, get_m6c_case_budget_config,
    get_m6c_collab_config, get_m6c_graph_config, get_propagation_harness_config,
)


@dataclass(frozen=True)
class CaseInputs:
    """Already loaded case dependencies; this object grants no Provider permission."""

    case_id: str
    sample: Sample
    coverage: str
    trace_root: Path
    wallet: PersistentExperimentBudget
    fact_worker_scope: EvidenceWorkerScope
    propagation_worker_scope: EvidenceWorkerScope
    index: BM25Index
    store_path: Path
    initial_manifest_path: Path
    propagation_spec: AgentSpec
    arbiter_spec: AgentSpec
    as_of: datetime | None = None
    group_id: str = "m5"
    variant: str = "fixed-v2"
    propagation_contract_version: Literal["v1.0", "v1.1"] = "v1.0"
    propagation_review_spec: AgentSpec | None = None
    m6b: bool = False
    m6c: bool = False


@dataclass
class CaseComposition:
    """One composition root; all model calls remain in the existing gateway."""

    inputs: CaseInputs
    context: CaseRuntimeContext = field(init=False)
    runs: dict[str, RunActionBinding] = field(default_factory=dict, init=False)
    last_decision: RunContext | None = field(default=None, init=False)
    arbiter_inputs: list[tuple[Any, ArbiterNodeCall, Any]] = field(default_factory=list, init=False)
    propagation_collab: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate identities and isolate runtime state before any Agent can run."""
        source = self.inputs
        if (not source.case_id or source.coverage not in {"full", "demo", "none", "mismatched"}
                or source.propagation_spec.agent != "propagation_analyst"
                or source.arbiter_spec.agent != "arbiter"
                or source.propagation_contract_version not in {"v1.0", "v1.1"}):
            raise ValueError("case_composition_input_invalid")
        if source.m6c and (not source.m6b or source.propagation_contract_version != "v1.1"):
            raise ValueError("m6c_requires_m6b_v11")
        if source.m6c:
            from src.yuqing.settings import get_m6c_run_budget_config
            get_m6c_run_budget_config()
            from experiments.archive.m6c_review_prompt import validate_m6c_review_prompt
            prepared = validate_m6c_review_prompt(offline=True)
            review_spec = source.propagation_review_spec
            if (review_spec is None or review_spec.agent != "propagation_analyst"
                    or review_spec.system_prompt != prepared.text):
                raise ValueError("m6c_review_spec_mismatch")
        elif source.propagation_review_spec is not None:
            raise ValueError("m6c_review_spec_without_m6c")
        if source.fact_worker_scope is source.propagation_worker_scope:
            raise ValueError("case_worker_scopes_must_be_distinct")
        if source.coverage != "none":
            load_initial_evidence_manifest(source.initial_manifest_path,
                source.sample.sample_id, source.sample.claim)
        self.context = CaseRuntimeContext(source.case_id, source.sample.sample_id)
        self.context.attach_usage_wallet(source.wallet)

    def assemble(self, *, external_calls_authorized: bool = False,
                 initial_state: dict[str, Any] | None = None,
                 m8_cached_first_round: object | None = None) -> CaseRuntimeAssembly:
        """Connect real services; callers still need explicit run authorization."""
        if initial_state is not None and "_m8_resume_group" in initial_state:
            raise ValueError("m8_cached_resume_not_authorized")
        if initial_state is not None and "_m8_capture_shared_decision" in initial_state:
            if (m8_cached_first_round is not None
                    or self.inputs.group_id != "shared-first-round"
                    or self.inputs.variant != "shared-first-round"
                    or initial_state.get("_m8_capture_shared_decision") is not True):
                raise ValueError("m8_shared_prefix_capture_not_authorized")
        adapter = FactNodeAdapter(get_graph_config().fact_node_version)
        fact, followup = bind_fact_node_handlers(adapter, self.context, self.fact_call,
            self.followup_call, lambda: self.inputs.wallet)
        bindings = CaseServiceBindings(claim=self.inputs.sample.claim, comments=(),
            coverage=self.inputs.coverage, fact_round1=fact,
            propagation_round1=bind_propagation_node_handler(self.propagation_call, self.context),
            arbitrate=bind_arbiter_v2_handler(self.arbiter_call, self.context), fact_followup=followup,
            action_snapshot=self.action_snapshot, references=self.references,
            score_inputs=self.score_inputs)
        if self.inputs.m6b:
            from src.yuqing.graph.propagation_collab import PropagationCollaboration
            self.propagation_collab = PropagationCollaboration(self, m6c=self.inputs.m6c)
            bindings = self.propagation_collab.bind(bindings)
        state = initial_state or {"case_id": self.inputs.case_id,
            "sample": self.inputs.sample.model_dump(mode="json"),
            "ticket": {"claim": self.inputs.sample.claim}}
        if self.inputs.coverage == "demo":
            state = {**state, "is_demo": True}
        state = {**state, "m6c_enabled": self.inputs.m6c}
        if self.inputs.m6c:
            state["collab_config"] = get_m6c_collab_config().model_dump(mode="json")
        if m8_cached_first_round is not None:
            return self._assemble_cached_first_round(state, bindings, m8_cached_first_round,
                external_calls_authorized=external_calls_authorized)
        return assemble_bound_case_runtime(state, bindings,
            config=get_m6c_graph_config() if self.inputs.m6c else get_graph_config(), eval_mode=True, context=self.context,
            external_calls_authorized=external_calls_authorized)

    def _assemble_cached_first_round(self, state: dict[str, Any], bindings: CaseServiceBindings,
                                     branch: object,
                                     *, external_calls_authorized: bool) -> CaseRuntimeAssembly:
        """Start at merge using revalidated M8 outputs; default M5 assembly is untouched."""
        from experiments.archive.m8_runner import M8GroupBranch
        from experiments.archive.m8_cache import _is_early_failure_state

        if (not isinstance(branch, M8GroupBranch) or branch.context is not self.context
                or branch.wallet is not self.inputs.wallet or branch.group_id != self.inputs.group_id
                or branch.snapshot.identity.sample_id != self.inputs.sample.sample_id):
            raise ValueError("m8_cached_snapshot_not_verified")
        if (self.inputs.group_id not in {"B4-nf", "B4-fixed"}
                or state.get("case_id") != self.inputs.case_id
                or (state.get("sample") or {}).get("sample_id") != self.inputs.sample.sample_id
                or state.get("thread_id") != self.context.thread_id):
            raise ValueError("m8_cached_group_identity_mismatch")
        expected_issues = state.get("issues")
        expected_compliance = state.get("compliance")
        handlers = build_bound_case_handlers(self.context, bindings)
        if _is_early_failure_state(state):
            handlers = replace(handlers, merge_round=lambda _state: {})
        else:
            rebuilt = handlers.merge_round(state)
            if (rebuilt.get("issues") != expected_issues
                    or rebuilt.get("compliance") != expected_compliance):
                raise ValueError("m8_cached_merge_reconstruction_mismatch")
        state = deepcopy(state)
        state.pop("_m8_capture_shared_decision", None)
        state["_m8_resume_group"] = self.inputs.group_id
        handlers = replace(handlers, prepare=self._m8_prepare(handlers.prepare, branch),
            fact_round1=lambda _state: {}, propagation_round1=lambda _state: {},
            merge_round=lambda _state: {},
            validate_action=handlers.validate_action)
        self._restore_m8_bindings(branch)
        return assemble_case_runtime(state, handlers, bindings.score_inputs,
            config=get_graph_config(), eval_mode=True, context=self.context,
            external_calls_authorized=external_calls_authorized,
            request_factory=bindings.request_factory)

    def _m8_prepare(self, original: Any, branch: object) -> Any:
        """Run existing preparation and record nf policy as an experiment event, not a rejection."""
        def prepare(state: dict[str, Any]) -> dict[str, Any]:
            """Record the policy stop only when its branch actually starts."""
            patch = original(state)
            patch["compliance"] = deepcopy(branch.snapshot.state["compliance"])
            if self.inputs.group_id == "B4-nf":
                from src.yuqing.harness.trace import RunContext, record_event, use_run_context
                from uuid import uuid4

                context = RunContext(run_id=uuid4().hex, sample_id=self.inputs.sample.sample_id,
                    agent="controller", trace_root=branch.trace_root)
                with use_run_context(context):
                    record_event("m8_policy_stop", {"group_id": "B4-nf",
                        "decision_run_id": None if branch.snapshot.arbiter_result is None
                            else branch.snapshot.arbiter_result.run_id,
                        "cache_sha256": branch.snapshot.cache_sha256,
                        "reason": "immediate_stop_control"})
            return patch
        return prepare

    def _restore_m8_bindings(self, branch: object) -> None:
        """Restore trusted run bindings so the existing budget gate sees shared spend."""
        from src.yuqing.collab.arbiter_budget_v2 import RunActionBinding
        from src.yuqing.harness.trace import RunContext
        from datetime import date

        self.runs = {item.run_id: RunActionBinding(run_id=item.run_id, role=item.role,
            decision_round=item.decision_round, request_ids=item.request_ids,
            target=item.target, shared_prefix=item.shared_prefix)
            for item in branch.snapshot.run_bindings}
        decision = branch.snapshot.arbiter_result
        if decision is None:
            self.last_decision = None
            return
        path = Path(decision.trace_path).resolve(strict=True)
        self.last_decision = RunContext(run_id=decision.run_id,
            sample_id=self.inputs.sample.sample_id, agent="arbiter",
            trace_root=path.parents[1], trace_day=date.fromisoformat(path.parent.name))


    def _run(self, role: str, target: str | None, state: dict[str, Any]) -> RunContext:
        """Allocate a program run identity and retain its wallet-binding metadata."""
        agent = "arbiter" if role == "decision" else (
            "fact_investigator" if target == "fact" else "propagation_analyst")
        ctx = RunContext(run_id=uuid4().hex, sample_id=self.inputs.sample.sample_id,
            agent=agent, trace_root=self.inputs.trace_root)
        round_no = len(state.get("decisions", [])) + 1 if role == "decision" else 1
        action_id = state.get("_program_action_id")
        pending = self.context._pending_intent
        self.runs[ctx.run_id] = RunActionBinding(ctx.run_id, role, round_no, (),
            action_id=action_id, target=target,
            action_kind=pending.action.kind if role == "action" and pending is not None else None,
            action_status="claimed" if role == "action" else None)
        return ctx

    def identity(self, ctx: RunContext) -> dict[str, str]:
        """Use the same group and variant for every request in this case."""
        return {"run_id": ctx.run_id, "sample_id": ctx.sample_id,
            "group_id": self.inputs.group_id, "variant": self.inputs.variant}

    def coordinator(self, budget: Any) -> DualBudgetCoordinator:
        """Bind each node tracker to the same existing persistent wallet."""
        return DualBudgetCoordinator(budget, self.inputs.wallet)

    def fact_call(self, state: dict[str, Any]) -> FactNodeCall:
        """Prepare the frozen fact node with a fresh first-round or action run."""
        role = "action" if state.get("_program_action_id") else "initial"
        ctx = self._run(role, "fact", state)
        return FactNodeCall(TaskInput(sample_id=ctx.sample_id, claim=self.inputs.sample.claim),
            self.inputs.case_id, ctx, self.inputs.index, self.inputs.fact_worker_scope,
            self.inputs.store_path, self.inputs.initial_manifest_path,
            self.coordinator, self.identity(ctx))

    def propagation_call(self, state: dict[str, Any]) -> PropagationNodeCall:
        """Prepare the existing propagation loop without replacing its verifier."""
        ctx = self._run("initial", "propagation", state)
        return self.propagation_call_view(ctx)

    def propagation_call_view(self, ctx: RunContext) -> PropagationNodeCall:
        """Bind an already allocated propagation run without allocating a second identity."""
        return PropagationNodeCall(self.inputs.propagation_spec, self.inputs.sample, ctx,
            get_propagation_harness_config(), self.inputs.propagation_worker_scope, self.inputs.as_of,
            coordinator_factory=self.coordinator, experiment_identity=self.identity(ctx),
            contract_version=self.inputs.propagation_contract_version)

    def _scope(self, state: dict[str, Any]) -> ArbiterBudgetScope:
        """Map every wallet request back to its program-created run and action."""
        if self.last_decision is None:
            raise ValueError("case_decision_run_missing")
        recovered, _ = _request_states(self.inputs.wallet)
        bindings = []
        for run_id, binding in self.runs.items():
            ids = tuple(key for key, value in recovered.requests.items()
                        if value.identity.get("run_id") == run_id)
            if binding.role == "action":
                status = next((row["status"] for row in state.get("actions", [])
                    if row["action_id"] == binding.action_id), "claimed")
                if not ids and status == "claimed":
                    continue
                if not ids and status != "claimed":
                    status = "cancelled"
                binding = replace(binding, action_status=status)
            bindings.append(replace(binding, request_ids=ids))
        current_cost = sum((item.actual or Decimal(0) for item in recovered.requests.values()
            if item.identity.get("run_id") == self.last_decision.run_id), Decimal(0))
        prop = self.context._propagation_result
        return ArbiterBudgetScope(str(recovered.manifest["experiment_id"]), self.inputs.case_id,
            self.inputs.sample.sample_id, self.inputs.group_id, self.inputs.variant,
            self.last_decision.run_id, self.runs[self.last_decision.run_id].decision_round,
            tuple(bindings), current_cost, self.inputs.coverage,
            bool(self.propagation_collab and self.propagation_collab.source or prop and (prop.comments or prop.stats and prop.stats.has_data)),
            prop is not None, self.propagation_collab.unreturned_count() if self.propagation_collab else 0,
            previous_progress=(state.get("task_results") or [{}])[-1].get("progress", {}).get("substantive"),
            completed_actions=sum(item.role == "action" for item in bindings),
            propagation_action_cap=self.propagation_collab.cap if self.propagation_collab else None)

    def references(self, state: dict[str, Any]) -> CaseReferenceSnapshot:
        """Derive fact and comment references only from accepted runtime artifacts."""
        scope = self._scope(state)
        prop = self.context._propagation_result
        source = self.propagation_collab.source if self.propagation_collab else None
        leads = {key: item.source for key, item in prop.leads_by_id.items()} if prop else {}
        comments = (frozenset(item.comment_id for item in prop.comments) if prop else
                    frozenset(item.comment_id for item in source.comments) if source else frozenset())
        facts = frozenset(f"F{item.scope.fact_round_no}:{e.evidence_id}"
            for item in self.context.fact_snapshots() for e in item.delivered)
        return CaseReferenceSnapshot(scope.case_id, scope.decision_round, leads,
            comments, facts, frozenset())

    def action_snapshot(self, state: dict[str, Any]) -> CaseActionSnapshot:
        """Read the verified shared wallet for the existing eight-rule gate."""
        config = get_m6c_collab_config() if self.inputs.m6c else get_collab_config()
        budget = get_m6c_case_budget_config() if self.inputs.m6c else get_m5_case_budget_config()
        scope = self._scope(state)
        view = build_arbiter_budget_view(self.inputs.wallet, scope, budget, collab_config=config)
        refs = self.references(state)
        executed = tuple(ActionSnapshot(row["kind"], row["target"], tuple(row["issue_ids"]),
            row.get("question", "")) for row in state.get("actions", [])
            if isinstance(row, dict) and row.get("status") in {"completed", "failed", "skipped"})
        result = CaseActionSnapshot(scope.case_id, scope.decision_round,
            tuple(issue_snapshot(item) for item in self.context.current_issues()), executed,
            state.get("task_results", [{}])[-1].get("progress", {}).get("substantive")
                if state.get("task_results") else None,
            ActionConfig(config.max_actions, frozenset(config.targets), config.review_enabled),
            TargetSnapshot(scope.fact_coverage, scope.propagation_has_data,
                refs.returned_comment_ids, refs.returned_comment_ids, scope.propagation_report_available),
            BudgetSnapshot(view.settled, view.pending,
                Decimal(str(budget.action_caps["fact_investigate"])), view.final_reserve,
                Decimal(str(budget.max_cost_cny))))
        return self.propagation_collab.snapshot(result, state) if self.propagation_collab else result

    def arbiter_call(self, state: dict[str, Any], material: Any) -> ArbiterNodeCall:
        """Prepare one real arbiter decision from current archives, wallet and board."""
        ctx = self._run("decision", None, state)
        self.last_decision = ctx
        prop = self.context._propagation_result
        snapshots = self.context.fact_snapshots()
        legacy = FactVerdict.model_validate_json(snapshots[-1].legacy_projection_json) if snapshots else None
        supplemental = ArbiterMaterial(case_id=self.inputs.case_id, claim=self.inputs.sample.claim,
            issues=self.context.current_issues(), fact_report=legacy, fact_citation_ok=bool(snapshots),
            propagation_report=prop.report if prop else None, stats=prop.stats if prop else None,
            compliance=ComplianceResult.model_validate(state["compliance"]),
            task_results=tuple(state.get("task_results", ())))
        call = ArbiterNodeCall(self.inputs.arbiter_spec, supplemental, ctx,
            get_m6c_arbiter_harness_config() if self.inputs.m6c else get_m5_arbiter_harness_config(),
            self.action_snapshot(state), self.references(state), self._scope(state), self.coordinator,
            self.identity(ctx), dict(prop.leads_by_id) if self.propagation_collab and prop else
                ({} if self.propagation_collab else None), "m6c" if self.inputs.m6c else "m5")
        active_budget = get_m6c_case_budget_config() if self.inputs.m6c else get_m5_case_budget_config()
        active_collab = get_m6c_collab_config() if self.inputs.m6c else get_collab_config()
        view = build_arbiter_budget_view(self.inputs.wallet, call.budget_scope, active_budget,
                                        collab_config=active_collab)
        self.arbiter_inputs.append((material, call, view))
        return call

    def followup_call(self, state: dict[str, Any], call: FactNodeCall) -> FactNodeFollowup:
        """Authorize the exact task and inherit only re-read source deliveries."""
        action = CollabAction.model_validate(state["decisions"][-1]["action"])
        if not self.context.fact_runs():
            raise ValueError("fact_followup_source_unavailable")
        source = self.context.fact_runs()[0]
        brief = FollowupBrief(action_id=state["_program_action_id"],
            round=state["_program_action_round"], target=action.target,
            issue_ids=action.issue_ids, issue_texts={item.issue_id: item.text
                for item in self.context.current_issues() if item.issue_id in action.issue_ids},
            budget_cny=Decimal(str(get_m5_case_budget_config().action_caps["fact_investigate"])),
            question=action.question, done_when=action.done_when,
            leads={key: self.context._propagation_result.leads_by_id[key]
                for key in action.lead_ids} if action.lead_ids else {},
            tried_queries=_tried_queries(self.context._fact_artifacts[1]),
            read_evidence_ids=[item.evidence_id for item in self.context.fact_snapshots()[0].delivered])
        proof = authorize_followup_action_v2(action=action, brief=brief,
            case_id=self.inputs.case_id, action_round=brief.round,
            decision_round=state["_program_decision_round"], snapshot=self.action_snapshot(state),
            references=self.references(state), issues=self.context.current_issues())
        scope = build_followup_scope_v2(task=call.task, case_id=call.case_id,
            run_id=call.trace.run_id, agent=call.trace.agent,
            fact_round_no=self.context.next_fact_round(), action_round=brief.round,
            decision_round=state["_program_decision_round"], brief=brief, proof=proof)
        source_ctx = RunContext(run_id=source.scope.run_id, sample_id=source.scope.sample_id,
            agent=source.scope.agent, trace_root=source.reader.trace_root)
        delivery = source.reader.read_deliveries(source_ctx, source.task, source.contract, source.scope)
        inherited = prepare_inherited_evidence_v2(source=delivery, target=scope, brief=brief, task=call.task)
        return FactNodeFollowup(call, scope, inherited.plan, proof)

    def score_inputs(self, state: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk from the selected fact report and real propagation/compliance."""
        choice = self.context.select_fact_conclusion()
        artifact = next((item for item in self.context.fact_snapshots()
            if item.scope.fact_round_no == choice.round_no), None)
        verdict = FactVerdict.model_validate_json(artifact.legacy_projection_json) if artifact else None
        prop = self.context._propagation_result
        decisions = state.get("decisions", [])
        decision = decisions[-1] if decisions else {}
        return {"fact_status": verdict.label if verdict else None,
            "fact_risk": compute_fact_risk(verdict) if verdict else None,
            "fact_coverage": self.inputs.coverage,
            "fact_failed": verdict is None and self.inputs.coverage != "none",
            "fact_citation_ok": verdict is not None,
            "emotion_level": prop.report.emotion_level if prop else None,
            "spread_level": prop.report.spread_level if prop else None,
            "compliance": ComplianceResult.model_validate(state["compliance"]),
            "unresolved_key_conflict": any(item.kind == "冲突" and item.key
                and item.epistemic == "conflicting" for item in self.context.current_issues()),
            "review_disagree": any(item.get("decision") == "disagree"
                for review in state.get("reviews", []) for item in review.get("items", [])),
            "arbiter_requests_review": (decision.get("needs_human_review", False)
                or state.get("m6c_force_review", False)),
            "adjustment": decision.get("adjustment", 0),
            "adjustment_reason": decision.get("adjustment_reason")}


def _tried_queries(artifact: FactRuntimeArtifact) -> list[str]:
    """Recover searched queries from the source run's verified Trace, not the evidence manifest."""
    archive = artifact.archive_input
    return archive.reader.read_executed_search_queries(archive.result, archive.task,
        archive.contract, archive.scope, artifact.snapshot.trace_sha256)
