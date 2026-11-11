"""Prepare one program-authorized propagation follow-up and inherit delivered previews."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from html import escape
import json
from typing import TYPE_CHECKING, Callable, Literal, Mapping

from pydantic import Field

from src.yuqing.collab.arbiter_budget_v2 import RunActionBinding
from src.yuqing.collab.propagation_delivery import PropagationDelivery, canonical, input_fingerprint
from src.yuqing.collab.validate import ActionValidation, CaseActionSnapshot, CaseReferenceSnapshot, issue_snapshot, validate_action
from src.yuqing.harness.trace import RunContext, record_event, use_run_context, write_archive
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.schemas import AgentSpec, CollabAction, FollowupBrief, Issue, Sample
from src.yuqing.settings import PolicyConfig, PropagationHarnessConfig

if TYPE_CHECKING:
    from src.yuqing.harness.budget import BudgetTracker
    from src.yuqing.agents.propagation import PropagationToolExecutor
    from src.yuqing.agents.propagation import PropagationResult
    from src.yuqing.eval.dual_budget import DualBudgetCoordinator


class PropagationFollowupHarnessConfig(PropagationHarnessConfig):
    """Validate the specification's step bound without adding any config defaults."""
    max_steps: int = Field(gt=0, le=3, strict=True)


@dataclass(frozen=True)
class PropagationFollowupInput:
    """Internal typed task; the only public factory consumes a program claim and gate state."""
    source: PropagationDelivery
    target_run_id: str
    round_no: int
    decision_round: int
    brief_json: str
    config_json: str


def prepare_propagation_followup(*, source: PropagationDelivery, sample: Sample,
        target: RunContext, action: CollabAction, claimed: RunActionBinding,
        snapshot: CaseActionSnapshot, references: CaseReferenceSnapshot,
        issues: tuple[Issue, ...], node_config: PropagationHarnessConfig,
        as_of: datetime | None, checked: ActionValidation | None = None) -> PropagationFollowupInput:
    """Validate the existing gate once and bind the claimed action to its trusted source."""
    config = PropagationFollowupHarnessConfig.model_validate(node_config.model_dump())
    if type(source) is not PropagationDelivery or source.input_sha256 != input_fingerprint(sample, as_of):
        raise ValueError("propagation_followup_snapshot_mismatch")
    returned = frozenset(item.comment_id for item in source.comments)
    if (target.agent != "propagation_analyst" or target.sample_id != source.sample_id
            or source.sample_id != sample.sample_id or target.run_id == source.run_id
            or source.case_id != snapshot.case_id or snapshot.config.max_actions != 1
            or snapshot.config.review_enabled or snapshot.target.visible_comment_ids != source.visible_ids
            or snapshot.target.returned_comment_ids != returned or references.returned_comment_ids != returned):
        raise ValueError("propagation_followup_scope_mismatch")
    if (claimed.role != "action" or claimed.run_id != target.run_id or not claimed.action_id
            or claimed.action_status != "claimed" or claimed.target != "propagation"
            or claimed.action_kind != "investigate" or action.target != "propagation"
            or action.kind != "investigate" or claimed.decision_round != snapshot.decision_round
            or claimed.request_ids or snapshot.budget.action_cap != Decimal(str(config.max_cost_cny))):
        raise ValueError("propagation_followup_claim_mismatch")
    current = {item.issue_id: item for item in issues}
    if len(current) != len(issues) or {issue_snapshot(item) for item in issues} != set(snapshot.issues):
        raise ValueError("propagation_followup_board_mismatch")
    verdict = checked if checked is not None else validate_action(action=action, case_id=source.case_id,
        decision_round=snapshot.decision_round, snapshot=snapshot, references=references)
    if not verdict.allowed:
        raise ValueError("propagation_followup_not_authorized:" + str(verdict.reason))
    queries = json.loads(source.queries_json)
    brief = FollowupBrief(action_id=claimed.action_id, round=len(snapshot.executed_actions) + 1,
        budget_cny=Decimal(str(config.max_cost_cny)), target="propagation", issue_ids=action.issue_ids,
        issue_texts={key: current[key].text for key in action.issue_ids}, question=action.question,
        done_when=action.done_when, returned_comment_ids=sorted(returned, key=lambda c: int(c[1:])),
        tried_queries=[q["arguments"]["query"] for q in queries if q["tool"] == "search_comments"])
    return PropagationFollowupInput(source, target.run_id, source.round_no + 1,
        snapshot.decision_round, canonical(brief.model_dump(mode="json")), canonical(config.model_dump()))


def _messages(base: list[dict], plan: PropagationFollowupInput, comments: list,
               stats: object | None) -> list[dict]:
    """Render only inert task data and the complete retained previews, never hidden tails."""
    from src.yuqing.agents.propagation_v11 import stats_receipt
    kept = {item.comment_id for item in comments}
    task = json.loads(plan.brief_json)
    task.update(source_run_id=plan.source.run_id, source_round=plan.source.round_no,
        propagation_round=plan.round_no, decision_round=plan.decision_round,
        query_receipts=json.loads(plan.source.queries_json),
        inherited_comment_ids=sorted(kept, key=lambda c: int(c[1:])),
        omitted_comment_ids=[c.comment_id for c in plan.source.comments if c.comment_id not in kept])
    material = {"comments": [item.model_dump(mode="json") for item in comments],
        "stats_receipt": stats_receipt(stats).model_dump(mode="json") if stats is not None else None}
    return [*base, {"role": "user", "content":
        '<followup_task encoding="json">' + escape(canonical(task), quote=False) + '</followup_task>'},
        {"role": "user", "content": '<inherited_propagation encoding="json">' +
            escape(canonical(material), quote=False) + '</inherited_propagation>'}]


def initialize_propagation_followup(plan: PropagationFollowupInput, *, sample: Sample,
        ctx: RunContext, as_of: datetime | None, config: PropagationHarnessConfig,
        executor: PropagationToolExecutor, coordinator: DualBudgetCoordinator | None,
        identity: Mapping[str, str] | None, messages: list[dict]) -> list[dict]:
    """Bind the execution wallet then register exactly the material fitting the real request."""
    from src.yuqing.agents.propagation import _PropagationFailure
    if type(plan) is not PropagationFollowupInput:
        raise _PropagationFailure("propagation_followup_input_invalid")
    source_identity = json.loads(plan.source.identity_json)
    expected = {**source_identity, "run_id": ctx.run_id}
    if (ctx.run_id != plan.target_run_id or ctx.sample_id != plan.source.sample_id
            or input_fingerprint(sample, as_of) != plan.source.input_sha256
            or canonical(config.model_dump()) != plan.config_json
            or executor.counter.contract_version != "v1.1" or dict(identity or {}) != expected
            or coordinator is None or coordinator.experiment is not plan.source.wallet
            or coordinator.experiment.manifest_snapshot()["experiment_id"] != plan.source.experiment_id):
        raise _PropagationFailure("propagation_followup_execution_mismatch")
    fitted = _fit_inherited_messages(plan, executor, messages)
    archive = write_archive("tool_result", {"program_material": "propagation_followup",
        "source_trace_sha256": plan.source.trace_sha256, "messages": fitted[len(messages):],
        "source_returned_ids": [c.comment_id for c in plan.source.comments]})
    record_event("propagation_followup_input", {"case_id": plan.source.case_id,
        "source_run_id": plan.source.run_id, "propagation_round": plan.round_no,
        "decision_round": plan.decision_round, "brief": json.loads(plan.brief_json),
        "inherited_comment_ids": list(executor.counter.comments), "archive": archive})
    for query in json.loads(plan.source.queries_json):
        if (query["tool"] == "search_comments" and query["status"] == "success"
                and query["delivered"] and query["reason"] == "no_query_match"):
            args = query["arguments"]
            executor.counter.empty_queries.add((args["query"], args["k"]))
    return fitted


def _fit_inherited_messages(plan: PropagationFollowupInput, executor: PropagationToolExecutor,
                             base: list[dict]) -> list[dict]:
    """Keep stats first and a deterministic complete-comment prefix within both request bounds."""
    from src.yuqing.agents.propagation import (PropagationToolPayload, _PropagationFailure, _build_request_messages,
        _startup_capacity_error)
    from src.yuqing.agents.propagation_v11 import stats_receipt, submission_model
    comments, stats = list(plan.source.comments), plan.source.stats
    state, registry, budget = executor.counter, executor.registry, executor.budget
    while True:
        state.comments, state.stats = {}, None
        spec = registry.get(budget.config.submission_tool)
        registry._tools[spec.name] = spec.model_copy(update={"args_model": submission_model(())})
        executor._record_delivered(PropagationToolPayload(status="success", reason="ok",
            comments=comments, stats=stats, stats_receipt=stats_receipt(stats) if stats else None))
        messages = _messages(base, plan, comments, stats)
        ordinary = _build_request_messages(messages, budget=budget, registry=registry, state=state, final=False)
        final = _build_request_messages(messages, budget=budget, registry=registry, state=state, final=True)
        error = _startup_capacity_error(ordinary, final, budget.config, registry, budget)
        if error is None:
            return messages
        if comments:
            comments.pop()
        elif stats is not None:
            stats = None
        else:
            raise _PropagationFailure("propagation_followup_required_input_exceeds_capacity:" + error)


def run_propagation_m6b(spec: AgentSpec, sample: Sample, ctx: RunContext, *,
        node_config: PropagationHarnessConfig, worker_scope: EvidenceWorkerScope,
        as_of: datetime | None = None, policy_config: PolicyConfig | None = None,
        coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None = None,
        experiment_identity: Mapping[str, str] | None = None,
        contract_version: Literal["v1.1"] = "v1.1", followup: PropagationFollowupInput | None = None,
        case_id: str | None = None
        ) -> PropagationResult | None:
    """Use the frozen initialization and loop while explicitly enabling M6b provenance/inputs."""
    from src.yuqing.agents import propagation as core
    from src.yuqing.collab.propagation_delivery import record_propagation_input
    state, result = core._ToolCounter(), None
    try:
        with use_run_context(ctx):
            if type(spec) is not AgentSpec or contract_version != "v1.1":
                raise core._PropagationFailure("invalid_m6b_agent_contract")
            sample, config, registry, executor, coordinator, messages = core._prepare_runtime(
                spec, sample, ctx, node_config, worker_scope, as_of, policy_config,
                coordinator_factory, experiment_identity, state, contract_version)
            if followup is not None and case_id != followup.source.case_id:
                raise core._PropagationFailure("propagation_followup_case_mismatch")
            record_propagation_input(sample, as_of, experiment_identity, case_id)
            if followup is not None:
                messages = initialize_propagation_followup(followup, sample=sample, ctx=ctx,
                    as_of=as_of, config=config, executor=executor, coordinator=coordinator,
                    identity=experiment_identity, messages=messages)
            result, state.reason, state.phase = core._iterate_propagation(spec, sample, ctx, config,
                registry, executor, state.budget, messages, coordinator, experiment_identity)
    except Exception as exc:
        result = None
        core._handle_run_failure(state, exc)
    try:
        core._record_run_outcome(ctx, result, state)
    except Exception as exc:
        core._handle_run_failure(state, core._PropagationAuditFailure("run_outcome_audit_failed", type(exc).__name__))
        return None
    return result
