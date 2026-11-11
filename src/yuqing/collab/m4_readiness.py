"""组装M4固定五样本的离线仲裁请求并检查请求容量。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from typing import Callable
from pydantic import BaseModel

from src.yuqing import llm
from src.yuqing.agents.arbiter_loop import ArbiterMaterial, _v2_messages
from src.yuqing.agents.compliance import check_compliance
from src.yuqing.collab.arbiter_material_v2 import ArbiterMaterialV2, build_arbiter_material_v2
from src.yuqing.collab.arbiter_budget_v2 import ArbiterBudgetScope, _make_cards
from src.yuqing.collab.board import (
    PropagationReferenceSnapshot, build_board, build_fact_board_v2,
)
from src.yuqing.collab.m4_fact_history_import import M4ImportedFactSource, load_m4_fact_history_imports
from src.yuqing.collab.m4_history import M4HistoricalPropagationInput, load_m4_history_inputs
from src.yuqing.collab.validate import (
    ActionConfig, BudgetSnapshot, CaseActionSnapshot, CaseReferenceSnapshot,
    TargetSnapshot, issue_snapshot,
)
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.tools import ToolRegistry, ToolSpec
from src.yuqing.agents.arbiter_loop import (
    ArbiterFeedback, _append_capability_cards, _v2_available_refs, _v2_decision_errors,
    _v2_feedback_issue,
)
from src.yuqing.schemas import AgentSpec, ArbiterDecisionV2, CapabilityCardV2, Issue, ToolValidationIssue
from src.yuqing.settings import (
    PROJECT_ROOT, get_collab_config, get_cost_config, get_evidence_sources_config,
    get_m4_arbiter_harness_config, get_m4_case_budget_config, get_m4_run_budget_config,
)
from src.yuqing.schemas import build_arbiter_decision_v2_schema

M4_SAMPLE_IDS = (
    "chef:dev:0", "chef:dev:1", "chef:dev:10",
    "ced:non_rumor:2602_AfRpJr42Z_2451549504",
    "ced:rumor:2421_z6uV4bCT2_2345130274",
)


@dataclass(frozen=True)
class M4PreparedCase:
    """一个已绑定固定来源的仲裁输入及其程序状态快照。"""

    material: ArbiterMaterialV2
    supplemental: ArbiterMaterial
    snapshot: CaseActionSnapshot
    references: CaseReferenceSnapshot


@dataclass(frozen=True)
class M4CapacityRow:
    """单次反馈代理值与累计多轮消息的容量估算。"""

    sample_id: str
    first_input_tokens: int
    schema_retry_input_tokens: int
    refusal_retry_input_tokens: int
    canonical_max_schema_retry_input_tokens: int
    refusal_then_schema_input_tokens: int
    first_allowed: bool
    schema_retry_allowed: bool
    refusal_retry_allowed: bool
    canonical_max_schema_retry_allowed: bool
    refusal_then_schema_allowed: bool
    all_retry_paths_allowed: bool
    retry_path_allowed: tuple[tuple[str, bool], ...]
    retry_path_request_tokens: tuple[tuple[str, int, int], ...]
    retry_path_input_tokens: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class M4ReadinessReport:
    """五样本来源、完整请求和容量门槛的离线核对结果。"""

    cases: tuple[M4PreparedCase, ...]
    capacity: tuple[M4CapacityRow, ...]


def load_m4_prepared_cases(root: Path = PROJECT_ROOT) -> tuple[M4PreparedCase, ...]:
    """从固定来源清单重验五条材料并组装同案仲裁输入。"""
    propagation = load_m4_history_inputs(root)
    facts = {item.archive_input.task.sample_id: item for item in load_m4_fact_history_imports(root)}
    if tuple(item.sample_id for item in propagation) != M4_SAMPLE_IDS:
        raise ValueError("m4_prepared_sample_order_invalid")
    if set(facts) != {"chef:dev:0", "chef:dev:1", "chef:dev:10"}:
        raise ValueError("m4_prepared_fact_sources_invalid")
    return tuple(_prepare_case(item, facts.get(item.sample_id)) for item in propagation)


def preflight_m4_five_samples(root: Path, prompt_path: Path) -> M4ReadinessReport:
    """在任何运行钱包或Provider入口之前核验固定材料与完整请求容量。"""
    cases = load_m4_prepared_cases(root)
    capacity = measure_m4_capacity(cases, prompt_path, root)
    if len(cases) != 5 or len(capacity) != len(cases):
        raise ValueError("m4_preflight_sample_count_invalid")
    for row in capacity:
        if not row.canonical_max_schema_retry_allowed:
            raise ValueError("m4_preflight_capacity_insufficient:" + row.sample_id
                + f":request=2:estimated={row.canonical_max_schema_retry_input_tokens}"
                + f":limit={get_m4_arbiter_harness_config().token_budget_per_call}")
        if not row.refusal_then_schema_allowed:
            raise ValueError("m4_preflight_capacity_insufficient:" + row.sample_id
                + f":path=business->schema:request=3:estimated={row.refusal_then_schema_input_tokens}"
                + f":limit={get_m4_arbiter_harness_config().token_budget_per_call}")
        if not row.all_retry_paths_allowed:
            path, _ = next(item for item in row.retry_path_allowed if not item[1])
            second, third = next((second, third) for name, second, third
                                 in row.retry_path_request_tokens if name == path)
            raise ValueError("m4_preflight_capacity_insufficient:" + row.sample_id
                + f":path={path}:request=2_or_3:estimated={second}/{third}"
                + f":limit={get_m4_arbiter_harness_config().token_budget_per_call}")
        if not row.first_allowed or not row.schema_retry_allowed or not row.refusal_retry_allowed:
            raise ValueError("m4_preflight_capacity_insufficient:" + row.sample_id)
    return M4ReadinessReport(cases, capacity)


def _prepare_case(item: M4HistoricalPropagationInput,
                  fact_source: M4ImportedFactSource | None) -> M4PreparedCase:
    """按固定案例身份组合已重验的事实与传播来源。"""
    case_id, round_no = f"m4:{item.sample_id}", 1
    prop_refs = _propagation_refs(item, case_id, round_no)
    prop_board = build_board(case_id=case_id, round_no=round_no, fact_verdict=None,
        fact_status="not_dispatched", fact_refs=None, propagation_report=item.report,
        propagation_refs=prop_refs)
    if prop_board.diagnostics:
        raise ValueError("m4_prepared_propagation_board_invalid")
    prop_issues = tuple(issue.model_copy(update={"issue_id": f"I{index}"})
        for index, issue in enumerate((row for row in prop_board.issues
            if row.origin_refs != [f"fact:F{round_no}"]), start=1))
    if fact_source is None:
        material = _no_fact_material(item, case_id, round_no, prop_board.issues)
    else:
        material = _fact_material(item, case_id, round_no, fact_source, prop_issues)
    comments = tuple((comment.comment_id, comment.text) for comment in item.comments)
    supplemental = ArbiterMaterial(case_id=case_id, claim=item.claim, issues=material.issues,
        fact_report=None, fact_citation_ok=False, propagation_report=item.report,
        stats=item.stats, compliance=check_compliance(item.claim, comments),
        fact_absence_reason="no_coverage" if fact_source is None else None)
    snapshot, refs = _case_snapshots(item, material, round_no, prop_refs, fact_source)
    return M4PreparedCase(material, supplemental, snapshot, refs)


def _propagation_refs(item: M4HistoricalPropagationInput, case_id: str,
                      round_no: int) -> PropagationReferenceSnapshot | None:
    """Bind a propagation report to its verified trace and returned comments."""
    if item.report is None:
        return None
    return PropagationReferenceSnapshot(case_id, round_no, True, item.review_accepted,
        frozenset(comment.comment_id for comment in item.comments),
        _report_digest(item.report))


def _fact_material(item: M4HistoricalPropagationInput, case_id: str, round_no: int,
                   source: M4ImportedFactSource, prop_issues: tuple[Issue, ...]) -> ArbiterMaterialV2:
    """重验CHEF的V2事实归档后加入独立验证过的传播问题。"""
    imported = source
    board = build_fact_board_v2(snapshot=imported.snapshot, expected_scope=imported.archive_input.scope,
                                action_round=round_no)
    numbered = tuple(issue.model_copy(update={"issue_id": f"I{len(board.issues) + index}"})
                     for index, issue in enumerate(prop_issues, start=1))
    board = replace(board, issues=board.issues + numbered)
    return build_arbiter_material_v2(case_id=case_id, sample_id=item.sample_id,
        decision_round=round_no, claim=item.claim, board=board,
        fact_runs=(imported.archive_input,))


def _no_fact_material(item: M4HistoricalPropagationInput, case_id: str,
                      round_no: int, issues: tuple[Issue, ...]) -> ArbiterMaterialV2:
    """只为固定CED无覆盖样本建立无事实材料，不伪造报告或审计指纹。"""
    if item.source != "ced" or item.coverage != "has_data" or not item.review_accepted:
        raise ValueError("m4_no_fact_source_invalid")
    if any(reference.startswith(("F", "fact:")) for issue in issues for reference in issue.evidence_refs):
        raise ValueError("m4_no_fact_contains_fact_reference")
    available = {f"issue:{issue.issue_id}" for issue in issues}
    available.update(reference for issue in issues for reference in issue.evidence_refs)
    return ArbiterMaterialV2(case_id=case_id, sample_id=item.sample_id, decision_round=round_no,
        claim=item.claim, claim_sha256=sha256(item.claim.encode("utf-8")).hexdigest(),
        issues=issues, fact_reports=(), evidence=(), available_refs=tuple(sorted(available)),
        fact_absence_reason="no_coverage")


def _case_snapshots(item: M4HistoricalPropagationInput, material: ArbiterMaterialV2,
                    round_no: int, prop_refs: PropagationReferenceSnapshot | None,
                    fact_source: object | None) -> tuple[CaseActionSnapshot, CaseReferenceSnapshot]:
    """构造只供离线预检的案件与引用快照。"""
    case_id, config = material.case_id, get_collab_config()
    comments = frozenset() if prop_refs is None else prop_refs.returned_comment_ids
    source = "chef_dev" if fact_source is not None else "ced"
    coverage = get_evidence_sources_config()[source].coverage
    budget = get_m4_case_budget_config()
    snapshot = CaseActionSnapshot(case_id, round_no,
        tuple(issue_snapshot(issue) for issue in material.issues), (), None,
        ActionConfig(config.max_actions, frozenset(config.targets), config.review_enabled),
        TargetSnapshot(coverage, item.stats is not None and item.stats.has_data,
            comments, comments, item.report is not None),
        BudgetSnapshot(Decimal(0), Decimal(0), Decimal(str(budget.action_caps["fact_investigate"])),
            Decimal(str(budget.final_decision_reserve)), Decimal(str(budget.max_cost_cny))))
    leads = {} if item.report is None else {
        f"L{index}": lead.source for index, lead in enumerate(item.report.leads, start=1)}
    fact_refs = frozenset(evidence.evidence_ref.removeprefix("fact:") for evidence in material.evidence)
    refs = CaseReferenceSnapshot(case_id, round_no, leads, comments, fact_refs, frozenset())
    return snapshot, refs


def _report_digest(report: object) -> str:
    """与问题台账一致地计算已核验传播报告的稳定指纹。"""
    import json

    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def measure_m4_capacity(cases: tuple[M4PreparedCase, ...], prompt_path: Path,
                        root: Path = PROJECT_ROOT) -> tuple[M4CapacityRow, ...]:
    """用实际五份材料和真实工具Schema估算首次及一次反馈请求。"""
    config = get_m4_arbiter_harness_config()
    prompt = (root / prompt_path).read_text(encoding="utf-8")
    if sha256(prompt.encode("utf-8")).hexdigest() != "298cf72780715713e165c8231e24d7cae86308bb6354a6933e1db0156b0ab8e8":
        raise ValueError("m4_arbiter_prompt_fingerprint_invalid")
    spec = AgentSpec(agent="arbiter", system_prompt=prompt)
    schema = build_arbiter_decision_v2_schema(get_collab_config().investigation_plan.max_text_chars)
    registry = ToolRegistry()
    registry.register(ToolSpec(name="submit_decision", risk="R0", args_model=schema,
        description="提交完整V2研判结论与只记录的调查计划；程序校验后才可接受。"))
    return tuple(_measure_case(case, spec, schema, registry, config)
                 for case in cases)


def _measure_case(case: M4PreparedCase, spec: AgentSpec, schema: type[BaseModel],
                  registry: ToolRegistry, config: object) -> M4CapacityRow:
    """Estimate first call and both retry shapes after one settled request."""
    cards = _preflight_cards(case)
    messages = _v2_messages(spec, case.material, case.supplemental, cards)
    tool = registry.get("submit_decision")
    if tool is None:
        raise ValueError("m4_capacity_tool_missing")
    definitions = llm._tool_definitions([tool], False, submission_tool=config.submission_tool)
    first = llm.estimate_request_input(messages, definitions, config.input_bound_margin_tokens)
    first_ok = BudgetTracker(config, get_cost_config()).check_model_call(
        first, config.max_output_tokens).allowed
    schema_retry = _last_retry_estimate(case, schema, messages, definitions, config, first, "schema")
    refusal_retry = _last_retry_estimate(case, schema, messages, definitions, config, first, "refusal")
    retry_paths = _cumulative_retry_estimates(case, schema, messages, definitions, config, first)
    max_legal_retry = max(pair[0] for pair in retry_paths.values())
    refusal_then_schema = retry_paths[("business", "schema")][1]
    schema_ok = _retry_allowed(config, first, schema_retry)
    refusal_ok = _retry_allowed(config, first, refusal_retry)
    cumulative_ok = _sequence_allowed(config, (first, max_legal_retry))
    retry_path_allowed = tuple((f"{a}->{b}", _sequence_allowed(config, (first, *pair)))
        for (a, b), pair in retry_paths.items())
    refusal_schema_ok = dict(retry_path_allowed)["business->schema"]
    all_retry_paths_allowed = all(allowed for _, allowed in retry_path_allowed)
    retry_path_requests = tuple((f"{a}->{b}", pair[0], pair[1])
        for (a, b), pair in retry_paths.items())
    return M4CapacityRow(case.material.sample_id, first, schema_retry, refusal_retry,
                         max_legal_retry, refusal_then_schema, first_ok, schema_ok, refusal_ok,
                         cumulative_ok, refusal_schema_ok, all_retry_paths_allowed,
                         retry_path_allowed, retry_path_requests,
                         tuple((f"{a}->{b}", value) for (a, b), (_, value) in retry_paths.items()))


def _cumulative_retry_estimates(case: M4PreparedCase, schema: type[BaseModel],
                                messages: list[dict[str, object]],
                                definitions: list[dict[str, object]], config: object,
                                first: int, *, cards_builder: Callable | None = None
                                ) -> dict[tuple[str, str], tuple[int, int]]:
    """Measure all two-retry paths with production schema-failure history semantics."""
    paths = {}
    for first_kind, second_kind in (("business", "business"), ("business", "schema"),
                                    ("schema", "business"), ("schema", "schema")):
        history = list(messages)
        first_spent = _max_request_cost(first, config)
        _append_failure(history, first_kind, "m4-capacity-first", case, schema,
                        first_spent, config, cards_builder=cards_builder)
        second = llm.estimate_request_input(history, definitions, config.input_bound_margin_tokens)
        second_spent = first_spent + _max_request_cost(second, config)
        _append_failure(history, second_kind, "m4-capacity-second", case, schema,
                        second_spent, config, cards_builder=cards_builder)
        third = llm.estimate_request_input(history, definitions, config.input_bound_margin_tokens)
        paths[(first_kind, second_kind)] = (second, third)
    return paths


def _append_failure(history: list[dict[str, object]], kind: str, call_id: str,
                    case: M4PreparedCase, schema: type[BaseModel], spent: Decimal,
                    config: object, *, cards_builder: Callable | None = None) -> None:
    """Append a retry transcript exactly as the production loop treats each failure class."""
    if kind == "schema":
        history.extend(_schema_feedback_messages(schema, config))
        cards = _preflight_cards(case, spent=spent) if cards_builder is None else cards_builder(case, spent)
        _append_capability_cards(history, cards)
        return
    arguments, _ = _max_schema_arguments(schema)
    feedback = _business_rejection_feedback(case, schema, arguments)
    _append_model_turn(history, call_id, arguments, feedback, case, spent, config,
                       cards_builder=cards_builder)


def _business_rejection_feedback(case: M4PreparedCase, schema: type[BaseModel],
                                 arguments: str) -> str:
    """Build the exact business-failure receipt via production validation and issue mapping."""
    decision = ArbiterDecisionV2.model_validate(schema.model_validate_json(arguments).model_dump())
    allowed_refs = _v2_available_refs(case.material, case.supplemental, case.references)
    errors = _v2_decision_errors(decision, case.material, case.supplemental,
        case.snapshot, case.references, allowed_refs)
    if not errors:
        raise ValueError("m4_capacity_pressure_answer_not_rejected")
    return ArbiterFeedback(accepted=False,
        issues=[_v2_feedback_issue(issue) for issue in errors]).model_dump_json()


def _max_schema_arguments(schema: type[BaseModel]) -> tuple[str, str]:
    """Build a Pydantic-valid tool call at every configured text/list boundary."""
    control = "\x01"
    ref = control * 48
    decision = {
        "evidence_sufficiency": "不足", "conflicts": ["关键事实缺证据", "证据之间冲突",
            "引用异常", "传播数据缺失", "疑似违规信号", "专家判断分歧"],
        "next_action": "investigate", "action": {"kind": "investigate", "target": "fact",
            "issue_ids": [ref, ref], "question": control * 80,
            "done_when": "找到一手来源", "lead_ids": [ref, ref],
            "review_fields": ["narrative", "emotion_level"]}, "register_conflict": None,
        "adjustment": 1, "adjustment_reason": control * 120, "needs_human_review": True,
        "rationale": [{"text": control * 120, "refs": [ref, ref, ref]} for _ in range(5)],
        "investigation_plan": {"uncertainty": control * 60, "action_note": control * 60,
            "stop_condition": control * 60, "intended_action": "investigate",
            "intended_target": "fact"}}
    parsed = schema.model_validate(decision)
    invalid = parsed.model_dump(mode="json")
    invalid["action"]["reason"] = "x"
    return (parsed.model_dump_json(),
            json.dumps(invalid, ensure_ascii=False, separators=(",", ":")))


def _append_model_turn(messages: list[dict[str, object]], call_id: str, arguments: str,
                       feedback: str, case: M4PreparedCase, spent: Decimal,
                       config: object, *, cards_builder: Callable | None = None) -> None:
    """Append one assistant tool call, its bounded reply, and refreshed capacity cards."""
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": call_id,
            "type": "function", "function": {"name": "submit_decision", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": call_id, "content": feedback},
    ])
    if len(feedback) > config.max_tool_result_chars:
        raise ValueError("m4_capacity_feedback_exceeds_tool_limit")
    cards = _preflight_cards(case, spent=spent) if cards_builder is None else cards_builder(case, spent)
    _append_capability_cards(messages, cards)


def _sequence_allowed(config: object, estimates: tuple[int, ...]) -> bool:
    """Check sequential calls against the same step, total-token, cost and final reserve rules."""
    tracker = BudgetTracker(config, get_cost_config())
    for index, estimate in enumerate(estimates):
        final = index == config.max_steps - 1
        if not tracker.check_model_call(estimate, config.max_output_tokens, final=final).allowed:
            return False
        if not tracker.reserve_model_call(estimate, config.max_output_tokens, final=final).allowed:
            return False
        tracker.settle_model_call(estimate, config.max_output_tokens)
    return True


def _last_retry_estimate(case: M4PreparedCase, schema: type[BaseModel], messages: list[dict[str, object]],
                         definitions: list[dict[str, object]], config: object, first: int,
                         retry: str) -> int:
    """Build an exact retry transcript and estimate it with the production gateway estimator."""
    retry_messages = list(messages)
    _append_retry_transcript(retry_messages, case, schema, config, retry,
                             spent=_max_request_cost(first, config))
    return llm.estimate_request_input(retry_messages, definitions, config.input_bound_margin_tokens)


def _retry_allowed(config: object, first: int, retry: int) -> bool:
    """Check a later request after the prior request used its configured output bound."""
    budget = BudgetTracker(config, get_cost_config())
    if not budget.reserve_model_call(first, config.max_output_tokens).allowed:
        return False
    budget.settle_model_call(first, config.max_output_tokens)
    return budget.check_model_call(retry, config.max_output_tokens).allowed


def _append_retry_transcript(messages: list[dict[str, object]], case: M4PreparedCase,
                             schema: type[BaseModel], config: object, retry: str,
                             spent: Decimal) -> None:
    """Append the same assistant/tool/user transcript used by the production retry path."""
    cards = _preflight_cards(case, spent=spent)
    if retry == "schema":
        messages.extend(_schema_feedback_messages(schema, config))
    elif retry == "refusal":
        messages.extend(_refusal_feedback_messages(schema, config))
    else:
        raise ValueError("m4_capacity_retry_kind_invalid")
    _append_capability_cards(messages, cards)


def _schema_feedback_messages(schema: type[BaseModel], config: object) -> list[dict[str, object]]:
    """Use production formatting for schema-invalid response feedback."""
    from src.yuqing.harness.feedback import format_retry_feedback

    issue = ToolValidationIssue(tool_name="submit_decision", loc=("action", "reason"),
                                error_type="extra_forbidden")
    feedback = format_retry_feedback("invalid_tool_arguments", (issue,),
        {"submit_decision": schema}, config.max_tool_result_chars)
    return [{"role": "user", "content": feedback}]


def _refusal_feedback_messages(schema: type[BaseModel], config: object) -> list[dict[str, object]]:
    """Use a schema-valid unknown reference to model the program refusal transcript."""
    decision = schema.model_validate(_invalid_reference_decision())
    arguments = decision.model_dump_json()
    call_id = "m4-capacity-refusal"
    assistant = {"role": "assistant", "content": None, "tool_calls": [{"id": call_id,
        "type": "function", "function": {"name": "submit_decision", "arguments": arguments}}]}
    feedback = ArbiterFeedback(accepted=False, issues=["unknown_reference"]).model_dump_json()
    if len(feedback) > config.max_tool_result_chars:
        raise ValueError("m4_capacity_feedback_exceeds_tool_limit")
    return [assistant, {"role": "tool", "tool_call_id": call_id, "content": feedback}]


def _invalid_reference_decision() -> dict[str, object]:
    """Return a structurally valid final decision with one intentionally unknown reference."""
    return {"evidence_sufficiency": "不足", "conflicts": [], "next_action": "finalize",
        "action": None, "register_conflict": None, "adjustment": 0, "adjustment_reason": None,
        "needs_human_review": True, "rationale": [{"text": "仍需人工核对", "refs": ["issue:__m4_unknown__"]}],
        "investigation_plan": {"uncertainty": "尚未确定", "action_note": "先保留疑问",
            "stop_condition": "不追加调查", "intended_action": "finalize", "intended_target": None}}




def _max_request_cost(input_tokens: int, config: object) -> Decimal:
    """Estimate the first request's maximum reserved cost from its input and output bounds."""
    usage = llm.TokenUsage(input_tokens=input_tokens, output_tokens=config.max_output_tokens)
    return Decimal(str(llm.compute_cost_cny(usage, get_cost_config())))


def _preflight_cards(case: M4PreparedCase, *, spent: Decimal = Decimal(0)) -> tuple[CapabilityCardV2, ...]:
    """Build capability cards before or after conservative request spending."""
    budget, config = get_m4_case_budget_config(), get_collab_config()
    coverage = get_evidence_sources_config()["chef_dev" if case.material.fact_reports else "ced"].coverage
    shared = Decimal(str(get_m4_run_budget_config().experiment_max_cost_cny))
    reserve = Decimal(str(budget.final_decision_reserve))
    case_balance = max(Decimal(0), Decimal(str(budget.max_cost_cny)) - reserve - spent)
    scope = ArbiterBudgetScope(
        experiment_id="m4-offline-preflight", case_id=case.material.case_id,
        sample_id=case.material.sample_id, group_id="m4", variant="capacity-only",
        current_run_id=uuid5(NAMESPACE_URL, case.material.case_id).hex,
        decision_round=case.material.decision_round, bindings=(), current_run_cost=Decimal(0),
        fact_coverage=coverage,
        propagation_has_data=bool(case.supplemental.stats and case.supplemental.stats.has_data),
        propagation_report_available=case.supplemental.propagation_report is not None,
        unreturned_comment_count=0)
    caps = {name: Decimal(str(value)) for name, value in budget.action_caps.items()}
    return _make_cards(scope, config=config, coverage=coverage, remaining=config.max_actions,
        reserve=reserve, case_balance=case_balance, action_pool=Decimal(str(budget.plan["action_pool"])),
        shared_balance=max(Decimal(0), shared - spent), caps=caps)
