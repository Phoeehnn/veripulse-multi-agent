"""构造配置驱动的事实 V2 Schema，并执行纯本地合同检查。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Literal, Mapping, Type
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, create_model, model_validator

from src.yuqing.schemas import (
    FactScopeV2, FactVerdictV2, FactVerdictV2Envelope, FactV2ProgramValidation,
    FollowupScopeV2, InheritedEvidencePlanV2, ProgramActionClaimV2,
    InvestigationPlanV2, KeyFactV2, KeyFactV2Followup, ProposedIssueV2,
    QuoteV2, ReasonV2, VerifiedQuoteV2,
)
from src.yuqing.settings import (FactV2LimitsConfig, get_collab_config,
                                 get_fact_v2_limits_config)
from src.yuqing.harness.quote_verifier import QuoteLocation, locate_quote, normalize_quote_text
from src.yuqing.harness.inherited_evidence import validate_inherited_plan, validate_followup_scope_v2


@dataclass(frozen=True)
class FactV2Schemas:
    """由已批准限制生成的首轮、补查及仲裁计划 Schema。"""

    initial: Type[BaseModel]
    followup: Type[BaseModel]
    initial_envelope: Type[BaseModel]
    followup_envelope: Type[BaseModel]
    proposed_issue: Type[BaseModel]
    investigation_plan: Type[BaseModel]


@dataclass(frozen=True)
class DeliveredEvidenceV2:
    """由可信调用方提供的正文和确认交付前缀长度。"""

    text: str
    read_end: int


@dataclass(frozen=True)
class FactV2Check:
    """严格核心解析结果与独立程序校验记录。"""

    report: BaseModel | None
    validation: FactV2ProgramValidation


@dataclass(frozen=True)
class FactRunContextV2:
    """由程序构造并复检的V2运行合同，不接受模型自报版本或模式。"""

    mode: Literal["initial", "followup"]
    limits: FactV2LimitsConfig
    plan_max_text_chars: int
    assigned_issue_ids: frozenset[str]
    schema: Type[BaseModel]
    response_schema: Type[BaseModel]
    initial_manifest_path: Path | None
    scope: FactScopeV2 | FollowupScopeV2 | None = None
    inherited_plan: InheritedEvidencePlanV2 | None = None
    action_claim: ProgramActionClaimV2 | None = None

    @property
    def schema_sha256(self) -> str:
        """返回本合同当前提交Schema的稳定指纹。"""
        return _canonical_sha256({"submission": self.schema.model_json_schema(),
                                  "response": self.response_schema.model_json_schema()})

    @property
    def contract_sha256(self) -> str:
        """绑定Schema、运行模式、限制与可选程序scope的完整合同指纹。"""
        return _canonical_sha256({"mode": self.mode, "schema_sha256": self.schema_sha256,
            "limits": self.limits.model_dump(mode="json"), "plan_max_text_chars": self.plan_max_text_chars,
            "assigned_issue_ids": sorted(self.assigned_issue_ids),
            "scope": self.scope.model_dump(mode="json") if self.scope is not None else None,
            "inherited_plan_sha256": (_canonical_sha256(self.inherited_plan.model_dump(mode="json"))
                                       if self.inherited_plan is not None else None),
            "action_claim_sha256": (_canonical_sha256(self.action_claim.model_dump(mode="json"))
                                    if self.action_claim is not None else None)})


def build_fact_run_context_v2(mode: Literal["initial", "followup"], *,
                              assigned_issue_ids: frozenset[str] = frozenset(),
                              initial_manifest_path: Path | None = None,
                              scope: FactScopeV2 | FollowupScopeV2 | None = None,
                              inherited_plan: InheritedEvidencePlanV2 | None = None,
                              action_claim: ProgramActionClaimV2 | None = None) -> FactRunContextV2:
    """按显式模式和配置构造首轮或补查运行合同。"""
    if mode not in ("initial", "followup"):
        raise ValueError("fact_v2_mode_invalid")
    if ((mode == "initial" and (assigned_issue_ids or initial_manifest_path is None))
            or (mode == "followup" and (not assigned_issue_ids or initial_manifest_path is not None))):
        raise ValueError("fact_v2_issue_scope_mismatch")
    _validate_followup_binding(mode, assigned_issue_ids, scope, inherited_plan, action_claim)
    limits = get_fact_v2_limits_config()
    plan_chars = get_collab_config().investigation_plan.max_text_chars
    schemas = build_fact_v2_schemas(limits, plan_max_text_chars=plan_chars)
    initial = mode == "initial"
    return FactRunContextV2(mode, limits, plan_chars, assigned_issue_ids,
                            schemas.initial if initial else schemas.followup,
                            schemas.initial_envelope if initial else schemas.followup_envelope,
                            initial_manifest_path, scope, inherited_plan, action_claim)


def _validate_followup_binding(
    mode: str, assigned_issue_ids: frozenset[str], scope: FactScopeV2 | FollowupScopeV2 | None,
    plan: InheritedEvidencePlanV2 | None, proof: ProgramActionClaimV2 | None,
) -> None:
    """Keep legacy unscoped schema tests compatible; require full bindings for real V2 follow-up runs."""
    if mode == "initial":
        if isinstance(scope, FollowupScopeV2) or plan is not None or proof is not None:
            raise ValueError("fact_v2_initial_scope_mismatch")
        return
    if scope is None:
        if plan is not None or proof is not None:
            raise ValueError("fact_v2_followup_scope_missing")
        return
    if (not isinstance(scope, FollowupScopeV2) or plan is None or proof is None
            or assigned_issue_ids != frozenset(scope.issue_ids)
            or plan.target_scope != scope):
        raise ValueError("fact_v2_followup_binding_mismatch")
    validate_inherited_plan(plan)
    validate_followup_scope_v2(scope, proof)


def build_fact_v2_schemas(limits: FactV2LimitsConfig,
                          *, plan_max_text_chars: int) -> FactV2Schemas:
    """显式传入配置后，生成可用于提示和严格解析的独立模型类。"""
    quote_model = _bounded_quote_model(limits)
    proposal_model = _bounded_proposal_model(limits)
    reason_model = _bounded_reason_model(limits)
    initial_key = _bounded_key_fact_model(KeyFactV2, "KeyFactV2", quote_model, limits)
    followup_key = _bounded_key_fact_model(KeyFactV2Followup, "KeyFactV2Followup", quote_model, limits)
    initial = _bounded_verdict_model("FactVerdictV2", initial_key, reason_model, proposal_model, limits)
    followup = _bounded_verdict_model(
        "FactVerdictV2Followup", followup_key, reason_model, proposal_model, limits)
    initial_envelope = _bounded_envelope_model("FactVerdictV2InitialEnvelope", initial, limits)
    followup_envelope = _bounded_envelope_model("FactVerdictV2FollowupEnvelope", followup, limits)
    plan = _bounded_plan_model(plan_max_text_chars)
    return FactV2Schemas(initial, followup, initial_envelope, followup_envelope,
                         proposal_model, plan)


def _bounded_envelope_model(name: str, core: Type[BaseModel],
                            limits: FactV2LimitsConfig) -> Type[BaseModel]:
    """严格解析核心字段，同时把可选建议附录保留为未经解释的数据。"""
    validator = model_validator(mode="after")(_report_shape_validator(limits))
    return create_model(name, __base__=FactVerdictV2Envelope,
        key_facts=(core.model_fields["key_facts"].annotation,
                   Field(min_length=1, max_length=limits.max_key_facts)),
        reasons=(core.model_fields["reasons"].annotation,
                 Field(min_length=1, max_length=limits.max_reasons)),
        proposed_issues=(Any, None), __validators__={"check_shape": validator})


def validate_fact_v2_core(
    envelope: FactVerdictV2Envelope | Mapping[str, object], *,
    delivered: Mapping[str, DeliveredEvidenceV2], limits: FactV2LimitsConfig,
    plan_max_text_chars: int, mode: Literal["initial", "followup"],
    assigned_issue_ids: frozenset[str] = frozenset(),
) -> FactV2Check:
    """独立校验严格核心；无效建议附录保留在外壳中，不拖垮核心。"""
    if mode not in ("initial", "followup"):
        raise ValueError("fact_v2_mode_invalid")
    if ((mode == "initial" and assigned_issue_ids)
            or (mode == "followup" and not assigned_issue_ids)):
        raise ValueError("fact_v2_issue_scope_mismatch")
    raw = _as_mapping(envelope)
    payload_sha = _canonical_sha256(raw)
    try:
        parsed_envelope = FactVerdictV2Envelope.model_validate(raw, strict=True)
        schemas = build_fact_v2_schemas(limits, plan_max_text_chars=plan_max_text_chars)
        schema = schemas.initial if mode == "initial" else schemas.followup
        core = {name: getattr(parsed_envelope, name)
                for name in ("key_facts", "reasons", "label", "confidence")}
        report = schema.model_validate(core, strict=True)
    except ValidationError as error:
        validation = FactV2ProgramValidation(canonical_payload_sha256=payload_sha,
                                             issues=list(_validation_codes(error)))
        return FactV2Check(report=None, validation=validation)
    issues, quote_records = _check_report(report, delivered, limits, mode, assigned_issue_ids)
    validation = FactV2ProgramValidation(canonical_payload_sha256=payload_sha,
                                         quote_records=quote_records, issues=issues)
    return FactV2Check(report=report, validation=validation)


def _bounded_quote_model(limits: FactV2LimitsConfig) -> Type[BaseModel]:
    """为引文文本应用已确认的 Unicode 字符上限。"""
    return create_model("QuoteV2", __base__=QuoteV2,
                        text=(str, Field(min_length=1, max_length=limits.max_quote_chars,
                                         pattern=r"\S")))


def _bounded_proposal_model(limits: FactV2LimitsConfig) -> Type[BaseModel]:
    """为建议问题应用已确认长度并保持来源字段封闭。"""
    return create_model("ProposedIssueV2", __base__=ProposedIssueV2,
                        question=(str, Field(min_length=1, max_length=limits.max_proposed_question_chars,
                                             pattern=r"\S")))


def _bounded_reason_model(limits: FactV2LimitsConfig) -> Type[BaseModel]:
    """为理由文字和编号列表应用确认的结构边界。"""
    validator = model_validator(mode="after")(_unique_reason_refs)
    return create_model("ReasonV2", __base__=ReasonV2,
                        text=(str, Field(min_length=1, max_length=limits.max_reason_chars,
                                         pattern=r"\S")),
                        __validators__={"unique_refs": validator})


def _unique_reason_refs(reason: BaseModel) -> BaseModel:
    """拒绝同一理由中重复列出的证据编号。"""
    if len(reason.evidence_ids) != len(set(reason.evidence_ids)):
        raise ValueError("reason_duplicate_evidence_ids")
    return reason


def _bounded_key_fact_model(base: Type[BaseModel], name: str, quote_model: Type[BaseModel],
                            limits: FactV2LimitsConfig) -> Type[BaseModel]:
    """分别生成禁止或必须 issue_id 的关键事实模型。"""
    fields: dict[str, tuple[object, object]] = {
        "text": (str, Field(min_length=1, max_length=limits.max_key_fact_chars, pattern=r"\S")),
        "evidence_ids": (list[str], Field(default_factory=list)),
        "quotes": (list[quote_model], Field(default_factory=list,
                                             max_length=limits.max_quotes_per_direction
                                             * len(("支持", "反驳")))),
    }
    if name == "KeyFactV2Followup":
        fields["issue_id"] = (str, Field(min_length=1, pattern=r"\S"))
    validator = model_validator(mode="after")(_key_fact_shape_validator(limits))
    return create_model(name, __base__=base, __validators__={"check_shape": validator}, **fields)


def _key_fact_shape_validator(limits: FactV2LimitsConfig
                              ) -> Callable[[BaseModel], BaseModel]:
    """构造关键事实的引文数量、方向、编号和重复项校验器。"""
    def check(fact: BaseModel) -> BaseModel:
        """保持事实方向与其引文字段形式一致。"""
        quotes = fact.quotes
        if len(fact.evidence_ids) != len(set(fact.evidence_ids)):
            raise ValueError("fact_duplicate_evidence_ids")
        quote_keys = [(quote.evidence_id, normalize_quote_text(quote.text)) for quote in quotes]
        if len(quote_keys) != len(set(quote_keys)):
            raise ValueError("fact_duplicate_quotes")
        quoted_ids = {quote.evidence_id for quote in quotes}
        if fact.status == "未提及":
            if quotes:
                raise ValueError("unmentioned_fact_has_quotes")
            return fact
        if fact.status in {"支持", "反驳"}:
            if not 1 <= len(quotes) <= limits.max_quotes_per_direction:
                raise ValueError("fact_quote_count_invalid")
            if any(quote.side is not None and quote.side != fact.status for quote in quotes):
                raise ValueError("fact_quote_side_mismatch")
        if fact.status == "冲突":
            for side in ("支持", "反驳"):
                side_quotes = [quote for quote in quotes if quote.side == side]
                if not 1 <= len(side_quotes) <= limits.max_quotes_per_direction:
                    raise ValueError("conflict_quote_sides_incomplete")
            if any(quote.side is None for quote in quotes):
                raise ValueError("conflict_quote_side_required")
        if fact.status != "未提及" and set(fact.evidence_ids) != quoted_ids:
            raise ValueError("fact_evidence_quote_set_mismatch")
        return fact
    return check


def _bounded_verdict_model(name: str, key_model: Type[BaseModel], reason_model: Type[BaseModel],
                           proposal_model: Type[BaseModel],
                           limits: FactV2LimitsConfig) -> Type[BaseModel]:
    """生成事实条数、理由条数和原始引文总量受限的报告模型。"""
    validator = model_validator(mode="after")(_report_shape_validator(limits))
    return create_model(
        name, __base__=FactVerdictV2,
        key_facts=(list[key_model], Field(min_length=1, max_length=limits.max_key_facts)),
        reasons=(list[reason_model], Field(min_length=1, max_length=limits.max_reasons)),
        confidence=(float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)),
        proposed_issues=(list[proposal_model],
                         Field(default_factory=list, max_length=limits.max_proposed_issues)),
        __validators__={"check_shape": validator},
    )


def _report_shape_validator(limits: FactV2LimitsConfig) -> Callable[[BaseModel], BaseModel]:
    """构造整份报告原始引文总数校验器。"""
    def check(report: BaseModel) -> BaseModel:
        """在去重前按原数组计数，避免无限重复绕过上限。"""
        quote_count = sum(len(fact.quotes) for fact in report.key_facts)
        if quote_count > limits.max_quotes_per_report:
            raise ValueError("report_quote_count_exceeded")
        return report
    return check


def _bounded_plan_model(max_text_chars: int) -> Type[BaseModel]:
    """为仲裁计划的三个说明字段应用显式配置长度。"""
    return create_model(
        "InvestigationPlanV2", __base__=InvestigationPlanV2,
        uncertainty=(str, Field(min_length=1, max_length=max_text_chars, pattern=r"\S")),
        action_note=(str, Field(min_length=1, max_length=max_text_chars, pattern=r"\S")),
        stop_condition=(str, Field(min_length=1, max_length=max_text_chars, pattern=r"\S")),
    )


def _check_report(report: BaseModel, delivered: Mapping[str, DeliveredEvidenceV2],
                  limits: FactV2LimitsConfig, mode: Literal["initial", "followup"],
                  assigned_issue_ids: frozenset[str]
                  ) -> tuple[list[str], list[VerifiedQuoteV2]]:
    """校验交付引用、issue归属、引文位置和五条标签一致性。"""
    issues: list[str] = []
    records: list[VerifiedQuoteV2] = []
    _check_issue_assignments(report, mode, assigned_issue_ids, issues)
    _check_delivered_references(report, delivered, issues)
    for fact_index, fact in enumerate(report.key_facts):
        _check_fact_quotes(fact_index, fact, delivered, limits, issues, records)
    _check_label_rules(report, issues)
    return list(dict.fromkeys(issues)), records


def _check_issue_assignments(report: BaseModel, mode: str,
                             assigned_issue_ids: frozenset[str], issues: list[str]) -> None:
    """首轮禁止 issue_id，补查只接受本任务单目标编号。"""
    if mode != "followup":
        return
    for index, fact in enumerate(report.key_facts):
        if fact.issue_id not in assigned_issue_ids:
            issues.append(f"key_facts[{index}].issue_id: not_assigned")


def _check_delivered_references(report: BaseModel, delivered: Mapping[str, DeliveredEvidenceV2],
                                issues: list[str]) -> None:
    """关键事实和理由只能引用当前请求确认交付的编号。"""
    for field in ("key_facts", "reasons"):
        for index, item in enumerate(getattr(report, field)):
            if len(item.evidence_ids) > len(delivered):
                issues.append(f"{field}[{index}].evidence_ids: exceeds_delivered_set")
            for evidence_id in item.evidence_ids:
                if evidence_id not in delivered:
                    issues.append(f"{field}[{index}].evidence_ids: undelivered")


def _check_fact_quotes(fact_index: int, fact: BaseModel,
                       delivered: Mapping[str, DeliveredEvidenceV2], limits: FactV2LimitsConfig,
                       issues: list[str], records: list[VerifiedQuoteV2]) -> None:
    """校验引文属于事实编号，并保存经复检的原文位置和上下文。"""
    for quote_index, quote in enumerate(fact.quotes):
        if quote.evidence_id not in fact.evidence_ids:
            issues.append(f"key_facts[{fact_index}].quotes[{quote_index}]: evidence_not_listed")
            continue
        evidence = delivered.get(quote.evidence_id)
        if evidence is None:
            continue
        _record_quote(fact_index, quote_index, quote, evidence, limits, issues, records)


def _record_quote(fact_index: int, quote_index: int, quote: BaseModel,
                  evidence: DeliveredEvidenceV2, limits: FactV2LimitsConfig,
                  issues: list[str], records: list[VerifiedQuoteV2]) -> None:
    """在实际交付前缀内匹配并形成不越界的程序记录。"""
    normalized = normalize_quote_text(quote.text)
    if not limits.min_normalized_quote_chars <= len(normalized) <= limits.max_quote_chars:
        issues.append(f"key_facts[{fact_index}].quotes[{quote_index}]: normalized_length_invalid")
        return
    location = locate_quote(evidence.text, quote.text, read_end=evidence.read_end)
    if location.status != "matched" or location.start is None or location.end is None:
        issues.append(f"key_facts[{fact_index}].quotes[{quote_index}]: {location.status}")
        return
    _append_quote_record(fact_index, quote_index, quote, evidence, limits, location, records)


def _append_quote_record(fact_index: int, quote_index: int, quote: BaseModel,
                         evidence: DeliveredEvidenceV2, limits: FactV2LimitsConfig,
                         location: QuoteLocation, records: list[VerifiedQuoteV2]) -> None:
    """生成带原始正文指纹和可见上下文的引用核验记录。"""
    start, end = location.start, location.end
    visible_end = min(evidence.read_end, end + limits.quote_context_chars)
    visible_start = max(0, start - limits.quote_context_chars)
    records.append(VerifiedQuoteV2(
        fact_index=fact_index, quote_index=quote_index, evidence_id=quote.evidence_id,
        original_quote=quote.text, normalized_text=location.normalized_text,
        body_sha256=sha256(evidence.text.encode("utf-8")).hexdigest(), start=start, end=end,
        left_context=evidence.text[visible_start:start],
        right_context=evidence.text[end:visible_end], ambiguous=location.ambiguous,
    ))


def _check_label_rules(report: BaseModel, issues: list[str]) -> None:
    """组合 V-1 与旧循环校验器中的五条确定性规则。"""
    statuses = {fact.status for fact in report.key_facts}
    if report.label == "支持":
        if "支持" not in statuses:
            issues.append("label: supported_without_supporting_fact")
        if "反驳" in statuses:
            issues.append("label: supported_with_refuting_fact")
        if "冲突" in statuses:
            issues.append("label: supported_with_conflicting_fact")
    if report.label == "反驳" and "反驳" not in statuses:
        issues.append("label: refuted_without_refuting_fact")
    if statuses == {"未提及"} and report.label != "信息不足":
        issues.append("label: all_facts_unmentioned")


def _as_mapping(value: FactVerdictV2Envelope | Mapping[str, object]
                ) -> Mapping[str, object]:
    """把已解码JSON字典或信封转换为不可变模型数据映射。"""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", exclude_unset=True)
    return value


def _canonical_sha256(value: Mapping[str, object]) -> str:
    """用稳定JSON序列化计算程序侧输入指纹。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _validation_codes(error: ValidationError) -> tuple[str, ...]:
    """仅输出字段位置与错误类型，不回显模型正文。"""
    return tuple(f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                 for item in error.errors())
