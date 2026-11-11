"""项目各层共享的输入、证据、研判和评测数据模型。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr, computed_field,
                      create_model, model_validator)


FactLabel = Literal["支持", "反驳", "信息不足"]

from src.yuqing.settings import get_propagation_limits_config

_PROPAGATION_LIMITS = get_propagation_limits_config()
PROPAGATION_SCHEMA_LIMITS = (
    _PROPAGATION_LIMITS.text_max_chars,
    _PROPAGATION_LIMITS.emotion_signal_max_items,
    _PROPAGATION_LIMITS.lead_max_items,
)


class ToolRequest(BaseModel):
    """经过参数校验的工具请求，仍须权限检查才能执行。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, object]


class ToolLLMResult(BaseModel):
    """模型的文本、工具请求、可续接消息及本次实际用量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    text: str
    tool_calls: list[ToolRequest]
    assistant_message: dict[str, object]
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    cost_cny: float = Field(ge=0, allow_inf_nan=False)
    latency_ms: int = Field(ge=0)


class ToolValidationIssue(BaseModel):
    """仅供程序使用的安全错误位置、类型与固定原因码，不保存模型原文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    tool_name: str | None = None
    loc: tuple[str | int, ...] = ()
    error_type: str
    reason_code: str | None = None
    trusted_feedback: str | None = None


class ToolCallOutcome(BaseModel):
    """程序产生的调用说明；成功才有可执行响应，不属于专家事实判断。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["success", "invalid_response", "budget_denied", "usage_unknown",
                    "reservation_exceeded", "startup_error", "audit_error"]
    reason: str = Field(min_length=1, pattern=r"\S")
    request_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    result: ToolLLMResult | None = None
    validation_issues: tuple[ToolValidationIssue, ...] = ()

    @model_validator(mode="after")
    def consistent_result(self) -> ToolCallOutcome:
        """成功必须携带同编号响应，失败禁止夹带可执行结果。"""
        if (self.status == "success") != (self.result is not None):
            raise ValueError("outcome_result_mismatch")
        if self.result is not None and self.request_id != self.result.request_id:
            raise ValueError("outcome_request_mismatch")
        return self


class BudgetStatus(BaseModel):
    """预算状态独立快照，停用原因不因后续拒绝或重复halt而消失。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    halted: bool
    pending: bool
    halt_reason: str | None

    @model_validator(mode="after")
    def consistent_halt(self) -> BudgetStatus:
        """停用必须有非空原因，未停用不携带停用原因。"""
        if self.halted != (self.halt_reason is not None) or self.halt_reason is not None and not self.halt_reason.strip():
            raise ValueError("budget_status_mismatch")
        return self


class ToolEvidence(BaseModel):
    """模型可见的匿名正文或预览，读取范围不包含截断尾部。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    text: str
    truncated: bool
    read_end: int = Field(ge=0, strict=True)


class EvidenceToolPayload(BaseModel):
    """完整序列化后受长度上限约束的工具内容，不含内部来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["success", "denied", "error", "timeout"]
    reason: str
    items: list[ToolEvidence] = Field(default_factory=list)


class ToolExecutionResult(BaseModel):
    """请求对应关系属于协议元信息，payload是发给模型的工具内容。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str
    name: str
    payload: EvidenceToolPayload


class RepostNode(BaseModel):
    """CED 中的一条转发或评论节点。"""

    mid: str
    uid: str
    parent: str
    text: str = ""
    date: datetime


class PropagationRaw(BaseModel):
    """一条原帖及其传播树的原始信息。"""

    root_time: datetime
    root_followers: int = 0
    root_verified: bool = False
    comments: int = 0
    reposts: int = 0
    likes: int = 0
    nodes: list[RepostNode] = Field(default_factory=list)


class Sample(BaseModel):
    """不同来源统一后的评测或推理样本。"""

    sample_id: str
    source: Literal["chef", "ced", "arg"]
    claim: str
    published_at: datetime | None = None
    gold_veracity: Literal["真", "假", "无法判定"] | None = None
    gold_risk: int | None = Field(default=None, ge=1, le=4)
    evidence_ids: list[str] = Field(default_factory=list)
    propagation: PropagationRaw | None = None


class EvidenceChunk(BaseModel):
    """可追溯到来源文档及位置的检索片段。"""

    chunk_id: str
    text: str
    source_doc: str
    position: int
    score: float = 0.0


class LedgerEntry(BaseModel):
    """程序保存的完整片段、全部来源与实际返回正文的前缀范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    text: str
    sources: list[EvidenceChunk]
    read_end: int = Field(default=0, ge=0, strict=True)


class AgentOutput(BaseModel):
    """四路专家共享的评分与理由结构。"""

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(min_length=1)
    citations: list[int] = Field(default_factory=list)


class FactCheck(AgentOutput):
    """旧版事实核查结构（deprecated），仅保留历史兼容定义。"""

    label: Literal["支持", "反驳", "信息不足"]


class Reason(BaseModel):
    """一条判断理由及用于解释该理由的证据编号。"""

    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class KeyFact(BaseModel):
    """声明中的关键事实、证据状态和相关证据编号。"""

    text: str
    status: Literal["支持", "反驳", "未提及", "冲突"]
    evidence_ids: list[str] = Field(default_factory=list)


class FactVerdict(BaseModel):
    """关键事实核查记录、判断理由、最终标签和判断置信度。"""

    model_config = ConfigDict(extra="forbid")

    key_facts: list[KeyFact] = Field(min_length=1)
    reasons: list[Reason] = Field(min_length=1)
    label: FactLabel
    confidence: float = Field(ge=0.0, le=1.0)


class QuoteV2(BaseModel):
    """事实 V2 的单条引文；范围和正文匹配由程序核验。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str = Field(min_length=1, pattern=r"\S")
    text: str = Field(min_length=1, pattern=r"\S")
    side: Literal["支持", "反驳"] | None = None


class ProposedIssueV2(BaseModel):
    """事实专家提出的候选问题，来源和入账字段由程序补全。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    question: str = Field(min_length=1, pattern=r"\S")
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_evidence_ids(self) -> ProposedIssueV2:
        """建议引用编号不得重复。"""
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("proposed_issue_duplicate_evidence_ids")
        return self


class KeyFactV2(BaseModel):
    """首轮事实 V2 结构；受限运行时使用配置生成的严格子类。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(min_length=1, pattern=r"\S")
    status: Literal["支持", "反驳", "未提及", "冲突"]
    evidence_ids: list[str] = Field(default_factory=list)
    quotes: list[QuoteV2] = Field(default_factory=list)


class KeyFactV2Followup(KeyFactV2):
    """补查关键事实必须逐条绑定一个被指派的问题。"""

    issue_id: str = Field(min_length=1, pattern=r"\S")


class ReasonV2(BaseModel):
    """事实 V2 的一条理由及其证据编号。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(min_length=1, pattern=r"\S")
    evidence_ids: list[str] = Field(min_length=1)


class FactVerdictV2(BaseModel):
    """V2 结构基型；实际调用必须使用按已确认限制生成的严格 Schema。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    key_facts: list[KeyFactV2] = Field(min_length=1)
    reasons: list[ReasonV2] = Field(min_length=1)
    label: FactLabel
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    proposed_issues: list[ProposedIssueV2] = Field(default_factory=list)


class FactVerdictV2Envelope(BaseModel):
    """容错解析外壳；附录保留为数据，核心稍后独立严格校验。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    key_facts: list[dict[str, Any]]
    reasons: list[dict[str, Any]]
    label: FactLabel
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
    proposed_issues: Any = None


class ProposedIssueRejectionV2(BaseModel):
    """程序记录被拒绝建议的原始位置和有限原因码。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_index: int = Field(ge=0, strict=True)
    reason: Literal["invalid_type", "blank_question", "question_too_long", "invalid_evidence",
                    "duplicate_evidence", "unknown_field", "duplicate_question",
                    "existing_issue_text_incomplete", "existing_issue_source_invalid"]


class ProposedIssueLinkV2(BaseModel):
    """Map one original appendix item to a created, replayed, or duplicate issue."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_index: int = Field(ge=0, strict=True)
    issue_id: str = Field(min_length=1, pattern=r"\S")
    disposition: Literal["created", "replay", "duplicate"]


class ProposedIssueAuditV2(BaseModel):
    """建议附录审计结果，不改变事实报告合格性或结论。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    appendix_status: Literal["absent", "accepted", "rejected_items", "mixed", "rejected_total"]
    appendix_rejection_reason: Literal["invalid_type", "too_many_items", "null_not_allowed",
                                       "existing_issue_text_incomplete",
                                       "existing_issue_source_invalid"] | None = None
    accepted: list[ProposedIssueV2] = Field(default_factory=list)
    rejected: list[ProposedIssueRejectionV2] = Field(default_factory=list)
    links: list[ProposedIssueLinkV2] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_appendix_status(self) -> ProposedIssueAuditV2:
        """总附录拒收原因只随整份附录拒绝状态记录。"""
        if (self.appendix_status == "rejected_total") != (self.appendix_rejection_reason is not None):
            raise ValueError("proposed_issue_audit_status_mismatch")
        if self.appendix_status == "absent" and (self.accepted or self.rejected):
            raise ValueError("absent_proposed_issue_has_items")
        if self.appendix_status == "accepted" and not self.accepted:
            raise ValueError("accepted_proposed_issue_missing_items")
        if self.appendix_status == "rejected_items" and (self.accepted or not self.rejected):
            raise ValueError("rejected_proposed_issue_items_mismatch")
        if self.appendix_status == "mixed" and (not self.accepted or not self.rejected):
            raise ValueError("mixed_proposed_issue_items_mismatch")
        if self.appendix_status == "accepted" and self.rejected:
            raise ValueError("accepted_proposed_issue_has_rejections")
        if self.appendix_status == "rejected_total" and self.accepted:
            raise ValueError("rejected_proposed_issue_has_accepted_items")
        if self.appendix_status == "rejected_total" and self.rejected:
            raise ValueError("total_rejection_has_item_rejections")
        link_indexes = [item.source_index for item in self.links]
        if (len(link_indexes) != len(set(link_indexes))
                or (self.appendix_status in {"absent", "rejected_total"} and self.links)):
            raise ValueError("proposed_issue_audit_links_mismatch")
        return self


class InvestigationPlanV2(BaseModel):
    """仲裁调查计划只作记录，字段不触发实际动作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    uncertainty: str = Field(min_length=1, pattern=r"\S")
    action_note: str = Field(min_length=1, pattern=r"\S")
    stop_condition: str = Field(min_length=1, pattern=r"\S")
    intended_action: Literal["finalize", "investigate", "review"]
    intended_target: Literal["fact", "propagation"] | None

    @model_validator(mode="after")
    def target_matches_intent_shape(self) -> InvestigationPlanV2:
        """定稿计划不带派发目标，其他动作必须明确目标。"""
        if (self.intended_action == "finalize") != (self.intended_target is None):
            raise ValueError("investigation_plan_target_mismatch")
        return self


class CapabilityCardV2(BaseModel):
    """仲裁可见的程序能力快照；实际门卫仍读取程序状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    expert: Literal["fact", "propagation"]
    action_kind: Literal["investigate", "review"]
    case_id: str = Field(min_length=1, pattern=r"\S")
    decision_round: int = Field(gt=0, strict=True)
    fact_coverage: Literal["full", "demo", "none", "mismatched"] | None = None
    propagation_has_data: bool | None = None
    propagation_report_available: bool | None = None
    unreturned_comment_count: int | None = Field(default=None, ge=0, strict=True)
    question_types: list[str]
    available: bool
    remaining_actions: int = Field(ge=0, strict=True)
    action_limit_cny: Decimal = Field(ge=0, strict=True)
    available_balance_cny: Decimal | None = Field(default=None, ge=0, strict=True)
    final_reserve_cny: Decimal = Field(ge=0, strict=True)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def expert_coverage_matches(self) -> CapabilityCardV2:
        """事实语料覆盖和传播数据状态使用各自的字段。"""
        if self.expert == "fact":
            if self.action_kind != "investigate" or self.fact_coverage is None:
                raise ValueError("fact_capability_requires_fact_coverage")
            if any(value is not None for value in (
                    self.propagation_has_data, self.propagation_report_available,
                    self.unreturned_comment_count)):
                raise ValueError("fact_capability_contains_propagation_state")
        elif self.fact_coverage is not None or self.propagation_has_data is None:
            raise ValueError("propagation_capability_state_mismatch")
        return self


class ProposedIssueOriginV2(BaseModel):
    """程序生成的建议来源证明，不接受模型自报 origin。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1, pattern=r"\S")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    fact_round_no: int = Field(gt=0, strict=True)
    created_round_no: int = Field(gt=0, strict=True)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_index: int = Field(ge=0, strict=True)
    evidence_refs: list[str] = Field(min_length=1)
    issue_id: str = Field(min_length=1, pattern=r"\S")
    origin_ref: str = Field(min_length=1, pattern=r"\S")


class IssueTextSourceV2(BaseModel):
    """Program-owned full issue text, bound to the accepted report and issue origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1, pattern=r"\S")
    sample_id: str = Field(min_length=1, pattern=r"\S")
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_id: str = Field(min_length=1, pattern=r"\S")
    origin_ref: str = Field(min_length=1, pattern=r"\S")
    source_kind: Literal["key_fact", "agent_proposed", "propagation"]
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    fact_round_no: int = Field(gt=0, strict=True)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_index: int = Field(ge=0, strict=True)
    full_text: str = Field(min_length=1, pattern=r"\S")
    full_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerifiedQuoteV2(BaseModel):
    """程序核验后保存的原文、坐标、规范化文本与已交付上下文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fact_index: int = Field(ge=0, strict=True)
    quote_index: int = Field(ge=0, strict=True)
    evidence_id: str = Field(min_length=1, pattern=r"\S")
    original_quote: str = Field(min_length=1, pattern=r"\S")
    normalized_text: str = Field(min_length=1, pattern=r"\S")
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    left_context: str
    right_context: str
    ambiguous: bool

    @model_validator(mode="after")
    def valid_range(self) -> VerifiedQuoteV2:
        """程序保存的原文范围必须是非空半开区间。"""
        if self.start >= self.end:
            raise ValueError("quote_original_range_invalid")
        return self


class FactV2ProgramValidation(BaseModel):
    """程序产生的引文与一致性记录，模型输出不能填写该结构。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote_records: list[VerifiedQuoteV2] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def accepted(self) -> bool:
        """无程序核验问题时才允许作为合格结果候选。"""
        return not self.issues


class FactV2ValidationSnapshot(BaseModel):
    """可信材料专用的深层不可变程序核验结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote_records: tuple[VerifiedQuoteV2, ...] = ()
    issues: tuple[str, ...] = ()

    @computed_field
    @property
    def accepted(self) -> bool:
        """无程序核验问题时才允许作为可信快照。"""
        return not self.issues


class FactScopeV2(BaseModel):
    """由程序绑定事实报告所属案件、样本、运行、轮次与原声明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1, pattern=r"\S")
    sample_id: str = Field(min_length=1, pattern=r"\S")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    agent: str = Field(min_length=1, pattern=r"\S")
    fact_round_no: int = Field(gt=0, strict=True)
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FollowupScopeV2(FactScopeV2):
    """补查运行的程序身份、目标问题和原动作许可快照。"""

    action_id: str = Field(min_length=1, pattern=r"\S")
    action_round: int = Field(gt=0, strict=True)
    decision_round: int = Field(gt=0, strict=True)
    followup_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    issue_texts_json: str = Field(min_length=2)
    question: str = Field(min_length=1, max_length=80, pattern=r"\S")
    done_when: Literal["找到一手来源", "确认或否定关键事实", "补充传播证据", "复核指定字段"]
    budget_cny: Decimal = Field(ge=0, allow_inf_nan=False)
    tried_queries: tuple[str, ...] = ()
    claim_proof_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProgramActionClaimV2(BaseModel):
    """由程序动作校验生成的补查许可凭据，不包含模型可填写的许可布尔值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1, pattern=r"\S")
    action_id: str = Field(min_length=1, pattern=r"\S")
    action_round: int = Field(gt=0, strict=True)
    decision_round: int = Field(gt=0, strict=True)
    action_json: str = Field(min_length=2)
    followup_brief_json: str = Field(min_length=2)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    references_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InheritedEvidenceItemV2(BaseModel):
    """一段从旧运行继承的正文、原始完整指纹和已读前缀。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    text: str = Field(min_length=1, pattern=r"\S")
    read_end: int = Field(gt=0, strict=True)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prefix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources_json: str = Field(min_length=2)
    origin_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_body_and_prefix(self) -> InheritedEvidenceItemV2:
        """分开核对完整正文身份与实际已确认前缀。"""
        if self.read_end > len(self.text):
            raise ValueError("inherited_read_prefix_out_of_range")
        if sha256(self.text.encode("utf-8")).hexdigest() != self.body_sha256:
            raise ValueError("inherited_body_sha_mismatch")
        if sha256(self.text[:self.read_end].encode("utf-8")).hexdigest() != self.prefix_sha256:
            raise ValueError("inherited_prefix_sha_mismatch")
        return self


class InheritedEvidencePlanV2(BaseModel):
    """绑定源可信交付、目标补查和新台账编号的不可变继承计划。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_scope: FactScopeV2 | FollowupScopeV2
    target_scope: FollowupScopeV2
    entries: tuple[InheritedEvidenceItemV2, ...] = Field(min_length=1)
    old_to_new_json: str
    source_delivery_json: str = Field(min_length=1)
    source_delivery_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_scope_and_ids(self) -> InheritedEvidencePlanV2:
        """拒绝跨案件、跨题、跨声明继承及新编号重复。"""
        source, target = self.source_scope, self.target_scope
        if (source.case_id != target.case_id or source.sample_id != target.sample_id
                or source.claim_sha256 != target.claim_sha256 or source.run_id == target.run_id
                or source.fact_round_no >= target.fact_round_no):
            raise ValueError("inherited_source_scope_mismatch")
        ids = [item.evidence_id for item in self.entries]
        if ids != list(dict.fromkeys(ids)):
            raise ValueError("inherited_evidence_id_duplicate")
        return self


class FactDeliveredEvidenceV2(BaseModel):
    """从真实请求和成功工具回执重建的正文及确认阅读前缀。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    text: str = Field(min_length=1, pattern=r"\S")
    read_end: int = Field(gt=0, strict=True)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources_json: str


class ValidatedFactSnapshotV2(BaseModel):
    """经程序重验并绑定可信交付的完整事实报告快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: FactScopeV2 | FollowupScopeV2
    report_json: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation: FactV2ValidationSnapshot
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    delivered: tuple[FactDeliveredEvidenceV2, ...]
    initial_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inherited_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_ids: tuple[str, ...]
    archive_manifest_json: str
    legacy_projection_json: str
    legacy_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    historical_import_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class VerifiedFactDeliveryV2(BaseModel):
    """独立于最终判断报告的可信证据交付快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    scope: FactScopeV2 | FollowupScopeV2
    delivered: tuple[FactDeliveredEvidenceV2, ...]
    delivery_request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    initial_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inherited_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_manifest_json: str


class SubmissionReview(BaseModel):
    """程序保存的不可修改判断快照及完整问题，验收结果由问题清单推导。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    verdict_json: str
    issues: tuple[str, ...]
    v2_validation: FactV2ProgramValidation | None = None
    v2_report_json: str | None = None

    @computed_field
    @property
    def accepted(self) -> bool:
        """只有没有校验问题才通过，不能由模型或调用方填写通过标记。"""
        return not self.issues


class SubmissionFeedback(BaseModel):
    """给模型的有界验收反馈，完整问题另存在内部复核结果中。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    accepted: bool
    issues: tuple[str, ...]
    truncated: bool


class SubmissionToolResult(BaseModel):
    """正式提交结果及内部快照，只有feedback可以作为工具内容发送模型。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str
    call_id: str
    reason: str
    feedback: SubmissionFeedback
    review: SubmissionReview | None


class ExpertOpinion(BaseModel):
    """程序为仲裁生成的专家意见，风险与引用质量分别记录。"""

    expert: Literal["fact", "emotion", "spread", "compliance"]
    risk: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    citation_ok: bool = True
    flags: list[str] = Field(default_factory=list)
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentSpec(BaseModel):
    """可信调用方提供的调查专家身份和系统规则，不由模型填写。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    agent: Literal["fact_investigator", "propagation_analyst", "arbiter"]
    system_prompt: str = Field(min_length=1, pattern=r"\S")


class EmotionSignal(BaseModel):
    """声明或已返回评论中的一条情绪信号。"""

    model_config = ConfigDict(extra="forbid")
    quote: str = Field(min_length=1, max_length=_PROPAGATION_LIMITS.text_max_chars)
    source: str = Field(pattern=r"^(claim|C\d+)$")
    kind: Literal["恐慌", "愤怒", "群体对立", "煽动号召", "其他"]


class Lead(BaseModel):
    """传播评论中待核实的线索，不属于事实证据。"""

    model_config = ConfigDict(extra="forbid")
    source: str = Field(pattern=r"^C\d+$")
    quote: str = Field(min_length=1, max_length=_PROPAGATION_LIMITS.text_max_chars)
    claim_to_verify: str = Field(min_length=1, max_length=_PROPAGATION_LIMITS.text_max_chars)
    kind: Literal["提及官方回应", "提及具体数据", "提及时间地点", "质疑声明", "其他"]


class PropagationReport(BaseModel):
    """传播情绪、扩散、时效与待核实线索的模型输出。"""

    model_config = ConfigDict(extra="forbid")
    emotion_level: Literal["低", "中", "高"]
    emotion_signals: list[EmotionSignal] = Field(max_length=_PROPAGATION_LIMITS.emotion_signal_max_items)
    spread_level: Literal["低", "中", "高", "未知"]
    spread_basis: list[str] = Field(default_factory=list)
    narrative: Literal["当前事件", "旧事重提", "无法判断"]
    narrative_basis: list[str] = Field(default_factory=list)
    leads: list[Lead] = Field(default_factory=list, max_length=_PROPAGATION_LIMITS.lead_max_items)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class RationaleItem(BaseModel):
    """仲裁结论的一条理由及其案件内依据编号。"""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=120)
    refs: list[str] = Field(min_length=1)


class CollabAction(BaseModel):
    """仲裁提出的补查或复核动作；是否可执行由程序另行检查。"""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["investigate", "review"]
    target: Literal["fact", "propagation"]
    issue_ids: list[str] = Field(min_length=1, max_length=2)
    question: str = Field(min_length=1, max_length=80)
    done_when: Literal["找到一手来源", "确认或否定关键事实", "补充传播证据", "复核指定字段"]
    lead_ids: list[str] = Field(default_factory=list, max_length=2)
    review_fields: list[Literal["narrative", "emotion_level"]] = Field(default_factory=list)


class ConflictNote(BaseModel):
    """仲裁登记的事实与传播专家意见分歧。"""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=60)
    fact_refs: list[str] = Field(min_length=1)
    prop_refs: list[str] = Field(min_length=1)


class ArbiterDecision(BaseModel):
    """仲裁Agent给出的完整结论和可选下一步动作。"""

    model_config = ConfigDict(extra="forbid")
    evidence_sufficiency: Literal["充分", "不足", "无覆盖"]
    conflicts: list[Literal["关键事实缺证据", "证据之间冲突", "引用异常", "传播数据缺失", "疑似违规信号", "专家判断分歧"]] = Field(default_factory=list)
    next_action: Literal["finalize", "investigate", "review"]
    action: CollabAction | None = None
    register_conflict: ConflictNote | None = None
    adjustment: Literal[-1, 0, 1] = 0
    adjustment_reason: str | None = None
    needs_human_review: bool
    rationale: list[RationaleItem] = Field(min_length=1, max_length=5)


class ArbiterRationaleItemV2(BaseModel):
    """V2仲裁理由与引用数上限，避免无限长的提交。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(min_length=1, max_length=120, pattern=r"\S")
    refs: list[Annotated[str, Field(min_length=1, max_length=48, pattern=r"\S")]] = Field(
        min_length=1, max_length=3)


class ArbiterActionV2(CollabAction):
    """V2仲裁动作沿用旧语义并限制引用标识长度。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    issue_ids: list[Annotated[str, Field(min_length=1, max_length=48, pattern=r"\S")]] = Field(
        min_length=1, max_length=2)
    lead_ids: list[Annotated[str, Field(min_length=1, max_length=48, pattern=r"\S")]] = Field(
        default_factory=list, max_length=2)
    review_fields: list[Literal["narrative", "emotion_level"]] = Field(default_factory=list,
                                                                          max_length=2)


class ArbiterDecisionV2(ArbiterDecision):
    """新仲裁合同：在旧结论字段上增加只记录、不触发动作的计划。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    conflicts: list[Literal["关键事实缺证据", "证据之间冲突", "引用异常", "传播数据缺失",
                            "疑似违规信号", "专家判断分歧"]] = Field(default_factory=list, max_length=6)
    action: ArbiterActionV2 | None = None
    register_conflict: None = None
    adjustment_reason: str | None = Field(default=None, max_length=120)
    rationale: list[ArbiterRationaleItemV2] = Field(min_length=1, max_length=5)
    investigation_plan: InvestigationPlanV2

    @model_validator(mode="after")
    def unique_conflicts(self) -> ArbiterDecisionV2:
        """避免重复冲突项挤占仲裁输出预算。"""
        if len(self.conflicts) != len(set(self.conflicts)):
            raise ValueError("arbiter_v2_duplicate_conflicts")
        return self


class ArbiterDecisionM6c(ArbiterDecisionV2):
    """M6c专用仲裁合同；冻结的M5 V2仍拒绝冲突登记。"""

    register_conflict: ConflictNote | None = None


def build_arbiter_decision_v2_schema(max_text_chars: int) -> type[BaseModel]:
    """Build a strict V2 submission schema using the approved plan text limit."""
    if type(max_text_chars) is not int or max_text_chars < 1:
        raise ValueError("arbiter_plan_text_limit_invalid")
    plan = create_model(
        "ArbiterInvestigationPlanV2",
        __base__=InvestigationPlanV2,
        uncertainty=(str, Field(min_length=1, max_length=max_text_chars, pattern=r"\S")),
        action_note=(str, Field(min_length=1, max_length=max_text_chars, pattern=r"\S")),
        stop_condition=(str, Field(min_length=1, max_length=max_text_chars, pattern=r"\S")),
    )
    return create_model("ArbiterDecisionV2Bounded", __base__=ArbiterDecisionV2,
                        investigation_plan=(plan, ...))


def build_arbiter_decision_m6c_schema(max_text_chars: int) -> type[BaseModel]:
    """仅为显式M6c配置开放已定义的冲突登记字段。"""
    bounded = build_arbiter_decision_v2_schema(max_text_chars)
    return create_model("ArbiterDecisionM6cBounded", __base__=bounded,
                        register_conflict=(ConflictNote | None, None))


class ArbiterResultV2(BaseModel):
    """保留完整V2提交、程序校验、未执行状态和输入身份指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    decision: ArbiterDecisionV2
    action_allowed: bool | None
    action_validation_issues: tuple[str, ...] = ()
    plan_action_mismatch: bool
    action_execution: Literal["not_executed"] = "not_executed"
    stop_reason: str
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    trace_path: str
    case_id: str = Field(min_length=1, pattern=r"\S")
    decision_round: int = Field(gt=0, strict=True)
    material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    references_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArbiterResultM6c(ArbiterResultV2):
    """保存M6c冲突登记的仲裁结果，不改变M5结果合同。"""

    decision: ArbiterDecisionM6c


class IssueEvent(BaseModel):
    """问题台账的一条程序生成历史记录。"""

    model_config = ConfigDict(extra="forbid")
    round: int = Field(gt=0, strict=True)
    action_id: str | None = None
    field: Literal["workflow", "epistemic", "evidence_refs", "key", "created"]
    old: str | None = None
    new: str
    by: Literal["program"]
    refs: list[str] = Field(default_factory=list)


class Issue(BaseModel):
    """案件中待解决的问题，分开保存工作进度和证据状态。"""

    model_config = ConfigDict(extra="forbid")
    issue_id: str = Field(min_length=1)
    text: str = Field(max_length=60)
    kind: Literal["关键事实", "线索", "传播性质", "冲突"]
    origin_refs: list[str]
    owner: Literal["fact", "propagation"]
    key: bool
    workflow: Literal["open", "assigned", "done", "failed", "skipped"]
    epistemic: Literal["unknown", "supported", "refuted", "insufficient", "conflicting", "unavailable"]
    evidence_refs: list[str] = Field(default_factory=list)
    history: list[IssueEvent] = Field(default_factory=list)


class FollowupBrief(BaseModel):
    """程序补全的协作任务单；字段来源对应design_v3 §7.4。"""

    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(min_length=1)
    round: int = Field(gt=0, strict=True)
    budget_cny: Decimal = Field(ge=0)
    target: Literal["fact", "propagation"]
    issue_ids: list[str] = Field(min_length=1, max_length=2)
    issue_texts: dict[str, str]
    question: str = Field(min_length=1, max_length=80)
    leads: dict[str, Lead] = Field(default_factory=dict, max_length=2)
    tried_queries: list[str] = Field(default_factory=list)
    read_evidence_ids: list[str] = Field(default_factory=list)
    returned_comment_ids: list[str] = Field(default_factory=list)
    done_when: Literal["找到一手来源", "确认或否定关键事实", "补充传播证据", "复核指定字段"]


class Progress(BaseModel):
    """程序判定本轮协作是否取得实质证据进展。"""

    model_config = ConfigDict(extra="forbid")
    substantive: bool
    resolved_issue_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    """传播专家对指定解释字段作出的复核意见。"""

    model_config = ConfigDict(extra="forbid")
    field: Literal["narrative", "emotion_level"]
    old_value: str
    decision: Literal["unchanged", "revised", "disagree"]
    new_value: str | None = None
    basis_refs: list[str] = Field(default_factory=list)
    peer_refs: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=120)


class ReviewResult(BaseModel):
    """一项传播复核任务的结构化结果。"""

    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(min_length=1)
    items: list[ReviewItem] = Field(min_length=1, max_length=2)


class PropagationReviewTaskItem(BaseModel):
    """程序交给传播复核Agent的一项字段任务与对方材料。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    field: Literal["narrative", "emotion_level"]
    old_value: str
    current_basis_refs: list[str] = Field(default_factory=list)
    peer_findings: list[str]
    peer_refs: list[str]


class PropagationReviewTask(BaseModel):
    """交叉复核的只读任务区块；不授予Agent写台账或改原始数据权限。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action_id: str = Field(min_length=1)
    items: list[PropagationReviewTaskItem] = Field(min_length=1, max_length=2)


class PropagationReviewRun(BaseModel):
    """带运行身份和费用元数据的合格复核提交。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    run_id: str
    review: ReviewResult
    comments: list[dict[str, Any]]
    stats: dict[str, Any] | None
    stop_reason: Literal["submitted", "forced_final"]
    steps: int = Field(ge=0, strict=True)
    input_output_tokens: int = Field(ge=0, strict=True)
    cost_cny: float = Field(ge=0, allow_inf_nan=False)
    trace_path: str


class TaskResult(BaseModel):
    """一次已派协作任务的执行结果，不把失败或未知费用伪装成成功。"""

    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(min_length=1)
    kind: Literal["investigate", "review"]
    target: Literal["fact", "propagation"]
    status: Literal["completed", "partial", "failed", "skipped", "cancelled"]
    stop_reason: str | None = None
    new_refs: list[str] = Field(default_factory=list)
    changed_fields: list[str] = Field(default_factory=list)
    issue_updates: list[IssueEvent] = Field(default_factory=list)
    validation_flags: dict[str, bool] = Field(default_factory=dict)
    progress: Progress
    usage: dict[str, int | float | Decimal | None]
    trace_path: str | None = None


class RuleHit(BaseModel):
    """程序合规规则的一条命中记录，评论命中可带C编号来源。"""

    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    severity: Literal["高", "中"]
    matched_text: str = Field(min_length=1)
    source: str | None = Field(default=None, pattern=r"^C\d+$")


class ComplianceResult(BaseModel):
    """声明与评论分别保存的程序合规检查结果。"""

    model_config = ConfigDict(extra="forbid")
    claim_status: Literal["not_checked", "checked_no_hit", "checked_with_hits", "check_failed", "rules_empty"]
    claim_hits: list[RuleHit] = Field(default_factory=list)
    comment_status: Literal["not_checked", "checked_no_hit", "checked_with_hits", "check_failed", "rules_empty", "no_comments"]
    comment_hits: list[RuleHit] = Field(default_factory=list)


class CaseRisk(BaseModel):
    """程序计算的案件风险结果、证据充分性和人工复核要求。"""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "system_failure"]
    level: int | None = Field(default=None, ge=1, le=4, strict=True)
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    used_dims: list[Literal["fact", "emotion", "spread", "compliance"]] = Field(default_factory=list)
    absent_dims: dict[str, str] = Field(default_factory=dict)
    weights_used: dict[str, float] = Field(default_factory=dict)
    base_level: int | None = Field(default=None, ge=1, le=4, strict=True)
    adjustment: Literal[-1, 0, 1] = 0
    adjustment_reason: str | None = None
    overrides: list[str] = Field(default_factory=list)
    evidence_sufficiency: Literal["充分", "不足", "无覆盖", "失败"]
    fact_status: Literal["支持", "反驳", "信息不足"] | None = None
    scenario: Literal["general", "content_platform"]
    disposition: str = Field(min_length=1)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_system_failure(self) -> CaseRisk:
        """系统完全失败时不产生正常风险等级或数值分数。"""
        if self.status == "system_failure" and any(
            value is not None for value in (self.level, self.base_level, self.score)
        ):
            raise ValueError("system_failure_has_no_risk_levels_or_score")
        if self.status == "ok" and (self.level is None or self.score is None):
            raise ValueError("completed_risk_missing_level_or_score")
        return self


class PropagationSubmissionFeedback(BaseModel):
    """传播报告验收反馈；通过状态仅由问题列表推导。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    issues: tuple[str, ...]
    _source_issues: tuple[str, ...] = PrivateAttr(default=())

    @classmethod
    def reviewed(cls, issues: tuple[str, ...], source_issues: tuple[str, ...]) -> PropagationSubmissionFeedback:
        """创建面向模型的反馈并把原始校验代码留在模型私有属性中。"""
        feedback = cls(issues=issues)
        object.__setattr__(feedback, "_source_issues", source_issues)
        return feedback

    @property
    def source_issues(self) -> tuple[str, ...]:
        """仅供程序归档的原始校验代码，不进入模型字段或JSON。"""
        return self._source_issues

    @computed_field
    @property
    def accepted(self) -> bool:
        """只有没有校验问题才通过。"""
        return not self.issues


class TaskInput(BaseModel):
    """循环处理的一道题，其编号必须匹配真实运行上下文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str = Field(min_length=1, pattern=r"\S")
    claim: str = Field(min_length=1, pattern=r"\S")


class AgentResult(BaseModel):
    """通过正式提交验收的事实判断与引用片段，不代表语义已获证明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    verdict: FactVerdict
    evidence: list[LedgerEntry]
    stop_reason: Literal["submitted", "forced_final"]
    steps: int = Field(ge=0, strict=True)
    input_output_tokens: int = Field(ge=0, strict=True)
    cost_cny: float = Field(ge=0, allow_inf_nan=False)
    trace_path: str


class AgentResultV2(BaseModel):
    """已通过程序核验的V2事实报告及本运行证据与费用快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    report_json: str
    validation: FactV2ProgramValidation
    evidence: list[LedgerEntry]
    stop_reason: Literal["submitted", "forced_final"]
    steps: int = Field(ge=0, strict=True)
    input_output_tokens: int = Field(ge=0, strict=True)
    cost_cny: float = Field(ge=0, allow_inf_nan=False)
    trace_path: str


class CitationRepairV2(BaseModel):
    """各理由对应的修正证据编号，理由文字保持原样。"""

    model_config = ConfigDict(extra="forbid")
    reasons: list[Reason] = Field(min_length=1)


class CitationRepair(BaseModel):
    """只允许引用修复调用返回理由与编号，保留原标签和评分。"""

    reasons: list[str] = Field(min_length=1)
    citations: list[int] = Field(default_factory=list)


class EmotionCheck(AgentOutput):
    """情绪与煽动性专家的文本信号。"""

    signals: list[str] = Field(default_factory=list)


class SpreadCheck(AgentOutput):
    """传播势能专家的传播特征与评分。"""

    velocity_1h: float = 0.0
    depth: int = 0
    breadth: int = 0
    key_nodes: list[str] = Field(default_factory=list)


class ComplianceCheck(AgentOutput):
    """合规规则命中结果。"""

    hit_rules: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    """仲裁后的风险等级与研判轨迹。"""

    risk_level: int = Field(ge=1, le=4)
    reason: str
    evidence_chain: list[int] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    debate_rounds: int = 0
    retries: int = 0
    confidence: float = Field(ge=0.0, le=1.0)
    needs_more_evidence: bool = False


class PredictionRow(BaseModel):
    """五组对照共用的一条预测记录。"""

    sample_id: str
    group: Literal["B0", "B1", "B2", "B3", "B4"]
    risk_level: int = Field(ge=1, le=4)
    veracity: str | None = None
    latency_ms: int = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_cny: float = Field(default=0.0, ge=0.0)
    degraded: list[str] = Field(default_factory=list)
    window: str = "full"
