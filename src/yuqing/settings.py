"""从项目根目录和环境变量读取运行配置。"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Annotated, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.yuqing.schemas import EvidenceToolPayload, FactLabel, ToolEvidence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """拒绝配置映射中重复出现的键。"""


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode,
                              deep: bool = False) -> dict[object, object]:
    """构造 YAML 字典并在键重复时停止加载。"""
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"duplicate key {key!r}", key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_yaml_no_duplicates(text: str) -> object:
    """读取安全 YAML，并拒绝任何重复映射键。"""
    return yaml.load(text, Loader=_UniqueKeySafeLoader)


class Settings(BaseSettings):
    """集中保存路径、模型和调用上界，避免各模块各写一份默认值。"""

    model_config = SettingsConfigDict(
        env_file=Path(os.environ.get("PROJECTA_ENV_FILE", str(PROJECT_ROOT / ".env"))),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    review_db_path: Path = PROJECT_ROOT / "data" / "review.db"
    checkpoints_db_path: Path = PROJECT_ROOT / "data" / "checkpoints.db"
    case_processing_expiry_minutes: int = Field(default=10, gt=0)
    results_dir: Path = PROJECT_ROOT / "results"
    weights_path: Path = PROJECT_ROOT / "config" / "weights.yaml"
    fact_v2_eval_path: Path = PROJECT_ROOT / "config" / "fact_v2_eval.yaml"
    chef_dev_path: Path = PROJECT_ROOT / "data/CHEF/dev.json"
    chef_test_path: Path = PROJECT_ROOT / "data/CHEF/test.json"
    ced_dataset_path: Path = PROJECT_ROOT / "data/raw/CED_Dataset"
    dev_questions_path: Path = PROJECT_ROOT / "eval/questions_30.jsonl"
    dev_store_path: Path = PROJECT_ROOT / "data/processed/chef_dev_full_chroma"
    test_store_path: Path = PROJECT_ROOT / "data/processed/chef_test_full_chroma"
    model_cache_dir: Path = PROJECT_ROOT / "data" / "processed" / "model_cache"

    deepseek_api_key: SecretStr | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0)
    llm_validation_retries: int = Field(default=1, ge=0)
    llm_thinking: bool = False


class RetrievalConfig(BaseModel):
    """集中校验检索链路的全部可调参数。"""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    embedding_model: str
    embedding_batch_size: int = Field(gt=0)
    embedding_use_fp16: bool
    collection_name: str
    vector_top_k: int = Field(gt=0)
    bm25_top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    final_top_k: int = Field(gt=0)
    rerank_model: str
    rerank_batch_size: int = Field(gt=0)
    rerank_use_fp16: bool
    rerank_query_max_length: int = Field(gt=0)
    rerank_passage_max_length: int = Field(gt=0)
    rerank_candidate_k: int = Field(gt=0)
    questions_per_label: int = Field(gt=0)
    diversify_results: bool
    corpus: Literal["question_only", "chef_dev_full"]


class AgentConfig(BaseModel):
    """集中校验 Agent 输出处理的数值参数。"""

    no_citation_weight_factor: float = Field(gt=0.0, le=1.0)
    citation_repair_retries: int = Field(ge=0)
    single_validation_retries: int = Field(ge=0, strict=True)


class PropagationToolsConfig(BaseModel):
    """传播纯数据工具可见窗口与输出边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    comment_preview_chars: int = Field(gt=0, strict=True)
    sample_max_k: int = Field(gt=0, strict=True)
    search_max_k: int = Field(gt=0, strict=True)
    search_query_max_chars: int = Field(gt=0, strict=True)
    window_hours: tuple[int, int, int]

    @model_validator(mode="after")
    def validate_windows(self) -> PropagationToolsConfig:
        """窗口必须严格递增且都是正整数。"""
        if any(type(value) is not int or value <= 0 for value in self.window_hours):
            raise ValueError("传播统计窗口必须是正整数")
        if tuple(sorted(set(self.window_hours))) != self.window_hours:
            raise ValueError("传播统计窗口必须严格递增")
        return self


class PropagationLimitsConfig(BaseModel):
    """传播报告与首轮步骤的规格上界。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    round1_max_steps: int = Field(gt=0)
    round1_max_cost_cny: float = Field(gt=0, strict=True)
    sample_max_calls: int = Field(gt=0)
    search_max_calls: int = Field(gt=0)
    text_max_chars: int = Field(gt=0)
    emotion_signal_max_items: int = Field(gt=0)
    lead_max_items: int = Field(gt=0)

    @model_validator(mode="after")
    def enforce_spec_upper_bounds(self) -> PropagationLimitsConfig:
        """传播节点只能使用设计规格给出的上界或更紧值。"""
        if (self.round1_max_steps > 4 or self.round1_max_cost_cny > 0.08
                or self.sample_max_calls > 2 or self.search_max_calls > 3
                or self.text_max_chars > 60 or self.emotion_signal_max_items > 5
                or self.lead_max_items > 3):
            raise ValueError("propagation_limits_exceed_design")
        return self


class FactRiskRule(BaseModel):
    """配置中的线性风险规则，整个置信度区间都必须落在零至一。"""

    base: float = Field(ge=0.0, le=1.0)
    slope: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_risk_range(self) -> FactRiskRule:
        """验证置信度为一时的端点，配合 base 边界约束守住整个区间。"""

        if not 0 <= self.base + self.slope <= 1:
            raise ValueError("fact_risk 在置信度区间内会算出超出 0 到 1 的风险")
        return self


class ArbiterScoreConfig(BaseModel):
    """校验程序计分默认值及其类别覆盖完整性。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    weights: dict[Literal["fact", "emotion", "spread", "compliance"], float]
    level_map: dict[Literal["emotion", "spread"], dict[Literal["低", "中", "高"], float]]
    compliance_score: dict[Literal["checked_no_hit", "中", "高"], float]
    thresholds: tuple[float, float, float]
    review_level: int = Field(ge=1, le=4, strict=True)
    compliance_high_min_level: int = Field(ge=1, le=4, strict=True)

    @model_validator(mode="after")
    def validate_score_tables(self) -> ArbiterScoreConfig:
        """计分权重必须齐全且合计为一，分界值严格递增。"""
        expected = {"fact", "emotion", "spread", "compliance"}
        if (set(self.weights) != expected or any(value <= 0 for value in self.weights.values())
                or abs(sum(self.weights.values()) - 1) > 1e-9):
            raise ValueError("arbiter weights must contain four dimensions summing to one")
        if any(set(table) != {"低", "中", "高"} for table in self.level_map.values()):
            raise ValueError("arbiter level map must define low, medium and high")
        if set(self.level_map) != {"emotion", "spread"} or set(self.compliance_score) != {
            "checked_no_hit", "中", "高"
        }:
            raise ValueError("arbiter score tables are incomplete")
        if tuple(sorted(set(self.thresholds))) != self.thresholds:
            raise ValueError("arbiter thresholds must be strictly increasing")
        if any(value < 0 or value > 1 for value in (*self.weights.values(),
            *self.compliance_score.values(), *self.thresholds,
            *(item for table in self.level_map.values() for item in table.values()))):
            raise ValueError("arbiter score values must be within zero and one")
        return self


class ScenarioConfig(BaseModel):
    """生效场景只能是已配置的场景之一。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    active: Literal["general", "content_platform"]


class CaseBudgetConfig(BaseModel):
    """校验整案预算、节点分配和动作池之间的静态关系。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    max_cost_cny: float = Field(gt=0, strict=True)
    plan: dict[str, float]
    action_caps: dict[str, float]
    final_decision_reserve: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_budget_plan(self) -> CaseBudgetConfig:
        """预算计划必须完整、非负且不超过整案额度。"""
        if set(self.plan) != {"fact_round1", "propagation_round1", "arbiter_decision", "action_pool"}:
            raise ValueError("case budget plan is incomplete")
        if set(self.action_caps) != {"fact_investigate", "propagation_investigate", "propagation_review"}:
            raise ValueError("case action caps are incomplete")
        values = [*self.plan.values(), *self.action_caps.values(), self.final_decision_reserve]
        if any(value < 0 for value in values) or sum(self.plan.values()) > self.max_cost_cny:
            raise ValueError("case budget plan exceeds or invalidates its case limit")
        if self.final_decision_reserve != self.plan["arbiter_decision"]:
            raise ValueError("case final reserve must match one arbiter decision allocation")
        return self


class InvestigationPlanConfig(BaseModel):
    """仲裁计划中每段说明文字的配置确定上限。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_text_chars: int = Field(gt=0, strict=True)


class CollabConfig(BaseModel):
    """基础协作功能开关；增强能力须由后续卡单独启用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_actions: int = Field(ge=1, le=2, strict=True)
    targets: list[Literal["fact", "propagation"]]
    review_enabled: bool
    generic_followup: bool
    investigation_plan: InvestigationPlanConfig

    @model_validator(mode="after")
    def validate_targets(self) -> CollabConfig:
        """拒绝重复目标或没有可执行目标的协作配置。"""
        if not self.targets or len(self.targets) != len(set(self.targets)):
            raise ValueError("collab targets must be nonempty and unique")
        if self.review_enabled and "propagation" not in self.targets:
            raise ValueError("collab review requires propagation target")
        return self


class FactV2LimitsConfig(BaseModel):
    """事实 V2 输出字段的单一配置来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    max_key_facts: int = Field(gt=0, strict=True)
    max_key_fact_chars: int = Field(gt=0, strict=True)
    max_quotes_per_direction: int = Field(gt=0, strict=True)
    max_quotes_per_report: int = Field(gt=0, strict=True)
    min_normalized_quote_chars: int = Field(gt=0, strict=True)
    max_quote_chars: int = Field(gt=0, strict=True)
    max_reasons: int = Field(gt=0, strict=True)
    max_reason_chars: int = Field(gt=0, strict=True)
    quote_context_chars: int = Field(ge=0, strict=True)
    max_proposed_issues: int = Field(ge=0, strict=True)
    max_proposed_question_chars: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_related_limits(self) -> FactV2LimitsConfig:
        """确保引文长度和总量上限内部一致。"""
        if self.min_normalized_quote_chars > self.max_quote_chars:
            raise ValueError("fact_v2_quote_length_limits_mismatch")
        if self.max_quotes_per_direction > self.max_quotes_per_report:
            raise ValueError("fact_v2_quote_count_limits_mismatch")
        return self


class FactV2Config(BaseModel):
    """事实 V2 相关设置的严格配置节。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    validation_sampling_seed: int = Field(ge=0, strict=True)
    limits: FactV2LimitsConfig


class FactV2EvaluationConfig(BaseModel):
    """F2评测专用候选预算；与历史E2共享配置隔离。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    input_estimate_version: Literal["json-chars-v1"]
    input_bound_margin_tokens: int = Field(ge=0, strict=True)
    token_budget_per_call: int = Field(gt=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    final_token_reserve: int = Field(gt=0, strict=True)
    max_total_tokens: int = Field(gt=0, strict=True)
    max_cost_cny: float = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_candidate_capacity(self) -> FactV2EvaluationConfig:
        """候选余量必须容纳一次普通请求和一次完整收尾请求。"""
        if self.final_token_reserve != self.token_budget_per_call + self.max_output_tokens:
            raise ValueError("fact_v2_invalid_final_reserve")
        if self.max_total_tokens <= self.final_token_reserve:
            raise ValueError("fact_v2_total_tokens_below_final_reserve")
        return self


class EvidenceSourceConfig(BaseModel):
    """单个证据来源的索引标识与覆盖状态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    index: str | None
    coverage: Literal["full", "demo", "none", "mismatched"]


class DispositionSuffixConfig(BaseModel):
    """合规命中与无覆盖情形的固定追加说明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    violation: str = Field(min_length=1)
    unverified: str = Field(min_length=1)


class EvaluationConfig(BaseModel):
    """固定事实核查验收选样及来源复核文件。"""

    gold_review_path: Path
    fact_check_prompt_path: Path
    fact_check_counts: dict[Literal["真", "假", "无法判定"], Annotated[int, Field(ge=0)]]
    acceptance_min_correct: int = Field(gt=0)
    dataset_per_label: int = Field(gt=0, strict=True)
    dev_sampling_seed: int = Field(strict=True)
    retrieval_metrics_k: int = Field(gt=0, strict=True)
    max_run_cost_cny: float = Field(gt=0, strict=True, allow_inf_nan=False)
    pilot_max_run_cost_cny: float = Field(gt=0, strict=True, allow_inf_nan=False)
    worker_wait_timeout_s: float = Field(gt=0, strict=True, allow_inf_nan=False)
    pilot_max_median_cost_cny: float = Field(gt=0, strict=True, allow_inf_nan=False)
    pilot_max_no_valid_results: int = Field(ge=0, strict=True)
    pilot_max_reservation_exceeded: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_experiment_limits(self) -> EvaluationConfig:
        """试跑上限只能收紧正式上限，金额必须显式配置。"""
        if self.pilot_max_run_cost_cny > self.max_run_cost_cny:
            raise ValueError("试跑费用上限不能超过正式上限")
        return self


class CostConfig(BaseModel):
    """人民币每百万 token 单价；两项都未填表示尚未定价。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_per_1m: float | None = Field(default=None, ge=0)
    output_per_1m: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_prices(self) -> CostConfig:
        """拒绝只填一项单价，避免把未填的另一项当成免费。"""

        if (self.input_per_1m is None) != (self.output_per_1m is None):
            raise ValueError("cost 输入与输出单价必须一起填写或一起留空")
        return self


class PolicyConfig(BaseModel):
    """工具权限名单、登记风险等级与外部动作模式，缺失字段不回退。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    allow: dict[str, list[str]]
    risk_tier: dict[str, Literal["R0", "R1", "R2"]]
    r2_mode: Literal["ask", "dry_run", "deny"]

    @model_validator(mode="after")
    def validate_allowlists(self) -> PolicyConfig:
        """拒绝重复、未知工具和给模型专家配置外部动作权限。"""
        for agent, names in self.allow.items():
            if not agent or len(names) != len(set(names)) or any(name not in self.risk_tier for name in names):
                raise ValueError("policy 权限名单含空身份、重复或未知工具")
            if agent != "alert_gate" and any(self.risk_tier[name] == "R2" for name in names):
                raise ValueError("R2 工具只能授予 alert_gate 程序节点")
        return self


class HarnessConfig(BaseModel):
    """每位专家单次任务的预算与工具超时，所有数值由配置提供。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    submission_tool: ClassVar[str] = "submit_verdict"
    input_estimate_version: Literal["json-chars-v1"]
    input_bound_margin_tokens: int = Field(ge=0, strict=True)
    repeat_query_limit: int = Field(gt=0, strict=True)
    no_new_evidence_limit: int = Field(gt=0, strict=True)
    max_consecutive_tool_errors: int = Field(gt=0, strict=True)
    max_format_nudges: int = Field(ge=0, strict=True)
    max_steps: int = Field(gt=0, strict=True)
    max_tool_calls: dict[str, Annotated[int, Field(gt=0, strict=True)]]
    token_budget_per_call: int = Field(gt=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    max_total_tokens: int = Field(gt=0, strict=True)
    max_cost_cny: float = Field(gt=0)
    final_token_reserve: int = Field(gt=0, strict=True)
    tool_timeout_s: dict[str, Annotated[float, Field(gt=0, strict=True)]]
    search_k: int = Field(gt=0, strict=True)
    gist_chars: int = Field(gt=0, strict=True)
    max_tool_result_chars: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_reserves(self) -> HarnessConfig:
        """预留完整一次模型调用，并确保工具次数与超时名单对应。"""
        from src.yuqing.schemas import EvidenceToolPayload, ToolEvidence

        if self.final_token_reserve != self.token_budget_per_call + self.max_output_tokens:
            raise ValueError("收尾token预留必须等于单次输入与输出上限之和")
        if self.max_total_tokens <= self.final_token_reserve:
            raise ValueError("总token预算必须大于收尾预留")
        if set(self.tool_timeout_s) != set(self.max_tool_calls) | {"submit_verdict"}:
            raise ValueError("工具超时名单必须覆盖所有限次工具及submit_verdict")
        minimum = EvidenceToolPayload(status="error", reason="result_limit", items=[
            ToolEvidence(evidence_id="E1", text="", truncated=True, read_end=0),
        ])
        if self.max_tool_result_chars < len(minimum.model_dump_json()):
            raise ValueError("工具结果上限不足以容纳最小结构")
        return self


class ArbiterHarnessConfig(BaseModel):
    """调用方显式提供的仲裁预算；不引入未经批准的生产默认数值。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    submission_tool: ClassVar[str] = "submit_decision"
    input_estimate_version: Literal["json-chars-v1"]
    input_bound_margin_tokens: int = Field(ge=0, strict=True)
    max_steps: int = Field(gt=0, le=3, strict=True)
    max_tool_calls: dict[str, Annotated[int, Field(gt=0, strict=True)]]
    token_budget_per_call: int = Field(gt=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    max_total_tokens: int = Field(gt=0, strict=True)
    max_cost_cny: float = Field(gt=0, strict=True)
    final_token_reserve: int = Field(gt=0, strict=True)
    tool_timeout_s: dict[str, Annotated[float, Field(gt=0, strict=True)]]
    max_tool_result_chars: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_reserves(self) -> ArbiterHarnessConfig:
        """限定只提交，并要求完整收尾预留。"""
        if self.max_tool_calls or set(self.tool_timeout_s) != {self.submission_tool}:
            raise ValueError("arbiter_submission_only")
        if self.final_token_reserve != self.token_budget_per_call + self.max_output_tokens:
            raise ValueError("arbiter_invalid_final_reserve")
        if self.max_total_tokens <= self.final_token_reserve:
            raise ValueError("arbiter_insufficient_total_tokens")
        return self


class M4RunBudgetConfig(BaseModel):
    """仅限M4五样本离线与正式运行使用的显式预算组合。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    experiment_max_cost_cny: float = Field(gt=0, strict=True)
    case_budget: CaseBudgetConfig
    arbiter_harness: ArbiterHarnessConfig

    @model_validator(mode="after")
    def validate_allocations(self) -> M4RunBudgetConfig:
        """仲裁请求上限必须由M4案件计划完整覆盖。"""
        if self.arbiter_harness.max_cost_cny > self.case_budget.plan["arbiter_decision"]:
            raise ValueError("M4仲裁节点额度超过案件分配")
        if self.case_budget.plan["arbiter_decision"] != self.case_budget.final_decision_reserve:
            raise ValueError("M4最终决策预留与节点计划不一致")
        return self


class M5RunBudgetConfig(BaseModel):
    """仅供M5图运行使用的独立案件和仲裁预算组合。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    case_budget: CaseBudgetConfig
    arbiter_harness: ArbiterHarnessConfig

    @model_validator(mode="after")
    def validate_allocations(self) -> M5RunBudgetConfig:
        """确保仲裁请求上限由M5案件中的单次决策分配覆盖。"""
        if self.arbiter_harness.max_cost_cny > self.case_budget.plan["arbiter_decision"]:
            raise ValueError("M5仲裁节点额度超过案件分配")
        if self.case_budget.plan["arbiter_decision"] != self.case_budget.final_decision_reserve:
            raise ValueError("M5最终决策预留与节点计划不一致")
        validate_case_budget_plan(self.case_budget, get_collab_config())
        return self


class M6cRunBudgetConfig(BaseModel):
    """交叉复核独立预算档；不改变M5默认配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    graph_recursion_limit: int = Field(gt=0, strict=True)
    case_budget: CaseBudgetConfig
    arbiter_harness: ArbiterHarnessConfig
    collab: CollabConfig

    @model_validator(mode="after")
    def validate_m6c_allocations(self) -> M6cRunBudgetConfig:
        """确保三次仲裁、两次动作和复核开关均由独立额度覆盖。"""
        if (self.collab.max_actions != 2 or not self.collab.review_enabled
                or set(self.collab.targets) != {"fact", "propagation"}
                or self.arbiter_harness.max_cost_cny > self.case_budget.plan["arbiter_decision"]
                or self.case_budget.plan["arbiter_decision"] != self.case_budget.final_decision_reserve):
            raise ValueError("M6c预算档与协作合同不匹配")
        validate_case_budget_plan(self.case_budget, self.collab)
        if self.graph_recursion_limit <= 3 * (self.collab.max_actions + 1) + 10:
            raise ValueError("m6c_graph_recursion_limit_too_low")
        return self


class GraphConfig(BaseModel):
    """M5图只允许登记的事实合同版本与经过校验的递归保护上限。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    fact_node_version: Literal["v2", "v1_1"]
    recursion_limit: int = Field(gt=0, strict=True)
    checkpointer: Literal["memory", "sqlite"] = "memory"

    @model_validator(mode="after")
    def validate_recursion_limit(self) -> GraphConfig:
        """递归保护阈值必须严格高于动作循环的规格公式。"""
        actions = get_collab_config().max_actions
        if self.recursion_limit <= 3 * (actions + 1) + 10:
            raise ValueError("graph_recursion_limit_too_low")
        return self


class PropagationHarnessConfig(BaseModel):
    """传播节点专用预算配置，不包含事实检索参数或可配置提交身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    submission_tool: ClassVar[str] = "submit_propagation_report"
    input_estimate_version: Literal["json-chars-v1"]
    input_bound_margin_tokens: int = Field(ge=0, strict=True)
    max_consecutive_tool_errors: int = Field(gt=0, strict=True)
    max_format_nudges: int = Field(ge=0, strict=True)
    max_steps: int = Field(gt=0, strict=True)
    max_tool_calls: dict[str, Annotated[int, Field(gt=0, strict=True)]]
    token_budget_per_call: int = Field(gt=0, strict=True)
    max_output_tokens: int = Field(gt=0, strict=True)
    max_total_tokens: int = Field(gt=0, strict=True)
    max_cost_cny: float = Field(gt=0, strict=True)
    final_token_reserve: int = Field(gt=0, strict=True)
    tool_timeout_s: dict[str, Annotated[float, Field(gt=0, strict=True)]]
    max_tool_result_chars: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_reserves(self) -> PropagationHarnessConfig:
        """校验收尾容量、传播调查工具及固定提交工具的键名。"""
        if self.final_token_reserve != self.token_budget_per_call + self.max_output_tokens:
            raise ValueError("收尾token预留必须等于单次输入与输出上限之和")
        if self.max_total_tokens <= self.final_token_reserve:
            raise ValueError("总token预算必须大于收尾预留")
        if set(self.max_tool_calls) != {"get_spread_stats", "sample_comments", "search_comments"}:
            raise ValueError("传播工具限次名单不匹配")
        if set(self.tool_timeout_s) != set(self.max_tool_calls) | {self.submission_tool}:
            raise ValueError("传播超时名单必须覆盖调查工具及提交工具")
        return self


class PropagationReviewHarnessConfig(PropagationHarnessConfig):
    """复核专用传播工具配置；只替换最终提交工具身份。"""

    submission_tool: ClassVar[str] = "submit_review"


@cache
def get_m6c_arbiter_harness_config() -> ArbiterHarnessConfig:
    """读取M6c仲裁请求上限。"""
    return get_m6c_run_budget_config().arbiter_harness


@cache
def get_m6c_case_budget_config() -> CaseBudgetConfig:
    """读取M6c单案预算与动作池。"""
    return get_m6c_run_budget_config().case_budget


@cache
def get_m6c_collab_config() -> CollabConfig:
    """读取M6c显式增强开关，不改变基础协作默认值。"""
    return get_m6c_run_budget_config().collab


@cache
def get_m6c_review_harness_config() -> PropagationReviewHarnessConfig:
    """从现有传播限额派生M6c复核额度，工具超时沿用同一已确认数值。"""
    base = get_propagation_harness_config().model_dump()
    cap = get_m6c_case_budget_config().action_caps["propagation_review"]
    timeouts = {key: value for key, value in base["tool_timeout_s"].items()
                if key != "submit_propagation_report"}
    timeouts[PropagationReviewHarnessConfig.submission_tool] = base["tool_timeout_s"][
        "submit_propagation_report"]
    return PropagationReviewHarnessConfig.model_validate({**base,
        "max_cost_cny": cap, "tool_timeout_s": timeouts})


@cache
def get_harness_config() -> HarnessConfig:
    """读取并校验预算配置，缺失字段和未知键直接报错。"""
    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "harness" not in content:
        raise ValueError("config/weights.yaml 缺少 harness 配置")
    return HarnessConfig.model_validate(content["harness"])


@cache
def get_policy_config() -> PolicyConfig:
    """从权重文件读取并校验工具权限配置，不猜测缺失字段。"""
    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "policy" not in content:
        raise ValueError("config/weights.yaml 缺少 policy 配置")
    return PolicyConfig.model_validate(content["policy"])


@cache
def get_cost_config() -> CostConfig:
    """读取费用配置，未定价使用 null，不猜测实际模型单价。"""

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "cost" not in content:
        raise ValueError("config/weights.yaml 缺少 cost 配置")
    return CostConfig.model_validate(content["cost"])


@cache
def get_settings() -> Settings:
    """按需加载并复用同一份运行配置。"""

    return Settings()


@cache
def get_retrieval_config() -> RetrievalConfig:
    """从统一的权重文件读取并校验检索参数。"""

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "retrieval" not in content:
        raise ValueError("config/weights.yaml 缺少 retrieval 配置")
    return RetrievalConfig.model_validate(content["retrieval"])


@cache
def get_agent_config() -> AgentConfig:
    """从统一权重文件读取并校验 Agent 参数。"""

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "agent" not in content:
        raise ValueError("config/weights.yaml 缺少 agent 配置")
    return AgentConfig.model_validate(content["agent"])


@cache
def get_propagation_tools_config() -> PropagationToolsConfig:
    """读取传播数据工具的规格上限。"""

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "propagation_tools" not in content:
        raise ValueError("config/weights.yaml 缺少 propagation_tools 配置")
    return PropagationToolsConfig.model_validate(content["propagation_tools"])


@cache
def get_propagation_limits_config() -> PropagationLimitsConfig:
    """读取传播报告和首轮循环的已定规格上界。"""

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "propagation_limits" not in content:
        raise ValueError("config/weights.yaml 缺少 propagation_limits 配置")
    return PropagationLimitsConfig.model_validate(content["propagation_limits"])


@cache
def get_propagation_harness_config() -> PropagationHarnessConfig:
    """读取作者明确提供的传播节点预算，不为缺失配置猜默认值。"""

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "propagation_harness" not in content:
        raise ValueError("config/weights.yaml 缺少 propagation_harness 配置")
    return PropagationHarnessConfig.model_validate(content["propagation_harness"])


@cache
def get_fact_risk_config() -> dict[FactLabel, FactRiskRule]:
    """读取已确认的三类风险规则，缺失或非法配置直接拒绝而不回退默认值。"""
    from src.yuqing.schemas import FactLabel

    content = yaml.safe_load(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "fact_risk" not in content:
        raise ValueError("config/weights.yaml 缺少 fact_risk 配置")
    rules = TypeAdapter(dict[FactLabel, FactRiskRule]).validate_python(content["fact_risk"])
    if set(rules) != set(get_args(FactLabel)):
        raise ValueError("fact_risk 必须同时包含支持、反驳和信息不足三类规则")
    return rules


@cache
def get_arbiter_score_config() -> ArbiterScoreConfig:
    """读取计分表并拒绝缺项、越界值或权重未归一。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "arbiter" not in content:
        raise ValueError("weights.yaml 缺少 arbiter 配置")
    return ArbiterScoreConfig.model_validate(content["arbiter"])


@cache
def get_scenario_config() -> ScenarioConfig:
    """读取默认场景配置。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "scenario" not in content:
        raise ValueError("weights.yaml 缺少 scenario 配置")
    return ScenarioConfig.model_validate(content["scenario"])


@cache
def get_case_budget_config() -> CaseBudgetConfig:
    """读取并校验整案预算的静态分配。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "case" not in content:
        raise ValueError("weights.yaml 缺少 case 配置")
    return CaseBudgetConfig.model_validate(content["case"])


@cache
def get_m4_run_budget_config() -> M4RunBudgetConfig:
    """读取仅用于五样本M4运行的共享、案件和节点预算组合。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "m4" not in content:
        raise ValueError("weights.yaml 缺少 m4 配置")
    return M4RunBudgetConfig.model_validate(content["m4"])


@cache
def get_m4_arbiter_harness_config() -> ArbiterHarnessConfig:
    """读取M4专用仲裁请求预算，不影响旧E2及默认仲裁路径。"""
    return get_m4_run_budget_config().arbiter_harness


@cache
def get_m4_case_budget_config() -> CaseBudgetConfig:
    """读取M4五样本专用案件预算。"""
    return get_m4_run_budget_config().case_budget


@cache
def get_m5_run_budget_config() -> M5RunBudgetConfig:
    """读取独立M5配置，不复用默认或M4预算档。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "m5" not in content:
        raise ValueError("weights.yaml 缺少 m5 配置")
    return M5RunBudgetConfig.model_validate(content["m5"])


@cache
def get_m6c_run_budget_config() -> M6cRunBudgetConfig:
    """读取交叉复核专用预算，不复用或覆盖M5额度。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "m6c" not in content:
        raise ValueError("weights.yaml 缺少 m6c 配置")
    return M6cRunBudgetConfig.model_validate(content["m6c"])


@cache
def get_m5_arbiter_harness_config() -> ArbiterHarnessConfig:
    """读取M5仲裁节点的独立请求额度。"""
    return get_m5_run_budget_config().arbiter_harness


@cache
def get_m5_case_budget_config() -> CaseBudgetConfig:
    """读取M5案件的独立总额和预算分配。"""
    return get_m5_run_budget_config().case_budget


@cache
def get_graph_config() -> GraphConfig:
    """读取M5图版本与递归兜底阈值。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "graph" not in content:
        raise ValueError("weights.yaml 缺少 graph 配置")
    return GraphConfig.model_validate(content["graph"])


@cache
def get_m6c_graph_config() -> GraphConfig:
    """仅为显式M6c运行替换配置确定的图递归上限。"""
    return GraphConfig.model_validate({**get_graph_config().model_dump(),
        "recursion_limit": get_m6c_run_budget_config().graph_recursion_limit})


@cache
def get_collab_config() -> CollabConfig:
    """读取基础协作开关。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "collab" not in content:
        raise ValueError("weights.yaml 缺少 collab 配置")
    return CollabConfig.model_validate(content["collab"])


@cache
def get_fact_v2_config() -> FactV2Config:
    """读取 V2 Schema 和引文校验共用的确认限制。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "fact_v2" not in content:
        raise ValueError("weights.yaml 缺少 fact_v2 配置")
    return FactV2Config.model_validate(content["fact_v2"])


@cache
def get_fact_v2_limits_config() -> FactV2LimitsConfig:
    """返回唯一来源中的 V2 字段限制。"""
    return get_fact_v2_config().limits


@cache
def get_fact_v2_evaluation_config() -> FactV2EvaluationConfig:
    """从独立YAML加载事实V2评测预算，不触碰旧E2配置。"""
    config_path = get_settings().fact_v2_eval_path
    content = load_yaml_no_duplicates(config_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("fact_v2_eval_config_must_be_mapping")
    return FactV2EvaluationConfig.model_validate(content)


@cache
def get_disposition_profiles() -> dict[str, dict[str, dict[str, str]]]:
    """读取两种场景的完整建议动作表，不对缺格作默认回退。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    raw = content.get("disposition_profiles") if isinstance(content, dict) else None
    return validate_disposition_profiles(raw)


def validate_disposition_profiles(raw: object) -> dict[str, dict[str, dict[str, str]]]:
    """校验显式传入或文件读取的全部建议动作表格。"""
    scenarios = {"general", "content_platform"}
    levels = {"system_failure", "1", "2", "3", "4"}
    labels = {"支持", "反驳", "信息不足"}
    if not isinstance(raw, dict) or set(raw) != scenarios:
        raise ValueError("disposition_profiles 场景缺失或未知")
    for scenario, table in raw.items():
        if not isinstance(table, dict) or set(table) != levels:
            raise ValueError(f"disposition_profiles {scenario} 等级缺失或未知")
        for level, row in table.items():
            if not isinstance(row, dict) or set(row) != labels or any(
                not isinstance(action, str) or not action.strip() for action in row.values()
            ):
                raise ValueError(f"disposition_profiles {scenario}/{level} 表格缺项")
            if any(term in row["支持"] for term in ("澄清", "辟谣", "降低推荐", "限流", "下架")):
                raise ValueError(f"disposition_profiles {scenario}/{level} 对支持事实包含不当干预")
    return raw


@cache
def get_disposition_suffix_config() -> DispositionSuffixConfig:
    """读取固定的合规与无覆盖追加说明。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "disposition_suffix" not in content:
        raise ValueError("weights.yaml 缺少 disposition_suffix 配置")
    return DispositionSuffixConfig.model_validate(content["disposition_suffix"])


def validate_case_budget_plan(budget: CaseBudgetConfig, collab: CollabConfig) -> None:
    """验证静态分配与当前动作数下的最坏费用都不超过整案上限。"""
    plan = budget.plan
    static = plan["fact_round1"] + plan["propagation_round1"] + (
        collab.max_actions + 1
    ) * plan["arbiter_decision"] + plan["action_pool"]
    worst_action = max(budget.action_caps.values())
    worst = plan["fact_round1"] + plan["propagation_round1"] + (
        collab.max_actions + 1
    ) * plan["arbiter_decision"] + collab.max_actions * worst_action
    if static > budget.max_cost_cny or worst > budget.max_cost_cny:
        raise ValueError("case budget plan exceeds the configured case limit")
    if collab.max_actions * worst_action > plan["action_pool"]:
        raise ValueError("case action pool cannot fund the configured number of actions")


@cache
def get_evidence_sources_config() -> dict[str, EvidenceSourceConfig]:
    """读取证据来源清单并验证每个来源的覆盖状态。"""
    content = load_yaml_no_duplicates(get_settings().weights_path.read_text(encoding="utf-8"))
    raw = content.get("evidence_sources") if isinstance(content, dict) else None
    if not isinstance(raw, dict) or set(raw) != {"chef_dev", "chef_test", "demo", "ced", "free_ticket"}:
        raise ValueError("weights.yaml 缺少 evidence_sources")
    return {name: EvidenceSourceConfig.model_validate(value) for name, value in raw.items()}


@cache
def get_evaluation_config() -> EvaluationConfig:
    """读取验收配置，并把相对文件路径解析到项目根目录。"""

    settings = get_settings()
    content = yaml.safe_load(settings.weights_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or "evaluation" not in content:
        raise ValueError("config/weights.yaml 缺少 evaluation 配置")
    config = EvaluationConfig.model_validate(content["evaluation"])
    if config.acceptance_min_correct > sum(config.fact_check_counts.values()):
        raise ValueError("验收正确数要求超过配置的题目数")
    return config.model_copy(update={
        "gold_review_path": (settings.project_root / config.gold_review_path).resolve(),
        "fact_check_prompt_path": (settings.project_root / config.fact_check_prompt_path).resolve(),
    })


def configure_model_cache() -> Path:
    """在导入模型库前把 Hugging Face 与 PyTorch 缓存定向到项目数据目录。"""

    root = get_settings().model_cache_dir.expanduser().resolve()
    folders = {
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "HF_XET_CACHE": root / "huggingface" / "xet",
        "TORCH_HOME": root / "torch",
        "SENTENCE_TRANSFORMERS_HOME": root / "sentence_transformers",
    }
    for variable, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(folder)
    return root
