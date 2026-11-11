"""把已校验的专家结果合成为程序风险和建议动作。"""

from __future__ import annotations

from typing import Literal

from src.yuqing.schemas import CaseRisk, ComplianceResult, FactLabel
from src.yuqing.agents.arbiter_loop import run_arbiter_agent
from src.yuqing.settings import (
    ArbiterScoreConfig,
    DispositionSuffixConfig,
    ScenarioConfig,
    get_arbiter_score_config,
    get_disposition_profiles,
    get_disposition_suffix_config,
    get_scenario_config,
    validate_disposition_profiles,
)


def score_case(*, fact_status: FactLabel | None, fact_risk: float | None,
               fact_coverage: Literal["full", "demo", "none", "mismatched"],
               fact_failed: bool, fact_citation_ok: bool, emotion_level: str | None,
               spread_level: str | None, compliance: ComplianceResult,
               unresolved_key_conflict: bool = False, review_disagree: bool = False,
               arbiter_requests_review: bool = False,
               adjustment: Literal[-1, 0, 1] = 0, adjustment_reason: str | None = None,
               scenario: str | None = None, score_config: ArbiterScoreConfig | None = None,
               disposition_profiles: dict[str, dict[str, dict[str, str]]] | None = None,
               suffix_config: DispositionSuffixConfig | None = None,
               scenario_config: ScenarioConfig | None = None) -> CaseRisk:
    """以显式规则配置进行纯计分；不调用模型、不执行建议动作。"""
    scoring = score_config or get_arbiter_score_config()
    profiles = (validate_disposition_profiles(disposition_profiles)
                if disposition_profiles is not None else get_disposition_profiles())
    suffixes = suffix_config if suffix_config is not None else get_disposition_suffix_config()
    active = ((scenario_config if scenario_config is not None else get_scenario_config()).active
              if scenario is None else scenario)
    if fact_status not in {None, "支持", "反驳", "信息不足"}:
        raise ValueError("invalid_fact_status")
    if fact_failed and (fact_status is not None or fact_risk is not None):
        raise ValueError("fact_failed_with_result")
    if fact_risk is not None and not 0 <= fact_risk <= 1:
        raise ValueError("fact_risk_out_of_range")
    if adjustment not in {-1, 0, 1}:
        raise ValueError("adjustment_out_of_range")
    if active not in profiles or (adjustment and not adjustment_reason):
        raise ValueError("invalid_scenario_or_adjustment_reason")
    sufficiency = _evidence_sufficiency(fact_status, fact_risk, fact_coverage, fact_failed,
                                       fact_citation_ok, unresolved_key_conflict)
    dimensions, absent = _dimensions(fact_status, fact_risk, fact_coverage, fact_failed,
                                     emotion_level, spread_level, compliance, scoring)
    if "fact" not in dimensions and not {"emotion", "spread"}.intersection(dimensions):
        return _system_failure(active, profiles, suffixes, absent, sufficiency, compliance)
    score, used_weights = _weighted_score(dimensions, scoring)
    base_level = _level_for(score, scoring)
    level = _adjust_level(base_level, adjustment, scoring)
    level, overrides = _apply_hard_floor(level, compliance, scoring)
    fact_column = fact_status if sufficiency == "充分" else "信息不足"
    disposition = _disposition(profiles, active, str(level), fact_column, compliance,
                               sufficiency, suffixes)
    review_reasons = _review_reasons(level, sufficiency, fact_coverage, fact_failed,
                                     fact_citation_ok, unresolved_key_conflict,
                                     review_disagree, arbiter_requests_review, compliance, scoring)
    return CaseRisk(status="ok", level=level, score=score, used_dims=list(dimensions),
        absent_dims=absent, weights_used=used_weights, base_level=base_level,
        adjustment=adjustment, adjustment_reason=adjustment_reason,
        overrides=overrides, evidence_sufficiency=sufficiency,
        fact_status=fact_status, scenario=active, disposition=disposition,
        needs_human_review=bool(review_reasons), review_reasons=review_reasons)


def _evidence_sufficiency(fact_status: FactLabel | None, fact_risk: float | None,
                          coverage: str, failed: bool,
                          citation_ok: bool, conflict: bool) -> str:
    """计算证据状态，不把未提及、无覆盖和运行失败混为一类。"""
    if coverage == "none":
        return "无覆盖"
    if failed and fact_status is None:
        return "失败"
    if (fact_status == "信息不足" or fact_status is None or fact_risk is None
            or not citation_ok or conflict):
        return "不足"
    return "充分"


def _dimensions(fact_status: FactLabel | None, fact_risk: float | None, coverage: str,
                failed: bool, emotion: str | None, spread: str | None,
                compliance: ComplianceResult, config: ArbiterScoreConfig
                ) -> tuple[dict[str, float], dict[str, str]]:
    """仅收集可用维度；缺席理由与分数分开保存。"""
    dimensions: dict[str, float] = {}
    absent: dict[str, str] = {}
    if coverage != "none" and not failed and fact_status is not None and fact_risk is not None:
        dimensions["fact"] = fact_risk
    else:
        absent["fact"] = "no_coverage" if coverage == "none" else "agent_unavailable"
    _add_level_dimension("emotion", emotion, config, dimensions, absent)
    _add_level_dimension("spread", spread if spread != "未知" else None, config, dimensions, absent)
    _add_compliance_dimension(compliance, config, dimensions, absent)
    return dimensions, absent


def _add_level_dimension(name: str, value: str | None, config: ArbiterScoreConfig,
                         dimensions: dict[str, float], absent: dict[str, str]) -> None:
    """将情绪或扩散等级映射成配置分数。"""
    table = config.level_map[name]
    if value in table:
        dimensions[name] = table[value]
    else:
        absent[name] = "agent_unavailable" if value is None else "unknown_value"


def _add_compliance_dimension(result: ComplianceResult, config: ArbiterScoreConfig,
                              dimensions: dict[str, float], absent: dict[str, str]) -> None:
    """声明检查有效时纳入合规分数；评论信号不参与等级分数。"""
    if result.claim_status == "checked_no_hit":
        dimensions["compliance"] = config.compliance_score["checked_no_hit"]
    elif result.claim_status == "checked_with_hits":
        severity = max((hit.severity for hit in result.claim_hits), default=None,
                       key=lambda item: 1 if item == "高" else 0)
        if severity is None:
            absent["compliance"] = "inconsistent_compliance_result"
        else:
            dimensions["compliance"] = config.compliance_score[severity]
    else:
        absent["compliance"] = result.claim_status


def _weighted_score(dimensions: dict[str, float], config: ArbiterScoreConfig
                    ) -> tuple[float, dict[str, float]]:
    """剔除缺席维度后重新归一化，避免把缺席误当零分。"""
    total = sum(config.weights[name] for name in dimensions)
    weights = {name: config.weights[name] / total for name in dimensions}
    return sum(dimensions[name] * weights[name] for name in dimensions), weights


def _level_for(score: float, config: ArbiterScoreConfig) -> int:
    """按三个配置边界映射到一至四级。"""
    return 1 + sum(score >= threshold for threshold in config.thresholds)


def _adjust_level(base: int, adjustment: int, config: ArbiterScoreConfig) -> int:
    """应用仲裁的一档调整，并限制在配置等级范围内。"""
    maximum = len(config.thresholds) + 1
    return max(1, min(maximum, base + adjustment))


def _apply_hard_floor(level: int, result: ComplianceResult,
                      config: ArbiterScoreConfig) -> tuple[int, list[str]]:
    """声明高严重度命中后，等级不得低于配置硬下限。"""
    if _has_high_claim_hit(result):
        floor = config.compliance_high_min_level
        if level < floor:
            return floor, ["claim_compliance_high_min_level"]
    return level, []


def _disposition(profiles: dict[str, dict[str, dict[str, str]]], scenario: str,
                 level: str, fact: str, compliance: ComplianceResult,
                 sufficiency: str, suffixes: DispositionSuffixConfig) -> str:
    """按已校验场景表查建议，并附加规格要求的提示。"""
    action = profiles[scenario]["system_failure" if level == "system_failure" else level][fact]
    if _has_high_claim_hit(compliance):
        action += "；" + suffixes.violation
    if sufficiency == "无覆盖":
        action += "；" + suffixes.unverified
    return action


def _system_failure(scenario: str, profiles: dict[str, dict[str, dict[str, str]]],
                    suffixes: DispositionSuffixConfig,
                    absent: dict[str, str], sufficiency: str,
                    compliance: ComplianceResult) -> CaseRisk:
    """所有计分维度缺席时返回不含正常风险等级的系统失败。"""
    reasons = ["all_dimensions_unavailable"]
    if compliance.comment_status == "check_failed":
        reasons.append("comment_compliance_check_failed")
    return CaseRisk(status="system_failure", absent_dims=absent,
        evidence_sufficiency="失败" if sufficiency != "无覆盖" else sufficiency,
        scenario=scenario, disposition=_disposition(profiles, scenario, "system_failure",
            "信息不足", compliance,
            sufficiency, suffixes), needs_human_review=True,
        review_reasons=reasons)


def _review_reasons(level: int, sufficiency: str, coverage: str, failed: bool,
                    citation_ok: bool, conflict: bool, disagree: bool, requested: bool,
                    compliance: ComplianceResult, config: ArbiterScoreConfig) -> list[str]:
    """汇总强制人工复核条件，不读取模型自由文本。"""
    reasons = []
    if level >= config.review_level:
        reasons.append("risk_level_requires_review")
    if sufficiency != "充分":
        reasons.append("evidence_sufficiency_requires_review")
    if coverage == "none":
        reasons.append("fact_coverage_unavailable")
    if failed:
        reasons.append("fact_agent_failed")
    if not citation_ok:
        reasons.append("citation_validation_failed")
    if conflict:
        reasons.append("unresolved_key_conflict")
    if disagree:
        reasons.append("review_disagreement")
    if requested:
        reasons.append("arbiter_requested_review")
    if compliance.comment_status == "check_failed":
        reasons.append("comment_compliance_check_failed")
    if compliance.comment_status == "checked_with_hits" and any(
        hit.severity == "高" for hit in compliance.comment_hits
    ):
        reasons.append("high_comment_compliance_signal")
    return list(dict.fromkeys(reasons))


def _has_high_claim_hit(result: ComplianceResult) -> bool:
    """只有声明检查状态与命中列表一致时，才应用高命中硬规则。"""
    return result.claim_status == "checked_with_hits" and any(
        hit.severity == "高" for hit in result.claim_hits
    )
