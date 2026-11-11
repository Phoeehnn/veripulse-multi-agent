"""事实核查专家与检索到研判的最小闭环。"""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from src.yuqing import llm
from src.yuqing.agents.base import BaseAgent, citation_issues, is_citation_only_edit
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.retrieval.search import search
from src.yuqing.schemas import CitationRepairV2, EvidenceChunk, ExpertOpinion, FactVerdict
from src.yuqing.settings import get_agent_config, get_fact_risk_config, get_harness_config, get_cost_config
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.trace import require_run_context

if TYPE_CHECKING:
    from src.yuqing.eval.dual_budget import DualBudgetCoordinator


logger = logging.getLogger(__name__)


def compute_fact_risk(verdict: FactVerdict) -> float:
    """根据已校验的类别、置信度和配置计算风险，不参考模型自由评分。"""

    rule = get_fact_risk_config()[verdict.label]
    return rule.base + rule.slope * verdict.confidence


def to_fact_opinion(
    verdict: FactVerdict, *, citation_ok: bool, flags: list[str]
) -> ExpertOpinion:
    """把事实判断转为专家意见，引用质量只作标记、不参与风险计算。"""

    evidence_ids = list(dict.fromkeys(
        evidence_id for reason in verdict.reasons for evidence_id in reason.evidence_ids
    ))
    return ExpertOpinion(
        expert="fact", risk=compute_fact_risk(verdict), confidence=verdict.confidence,
        citation_ok=citation_ok, flags=list(flags),
        summary="；".join(reason.text for reason in verdict.reasons),
        evidence_ids=evidence_ids,
    )


class FactCheckAgent(BaseAgent[FactVerdict]):
    """仅使用外部传入的系统提示词进行事实核查。"""

    name = "fact"
    output_model = FactVerdict

    def __init__(self, system_prompt: str, *, structured_context: llm.StructuredBudgetContext | None = None) -> None:
        """要求调用者提供外部传入的系统提示词。"""

        if not system_prompt.strip():
            raise ValueError("事实核查系统提示词不能为空")
        self.system_prompt = system_prompt
        self.structured_context = structured_context
        self.citation_flags: list[str] = []

    def build_prompt(
        self, claim: str, evidence: list[EvidenceChunk], extra: dict | None = None
    ) -> str:
        """将声明和带编号的证据封装为可引用的用户消息。"""

        lines = [
            f"E{number}：{escape(item.text, quote=False)}"
            for number, item in enumerate(evidence, start=1)
        ]
        evidence_text = "\n".join(lines)
        return (f"<task>fact_check</task>\n<claim>{escape(claim, quote=False)}</claim>\n"
                f"<evidence>\n{evidence_text}\n</evidence>")


    def run(
        self, claim: str, evidence: list[EvidenceChunk], extra: dict | None = None
    ) -> FactVerdict | None:
        """取得首次判断并有限次修复引用，附加修复失败保留判断。"""

        self.citation_flags = []
        if not evidence:
            logger.warning("事实核查没有证据，不能编造理由引用")
            return None
        out = super().run(claim, evidence, extra)
        if out is None:
            return None
        try:
            out, self.citation_flags = self.repair_citations(
                out, self.build_prompt(claim, evidence, extra), len(evidence)
            )
        except Exception:
            logger.exception("事实核查引用修复准备失败，保留原判断")
            self.citation_flags = ["citation_repair_failed"]
        return out

    def repair_citations(
        self, out: FactVerdict, prompt: str, evidence_count: int
    ) -> tuple[FactVerdict, list[str]]:
        """只更新理由引用，失败留标记，正文或其他判断字段不能改。"""

        flags: list[str] = []
        for _ in range(get_agent_config().citation_repair_retries):
            issues = citation_issues(out, evidence_count)
            if not issues:
                break
            repair_prompt = (
                f"{prompt.replace('<task>fact_check</task>', '<task>repair_citations</task>')}\n"
                f"<original_verdict>{escape(out.model_dump_json(), quote=False)}</original_verdict>\n"
                f"<citation_issues>{escape('；'.join(issues), quote=False)}</citation_issues>"
            )
            logger.info("事实核查请求引用修复：%s", "；".join(issues))
            try:
                fixed = llm.call_structured(repair_prompt, CitationRepairV2, system=self.system_prompt, **self.structured_kwargs())
            except Exception:
                logger.exception("事实核查引用修复异常，保留原判断")
                flags.append("citation_repair_failed")
                break
            if fixed is None:
                logger.warning("事实核查引用修复调用失败，保留原判断")
                flags.append("citation_repair_failed")
                break
            if not is_citation_only_edit(out.reasons, fixed.reasons):
                logger.warning("事实核查引用修复改动理由正文，拒绝采用并保留原输出")
                flags.append("citation_repair_rejected")
                continue
            out = out.model_copy(update={"reasons": fixed.reasons})
        return out, flags

    def postprocess(self, out: FactVerdict, evidence_count: int) -> ExpertOpinion:
        """风险独立计算，编号与尚未验证的读取问题只作引用质量标记。"""

        issues = citation_issues(out, evidence_count)
        if issues:
            logger.warning("事实核查引用不完整：%s", "；".join(issues))
        flags = list(dict.fromkeys(self.citation_flags + issues + ["citation_read_unverified"]))
        return to_fact_opinion(out, citation_ok=False, flags=flags)


def _single_budget_context(factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None,
                           identity: Mapping[str, str] | None) -> llm.StructuredBudgetContext | None:
    """显式装配单题预算，首次判断和引用修复只取得同一个上下文。"""
    if (factory is None) != (identity is None):
        raise ValueError("incomplete_dual_budget_arguments")
    if factory is None:
        return None
    ctx = require_run_context()
    if identity.get("run_id") != ctx.run_id or identity.get("sample_id") != ctx.sample_id:
        raise ValueError("experiment_identity_mismatch")
    budget = BudgetTracker(get_harness_config(), get_cost_config())
    return llm.StructuredBudgetContext(budget, factory(budget), identity,
                                      validation_retries=get_agent_config().single_validation_retries)


def verify(
    claim: str,
    index: BM25Index,
    system_prompt: str,
    store_path: Path | None = None,
    *, coordinator_factory: Callable[[BudgetTracker], DualBudgetCoordinator] | None = None,
    experiment_identity: Mapping[str, str] | None = None,
) -> tuple[FactVerdict, ExpertOpinion, list[EvidenceChunk]] | None:
    """检索证据并返回事实判断、程序计算的专家意见及编号所指的原始片段。"""

    try:
        if not claim.strip():
            logger.warning("事实核查缺少非空声明")
            return None
        context = _single_budget_context(coordinator_factory, experiment_identity)
        agent = FactCheckAgent(system_prompt, **({"structured_context": context} if context is not None else {}))
        evidence = search(claim, index, path=store_path)
        output = agent.run(claim, evidence)
        return None if output is None else (output, agent.postprocess(output, len(evidence)), evidence)
    except Exception:
        logger.exception("事实核查检索或初始化失败")
        return None
