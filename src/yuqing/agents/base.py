"""专家共用的结构化调用入口与理由编号检查。"""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

from pydantic import BaseModel

from src.yuqing import llm
from src.yuqing.schemas import EvidenceChunk, FactVerdict, Reason


logger = logging.getLogger(__name__)
OutputT = TypeVar("OutputT", bound=BaseModel)


def citation_issues(out: FactVerdict, evidence_count: int) -> list[str]:
    """逐条检查理由的 E 编号是否属于本次证据，不证明已读或语义支持。"""

    issues: list[str] = []
    available = {f"E{number}" for number in range(1, evidence_count + 1)}
    for index, reason in enumerate(out.reasons):
        if not reason.evidence_ids:
            issues.append(f"reasons[{index}] 缺少证据编号")
        invalid = sorted(set(reason.evidence_ids) - available)
        if invalid:
            issues.append(f"reasons[{index}] 含无效证据编号：{invalid}")
    return issues


def is_citation_only_edit(original: list[Reason], repaired: list[Reason]) -> bool:
    """只允许修改结构化引用编号，拒绝删改、增加或调换理由正文。"""

    if len(original) != len(repaired):
        return False
    return all(before.text == after.text for before, after in zip(original, repaired, strict=True))


class BaseAgent(Generic[OutputT]):
    """把专家的结构化模型调用收敛到一个入口。"""

    name: str
    output_model: type[OutputT]
    system_prompt: str
    structured_context: llm.StructuredBudgetContext | None = None

    def structured_kwargs(self) -> dict[str, object]:
        """仅向显式受保护路径传上下文，旧调用参数保持兼容。"""
        return {} if self.structured_context is None else {"budget_context": self.structured_context}

    def build_prompt(
        self, claim: str, evidence: list[EvidenceChunk], extra: dict | None = None
    ) -> str:
        """由具体专家把输入组装成用户消息。"""

        raise NotImplementedError

    def run(
        self, claim: str, evidence: list[EvidenceChunk], extra: dict | None = None
    ) -> OutputT | None:
        """调用唯一 LLM 出口，首次判断失败时返回 None 供仲裁降级。"""

        try:
            if not claim.strip():
                logger.warning("%s Agent 缺少非空声明", self.name)
                return None
            prompt = self.build_prompt(claim, evidence, extra)
            return llm.call_structured(prompt, self.output_model, system=self.system_prompt, **self.structured_kwargs())
        except Exception:
            logger.exception("%s Agent 执行失败", self.name)
            return None
