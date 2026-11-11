"""Optional public sample side task and projection; never writes legacy risk inputs."""
from collections.abc import Callable
from datetime import datetime, timezone
from html import escape
import logging

from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask, PublicSignalResult
from src.yuqing.retrieval.public_materials import qualify_public_quote
from src.yuqing.serve.research_contracts import ResearchAgent, ResearchCitation, ResearchSnapshot

logger = logging.getLogger(__name__)
PUBLIC_REASON = "public_sample:"
APPENDIX_TITLE = "### 附录：公开传播材料样本分析"


def _plain(text: str) -> str:
    """Escape original strings as literal Markdown text, including images and links."""
    value = escape(text)
    for char in ("\\", "`", "*", "_", "[", "]", "(", ")", "#", "!", "|", ">"):
        value = value.replace(char, "\\" + char)
    return value.replace("\n", " ").replace("\r", " ")


class PublicSampleSession:
    """Own one explicitly injected sample job without modifying Graph scoring state."""

    def __init__(self, task: PublicPropagationTask, run: Callable[[], PublicSignalResult | None]) -> None:
        """Bind a case-specific task and a caller-authorized job; construction never executes it."""
        self.task = PublicPropagationTask.model_validate(task.model_dump(exclude={"batch": {"discussion_status", "media_status"}}))
        self.run = run
        self.status = "not_started"
        self.result: PublicSignalResult | None = None
        self.reason = "not_run"
        self.observed_at: datetime | None = None

    def round1(self, _state: dict[str, object]) -> dict[str, object]:
        """Execute only the supplied side task and return no legacy propagation score patch."""
        self.status, self.reason = "running", "sample_only"
        try:
            result = self.run()
            if result is None:
                self.status, self.reason = "failed", "no_qualified_sample_report"
            else:
                result = PublicSignalResult.model_validate(result.model_dump())
                validate_public_result(self.task, result)
                self.result, self.status, self.reason = result, "completed", "sample_only"
        except Exception as exc:
            logger.error("Public sample failed type=%s", type(exc).__name__)
            self.result, self.status, self.reason = None, "failed", "sample_task_failed"
        self.observed_at = datetime.now(timezone.utc)
        return {}


def validate_public_result(task: PublicPropagationTask, result: PublicSignalResult) -> None:
    """Retain original case/claim, source-kind and delivered quote gates at result ingestion."""
    if (result.sample_id != task.sample_id or result.coverage.case_id != task.sample_id
            or result.coverage.provenance != task.batch.provenance):
        raise ValueError("public_result_case_mismatch")
    for receipt in result.receipts:
        if receipt.case_id != task.sample_id or receipt.claim_sha256 != task.batch.claim_sha256:
            raise ValueError("public_receipt_case_mismatch")
    delivered = {receipt.material_id: receipt for receipt in result.receipts}
    for signal in result.report.signals:
        receipt = delivered.get(signal.material_id)
        location = qualify_public_quote(task.batch, signal.material_id, signal.quote, result.receipts)
        if receipt is None or receipt.source_kind != signal.source_kind or location is None or location.status != "matched":
            raise ValueError("public_result_quote_invalid")
        if (signal.kind == "public_expression" and signal.source_kind == "news_report"
                or signal.kind == "media_narrative" and signal.source_kind != "news_report"):
            raise ValueError("public_result_kind_invalid")


def project_public_sample(snapshot: ResearchSnapshot, session: PublicSampleSession | None, *, sample_id: str) -> ResearchSnapshot:
    """Append sample-only observations and W citations without altering scores, cost or overall outcome."""
    status, reason = (session.status, session.reason) if session else ("not_started", "not_run")
    result = session.result if session and status == "completed" else None
    if session and session.task.sample_id != sample_id:
        raise ValueError("public_snapshot_identity_mismatch")
    if result and snapshot.execution_mode == "live" and result.coverage.provenance == "synthetic_fixture":
        raise ValueError("synthetic_public_result_cannot_be_live")
    citations = list(snapshot.citations)
    appendix = [APPENDIX_TITLE, "范围：仅已取得材料的样本分析，不代表全网；不参加全网风险计分。",
                "总体传播规模：`unavailable` · 总体情绪：`unavailable`", f"样本任务状态：`{status}`"]
    if result:
        appendix.append("材料来源：" + ("人工合成离线夹具" if result.coverage.provenance == "synthetic_fixture" else "已归档网页材料"))
        delivered = {receipt.material_id: receipt for receipt in result.receipts}
        for index, signal in enumerate(result.report.signals):
            receipt = delivered[signal.material_id]
            appendix.append(f"- [{signal.material_id} · {signal.source_kind}] {_plain(signal.interpretation)}\n  原文：{_plain(signal.quote)}")
            citations.append(ResearchCitation(citation_id=f"public:{result.run_id}:{index}", run_id=result.run_id,
                evidence_id=signal.material_id, quote=signal.quote, source_url=str(receipt.source_url),
                source_label=f"公开样本 · {signal.source_kind}", body_sha256=receipt.body_sha256, validation="passed"))
        appendix.extend("- 待解决：" + _plain(item) for item in result.report.unresolved_questions)
    else:
        appendix.append("本次样本任务未产生合格报告。" if status == "failed" else "本次未运行公开材料样本任务，不能据此推断没有网上讨论。")
    agents = tuple(agent for agent in snapshot.agents if agent.role != "propagation") + (
        ResearchAgent(role="propagation", status=status, reason=PUBLIC_REASON + reason),)
    return snapshot.model_copy(update={"report": (snapshot.report or "未产生主调查报告。") + "\n\n" + "\n\n".join(appendix),
        "citations": tuple(citations), "agents": agents, "propagation_status": "unavailable"})
