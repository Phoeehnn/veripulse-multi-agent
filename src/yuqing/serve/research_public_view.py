"""Lightweight snapshot-only public appendix view; usable without Agent dependencies."""
from hashlib import sha256
from html import escape
from pathlib import Path
from src.yuqing.serve.research_contracts import ResearchEvent, ResearchSnapshot


def public_report_view(snapshot: ResearchSnapshot) -> str:
    """Show explicit missing archival data for old histories without reconstructing a report."""
    report = snapshot.report or "未产生合格报告。"
    if not any((item.reason or "").startswith("public_sample:") for item in snapshot.agents):
        report += ("\n\n### 附录：公开传播材料样本分析\n\n"
                   "样本任务：未运行或未归档（旧快照无此记录），不重建结果。\n\n"
                   "总体传播规模：`unavailable` · 总体情绪：`unavailable`。样本分析不参加全网风险计分。")
    return report


def public_business_timeline(case: Path, snapshot: ResearchSnapshot) -> list[str]:
    """Read only matching snapshot-derived events; browsing never generates business activity."""
    path = case / "business-events.jsonl"
    if not path.exists():
        return []
    source = "snapshot-sha256:" + sha256((case / "snapshot.json").read_bytes()).hexdigest()
    events = [ResearchEvent.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [f"- {event.occurred_at.isoformat()} · 样本任务快照 · {escape(event.status)} · {escape(event.summary)}"
            for event in events if event.research_id == snapshot.research_id
            and event.execution_mode == snapshot.execution_mode and event.source_event_ref == source]


def public_agent_label(snapshot: ResearchSnapshot) -> str:
    """Distinguish an executed sample task from absent global propagation data."""
    return "传播 Agent（样本任务）" if any((item.reason or "").startswith("public_sample:")
        for item in snapshot.agents) else "传播 Agent"
