"""Project archived Trace rows into the stable public event contract."""

import json
from pathlib import Path

from src.yuqing.serve.research_contracts import ResearchEvent, ResearchSnapshot, ResearchAgent


AGENTS = {"fact_investigator": "fact", "propagation_analyst": "propagation", "arbiter": "arbiter"}


def load_trace(path: Path, *, research_id: str, execution_mode: str = "historical_replay") -> tuple[ResearchSnapshot, tuple[ResearchEvent, ...]]:
    """Read one explicitly selected JSONL trace and expose only safe event fields."""
    if execution_mode != "historical_replay" or not path.is_file() or path.suffix != ".jsonl":
        raise ValueError("unsupported_trace_source")
    rows = _read_rows(path)
    if not rows:
        raise ValueError("empty_trace")
    observed_roles = {AGENTS.get(row.get("agent")) for row in rows}
    run_id = rows[0].get("run_id")
    sample_id = rows[0].get("sample_id")
    if not isinstance(run_id, str) or any(row.get("run_id") != run_id or row.get("sample_id") != sample_id for row in rows):
        raise ValueError("trace_identity_mismatch")
    events: list[ResearchEvent] = []
    role_outcomes: dict[str, dict[str, object]] = {}
    outcome: dict[str, object] = {}
    for row in rows:
        event_type, agent, data = row.get("type"), row.get("agent"), row.get("data", {})
        if not isinstance(data, dict):
            continue
        mapped = _event_kind(event_type)
        if mapped is None:
            continue
        role = AGENTS.get(agent, "program")
        status = _status(event_type, data)
        summary = _summary(event_type, data)
        if (event_type == "tool_call" and data.get("tool") == "submit_verdict"
                and data.get("accepted") is False):
            status = "rejected"
        source = data.get("archive")
        archive_hash = source.get("sha256") if isinstance(source, dict) else None
        ref = f"archive:{archive_hash}" if isinstance(archive_hash, str) else f"trace-event:{len(events)+1}"
        events.append(_make_event(row, research_id, execution_mode, mapped, role, status, summary, ref, run_id, len(events)+1))
        if event_type == "run_outcome":
            role_outcomes[role] = data
    outcome = next(iter(role_outcomes.values()), {})
    agents = tuple(_agent_status(role, role_outcomes.get(role), role in observed_roles)
        for role in ("fact", "propagation", "arbiter"))
    snapshot = ResearchSnapshot(research_id=research_id, execution_mode="historical_replay",
        status=_terminal_status(outcome, next(iter(role_outcomes), "program")) if outcome else "interrupted", agents=agents,
        known_cost_cny=_cost_text(outcome), cost_status=_cost_status(outcome),
        last_seq=len(events), trace_ref=path.stem)
    return snapshot, tuple(events)


def _agent_status(role: str, outcome: dict[str, object] | None, observed: bool) -> ResearchAgent:
    """Derive completion only from that Agent's recorded terminal outcome."""
    if outcome is None:
        if observed:
            return ResearchAgent(role=role, status="interrupted", reason="missing_terminal_outcome")
        return ResearchAgent(role=role, status="unavailable", reason="not_observed_in_trace")
    status = _terminal_status(outcome, role)
    return ResearchAgent(role=role, status=status, reason=str(outcome.get("stop_reason") or outcome.get("reason") or "recorded_outcome"))


def _terminal_status(outcome: dict[str, object], role: str) -> str:
    """Map recognized terminal result classes without inferring success."""
    result = outcome.get("outcome_class")
    if role == "arbiter" and "outcome_class" not in outcome:
        if outcome.get("submitted") is True:
            return "completed"
        if outcome.get("submitted") is False:
            return "failed"
    if isinstance(result, str) and result.startswith("submitted_"):
        return "completed"
    if result in {"no_valid_result", "failed", "error"}:
        return "failed"
    return "interrupted"


def _cost_text(outcome: dict[str, object]) -> str | None:
    """Display only a known total or an explicitly recorded known subtotal."""
    amount = outcome.get("cost_cny") if outcome.get("usage_known") is True else outcome.get("known_cost_subtotal_cny")
    return f"{amount:.6f}" if isinstance(amount, (int, float)) else None


def _cost_status(outcome: dict[str, object]) -> str:
    """Keep missing and partially known usage distinct."""
    if outcome.get("usage_known") is True and isinstance(outcome.get("cost_cny"), (int, float)):
        return "known"
    return "partial" if isinstance(outcome.get("known_cost_subtotal_cny"), (int, float)) else "unknown"


def _read_rows(path: Path) -> list[dict[str, object]]:
    """Parse archived rows and reject malformed JSON at the file boundary."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_event(row: dict[str, object], research_id: str, mode: str, kind: str,
                actor: str, status: str, summary: str, ref: str, run_id: str, seq: int) -> ResearchEvent:
    """Validate one projected event against the public contract."""
    return ResearchEvent(research_id=research_id, seq=seq,
        execution_mode=mode, occurred_at=row["ts"], kind=kind, actor=actor,
        status=status, summary=summary, run_id=run_id, source_event_ref=ref)


def _event_kind(name: object) -> str | None:
    """Map only explicitly supported Trace event types."""
    if name in {"llm_call", "tool_call", "arbiter_v2_decision", "arbiter_v2_action_validation", "run_outcome", "stop"}:
        return "outcome" if name in {"run_outcome", "stop"} else "tool_call" if name == "tool_call" else "node_status"
    if name in {"submission", "fact_v2_submission_validation"}:
        return "citation_validation"
    return None


def _status(name: str, data: dict[str, object]) -> str:
    """Expose recorded status or a conservative event-specific label."""
    for key in ("status", "reason", "outcome_class", "stop_reason"):
        value = data.get(key)
        if isinstance(value, str):
            return value[:80]
    return "observed"


def _summary(name: str, data: dict[str, object]) -> str:
    """Build a concise display label without exposing archived payload text."""
    labels = {"llm_call": "模型调用已完成", "tool_call": "工具调用", "run_outcome": "运行终局",
        "stop": "运行停止", "arbiter_v2_decision": "仲裁决策", "arbiter_v2_action_validation": "补查门卫校验",
        "submission": "交卷记录", "fact_v2_submission_validation": "引用与交卷校验"}
    tool = data.get("tool")
    return f"{labels.get(name, '运行事件')}：{tool[:60]}" if isinstance(tool, str) else labels.get(name, "运行事件")


def public_snapshot_events(snapshot: ResearchSnapshot, *, occurred_at: object, source_ref: str) -> tuple[ResearchEvent, ...]:
    """Publish only state, report and W validations actually recorded in this snapshot."""
    agent = next((item for item in snapshot.agents if item.role == "propagation"), None)
    if agent is None or not (agent.reason or "").startswith("public_sample:"):
        return ()
    entries = [("node_status", agent.status, "快照记录：公开样本任务状态；总体规模/情绪unavailable", None)]
    if agent.status == "completed":
        entries.append(("report", "completed", "样本报告已归档，不参加全网风险计分", None))
        entries.extend(("citation_validation", citation.validation, f"{citation.evidence_id} · 公开样本原文引用", citation.run_id)
            for citation in snapshot.citations if citation.citation_id.startswith("public:"))
    return tuple(ResearchEvent(research_id=snapshot.research_id, seq=snapshot.last_seq + number,
        execution_mode=snapshot.execution_mode, occurred_at=occurred_at, kind=kind, actor="propagation",
        status=status, summary=summary, run_id=run_id, source_event_ref=source_ref)
        for number, (kind, status, summary, run_id) in enumerate(entries, start=1))
