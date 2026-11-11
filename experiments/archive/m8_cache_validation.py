"""Trace-side verification for persisted M8 propagation snapshots."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from src.yuqing.agents.propagation import PropagationResult, PropagationToolPayload
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.settings import get_propagation_harness_config


def verify_cached_propagation(result: PropagationResult, root: Path) -> None:
    """只接受身份一致且留有成功提交归档的传播Trace。"""
    path = Path(result.trace_path).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if not path.is_relative_to(resolved_root) or path.suffix != ".jsonl":
        raise ValueError("m8_cache_propagation_trace_invalid")
    run_root = path.parents[1]
    if not run_root.is_relative_to(resolved_root):
        raise ValueError("m8_cache_propagation_trace_invalid")
    events = [TraceEvent.model_validate_json(line, strict=True)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any((event.run_id, event.sample_id, event.agent) !=
           (result.run_id, result.sample_id, "propagation_analyst") for event in events):
        raise ValueError("m8_cache_propagation_trace_identity_mismatch")
    outcomes = [event for event in events if event.type == "run_outcome"]
    if len(outcomes) != 1 or not _outcome_matches(outcomes[0], result):
        raise ValueError("m8_cache_propagation_outcome_mismatch")
    _verify_propagation_submission(events, result, run_root)
    _verify_propagation_deliveries(events, result, run_root)


def _verify_propagation_submission(events: list[TraceEvent], result: PropagationResult,
                                   root: Path) -> None:
    """验证Trace中的唯一成功提交确实与缓存报告逐字段一致。"""
    name = get_propagation_harness_config().submission_tool
    accepted = []
    for event in events:
        if event.type != "tool_call" or event.data.get("tool") != name:
            continue
        archive = _read_tool_archive(event, root)
        payload = archive["payload"]
        if (payload.get("status") == "success"
                and payload.get("feedback", {}).get("accepted") is True
                and archive.get("request", {}).get("arguments") == result.report.model_dump(mode="json")):
            accepted.append(payload)
    if len(accepted) != 1:
        raise ValueError("m8_cache_propagation_submission_mismatch")


def _verify_propagation_deliveries(events: list[TraceEvent], result: PropagationResult,
                                   root: Path) -> None:
    """重建已交付评论和统计，防止缓存报告引用未出现的运行材料。"""
    comments: dict[str, dict[str, Any]] = {}
    stats = None
    for event in events:
        if event.type != "tool_call" or event.data.get("tool") == get_propagation_harness_config().submission_tool:
            continue
        archive = _read_tool_archive(event, root)
        payload = archive["payload"]
        if payload.get("status") != "success":
            continue
        decoded = _decode_propagation_payload(payload)
        if decoded.stats is not None:
            stats = decoded.stats.model_dump(mode="json")
        comments.update({item.comment_id: item.model_dump(mode="json") for item in decoded.comments})
    expected = {item.comment_id: item.model_dump(mode="json") for item in result.comments}
    actual_stats = stats
    result_stats = result.stats.model_dump(mode="json") if result.stats else None
    if comments != expected or actual_stats != result_stats:
        raise ValueError("m8_cache_propagation_delivery_mismatch")


def _decode_propagation_payload(payload: dict[str, Any]) -> PropagationToolPayload:
    """Validate an archived JSON payload using JSON's strict decoding rules."""
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return PropagationToolPayload.model_validate_json(raw, strict=True)


def _read_tool_archive(event: TraceEvent, root: Path) -> dict[str, Any]:
    """读取并校验一条传播工具归档的路径、散列和运行身份。"""
    archive = event.data.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("path"), str):
        raise ValueError("m8_cache_tool_archive_missing")
    path = (root / archive["path"]).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest() != archive.get("sha256"):
        raise ValueError("m8_cache_tool_archive_hash_mismatch")
    document = TypeAdapter(dict[str, Any]).validate_json(path.read_bytes(), strict=True)
    if (document.get("kind") != "tool_result" or document.get("run_id") != event.run_id
            or document.get("sample_id") != event.sample_id or document.get("agent") != event.agent):
        raise ValueError("m8_cache_tool_archive_identity_mismatch")
    payload = document.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("payload"), dict):
        raise ValueError("m8_cache_tool_archive_payload_invalid")
    return payload


def _outcome_matches(event: TraceEvent, result: PropagationResult) -> bool:
    """核对终局Trace与可信结果的费用、token和停止原因。"""
    data = event.data
    return (data.get("outcome_class") == "submitted_report"
        and data.get("stop_reason") == result.stop_reason
        and data.get("input_output_tokens") == result.input_output_tokens
        and data.get("cost_cny") == result.cost_cny)



def _validate_first_round_state(state: dict[str, Any], identity: Any,
                                fact: Any | None,
                                prop: PropagationResult | None) -> None:
    """确保缓存只含merge前数据，且与两份可信Agent结果匹配。"""
    forbidden = {"decisions", "actions", "task_results", "issues", "pending_task",
        "action_validation", "risk", "approval", "review", "alert", "brief"}
    if (forbidden.intersection(state) or (state.get("case_id"),
            (state.get("sample") or {}).get("sample_id")) !=
            (identity.case_id, identity.sample_id) or state.get("decisions") or state.get("actions")
            or state.get("task_results") or state.get("issues")
            or state.get("coverage") != identity.coverage):
        raise ValueError("m8_cache_state_not_first_round")
    if identity.coverage == "none" and fact is not None:
        raise ValueError("m8_cache_ced_fact_must_not_run")
    if fact is None and identity.coverage != "none" and state.get("fact") is not None:
        raise ValueError("m8_cache_unverified_fact_result")
    if fact is not None and state.get("fact") != fact.summary.report:
        raise ValueError("m8_cache_fact_state_mismatch")
    _validate_fact_round_rows(state, identity, fact)
    if prop is not None and state.get("propagation") != prop.report.model_dump(mode="json"):
        raise ValueError("m8_cache_propagation_state_mismatch")
    if prop is None and state.get("propagation") is not None:
        raise ValueError("m8_cache_unverified_propagation_result")
    _validate_propagation_rows(state, prop)
    if prop is not None and state.get("leads") != [item.model_dump(mode="json")
            for item in prop.leads_by_id.values()]:
        raise ValueError("m8_cache_propagation_leads_mismatch")


def _validate_fact_round_rows(state: dict[str, Any], identity: Any,
                              fact: Any | None) -> None:
    """确保事实结果与首轮 State 行数和摘要一一对应。"""
    rows = state.get("fact_rounds", [])
    if identity.coverage == "none":
        if rows != [{"status": "not_dispatched", "reason": "evidence_unavailable"}]:
            raise ValueError("m8_cache_ced_fact_round_mismatch")
        return
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("m8_cache_fact_round_count_mismatch")
    expected = ({"round": 1, "status": "failed", "stop_reason": "no_valid_result"}
        if fact is None else _fact_row(fact))
    if rows[0] != expected:
        raise ValueError("m8_cache_fact_round_summary_mismatch")


def _fact_row(fact: Any) -> dict[str, Any]:
    """Build the exact State summary emitted by the verified fact adapter."""
    result = fact.summary
    return {"round": fact.archive_input.scope.fact_round_no, "status": result.status,
        "stop_reason": result.stop_reason, "ledger_refs": list(result.ledger_refs),
        "delivery_refs": list(result.delivery_refs), "archive_path": result.archive_path,
        "archive_sha256": result.archive_sha256, "usage": result.usage}


def _validate_propagation_rows(state: dict[str, Any], prop: PropagationResult | None) -> None:
    """确保传播State行与Trace校验后的结果完全对应。"""
    rows = state.get("prop_rounds", [])
    expected = ([{"status": "failed"}] if prop is None else [{
        "status": prop.stop_reason, "trace_path": prop.trace_path, "run_id": prop.run_id,
        "input_output_tokens": prop.input_output_tokens, "cost_cny": prop.cost_cny}])
    if rows != expected:
        raise ValueError("m8_cache_propagation_round_summary_mismatch")

