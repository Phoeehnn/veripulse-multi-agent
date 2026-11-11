"""Reconstruct propagation deliveries from same-snapshot, wallet-backed archives."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from src.yuqing.collab.fact_archive import _safe_file, _safe_trace, _response_tool_calls
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.features.propagation import CommentItem, SpreadStats, _visible_nodes, _comment_items, get_spread_stats
from src.yuqing.harness.trace import RunContext, TraceEvent, record_event
from src.yuqing.report.case_usage import summarize_case_usage
from src.yuqing.schemas import Sample, ToolRequest
from src.yuqing.settings import PROJECT_ROOT


def canonical(value: object) -> str:
    """Encode immutable data blocks and fingerprints deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_fingerprint(sample: Sample, as_of: datetime | None) -> str:
    """Bind the claim and propagation snapshot without including labels."""
    value = {"sample_id": sample.sample_id, "claim": sample.claim,
        "propagation": sample.propagation.model_dump(mode="json") if sample.propagation else None,
        "as_of": as_of.isoformat() if as_of is not None else None}
    return sha256(canonical(value).encode()).hexdigest()


def record_propagation_input(sample: Sample, as_of: datetime | None,
                             identity: Mapping[str, str] | None, case_id: str | None = None) -> None:
    """Record the source snapshot identity for future explicit v1.1 follow-ups."""
    record_event("propagation_input", {"input_sha256": input_fingerprint(sample, as_of),
        "identity": dict(identity or {}), "contract_version": "v1.1", "case_id": case_id})


@dataclass(frozen=True)
class PropagationDelivery:
    """Internal archive-verified source; never accepted from a model or CLI JSON."""
    case_id: str
    sample_id: str
    run_id: str
    round_no: int
    input_sha256: str
    identity_json: str
    comments: tuple[CommentItem, ...]
    stats: SpreadStats | None
    queries_json: str
    visible_ids: frozenset[str]
    trace_sha256: str
    wallet: object
    experiment_id: str


def _archive(ctx: RunContext, event: TraceEvent, kind: str) -> dict:
    """Validate archive path, bytes, envelope and event identity at the file boundary."""
    reference = event.data.get("archive")
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or any(not isinstance(reference[key], str) for key in ("path", "sha256"))):
        raise ValueError("propagation_archive_reference_invalid")
    path = _safe_file(ctx.trace_root.resolve(), reference["path"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("propagation_archive_sha_mismatch")
    doc = json.loads(raw)
    if (not isinstance(doc, dict) or doc.get("kind") != kind or doc.get("run_id") != ctx.run_id
            or doc.get("sample_id") != ctx.sample_id or doc.get("agent") != ctx.agent
            or doc.get("step") != event.step or not isinstance(doc.get("payload"), dict)):
        raise ValueError("propagation_archive_identity_mismatch")
    return doc["payload"]


def _request_chain(ctx: RunContext, events: list[TraceEvent]) -> tuple[dict, dict]:
    """Index unique request archives and the exact SDK tool requests they produced."""
    requests, responses, calls = {}, {}, {}
    for event in events:
        if event.type not in {"llm_request", "model_response_archive"}:
            continue
        kind = "model_request" if event.type == "llm_request" else "model_response"
        payload = _archive(ctx, event, kind)
        rid = event.data.get("request_id")
        target = requests if kind == "model_request" else responses
        if not isinstance(rid, str) or rid in target or payload.get("request_id") != rid:
            raise ValueError("propagation_request_identity_invalid")
        target[rid] = (event, payload)
    if not requests or requests.keys() != responses.keys():
        raise ValueError("propagation_request_response_mismatch")
    invalid_steps = {e.step for e in events if e.type == "verifier"
        and e.data.get("schema") == "ToolLLMResult" and e.data.get("reason")}
    for rid, (event, payload) in responses.items():
        if event.step != requests[rid][0].step:
            raise ValueError("propagation_response_step_mismatch")
        if event.step in invalid_steps:
            continue
        for call in _response_tool_calls(payload):
            cid = call.get("id")
            if not isinstance(cid, str) or cid in calls:
                raise ValueError("propagation_source_call_duplicate")
            calls[cid] = (event.step, call["function"])
    return requests, calls


def _verify_wallet(wallet: object, ctx: RunContext, identity: Mapping[str, str],
                   request_ids: set[str],
                   allowed_pending: Mapping[str, Mapping[str, str]] | None = None) -> str:
    """Require every source request to be settled in the same explicitly bound wallet."""
    manifest = wallet.manifest_snapshot()
    status = wallet.snapshot()
    if status.halted or status.unknown_requests:
        raise ValueError("propagation_source_wallet_unavailable")
    state = replay_history(wallet.load_history(), manifest)
    pending = {rid: item for rid, item in state.requests.items()
        if item.state in {"reserved", "dispatched"}}
    allowed_pending = allowed_pending or {}
    if status.pending_requests != len(pending) or any(
            item.identity.get("run_id") not in allowed_pending
            or dict(item.identity) != dict(allowed_pending[item.identity.get("run_id")])
            for item in pending.values()):
        raise ValueError("propagation_source_wallet_unavailable")
    rows = {rid: item for rid, item in state.requests.items()
        if item.identity.get("run_id") == ctx.run_id and item.state != "cancelled"}
    if set(rows) != request_ids or any(item.state != "settled" or item.actual is None
            or dict(item.identity) != dict(identity) for item in rows.values()):
        raise ValueError("propagation_source_wallet_mismatch")
    usage = summarize_case_usage((ctx,), wallet)
    if not usage["usage_complete"] or usage["estimated_cost_cny"] is None:
        raise ValueError("propagation_source_usage_untrusted")
    return manifest["experiment_id"]


def _deliveries(ctx: RunContext, events: list[TraceEvent], requests: dict, calls: dict,
                sample: Sample, as_of: datetime | None, inherited: tuple = ((), None)) -> tuple[tuple, SpreadStats | None, str]:
    """Recover only successful previews actually present in a later model request."""
    from src.yuqing.agents.propagation import PropagationToolPayload, _serialize_tool_payload
    from src.yuqing.agents.propagation_v11 import stats_receipt
    visible = {c.comment_id: c for c in _comment_items(_visible_nodes(sample.propagation, as_of))}
    comments, stats, queries, seen = {c.comment_id: c for c in inherited[0]}, inherited[1], [], set()
    for event in events:
        if event.type != "tool_call":
            continue
        archive = _archive(ctx, event, "tool_result")
        request = ToolRequest.model_validate(archive["request"])
        pair = calls.get(request.call_id)
        if (pair is None or pair[0] != event.step or request.call_id in seen
                or pair[1].get("name") != request.name
                or json.loads(pair[1].get("arguments", "{}")) != request.arguments):
            raise ValueError("propagation_source_tool_mismatch")
        seen.add(request.call_id)
        if request.name == "submit_propagation_report":
            continue
        payload = PropagationToolPayload.model_validate(archive["payload"])
        content = archive["content"]
        if json.loads(content) != json.loads(_serialize_tool_payload(payload)):
            raise ValueError("propagation_source_receipt_mismatch")
        sent = any(e.step > event.step and any(m.get("role") == "tool"
            and m.get("tool_call_id") == request.call_id and m.get("content") == content
            for m in doc["body"]["messages"]) for e, doc in requests.values())
        if request.name in {"sample_comments", "search_comments"}:
            queries.append({"tool": request.name, "arguments": request.arguments,
                "status": payload.status, "reason": payload.reason, "delivered": sent,
                "returned_comment_ids": [c.comment_id for c in payload.comments] if sent else []})
        if sent and payload.status == "success":
            if payload.stats is not None:
                if payload.stats != get_spread_stats(sample.propagation, as_of=as_of) or payload.stats_receipt != stats_receipt(payload.stats):
                    raise ValueError("propagation_source_stats_mismatch")
                stats = payload.stats
            for comment in payload.comments:
                if visible.get(comment.comment_id) != comment:
                    raise ValueError("propagation_source_comment_mismatch")
                comments[comment.comment_id] = comment
    if seen != set(calls):
        raise ValueError("propagation_source_tool_receipt_missing")
    return tuple(comments[key] for key in sorted(comments, key=lambda c: int(c[1:]))), stats, canonical(queries)


def read_propagation_delivery(*, case_id: str, sample: Sample, ctx: RunContext,
                              as_of: datetime | None, wallet: object,
                              identity: Mapping[str, str], round_no: int = 1,
                              followup: object | None = None,
                              allowed_pending: Mapping[str, Mapping[str, str]] | None = None
                              ) -> PropagationDelivery:
    """Validate one inactive source run at the external archive boundary exactly once."""
    root = ctx.trace_root.absolute()
    if root != root.resolve() or not root.is_relative_to(PROJECT_ROOT / "traces"):
        raise ValueError("propagation_source_trace_outside_project")
    path = _safe_trace(root, ctx.trace_path)
    raw = path.read_bytes()
    events = [TraceEvent.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if (not case_id or round_no != (followup.round_no if followup is not None else 1) or ctx.agent != "propagation_analyst"
            or ctx.sample_id != sample.sample_id or any(e.run_id != ctx.run_id
            or e.sample_id != sample.sample_id or e.agent != ctx.agent for e in events)):
        raise ValueError("propagation_source_scope_mismatch")
    expected = {"run_id": ctx.run_id, "sample_id": sample.sample_id,
        "group_id": identity.get("group_id"), "variant": identity.get("variant")}
    scopes = [e.data for e in events if e.type == "propagation_input"]
    if (dict(identity) != expected or not expected["group_id"] or not expected["variant"]
            or scopes != [{"input_sha256": input_fingerprint(sample, as_of),
                "identity": expected, "contract_version": "v1.1", "case_id": case_id}]
            or len([e for e in events if e.type == "run_outcome"]) != 1):
        raise ValueError("propagation_source_snapshot_mismatch")
    requests, calls = _request_chain(ctx, events)
    experiment = _verify_wallet(wallet, ctx, expected, set(requests), allowed_pending)
    inherited = _read_inheritance(ctx, events, requests, followup) if followup is not None else ((), None)
    comments, stats, queries = _deliveries(ctx, events, requests, calls, sample, as_of, inherited)
    return PropagationDelivery(case_id, sample.sample_id, ctx.run_id, round_no,
        input_fingerprint(sample, as_of), canonical(expected), comments, stats, queries,
        frozenset(c.comment_id for c in _comment_items(_visible_nodes(sample.propagation, as_of))),
        sha256(raw).hexdigest(), wallet, experiment)


def _read_inheritance(ctx: RunContext, events: list[TraceEvent], requests: dict, plan: object) -> tuple:
    """Verify the retained whole-prefix material against the original plan and actual requests."""
    from html import unescape
    from src.yuqing.collab.propagation_followup import _messages
    from src.yuqing.agents.propagation_v11 import stats_receipt
    matches = [e for e in events if e.type == "propagation_followup_input"]
    if len(matches) != 1 or ctx.run_id != plan.target_run_id:
        raise ValueError("propagation_inheritance_scope_mismatch")
    event = matches[0]
    if (event.data.get("case_id") != plan.source.case_id
            or event.data.get("source_run_id") != plan.source.run_id
            or event.data.get("brief") != json.loads(plan.brief_json)
            or event.data.get("propagation_round") != plan.round_no
            or event.data.get("decision_round") != plan.decision_round):
        raise ValueError("propagation_inheritance_task_mismatch")
    archive = _archive(ctx, event, "tool_result")
    blocks = archive["messages"]
    raw = unescape(blocks[-1]["content"].split('>', 1)[1].rsplit('</', 1)[0])
    data = json.loads(raw)
    comments = tuple(CommentItem.model_validate(c) for c in data["comments"])
    receipt = data["stats_receipt"]
    stats = plan.source.stats if receipt is not None else None
    if (comments != plan.source.comments[:len(comments)]
            or (receipt is not None and (stats is None or receipt != stats_receipt(stats).model_dump(mode="json")))
            or archive.get("source_trace_sha256") != plan.source.trace_sha256
            or archive.get("source_returned_ids") != [c.comment_id for c in plan.source.comments]
            or blocks != _messages([], plan, list(comments), stats)
            or event.data.get("inherited_comment_ids") != [c.comment_id for c in comments]):
        raise ValueError("propagation_inheritance_material_mismatch")
    if any([m for m in doc["body"]["messages"] if m in blocks] != blocks for _, doc in requests.values()):
        raise ValueError("propagation_inheritance_not_delivered")
    return comments, stats


def read_qualified_propagation(result: object, *, case_id: str, sample: Sample,
        ctx: RunContext, as_of: datetime | None, wallet: object,
        identity: Mapping[str, str], followup: object | None = None,
        allowed_pending: Mapping[str, Mapping[str, str]] | None = None) -> PropagationDelivery:
    """Revalidate a final accepted report and its cited material before graph publication."""
    from src.yuqing.agents.propagation import PropagationResult, review_propagation_report
    checked = PropagationResult.model_validate(result.model_dump())
    delivery = read_propagation_delivery(case_id=case_id, sample=sample, ctx=ctx, as_of=as_of,
        wallet=wallet, identity=identity, round_no=followup.round_no if followup else 1,
        followup=followup, allowed_pending=allowed_pending)
    events = [TraceEvent.model_validate_json(line) for line in ctx.trace_path.read_bytes().splitlines() if line.strip()]
    accepted = []
    for event in events:
        if event.type == "tool_call" and event.data.get("tool") == "submit_propagation_report":
            doc = _archive(ctx, event, "tool_result")
            if json.loads(doc["content"]).get("feedback", {}).get("accepted") is True:
                accepted.append(doc["request"]["arguments"])
    outcome = next(e.data for e in events if e.type == "run_outcome")
    feedback = review_propagation_report(checked.report, claim=sample.claim, raw=sample.propagation,
        returned_comments={c.comment_id: c for c in delivery.comments}, stats=delivery.stats, as_of=as_of)
    if (checked.run_id != ctx.run_id or checked.sample_id != sample.sample_id
            or accepted != [checked.report.model_dump(mode="json")] or not feedback.accepted
            or len(checked.comments) != len(delivery.comments)
            or {c.comment_id: c for c in checked.comments} != {c.comment_id: c for c in delivery.comments}
            or checked.stats != delivery.stats
            or outcome.get("outcome_class") != "submitted_report"
            or outcome.get("cost_cny") != checked.cost_cny
            or outcome.get("input_output_tokens") != checked.input_output_tokens):
        raise ValueError("propagation_final_report_mismatch")
    return delivery
