"""从可信模型请求和工具回执重建实际交付给模型的证据正文。"""

from __future__ import annotations

from hashlib import sha256
import html
import json
import re
from typing import Mapping

from src.yuqing.collab.fact_archive import (ArchiveReadSession, _archive_record, _canonical,
    _mapping, _message_tool_calls, _response_tool_calls)
from src.yuqing.harness.initial_evidence import (InitialEvidenceManifestV2,
    InitialEvidenceReceiptItemV2, parse_delivery_message)
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.harness.inherited_evidence import followup_task_block_from_scope, inherited_evidence_block
from src.yuqing.schemas import (EvidenceToolPayload, FactDeliveredEvidenceV2, FactScopeV2,
    InheritedEvidencePlanV2, TaskInput)


def _digest(payload: bytes) -> str:
    """Return the SHA-256 fingerprint of exact bytes."""
    return sha256(payload).hexdigest()


def _reconstruct_delivered(final: TraceEvent, manifest: InitialEvidenceManifestV2 | InheritedEvidencePlanV2, task: TaskInput,
                           events: list[TraceEvent], requests: dict[str, TraceEvent],
                           responses: dict[str, TraceEvent], archives: ArchiveReadSession, scope: FactScopeV2
                           ) -> tuple[dict[str, FactDeliveredEvidenceV2], list[dict[str, str]]]:
    """Rebuild only initial evidence and successful read bodies in the final request."""
    if isinstance(manifest, InheritedEvidencePlanV2):
        return _reconstruct_followup_delivered(final, manifest, task, events, requests, responses, archives, scope)
    request_id = str(final.data["request_id"])
    request_payload, _ = archives.read(_mapping(requests[request_id].data.get("archive")), scope, "model_request")
    body = _mapping(request_payload.get("body"))
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("fact_final_request_messages_missing")
    parsed_initial = _verify_delivery_receipt(events, request_id, messages, manifest, task)
    delivered = _initial_delivery(parsed_initial, manifest)
    prior_events, prior_responses = _prior_request_events(events, responses, request_id)
    tool_hashes = _read_tool_deliveries(messages, delivered, prior_events, prior_responses, archives, scope)
    return delivered, tool_hashes


def _reconstruct_followup_delivered(final: TraceEvent, plan: InheritedEvidencePlanV2, task: TaskInput,
                                   events: list[TraceEvent], requests: dict[str, TraceEvent],
                                   responses: dict[str, TraceEvent], archives: ArchiveReadSession,
                                   scope: FactScopeV2
                                   ) -> tuple[dict[str, FactDeliveredEvidenceV2], list[dict[str, str]]]:
    """Verify inherited request bytes and receipt before adding this run's confirmed tool reads."""
    request_id = str(final.data["request_id"])
    request, _ = archives.read(_mapping(requests[request_id].data.get("archive")), scope, "model_request")
    messages = _mapping(request.get("body")).get("messages")
    if not isinstance(messages, list):
        raise ValueError("fact_final_request_messages_missing")
    receipts = [event.data for event in events if event.type == "evidence_delivery"
                and event.data.get("request_id") == request_id]
    expected = [{"evidence_id": item.evidence_id, "origin": "inherited",
        "origin_refs": list(item.origin_refs), "read_end": item.read_end,
        "body_sha256": item.body_sha256, "prefix_sha256": item.prefix_sha256} for item in plan.entries]
    if (len(receipts) != 1 or receipts[0].get("status") != "included_in_request"
            or receipts[0].get("messages_sha256") != _digest(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode())
            or receipts[0].get("inherited_plan_sha256") != _digest(plan.model_dump_json().encode())
            or receipts[0].get("inherited_evidence") != expected):
        raise ValueError("fact_followup_delivery_receipt_mismatch")
    user_messages = [item.get("content") for item in messages if isinstance(item, dict) and item.get("role") == "user"]
    claims = [item for item in user_messages if isinstance(item, str) and "<claim>" in item]
    blocks = [item for item in user_messages if isinstance(item, str)
              and ("<followup_task_v2" in item or "<inherited_evidence_v2" in item)]
    if (claims != [f"<claim>{html.escape(task.claim, quote=False)}</claim>"]
            or blocks != [followup_task_block_from_scope(task, plan.target_scope), inherited_evidence_block(plan)]):
        raise ValueError("fact_followup_request_block_mismatch")
    delivered = {item.evidence_id: FactDeliveredEvidenceV2(evidence_id=item.evidence_id,
        text=item.text, read_end=item.read_end, body_sha256=item.body_sha256, sources_json=item.sources_json)
        for item in plan.entries}
    prior_events, prior_responses = _prior_request_events(events, responses, request_id)
    hashes = _read_tool_deliveries(messages, delivered, prior_events, prior_responses, archives, scope)
    return delivered, hashes


def _prior_request_events(events: list[TraceEvent], responses: dict[str, TraceEvent],
                          request_id: str) -> tuple[list[TraceEvent], dict[str, TraceEvent]]:
    """Limit read evidence to tool calls completed before the selected delivered request."""
    matches = [index for index, event in enumerate(events)
               if event.type == "llm_request" and event.data.get("request_id") == request_id]
    if len(matches) != 1:
        raise ValueError("fact_delivery_request_boundary_invalid")
    boundary = matches[0]
    response_indexes = {str(event.data.get("request_id")): index
                        for index, event in enumerate(events) if event.type == "model_response_archive"}
    prior_responses = {key: event for key, event in responses.items()
                       if response_indexes.get(key, boundary) < boundary}
    return events[:boundary], prior_responses


def _verify_delivery_receipt(events: list[TraceEvent], request_id: str, messages: list[object],
                             manifest: InitialEvidenceManifestV2,
                             task: TaskInput) -> tuple[InitialEvidenceReceiptItemV2, ...]:
    """Check actual final request bytes against the program's successful delivery receipt."""
    receipt = [event for event in events if event.type == "evidence_delivery"
               and event.data.get("request_id") == request_id]
    raw_messages = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(receipt) != 1 or receipt[0].data.get("status") != "included_in_request":
        raise ValueError("fact_final_delivery_receipt_missing")
    if receipt[0].data.get("messages_sha256") != _digest(raw_messages):
        raise ValueError("fact_final_delivery_message_hash_mismatch")
    claim = next((message.get("content") for message in messages if isinstance(message, dict)
                  and message.get("role") == "user" and "<claim>" in str(message.get("content"))), None)
    match = re.search(r"<claim>(.*?)</claim>", claim, re.S) if isinstance(claim, str) else None
    if match is None or _digest(html.unescape(match.group(1)).encode()) != _digest(task.claim.encode()):
        raise ValueError("fact_final_claim_mismatch")
    initial_messages = [str(message.get("content")) for message in messages
                        if isinstance(message, dict) and message.get("role") == "user"
                        and "<initial_evidence_v2" in str(message.get("content", ""))]
    if not initial_messages:
        raise ValueError("fact_final_initial_evidence_missing")
    parsed = [parse_delivery_message(message) for message in initial_messages]
    initial = parsed[0]
    if any(item != initial for item in parsed[1:]):
        raise ValueError("fact_initial_delivery_changed")
    event = receipt[0].data
    expected_initial = [{"evidence_id": item.evidence_id, "read_start": 0,
        "read_end": len(item.text), "body_sha256": _digest(item.text.encode())} for item in initial]
    if (event.get("initial_evidence_ids") != sorted(item.evidence_id for item in initial)
            or event.get("initial_evidence") != expected_initial
            or _digest(task.claim.encode()) != getattr(manifest, "claim_sha256", None)):
        raise ValueError("fact_final_initial_receipt_mismatch")
    return initial


def _initial_delivery(parsed: tuple[InitialEvidenceReceiptItemV2, ...], manifest: InitialEvidenceManifestV2
                      ) -> dict[str, FactDeliveredEvidenceV2]:
    """Bind one parsed initial evidence block to frozen manifest bodies and ids."""
    if len(parsed) != len(manifest.entries):
        raise ValueError("fact_initial_delivery_count_mismatch")
    delivered: dict[str, FactDeliveredEvidenceV2] = {}
    for item, entry in zip(parsed, manifest.entries, strict=True):
        if item.text != entry.text or item.evidence_id in delivered:
            raise ValueError("fact_initial_delivery_body_mismatch")
        delivered[item.evidence_id] = FactDeliveredEvidenceV2(evidence_id=item.evidence_id,
            text=item.text, read_end=len(item.text), body_sha256=_digest(item.text.encode()),
            sources_json=_canonical([source.model_dump(mode="json") for source in entry.sources]))
    return delivered


def _read_tool_deliveries(messages: list[object], delivered: dict[str, FactDeliveredEvidenceV2],
                          events: list[TraceEvent], responses: dict[str, TraceEvent],
                          archives: ArchiveReadSession, scope: FactScopeV2) -> list[dict[str, str]]:
    """Verify every read tool message against its response call and full ledger archive."""
    tools = [event for event in events if event.type == "tool_call" and event.data.get("tool") == "read_evidence"]
    by_call = {event.data.get("call_id"): event for event in tools}
    if len(by_call) != len(tools):
        raise ValueError("fact_read_call_id_duplicate")
    model_calls = _read_calls(messages)
    if set(model_calls) != set(by_call):
        raise ValueError("fact_read_call_pair_mismatch")
    response_index = _index_response_calls(responses, archives, scope)
    hashes: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if call_id not in by_call:
            continue
        event = by_call[str(call_id)]
        archive_sha = _verify_one_read(str(call_id), message, event, model_calls[str(call_id)],
                                       delivered, archives, scope)
        matches = response_index.get(str(call_id), [])
        if len(matches) != 1:
            raise ValueError("fact_tool_response_request_ambiguous")
        _, response_event, response, response_sha = matches[0]
        _verify_response_call(response, str(call_id), model_calls[str(call_id)])
        hashes.extend((_archive_record("model_response", response_event, response_sha),
                       _archive_record("tool_result", event, archive_sha)))
    return hashes


def _index_response_calls(responses: dict[str, TraceEvent], archives: ArchiveReadSession,
                          scope: FactScopeV2) -> dict[str, list[tuple[str, TraceEvent, Mapping[str, object], str]]]:
    """Index response tool calls by id without reopening response archives."""
    index: dict[str, list[tuple[str, TraceEvent, Mapping[str, object], str]]] = {}
    for request_id, event in responses.items():
        response, digest = archives.read(_mapping(event.data.get("archive")), scope, "model_response")
        calls = _response_tool_calls(response)
        for call in calls:
            call_id = call.get("id")
            if isinstance(call_id, str):
                index.setdefault(call_id, []).append((request_id, event, response, digest))
    return index


def _read_calls(messages: list[object]) -> dict[str, dict[str, object]]:
    """Collect read tool requests from the actual messages later delivered to the model."""
    calls: dict[str, dict[str, object]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in _message_tool_calls(message, error_code="fact_read_message_invalid"):
            function = call.get("function", {})
            if function.get("name") == "read_evidence":
                call_id = call.get("id")
                if not isinstance(call_id, str) or call_id in calls:
                    raise ValueError("fact_read_call_id_invalid")
                calls[call_id] = call
    return calls


def _verify_response_call(response: Mapping[str, object], call_id: str,
                          expected: Mapping[str, object]) -> None:
    """Require the archived SDK response to contain the exact delivered tool request."""
    calls = _response_tool_calls(response)
    actual = [call for call in calls if call.get("id") == call_id]
    if len(actual) != 1 or actual[0] != expected:
        raise ValueError("fact_read_response_call_mismatch")


def _verify_one_read(call_id: str, message: dict[str, object], event: TraceEvent,
                     model_call: Mapping[str, object], delivered: dict[str, FactDeliveredEvidenceV2], archives: ArchiveReadSession,
                     scope: FactScopeV2) -> str:
    """Verify a successful read's delivered prefix and archive-backed full body."""
    archive, archive_sha = archives.read(_mapping(event.data.get("archive")), scope, "tool_result")
    request, result = _mapping(archive.get("request")), _mapping(archive.get("result"))
    payload = EvidenceToolPayload.model_validate_json(str(message.get("content")), strict=True)
    model_arguments = json.loads(model_call["function"]["arguments"])
    if (request.get("name") != "read_evidence" or request.get("call_id") != call_id
            or request.get("arguments") != model_arguments
            or result.get("call_id") != call_id or payload.model_dump(mode="json") != result.get("payload")
            or payload.status != event.data.get("status")):
        raise ValueError("fact_read_archive_mismatch")
    if payload.status != "success":
        return archive_sha
    if len(payload.items) != 1:
        raise ValueError("fact_read_result_count_invalid")
    item = payload.items[0]
    archived = [entry for entry in archive.get("evidence", [])
                if isinstance(entry, dict) and entry.get("evidence_id") == item.evidence_id]
    if len(archived) != 1:
        raise ValueError("fact_read_body_archive_missing")
    entry = archived[0]
    text, end = entry.get("text"), entry.get("read_end")
    evidence_audit = [entry for entry in event.data.get("evidence", [])
                      if entry.get("evidence_id") == item.evidence_id]
    if (len(evidence_audit) != 1 or not isinstance(text, str) or type(end) is not int or end != item.read_end
            or not text.startswith(item.text) or len(item.text) != item.read_end
            or sha256(text.encode()).hexdigest() != evidence_audit[0].get("body_sha256")
            or sha256(item.text.encode()).hexdigest() != evidence_audit[0].get("returned_sha256")):
        raise ValueError("fact_read_body_prefix_mismatch")
    sources = _canonical(entry.get("sources", []))
    candidate = FactDeliveredEvidenceV2(evidence_id=item.evidence_id, text=text, read_end=end,
                                         body_sha256=_digest(text.encode()), sources_json=sources)
    prior = delivered.get(item.evidence_id)
    if prior is not None and prior.body_sha256 != candidate.body_sha256:
        raise ValueError("fact_evidence_body_changed")
    if prior is None or candidate.read_end > prior.read_end:
        delivered[item.evidence_id] = candidate
    return archive_sha
