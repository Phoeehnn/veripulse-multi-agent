"""Reconcile M8's immutable slot journal with its Trace files and shared wallet."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from experiments.archive.m8_run_journal import M8RunJournal, _read_events, _safe_child
from src.yuqing.harness.trace import TraceEvent


def audit_m8_run(journal: M8RunJournal, wallet: object) -> dict[str, object]:
    """Verify one run's answer slots, gateway receipts, and durable budget history."""
    journal._ensure_open()
    rows = _read_events(journal.run_dir / "events.jsonl", journal.manifest)
    manifest = dict(wallet.manifest_snapshot())
    if (manifest.get("experiment_id") != journal.manifest["experiment_id"]
            or Decimal(str(manifest.get("max_cost_cny"))) != Decimal(
                journal.manifest["max_cost_cny"])
            or manifest.get("conditions_fingerprint") != journal.manifest["conditions_sha256"]):
        raise ValueError("m8_journal_wallet_manifest_mismatch")
    from src.yuqing.eval.experiment_budget_recovery import replay_history

    recovered = replay_history(wallet.load_history(), manifest)
    requests = recovered.requests
    trace_requests, trace_calls, assigned = _trace_index(journal.run_dir, rows)
    _match_wallet(requests, trace_requests, trace_calls, assigned, journal.manifest["slots"])
    status = recovered.snapshot()
    if status != wallet.snapshot():
        raise ValueError("m8_journal_wallet_snapshot_mismatch")
    return _summary(journal, rows, status)


def audit_m8_slot(journal: M8RunJournal, wallet: object, slot_index: int) -> None:
    """Verify the newest finished slot against its Trace and wallet before advancing."""
    journal._ensure_open()
    rows = _read_events(journal.run_dir / "events.jsonl", journal.manifest)
    current = next((row for row in rows if row.get("slot_index") == slot_index
        and row["event_type"] == "slot_finished"), None)
    if current is None or current["status"] == "unexecuted":
        raise ValueError("m8_journal_slot_audit_target_invalid")
    manifest, recovered = _wallet_requests(wallet, journal)
    assigned = [request_id for row in rows if row["event_type"] == "slot_finished"
        and row["status"] != "unexecuted" for request_id in row["request_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(recovered.requests):
        raise ValueError("m8_journal_slot_wallet_request_mismatch")
    slot = journal.manifest["slots"][slot_index - 1]
    requests, calls, slot_requests = {}, {}, set()
    for relative in current["trace_paths"]:
        path = _safe_child(journal.run_dir / relative, journal.run_dir, strict=True)
        if not path.is_file():
            raise ValueError("m8_journal_trace_missing")
        _index_trace_file(path, current, requests, calls, slot_requests)
    if slot_requests != set(current["request_ids"]):
        raise ValueError("m8_journal_slot_trace_request_mismatch")
    if set(calls) - set(requests):
        raise ValueError("m8_journal_trace_orphan_receipt")
    for request_id in slot_requests:
        request = recovered.requests.get(request_id)
        if request is None:
            raise ValueError("m8_journal_slot_wallet_request_missing")
        _verify_wallet_request(request, requests[request_id], calls.get(request_id), slot)
    if recovered.snapshot() != wallet.snapshot():
        raise ValueError("m8_journal_wallet_snapshot_mismatch")


def _wallet_requests(wallet: object, journal: M8RunJournal) -> tuple[dict, Any]:
    """Load one matching wallet history for a bounded slot audit."""
    from src.yuqing.eval.experiment_budget_recovery import replay_history

    manifest = dict(wallet.manifest_snapshot())
    if (manifest.get("experiment_id") != journal.manifest["experiment_id"]
            or Decimal(str(manifest.get("max_cost_cny"))) != Decimal(
                journal.manifest["max_cost_cny"])
            or manifest.get("conditions_fingerprint") != journal.manifest["conditions_sha256"]):
        raise ValueError("m8_journal_wallet_manifest_mismatch")
    recovered = replay_history(wallet.load_history(), manifest)
    return manifest, recovered


def _trace_index(run_dir: Path, events: list[dict[str, Any]]) -> tuple[dict, dict, dict[int, set[str]]]:
    """Read only the listed in-run traces and bind each request to its slot."""
    requests, calls, assigned = {}, {}, {}
    for event in events:
        if event["event_type"] != "slot_finished" or event["status"] == "unexecuted":
            continue
        index = int(event["slot_index"])
        assigned[index] = set(event["request_ids"])
        slot_trace_requests: set[str] = set()
        for relative in event["trace_paths"]:
            path = _safe_child(run_dir / relative, run_dir, strict=True)
            if not path.is_file():
                raise ValueError("m8_journal_trace_missing")
            _index_trace_file(path, event, requests, calls, slot_trace_requests)
        if slot_trace_requests != assigned[index]:
            raise ValueError("m8_journal_slot_trace_request_mismatch")
    return requests, calls, assigned


def _index_trace_file(path: Path, row: dict[str, Any], requests: dict, calls: dict,
                      slot_requests: set[str]) -> None:
    """Index unique request/receipt events after validating their sample identity."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trace = TraceEvent.model_validate_json(line, strict=True)
        if trace.sample_id != row["sample_id"]:
            raise ValueError("m8_journal_trace_sample_mismatch")
        if trace.type in {"llm_request", "llm_call"}:
            target = requests if trace.type == "llm_request" else calls
            request_id = trace.data.get("request_id")
            if not isinstance(request_id, str) or request_id in target:
                raise ValueError("m8_journal_trace_request_duplicate")
            target[request_id] = trace
            if trace.type == "llm_request":
                slot_requests.add(request_id)


def _match_wallet(wallet_requests: Any, trace_requests: dict, trace_calls: dict,
                  assigned: dict[int, set[str]], slots: list[dict[str, Any]]) -> None:
    """Require one-to-one request assignment and matching Trace/wallet identities."""
    assigned_ids = [request_id for values in assigned.values() for request_id in values]
    if len(assigned_ids) != len(set(assigned_ids)) or set(assigned_ids) != set(wallet_requests):
        raise ValueError("m8_journal_slot_wallet_request_mismatch")
    if set(trace_requests) != set(wallet_requests) or set(trace_calls) - set(trace_requests):
        raise ValueError("m8_journal_trace_wallet_request_mismatch")
    slot_by_index = {slot["slot_index"]: slot for slot in slots}
    assigned_slot = {request_id: slot_by_index[index] for index, ids in assigned.items()
        for request_id in ids}
    for request_id, request in wallet_requests.items():
        _verify_wallet_request(request, trace_requests[request_id], trace_calls.get(request_id),
            assigned_slot[request_id])


def _verify_wallet_request(request: Any, start: TraceEvent, receipt: TraceEvent | None,
                           slot: dict[str, Any]) -> None:
    """Check the request's sample/group, response state, and trusted cost receipt."""
    identity = request.identity
    groups = {slot["execution_group"], "shared-first-round"}
    if ((start.run_id, start.sample_id) != (identity.get("run_id"), identity.get("sample_id"))
            or identity.get("sample_id") != slot["sample_id"]
            or identity.get("group_id") not in groups):
        raise ValueError("m8_journal_wallet_trace_identity_mismatch")
    if request.state in {"settled", "unknown"}:
        _verify_receipt(request, start, receipt)
    elif request.state == "cancelled":
        if receipt is not None:
            raise ValueError("m8_journal_cancelled_request_has_receipt")
    elif request.state in {"reserved", "dispatched"}:
        if receipt is not None:
            raise ValueError("m8_journal_pending_request_has_receipt")
    else:
        raise ValueError("m8_journal_wallet_state_invalid")


def _verify_receipt(request: Any, start: TraceEvent, receipt: TraceEvent | None) -> None:
    """Require a matching usage-known flag and exact identity/cost for a settled request."""
    if receipt is None or receipt.data.get("usage_known") != (request.state == "settled"):
        raise ValueError("m8_journal_wallet_receipt_missing")
    if (receipt.run_id, receipt.sample_id, receipt.agent, receipt.step) != (
            start.run_id, start.sample_id, start.agent, start.step):
        raise ValueError("m8_journal_wallet_receipt_identity_mismatch")
    if request.state == "settled" and Decimal(str(receipt.data.get("cost_cny"))) != request.actual:
        raise ValueError("m8_journal_wallet_receipt_cost_mismatch")


def _summary(journal: M8RunJournal, rows: list[dict[str, Any]], status: Any) -> dict[str, object]:
    """Report original denominator and certainty without treating failures as missing rows."""
    executed = sum(row["event_type"] == "slot_finished" and row["status"] != "unexecuted"
        for row in rows)
    terminal = sum(row["event_type"] == "slot_finished" for row in rows)
    return {"status": "verified", "planned": len(journal.manifest["slots"]),
        "terminal_records": terminal, "executed": executed, "unexecuted": terminal - executed,
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_reserved_cny": str(status.pending_reserved_cny),
        "unknown_requests": status.unknown_requests, "pending_requests": status.pending_requests,
        "complete": terminal == len(journal.manifest["slots"]) and terminal == executed
            and status.unknown_requests == 0 and status.pending_requests == 0
            and status.cost_cny is not None and not status.halted}
