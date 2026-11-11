"""Audit fixed-ten answers, Trace identities and the shared wallet before publication."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from src.yuqing.eval.experiment_budget_recovery import replay_history
from experiments.archive.m5_inputs import FIXED_CAP, M5Inputs, _json_bytes, _sha
from src.yuqing.harness.trace import TraceEvent


def audit_m5_checkpoint(root: Path, run_dir: Path, experiment_id: str,
                        inputs: M5Inputs, rows: list[dict[str, Any]], wallet: Any,
                        conditions: dict[str, str]) -> dict[str, object]:
    """Refuse official output unless all ten slots, traces and ledger reconcile."""
    _audit_claim(root, run_dir, experiment_id, conditions)
    _audit_rows(inputs, rows, experiment_id)
    saved = _read_answers(run_dir / "answers.jsonl", experiment_id)
    if saved != rows:
        raise ValueError("m5_answers_journal_mismatch")
    trace_requests, trace_calls = _read_slot_traces(run_dir, rows)
    history = wallet.load_history()
    manifest = wallet.manifest_snapshot()
    if (manifest.get("experiment_id") != experiment_id
            or Decimal(str(manifest.get("max_cost_cny"))) != FIXED_CAP
            or manifest.get("conditions_fingerprint") != _sha(_json_bytes(conditions))):
        raise ValueError("m5_wallet_manifest_mismatch")
    state = replay_history(history, manifest)
    _audit_wallet(trace_requests, trace_calls, state.requests, inputs)
    status = state.snapshot()
    if status.pending_requests:
        raise ValueError("m5_wallet_has_unsettled_requests")
    if status != wallet.snapshot():
        raise ValueError("m5_wallet_snapshot_mismatch")
    _audit_cost_rows(rows, state.requests)
    executed = sum(row["status"] != "unexecuted" for row in saved)
    return {"status": "verified", "planned": 10, "executed": executed,
        "trace_requests": len(trace_requests), "trace_receipts": len(trace_calls),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_requests": status.unknown_requests, "pending_requests": status.pending_requests}


def _audit_claim(root: Path, run_dir: Path, experiment_id: str,
                 conditions: dict[str, str]) -> None:
    """Bind the existing one-time claim and wallet manifest to this exact run."""
    claim = json.loads((root / "traces/m5-fixed-ten.claim").read_text(encoding="utf-8"))
    relative = run_dir.resolve().relative_to((root / "traces").resolve()).as_posix()
    expected = {"experiment_id": experiment_id, "run_dir": relative,
        "conditions_sha256": _sha(_json_bytes(conditions)), "status": "claimed"}
    if claim != expected:
        raise ValueError("m5_claim_identity_mismatch")


def _audit_rows(inputs: M5Inputs, rows: list[dict[str, Any]], experiment_id: str) -> None:
    """Require the exact frozen ten-slot order with no duplicate or foreign answer."""
    expected = [(case.order, case.sample.sample_id) for case in inputs.cases]
    actual = [(row.get("order"), row.get("sample_id")) for row in rows]
    if len(rows) != 10 or actual != expected:
        raise ValueError("m5_result_slots_not_exact_fixed_ten")
    if any(row.get("experiment_id") != experiment_id or row.get("status") not in
            {"completed", "failed", "unexecuted"} for row in rows):
        raise ValueError("m5_result_slot_identity_or_status_invalid")
    unexecuted = [index for index, row in enumerate(rows) if row["status"] == "unexecuted"]
    if unexecuted and unexecuted != list(range(unexecuted[0], len(rows))):
        raise ValueError("m5_unexecuted_slots_not_final_suffix")


def _read_answers(path: Path, experiment_id: str) -> list[dict[str, Any]]:
    """Read only complete durable rows and reject malformed or cross-run records."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("m5_answers_journal_partial_line")
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(not isinstance(row, dict) or row.get("experiment_id") != experiment_id for row in rows):
        raise ValueError("m5_answers_journal_identity_invalid")
    return rows


def _read_slot_traces(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[dict, dict]:
    """Read every executed slot Trace under its assigned run directory."""
    requests, calls = {}, {}
    for row in rows:
        if row["status"] == "unexecuted":
            if row.get("trace_root") is not None:
                raise ValueError("m5_unexecuted_slot_has_trace")
            continue
        relative = row.get("trace_root")
        if not isinstance(relative, str):
            raise ValueError("m5_executed_slot_trace_missing")
        folder = (run_dir / relative).resolve()
        if not folder.is_relative_to(run_dir.resolve()) or not folder.is_dir():
            raise ValueError("m5_trace_path_invalid")
        for path in folder.rglob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                event = TraceEvent.model_validate_json(line, strict=True)
                if event.sample_id != row["sample_id"]:
                    raise ValueError("m5_trace_sample_identity_mismatch")
                _index_trace_event(event, requests, calls)
    return requests, calls


def _index_trace_event(event: TraceEvent, requests: dict, calls: dict) -> None:
    """Index gateway request/receipt events and reject duplicate identities."""
    if event.type not in {"llm_request", "llm_call"}:
        return
    request_id = event.data.get("request_id")
    target = requests if event.type == "llm_request" else calls
    if not isinstance(request_id, str) or request_id in target:
        raise ValueError("m5_trace_request_identity_invalid")
    target[request_id] = event


def _audit_wallet(trace_requests: dict, trace_calls: dict, requests: Any,
                  inputs: M5Inputs) -> None:
    """Match every traced call to one lifecycle record and its selected sample/run."""
    allowed_samples = {case.sample.sample_id for case in inputs.cases}
    if set(trace_calls) - set(trace_requests) or set(trace_requests) != set(requests):
        raise ValueError("m5_trace_wallet_request_set_mismatch")
    for request_id, request in requests.items():
        identity = request.identity
        sample_id, run_id = identity.get("sample_id"), identity.get("run_id")
        if sample_id not in allowed_samples:
            raise ValueError("m5_wallet_sample_identity_invalid")
        if request_id not in trace_requests:
            raise ValueError("m5_wallet_request_trace_missing")
        event = trace_requests[request_id]
        if (event.run_id, event.sample_id) != (run_id, sample_id) or not run_id:
            raise ValueError("m5_wallet_trace_identity_mismatch")
        receipt = trace_calls.get(request_id)
        if request.state == "settled":
            if receipt is None or receipt.data.get("usage_known") is not True:
                raise ValueError("m5_settled_wallet_request_receipt_missing")
            _audit_receipt_identity(event, receipt)
            if Decimal(str(receipt.data.get("cost_cny"))) != request.actual:
                raise ValueError("m5_wallet_trace_cost_mismatch")
        elif request.state == "unknown":
            if receipt is None or receipt.data.get("usage_known") is not False:
                raise ValueError("m5_unknown_wallet_request_receipt_mismatch")
            _audit_receipt_identity(event, receipt)
        elif request.state == "cancelled":
            if receipt is not None:
                raise ValueError("m5_cancelled_request_has_receipt")
        elif request.state in {"reserved", "dispatched"}:
            if receipt is not None:
                raise ValueError("m5_pending_request_has_receipt")
        else:
            raise ValueError("m5_wallet_request_state_invalid")


def _audit_receipt_identity(request_event: TraceEvent, receipt: TraceEvent) -> None:
    """Require the usage receipt to belong to the exact model call in Trace."""
    if (receipt.run_id, receipt.sample_id, receipt.agent, receipt.step) != (
            request_event.run_id, request_event.sample_id, request_event.agent, request_event.step):
        raise ValueError("m5_trace_receipt_identity_mismatch")


def _audit_cost_rows(rows: list[dict[str, Any]], requests: Any) -> None:
    """Reconcile per-sample known subtotals and terminal unknown/pending state."""
    totals = {sample: {"known": Decimal(0), "unknown": False, "pending": 0}
        for sample in (row["sample_id"] for row in rows)}
    for request in requests.values():
        sample = request.identity.get("sample_id")
        if sample not in totals:
            raise ValueError("m5_wallet_sample_identity_invalid")
        if request.state == "settled":
            totals[sample]["known"] += request.actual or Decimal(0)
        elif request.state == "unknown":
            totals[sample]["unknown"] = True
        elif request.state in {"reserved", "dispatched"}:
            totals[sample]["pending"] += 1
    for row in rows:
        sample = row["sample_id"]
        known = totals[sample]["known"]
        if Decimal(str(row.get("known_cost_delta_cny", "0"))) != known:
            raise ValueError("m5_answer_cost_delta_mismatch")
        if bool(row.get("usage_unknown")) != totals[sample]["unknown"]:
            raise ValueError("m5_answer_unknown_usage_mismatch")
        if row.get("pending_requests") != totals[sample]["pending"]:
            raise ValueError("m5_answer_pending_request_mismatch")
