"""Validate the one author-approved replacement for the failed M8 attempt."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from types import MappingProxyType
from urllib.parse import quote

from src.yuqing.eval.experiment_budget import ExperimentBudgetStatus
from src.yuqing.eval.experiment_budget_recovery import RecoveredState, replay_history
from src.yuqing.eval.experiment_budget_store import (
    LedgerHistory, _GENESIS, _digest as _ledger_digest, _event_data, _json,
    _manifest_data,
)
from experiments.archive.m8_run_journal import (
    M8RunJournal, _read_events, _read_manifest, _unique,
)
from experiments.archive.m8_run_preflight import M8PreflightReport

PARENT_RUN = "m8-fixed-100-94c91dd"
PARENT_MANIFEST_SHA256 = "1dab527df085a224834c55c4048984a878f3dc0cab13b6668e89c2db985d6ba8"
PARENT_TREE_SHA256 = "89205772b3881c6a6e448980625db9c7c502bec46a7207050a07e87f6b455064"
PARENT_COST_CNY = Decimal("0.055072")
TOTAL_CAP_CNY = Decimal("20")


@dataclass(frozen=True)
class ReadOnlyWallet:
    """Expose only the validated history and snapshot required by M8 audit."""

    manifest: MappingProxyType
    history: LedgerHistory
    state: RecoveredState

    def manifest_snapshot(self) -> MappingProxyType:
        """Return the immutable wallet identity checked against the parent journal."""
        return self.manifest

    def load_history(self) -> LedgerHistory:
        """Return the append-only history loaded from SQLite in read-only mode."""
        return self.history

    def snapshot(self) -> ExperimentBudgetStatus:
        """Return costs recomputed by replaying the validated wallet history."""
        return self.state.snapshot()


@dataclass(frozen=True)
class ApprovedParentAudit:
    """Carry the verified parent input identity and settled spend to retry gates."""

    summary_sha256: str
    archive_sha256: str
    slot_plan_sha256: str
    known_cost_cny: Decimal


def audit_approved_parent(root: Path, parent_run_dir: Path) -> ApprovedParentAudit:
    """Audit the exact failed M8 attempt and return its frozen identity and spend."""
    traces = (root / "traces").resolve(strict=True)
    parent = parent_run_dir.resolve(strict=False)
    if parent != traces / PARENT_RUN:
        raise ValueError("m8_retry_parent_not_authorized")
    manifest = _read_manifest(parent / "manifest.json")
    if manifest["manifest_sha256"] != PARENT_MANIFEST_SHA256:
        raise ValueError("m8_retry_parent_manifest_changed")
    if _parent_tree_sha256(parent) != PARENT_TREE_SHA256:
        raise ValueError("m8_retry_parent_artifacts_changed")
    _verify_parent_claim(traces / "m8-fixed-100.claim", manifest)
    events = _read_events(parent / "events.jsonl", manifest)
    _verify_parent_slots(events)
    wallet = _read_only_wallet(parent / "wallet.sqlite")
    state = replay_history(wallet.history, wallet.manifest)
    if wallet.snapshot() != state.snapshot():
        raise ValueError("m8_retry_parent_wallet_mismatch")
    journal = M8RunJournal.resume(root, parent, expected_manifest=manifest)
    try:
        journal.audit(wallet)
    finally:
        journal.close()
    status = state.snapshot()
    if (status.known_cost_subtotal_cny != PARENT_COST_CNY or status.unknown_requests
            or status.pending_requests or status.unknown_reserved_cny or status.pending_reserved_cny
            or status.halt_reason != "m8_slot_runtime_failure"):
        raise ValueError("m8_retry_parent_spend_unresolved")
    return ApprovedParentAudit(str(manifest["summary_sha256"]),
        str(manifest["archive_sha256"]), str(manifest["slot_plan_sha256"]),
        status.known_cost_subtotal_cny)


def validate_retry_input_identity(report: M8PreflightReport,
                                  parent: ApprovedParentAudit) -> None:
    """Require the replacement to use the same summary, archive, and slot order."""
    if (report.summary_sha256 != parent.summary_sha256
            or report.archive_sha256 != parent.archive_sha256
            or report.slot_plan_sha256 != parent.slot_plan_sha256):
        raise ValueError("m8_retry_input_identity_changed")


def remaining_cap(known_spend: Decimal) -> Decimal:
    """Subtract the verified old spend from the fixed cumulative M8 ceiling."""
    if known_spend != PARENT_COST_CNY:
        raise ValueError("m8_retry_parent_spend_changed")
    return TOTAL_CAP_CNY - known_spend


def validate_retry_cap(requested: Decimal, known_spend: Decimal) -> None:
    """Reject a replacement cap that would exceed the original cumulative ceiling."""
    if requested <= 0 or requested > remaining_cap(known_spend):
        raise ValueError("m8_retry_cumulative_cap_exceeded")


def _verify_parent_claim(path: Path, manifest: dict[str, object]) -> None:
    """Require the unchanged one-time claim to reference the pinned parent run."""
    claim = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    if claim != {"run_dir": PARENT_RUN, "experiment_id": "m8-fixed-100",
            "manifest_sha256": PARENT_MANIFEST_SHA256, "status": "claimed"}:
        raise ValueError("m8_retry_parent_claim_changed")
    if manifest["experiment_id"] != claim["experiment_id"]:
        raise ValueError("m8_retry_parent_claim_mismatch")


def _verify_parent_slots(events: list[dict[str, object]]) -> None:
    """Allow replacement only for the known first-slot failure and untouched suffix."""
    finished = [event for event in events if event.get("event_type") == "slot_finished"]
    result = finished[0].get("result") if finished else None
    if (len(finished) != 160 or finished[0].get("status") != "failed"
            or finished[0].get("slot_index") != 1
            or not isinstance(result, dict) or result.get("failure_type") != "ValidationError"
            or any(event.get("status") != "unexecuted" for event in finished[1:])):
        raise ValueError("m8_retry_parent_slot_history_not_replaceable")


def _read_only_wallet(path: Path) -> ReadOnlyWallet:
    """Read and validate the exact SQLite hash chain without opening it for writes."""
    uri = "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        rows = connection.execute(
            "SELECT revision, event_json, previous_hash, event_hash FROM events ORDER BY revision"
        ).fetchall()
    finally:
        connection.close()
    history = _validated_history(metadata, rows)
    state = replay_history(history, history.manifest)
    return ReadOnlyWallet(MappingProxyType(dict(history.manifest)), history, state)


def _validated_history(metadata: dict[str, str], rows: list[tuple]) -> LedgerHistory:
    """Validate wallet metadata, canonical rows, and every chained SQLite digest."""
    required = {"format", "manifest", "manifest_hash", "revision", "head_hash"}
    if set(metadata) != required or metadata["format"] != "1":
        raise ValueError("m8_retry_wallet_metadata_invalid")
    raw = metadata["manifest"]
    if sha256(raw.encode()).hexdigest() != metadata["manifest_hash"]:
        raise ValueError("m8_retry_wallet_manifest_digest_mismatch")
    manifest = _manifest_data(json.loads(raw))
    if _json(manifest) != raw:
        raise ValueError("m8_retry_wallet_manifest_not_canonical")
    events, hashes, previous = [], [], _GENESIS
    for expected, (revision, event_json, prior, digest) in enumerate(rows, start=1):
        event = _event_data(json.loads(event_json))
        if revision != expected or prior != previous or _json(event) != event_json:
            raise ValueError("m8_retry_wallet_event_chain_invalid")
        if _ledger_digest(revision, prior, event_json) != digest:
            raise ValueError("m8_retry_wallet_event_chain_invalid")
        events.append(MappingProxyType(event))
        hashes.append(digest)
        previous = digest
    if int(metadata["revision"]) != len(rows) or metadata["head_hash"] != previous:
        raise ValueError("m8_retry_wallet_revision_invalid")
    return LedgerHistory(MappingProxyType(manifest), len(rows), tuple(events), tuple(hashes))


def _parent_tree_sha256(parent: Path) -> str:
    """Hash every immutable parent artifact except the OS lock handle file."""
    base = parent.resolve(strict=True)
    rows = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == "run.lock":
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(base):
            raise ValueError("m8_retry_parent_path_escape")
        rows.append({"path": resolved.relative_to(base).as_posix(),
            "sha256": sha256(resolved.read_bytes()).hexdigest()})
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()
