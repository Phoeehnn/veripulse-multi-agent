"""M6b固定十案的输入归档、一次性认领与运行审计。"""

from __future__ import annotations

from decimal import Decimal
import json
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.archive import joint_demo
from src.yuqing.eval.experiment_budget_recovery import replay_history
from experiments.archive.m5_checkpoint_audit import _audit_cost_rows, _audit_wallet, _read_slot_traces
from experiments.archive.m5_inputs import _initial_rows, _json_bytes
from src.yuqing.harness.initial_evidence import load_initial_evidence_manifest

M6B_CLAIM = "traces/m6b-fixed-ten.claim"


def archive_inputs(root: Path, selection: dict[str, Any], cases: tuple[Any, ...],
                   conditions: dict[str, str]) -> dict[str, Any]:
    """Build deterministic model-visible inputs before creating the experiment or wallet."""
    files: dict[str, bytes] = {"selection.json": (root / "docs/cards/M6b-C_selection.json").read_bytes(),
        "conditions.json": _json_bytes(conditions) + b"\n"}
    manifests = _initial_rows(root)
    for case in cases:
        folder = f"case-{case.order:02d}"
        sample = case.sample.model_dump_json().encode("utf-8")
        files[f"{folder}/sample.json"] = sample
        initial = _initial_bytes(case, manifests)
        if initial is not None:
            files[f"{folder}/initial-evidence.json"] = initial
        files[f"{folder}/case.json"] = _json_bytes({"order": case.order,
            "sample_id": case.sample.sample_id, "group": case.group,
            "sample_sha256": _sha(sample), "initial_sha256": _sha(initial) if initial else None}) + b"\n"
    unsigned = {"version": "m6b-inputs-v1", "selection_sha256": _sha(files["selection.json"]),
        "files": {name: _sha(payload) for name, payload in sorted(files.items())}}
    manifest = _json_bytes(unsigned) + b"\n"
    files["manifest.json"] = manifest
    return {"files": files, "sha256": _sha(manifest)}


def _initial_bytes(case: Any, manifests: dict[str, Any]) -> bytes | None:
    """Return the exact applicable initial-evidence bundle, validating its case binding."""
    if case.group == "ced_new":
        return None
    path = case.joint_input.store / "initial.json" if case.group == "joint_reuse" else None
    if path is not None:
        load_initial_evidence_manifest(path, case.sample.sample_id, case.sample.claim)
        return path.read_bytes()
    manifest = manifests.get(case.sample.sample_id)
    if manifest is None:
        raise ValueError("m6b_chef_initial_manifest_missing")
    if (manifest.sample_id != case.sample.sample_id
            or manifest.claim_sha256 != sha256(case.sample.claim.encode("utf-8")).hexdigest()):
        raise ValueError("m6b_chef_initial_manifest_identity_mismatch")
    raw = manifest.model_dump_json().encode("utf-8")
    type(manifest).model_validate_json(raw, strict=True)
    return raw


def claim_run(root: Path, run_dir: Path, experiment_id: str,
              conditions: dict[str, str]) -> None:
    """Atomically claim the sole M6b fixed-ten attempt; never replace a prior claim."""
    path = root / M6B_CLAIM
    payload = {"experiment_id": experiment_id,
        "run_dir": run_dir.relative_to((root / "traces").resolve()).as_posix(),
        "conditions_sha256": _digest(conditions), "status": "claimed"}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m6b_fixed_ten_already_claimed") from exc


def conditions_unchanged(root: Path, run_dir: Path, selection: dict[str, Any],
                         expected: dict[str, str]) -> bool:
    """Recheck registered runtime sources and the archived input manifest before each slot."""
    from experiments.archive.m6b_runner import _conditions_unchanged

    if not _conditions_unchanged(root, selection,
            {key: value for key, value in expected.items() if key != "input_archive_sha256"}):
        return False
    try:
        _audit_archive(run_dir, selection, expected["input_archive_sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Durably append one completed slot before the next fixed identity starts."""
    with path.open("ab") as stream:
        stream.write(_json_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_run(run_dir: Path, value: dict[str, Any]) -> None:
    """Atomically replace only this newly-created run's progress summary."""
    path = run_dir / "run.json"
    temporary = run_dir / "run.json.partial"
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def audit_run(root: Path, run_dir: Path, experiment_id: str, selection: dict[str, Any],
              cases: tuple[Any, ...], rows: list[dict[str, Any]], wallet: Any,
              conditions: dict[str, str], archive_sha: str) -> dict[str, Any]:
    """Reconcile frozen slots, archive, one-time claim, Trace and durable wallet history."""
    _audit_archive(run_dir, selection, archive_sha)
    _audit_claim(root, run_dir, experiment_id, conditions)
    _audit_rows(cases, rows, experiment_id)
    if _read_rows(run_dir / "answers.jsonl") != rows:
        raise ValueError("m6b_answer_journal_mismatch")
    requests, calls = _read_slot_traces(run_dir, rows)
    history, manifest = wallet.load_history(), wallet.manifest_snapshot()
    if (manifest.get("experiment_id") != experiment_id
            or Decimal(str(manifest.get("max_cost_cny"))) != Decimal("3")
            or manifest.get("conditions_fingerprint") != _digest(conditions)):
        raise ValueError("m6b_wallet_manifest_mismatch")
    state = replay_history(history, manifest)
    _audit_wallet(requests, calls, state.requests, SimpleNamespace(cases=cases))
    _audit_request_groups(state.requests)
    status = state.snapshot()
    if status != wallet.snapshot():
        raise ValueError("m6b_wallet_snapshot_mismatch")
    _audit_cost_rows(rows, state.requests)
    audit_status = "verified"
    if status.pending_requests:
        audit_status = "verified_stopped_with_pending"
    elif status.unknown_requests:
        audit_status = "verified_stopped_with_unknown_cost"
    return {"status": audit_status, "planned": 10,
        "executed": sum(row["status"] != "unexecuted" for row in rows),
        "trace_requests": len(requests), "trace_receipts": len(calls),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_requests": status.unknown_requests,
        "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_requests": status.pending_requests}


def _audit_archive(run_dir: Path, selection: dict[str, Any], expected_sha: str) -> None:
    """Verify every archived input byte and its frozen selection identity."""
    folder = run_dir / "input_archive"
    raw = (folder / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if _sha(raw) != expected_sha or manifest.get("selection_sha256") != _sha((folder / "selection.json").read_bytes()):
        raise ValueError("m6b_input_archive_manifest_mismatch")
    if json.loads((folder / "selection.json").read_bytes()) != selection:
        raise ValueError("m6b_archived_selection_changed")
    for name, digest in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or _sha(path.read_bytes()) != digest:
            raise ValueError("m6b_archived_input_changed")


def _audit_claim(root: Path, run_dir: Path, experiment_id: str,
                 conditions: dict[str, str]) -> None:
    """Require this run to own the immutable global M6b claim."""
    claim = json.loads((root / M6B_CLAIM).read_bytes())
    expected = {"experiment_id": experiment_id,
        "run_dir": run_dir.relative_to((root / "traces").resolve()).as_posix(),
        "conditions_sha256": _digest(conditions), "status": "claimed"}
    if claim != expected:
        raise ValueError("m6b_claim_identity_mismatch")


def _audit_rows(cases: tuple[Any, ...], rows: list[dict[str, Any]], experiment_id: str) -> None:
    """Require all ten fixed rows in order and only an unexecuted suffix."""
    expected = [(case.order, case.sample.sample_id) for case in cases]
    if len(rows) != 10 or [(row.get("order"), row.get("sample_id")) for row in rows] != expected:
        raise ValueError("m6b_result_slots_not_exact")
    if any(row.get("experiment_id") != experiment_id or row.get("status") not in
            {"completed", "failed", "unexecuted"} for row in rows):
        raise ValueError("m6b_result_identity_or_status_invalid")
    indices = [i for i, row in enumerate(rows) if row["status"] == "unexecuted"]
    if indices and indices != list(range(indices[0], 10)):
        raise ValueError("m6b_unexecuted_slots_not_final_suffix")


def _audit_request_groups(requests: dict[str, Any]) -> None:
    """Keep all model requests inside the single fixed M6b experimental group."""
    if any(item.identity.get("group_id") != "m6b_development"
            or item.identity.get("variant") != "m6b-offline-candidate" for item in requests.values()):
        raise ValueError("m6b_wallet_request_group_mismatch")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Reject partial lines and malformed answer records before comparing them."""
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("m6b_answer_journal_partial_line")
    return [json.loads(line) for line in raw.splitlines()]


def _digest(value: object) -> str:
    """Hash canonical JSON without depending on mapping insertion order."""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(payload: bytes) -> str:
    """Hash exact archive or ledger identity bytes."""
    return sha256(payload).hexdigest()
