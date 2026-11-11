"""M6c fixed-eight input archive, one-time claim, durable journal, and audit."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from experiments.archive.m5_inputs import _initial_rows, _json_bytes
from experiments.archive import m6b_execution, m6b_run

M6C_CLAIM = "traces/m6c-fixed-eight.claim"


def archive_inputs(root: Path, selection: dict[str, Any], cases: tuple[Any, ...],
                   conditions: dict[str, str]) -> dict[str, Any]:
    """Snapshot the selected cases and the exact initial evidence shown to the graph."""
    files = {"selection.json": _json_bytes(selection) + b"\n",
        "conditions.json": _json_bytes(conditions) + b"\n"}
    initials = _initial_rows(root)
    for case in cases:
        prefix = f"case-{case.order:02d}"
        sample = case.sample.model_dump_json().encode("utf-8")
        files[f"{prefix}/sample.json"] = sample
        initial = _initial_bytes(case, initials)
        if initial is not None:
            files[f"{prefix}/initial-evidence.json"] = initial
        files[f"{prefix}/identity.json"] = _json_bytes({"order": case.order,
            "sample_id": case.sample.sample_id, "group": case.group,
            "sample_sha256": _sha(sample), "initial_sha256": _sha(initial) if initial else None}) + b"\n"
    body = {"version": "m6c-inputs-v1", "selection_sha256": _sha(files["selection.json"]),
        "files": {name: _sha(data) for name, data in sorted(files.items())}}
    manifest = _json_bytes(body) + b"\n"
    files["manifest.json"] = manifest
    return {"files": files, "sha256": _sha(manifest)}


def _initial_bytes(case: Any, initials: dict[str, Any]) -> bytes | None:
    """Return the case-bound initial evidence manifest; never synthesize one."""
    if case.joint_input is not None:
        path = case.joint_input.store / "initial.json"
        from src.yuqing.harness.initial_evidence import load_initial_evidence_manifest
        load_initial_evidence_manifest(path, case.sample.sample_id, case.sample.claim)
        return path.read_bytes()
    manifest = initials.get(case.sample.sample_id)
    if manifest is None:
        raise ValueError("m6c_chef_initial_evidence_missing")
    if manifest.sample_id != case.sample.sample_id or manifest.claim_sha256 != sha256(
            case.sample.claim.encode("utf-8")).hexdigest():
        raise ValueError("m6c_chef_initial_identity_mismatch")
    type(manifest).model_validate_json(manifest.model_dump_json().encode("utf-8"), strict=True)
    return manifest.model_dump_json().encode("utf-8")


def claim_run(root: Path, run_dir: Path, run_id: str, conditions: dict[str, str]) -> None:
    """Atomically claim the only fixed-eight attempt; an existing claim is never replaced."""
    target = run_dir.resolve()
    payload = {"run_id": run_id, "run_dir": target.relative_to((root / "traces").resolve()).as_posix(),
        "conditions_sha256": _digest(conditions), "status": "claimed"}
    path = root / M6C_CLAIM
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m6c_fixed_eight_already_claimed") from exc


def audit_archive(run_dir: Path, expected_sha: str) -> None:
    """Verify every archived byte and reject path traversal or a changed manifest."""
    folder = run_dir / "input_archive"
    raw = (folder / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if _sha(raw) != expected_sha or manifest.get("selection_sha256") != _sha(
            (folder / "selection.json").read_bytes()):
        raise ValueError("m6c_input_archive_manifest_mismatch")
    for name, digest in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or _sha(path.read_bytes()) != digest:
            raise ValueError("m6c_input_archive_file_mismatch")


def _digest(value: object) -> str:
    """Hash canonical JSON independent of mapping insertion order."""
    return m6b_execution._digest(value)


def _sha(data: bytes) -> str:
    """Hash exact bytes in an archive or claim."""
    return sha256(data).hexdigest()
