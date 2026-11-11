"""Durable M8 slot journal with fail-closed resume and wallet reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from src.yuqing.eval.experiment_budget import ExperimentBudgetStatus
from experiments.archive.m8_run_preflight import M8ExecutionSlot

_GENESIS = "genesis"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL = {"completed", "failed", "unexecuted"}
_RETRY_CLAIM = "m8-fixed-100-retry1.claim"
_RETRY_EXPERIMENT_ID = "m8-fixed-100-retry1"
_RETRY_PARENT_MANIFEST_SHA256 = "1dab527df085a224834c55c4048984a878f3dc0cab13b6668e89c2db985d6ba8"


def _json_bytes(value: object) -> bytes:
    """Encode stable UTF-8 JSON and reject non-finite values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")


def _digest(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return sha256(raw).hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys instead of silently keeping the last value."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("m8_journal_duplicate_json_key")
        result[key] = value
    return result


def _slot_data(slot: M8ExecutionSlot) -> dict[str, object]:
    """Convert a frozen preflight slot to its canonical persisted identity."""
    return {"order": slot.order, "sample_id": slot.sample_id, "source": slot.source,
        "execution_group": slot.execution_group, "logical_groups": list(slot.logical_groups)}


def _validate_slots(slots: tuple[M8ExecutionSlot, ...]) -> tuple[dict[str, object], ...]:
    """Require frozen source order with paired CHEF and single-run CED slots."""
    raw_rows = tuple(_slot_data(slot) for slot in slots)
    if len(raw_rows) != 160:
        raise ValueError("m8_journal_slot_plan_invalid")
    if any(type(row["order"]) is not int or not row["sample_id"] for row in raw_rows):
        raise ValueError("m8_journal_slot_identity_invalid")
    rows = tuple(dict(slot_index=index, **row) for index, row in enumerate(raw_rows, start=1))
    if len({(row["sample_id"], row["execution_group"]) for row in rows}) != len(rows):
        raise ValueError("m8_journal_duplicate_slot")
    groups: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(int(row["order"]), []).append(row)
    if set(groups) != set(range(1, 101)):
        raise ValueError("m8_journal_sample_order_invalid")
    sources: list[str] = []
    for order, items in groups.items():
        source = items[0]["source"]
        if source not in {"chef", "ced"} or any(row["source"] != source for row in items):
            raise ValueError("m8_journal_sample_group_layout_invalid")
        sources.append(str(source))
        chef = source == "chef"
        expected_groups = ("B4-nf", "B4-fixed") if chef and order % 2 else (
            ("B4-fixed", "B4-nf") if chef else ("CED-shared",))
        if (len(items) != len(expected_groups) or len({row["sample_id"] for row in items}) != 1
                or tuple(row["execution_group"] for row in items)
                != expected_groups or any(row["source"] != ("chef" if chef else "ced")
                for row in items)):
            raise ValueError("m8_journal_sample_group_layout_invalid")
    if sources.count("chef") != 60 or sources.count("ced") != 40:
        raise ValueError("m8_journal_source_count_invalid")
    return rows


def _run_root(root: Path) -> Path:
    """Resolve the project trace directory and require the configured D-drive root."""
    project = root.resolve(strict=True)
    traces = (project / "traces").resolve(strict=True)
    if project.drive.upper() != "D:" or not traces.is_relative_to(project):
        raise ValueError("m8_journal_project_root_invalid")
    return traces


def _safe_child(path: Path, parent: Path, *, strict: bool = False) -> Path:
    """Resolve a run artifact and reject paths escaping its assigned directory."""
    resolved = path.resolve(strict=strict)
    base = parent.resolve(strict=True)
    if resolved == base or not resolved.is_relative_to(base):
        raise ValueError("m8_journal_path_escape")
    return resolved


def _write_new(path: Path, payload: bytes) -> None:
    """Durably create an immutable file without replacing an existing artifact."""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class M8RunJournal:
    """One locked M8 run with an immutable plan and hash-chained per-slot records."""

    def __init__(self, root: Path, run_dir: Path, handle: Any, manifest: dict[str, Any],
                 events: list[dict[str, Any]]) -> None:
        """Keep the held process lock and the already validated journal state."""
        self.root, self.run_dir, self._handle = root.resolve(), run_dir.resolve(), handle
        self.manifest, self._events, self._closed = manifest, events, False
        self._storage_uncertain = False

    @classmethod
    def create(cls, root: Path, run_dir: Path, *, experiment_id: str,
               summary_sha256: str, archive_sha256: str, conditions_sha256: str,
               slot_plan_sha256: str, max_cost_cny: Decimal,
               slots: tuple[M8ExecutionSlot, ...],
               retry_parent_manifest_sha256: str | None = None) -> M8RunJournal:
        """Claim the single M8 attempt and publish a new run journal without overwrite."""
        traces = _run_root(root)
        target = _safe_child(run_dir, traces)
        retry = retry_parent_manifest_sha256 is not None
        claim_path = _validate_create_identity(traces, target, experiment_id,
            (summary_sha256, archive_sha256, conditions_sha256, slot_plan_sha256),
            retry_parent_manifest_sha256)
        if (not isinstance(max_cost_cny, Decimal) or not max_cost_cny.is_finite()
                or not Decimal("0") < max_cost_cny <= Decimal("20")):
            raise ValueError("m8_journal_budget_invalid")
        slot_rows = _validate_slots(slots)
        if _digest(_json_bytes([{key: value for key, value in row.items()
                if key != "slot_index"} for row in slot_rows])) != slot_plan_sha256:
            raise ValueError("m8_journal_slot_plan_fingerprint_mismatch")
        manifest = {"format": "m8-run-v2" if retry else "m8-run-v1",
            "experiment_id": experiment_id,
            "summary_sha256": summary_sha256, "archive_sha256": archive_sha256,
            "conditions_sha256": conditions_sha256, "slot_plan_sha256": slot_plan_sha256,
            "max_cost_cny": str(max_cost_cny), "slots": slot_rows}
        if retry:
            manifest["retry_parent_manifest_sha256"] = retry_parent_manifest_sha256
        manifest["manifest_sha256"] = _digest(_json_bytes(manifest))
        target.mkdir(parents=True, exist_ok=False)
        handle = _lock_run(target / "run.lock")
        try:
            _write_new(target / "manifest.json", _json_bytes(manifest) + b"\n")
            _write_new(target / "events.jsonl", b"")
            _claim_once(claim_path, target, manifest)
            return cls(root, target, handle, manifest, [])
        except BaseException:
            handle.close()
            raise

    @classmethod
    def resume(cls, root: Path, run_dir: Path, *, expected_manifest: dict[str, Any]) -> M8RunJournal:
        """Resume only an identical run with a clean terminal prefix and no started slot."""
        traces = _run_root(root)
        target = _safe_child(run_dir, traces, strict=True)
        handle = _lock_run(target / "run.lock")
        try:
            manifest = _read_manifest(target / "manifest.json")
            expected = json.loads(_json_bytes(expected_manifest), object_pairs_hook=_unique)
            if manifest != expected:
                raise ValueError("m8_journal_conditions_changed")
            _verify_claim(_claim_path(traces, manifest), target, manifest)
            events = _read_events(target / "events.jsonl", manifest)
            if events and events[-1]["event_type"] == "slot_started":
                raise ValueError("m8_resume_inflight_slot_requires_manual_review")
            return cls(root, target, handle, manifest, events)
        except BaseException:
            handle.close()
            raise

    @property
    def completed_slots(self) -> int:
        """Return the number of slots with a durable terminal record."""
        return sum(event["event_type"] == "slot_finished" for event in self._events)

    def finished_result(self, order: int, execution_group: str) -> dict[str, Any] | None:
        """Return a copy of one completed slot result for safe paired-branch recovery."""
        self._ensure_open()
        matches = [event for event in self._events
            if event["event_type"] == "slot_finished" and event.get("status") != "unexecuted"
            and event.get("sample_order") == order
            and event.get("execution_group") == execution_group]
        if len(matches) > 1:
            raise ValueError("m8_journal_finished_slot_duplicate")
        return None if not matches else deepcopy(matches[0].get("result", {}))

    def start_next(self, conditions_sha256: str,
                   wallet_status: ExperimentBudgetStatus) -> M8ExecutionSlot:
        """Durably mark exactly the next frozen slot started before its model work."""
        self.require_conditions(conditions_sha256)
        self._require_wallet_clear(wallet_status)
        if self._events and self._events[-1]["event_type"] == "slot_started":
            raise ValueError("m8_journal_slot_already_inflight")
        if self._events and self._events[-1].get("status") == "unexecuted":
            raise ValueError("m8_journal_stopped_suffix_is_terminal")
        index = self.completed_slots
        slots = self.manifest["slots"]
        if index >= len(slots):
            raise ValueError("m8_journal_all_slots_finished")
        row = slots[index]
        self._append({"event_type": "slot_started", "slot_index": row["slot_index"],
            "sample_order": row["order"],
            "sample_id": row["sample_id"], "execution_group": row["execution_group"],
            "slot_sha256": _digest(_json_bytes(row)), "timestamp": _now()})
        return M8ExecutionSlot(order=row["order"], sample_id=row["sample_id"],
            source=row["source"], execution_group=row["execution_group"],
            logical_groups=tuple(row["logical_groups"]))

    def _require_wallet_clear(self, status: ExperimentBudgetStatus) -> None:
        """Prevent progress when wallet identity differs or a request is unresolved."""
        if (not isinstance(status, ExperimentBudgetStatus)
                or status.experiment_id != self.manifest["experiment_id"]
                or status.max_cost_cny != Decimal(self.manifest["max_cost_cny"])
                or status.halted or status.unknown_requests or status.pending_requests
                or status.cost_cny is None):
            raise ValueError("m8_journal_wallet_not_resumable")

    def require_conditions(self, conditions_sha256: str) -> None:
        """Stop before a slot when the live runtime no longer matches its frozen snapshot."""
        self._ensure_open()
        if conditions_sha256 != self.manifest["conditions_sha256"]:
            raise ValueError("m8_journal_conditions_changed")

    def finish_slot(self, slot: M8ExecutionSlot, status: Literal["completed", "failed"],
                    *, result: dict[str, Any], request_ids: tuple[str, ...],
                    trace_paths: tuple[str, ...]) -> None:
        """Persist a terminal slot result and its exact wallet/Trace references."""
        self._ensure_open()
        if status not in _TERMINAL - {"unexecuted"} or not isinstance(result, dict):
            raise ValueError("m8_journal_result_invalid")
        if not self._events or self._events[-1]["event_type"] != "slot_started":
            raise ValueError("m8_journal_slot_not_started")
        started = self._events[-1]
        expected = self.manifest["slots"][int(started["slot_index"]) - 1]
        clean_expected = {key: value for key, value in expected.items() if key != "slot_index"}
        if (started["sample_order"] != slot.order or clean_expected != _slot_data(slot)
                or started["slot_sha256"] != _digest(_json_bytes(expected))):
            raise ValueError("m8_journal_slot_identity_mismatch")
        _validate_refs(request_ids, trace_paths)
        payload = {"event_type": "slot_finished", "slot_index": started["slot_index"],
            "sample_order": slot.order,
            "sample_id": slot.sample_id, "execution_group": slot.execution_group,
            "status": status, "result": result, "request_ids": list(request_ids),
            "trace_paths": list(trace_paths), "timestamp": _now()}
        self._append(payload)

    def mark_unexecuted_suffix(self, reason: str) -> None:
        """Record every remaining frozen slot as unexecuted after a fail-closed stop."""
        self._ensure_open()
        if not _valid_reason(reason) or (self._events and
                self._events[-1]["event_type"] == "slot_started"):
            raise ValueError("m8_journal_unexecuted_suffix_invalid")
        slots = self.manifest["slots"]
        for row in slots[self.completed_slots:]:
            self._append({"event_type": "slot_finished", "slot_index": row["slot_index"],
                "sample_order": row["order"],
                "sample_id": row["sample_id"], "execution_group": row["execution_group"],
                "status": "unexecuted", "reason": reason, "request_ids": [],
                "trace_paths": [], "timestamp": _now()})

    def audit(self, wallet: object) -> dict[str, object]:
        """Reconcile durable slot rows, Trace request receipts, and the shared wallet."""
        from experiments.archive.m8_run_audit import audit_m8_run

        return audit_m8_run(self, wallet)

    def close(self) -> None:
        """Release the process lock without deleting or changing durable records."""
        if not self._closed:
            self._closed = True
            self._handle.close()

    def _append(self, payload: dict[str, Any]) -> None:
        """Hash-chain and fsync one ordered event before publishing it in memory."""
        previous = self._events[-1]["event_sha256"] if self._events else _GENESIS
        event = dict(payload, previous_sha256=previous)
        event["event_sha256"] = _digest(_json_bytes(event))
        raw = _json_bytes(event) + b"\n"
        try:
            with (self.run_dir / "events.jsonl").open("ab") as stream:
                if stream.write(raw) != len(raw):
                    raise OSError("m8_journal_short_write")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            self._storage_uncertain = True
            raise ValueError("m8_journal_storage_uncertain") from exc
        self._events.append(event)

    def _ensure_open(self) -> None:
        """Reject any operation after the cross-process run lock was released."""
        if self._closed:
            raise ValueError("m8_journal_closed")
        if self._storage_uncertain:
            raise ValueError("m8_journal_storage_uncertain")


def _now() -> str:
    """Return a UTC timestamp for audit ordering, not for run identity."""
    return datetime.now(timezone.utc).isoformat()


def _valid_identity(value: str) -> bool:
    """Validate a stable experiment identifier before any run claim is written."""
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None


def _valid_sha_set(*values: str) -> bool:
    """Require lowercase SHA-256 identifiers for all frozen run inputs."""
    return all(isinstance(value, str) and _SHA.fullmatch(value) for value in values)


def _valid_reason(value: str) -> bool:
    """Accept concise stable stop codes only, not prompt or evidence text."""
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", value) is not None


def _validate_refs(request_ids: tuple[str, ...], trace_paths: tuple[str, ...]) -> None:
    """Reject duplicate IDs and non-relative trace references in one slot result."""
    if len(set(request_ids)) != len(request_ids) or any(
            not re.fullmatch(r"[0-9a-f]{32}", item) for item in request_ids):
        raise ValueError("m8_journal_request_ids_invalid")
    if len(set(trace_paths)) != len(trace_paths) or any(Path(item).is_absolute() or ".." in Path(item).parts
            for item in trace_paths):
        raise ValueError("m8_journal_trace_paths_invalid")


def _lock_run(path: Path) -> Any:
    """Acquire a non-blocking cross-process OS lock, retaining the handle while open."""
    handle = path.open("a+b")
    if os.name == "nt":
        import msvcrt

        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise ValueError("m8_journal_busy") from exc
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ValueError("m8_journal_busy") from exc
    return handle


def _claim_once(path: Path, run_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically reserve the single fixed M8 run after all preflight checks pass."""
    traces = path.parent.resolve(strict=True)
    payload = {"run_dir": run_dir.relative_to(traces).as_posix(),
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest["manifest_sha256"], "status": "claimed"}
    try:
        _write_new(path, _json_bytes(payload) + b"\n")
    except FileExistsError as exc:
        raise ValueError("m8_run_already_claimed") from exc


def _verify_claim(path: Path, run_dir: Path, manifest: dict[str, Any]) -> None:
    """Require resume to own the exact immutable one-time claim."""
    traces = path.parent.resolve(strict=True)
    expected = {"run_dir": run_dir.relative_to(traces).as_posix(),
        "experiment_id": manifest["experiment_id"],
        "manifest_sha256": manifest["manifest_sha256"], "status": "claimed"}
    if json.loads(path.read_bytes(), object_pairs_hook=_unique) != expected:
        raise ValueError("m8_run_claim_mismatch")


def _claim_path(traces: Path, manifest: dict[str, Any]) -> Path:
    """Select the immutable one-time claim associated with a journal version."""
    return traces / (_RETRY_CLAIM if manifest.get("format") == "m8-run-v2"
        else "m8-fixed-100.claim")


def _validate_create_identity(traces: Path, target: Path, experiment_id: str,
                              hashes: tuple[str, ...], retry_parent_sha256: str | None) -> Path:
    """Validate the unique initial claim or the one fixed replacement claim."""
    retry = retry_parent_sha256 is not None
    claim_path = traces / (_RETRY_CLAIM if retry else "m8-fixed-100.claim")
    if target.exists() or claim_path.exists() or not _valid_identity(experiment_id) or not _valid_sha_set(*hashes):
        raise ValueError("m8_journal_create_identity_invalid")
    if not retry:
        if experiment_id != "m8-fixed-100":
            raise ValueError("m8_journal_create_identity_invalid")
        return claim_path
    if (experiment_id != _RETRY_EXPERIMENT_ID
            or target.name != "m8-fixed-100-retry1"
            or retry_parent_sha256 != _RETRY_PARENT_MANIFEST_SHA256
            or not (traces / "m8-fixed-100.claim").is_file()):
        raise ValueError("m8_journal_retry_identity_invalid")
    _verify_retry_parent_claim(traces / "m8-fixed-100.claim")
    parent = traces / "m8-fixed-100-94c91dd"
    if _read_manifest(parent / "manifest.json")["manifest_sha256"] != _RETRY_PARENT_MANIFEST_SHA256:
        raise ValueError("m8_journal_retry_parent_manifest_invalid")
    return claim_path


def _verify_retry_parent_claim(path: Path) -> None:
    """Require the original claim to remain byte-consistent with the pinned failed run."""
    expected = {"run_dir": "m8-fixed-100-94c91dd", "experiment_id": "m8-fixed-100",
        "manifest_sha256": _RETRY_PARENT_MANIFEST_SHA256, "status": "claimed"}
    if json.loads(path.read_bytes(), object_pairs_hook=_unique) != expected:
        raise ValueError("m8_journal_retry_parent_claim_invalid")


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read the immutable manifest and verify its self-excluding digest."""
    record = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    common = {"format", "experiment_id", "summary_sha256", "archive_sha256",
        "conditions_sha256", "slot_plan_sha256", "max_cost_cny", "slots", "manifest_sha256"}
    expected = common | ({"retry_parent_manifest_sha256"}
        if record.get("format") == "m8-run-v2" else set())
    if set(record) != expected:
        raise ValueError("m8_journal_manifest_fields_invalid")
    if record.get("format") not in {"m8-run-v1", "m8-run-v2"}:
        raise ValueError("m8_journal_manifest_fields_invalid")
    if record["format"] == "m8-run-v2" and (
            record.get("experiment_id") != _RETRY_EXPERIMENT_ID
            or record.get("retry_parent_manifest_sha256") != _RETRY_PARENT_MANIFEST_SHA256):
        raise ValueError("m8_journal_retry_identity_invalid")
    digest = record.pop("manifest_sha256", None)
    if not isinstance(digest, str) or _digest(_json_bytes(record)) != digest:
        raise ValueError("m8_journal_manifest_hash_mismatch")
    record["manifest_sha256"] = digest
    stored_slots = record.get("slots")
    if not isinstance(stored_slots, list):
        raise ValueError("m8_journal_slot_plan_invalid")
    slots = tuple(M8ExecutionSlot(order=row["order"], sample_id=row["sample_id"],
        source=row["source"], execution_group=row["execution_group"],
        logical_groups=tuple(row["logical_groups"])) for row in stored_slots)
    if list(_validate_slots(slots)) != stored_slots:
        raise ValueError("m8_journal_slot_plan_invalid")
    return record


def _read_events(path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify complete lines, event hashes, slot order, and state transitions."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("m8_journal_partial_line")
    events, previous, next_index, active = [], _GENESIS, 0, False
    for line in raw.splitlines():
        event = json.loads(line, object_pairs_hook=_unique)
        digest = event.pop("event_sha256", None)
        if event.get("previous_sha256") != previous or _digest(_json_bytes(event)) != digest:
            raise ValueError("m8_journal_event_hash_mismatch")
        order, kind = event.get("slot_index"), event.get("event_type")
        if next_index >= len(manifest["slots"]) or order != manifest["slots"][next_index]["slot_index"]:
            raise ValueError("m8_journal_event_slot_order_invalid")
        expected_slot = manifest["slots"][next_index]
        if (event.get("sample_id"), event.get("execution_group"), event.get("sample_order")) != (
                expected_slot["sample_id"], expected_slot["execution_group"], expected_slot["order"]):
            raise ValueError("m8_journal_event_slot_identity_invalid")
        if kind == "slot_started" and not active:
            if event.get("slot_sha256") != _digest(_json_bytes(expected_slot)):
                raise ValueError("m8_journal_started_slot_fingerprint_invalid")
            active = True
        elif kind == "slot_finished" and event.get("status") in _TERMINAL:
            if event["status"] != "unexecuted" and not active:
                raise ValueError("m8_journal_finish_without_start")
            if event["status"] == "unexecuted" and active:
                raise ValueError("m8_journal_unexecuted_while_active")
            active, next_index = False, next_index + 1
        else:
            raise ValueError("m8_journal_event_transition_invalid")
        event["event_sha256"] = digest
        events.append(event)
        previous = digest
    return events
