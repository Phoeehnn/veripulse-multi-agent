"""受限读取并核验事实运行的 trace、请求归档和费用记录。"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
import json
import os
import math
from pathlib import Path
import stat
from typing import Mapping

from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.harness.fact_v2_contract import FactRunContextV2
from src.yuqing.harness.initial_evidence import InitialEvidenceManifestV2, read_initial_evidence_manifest
from src.yuqing.harness.trace import RunContext, TraceEvent
from src.yuqing.schemas import AgentResultV2, FactScopeV2, TaskInput
from src.yuqing.settings import PROJECT_ROOT


def _digest(payload: bytes) -> str:
    """Return the SHA-256 fingerprint of exact bytes."""
    return sha256(payload).hexdigest()


def _canonical(value: object) -> str:
    """Serialize data deterministically for stored fingerprints."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)




def _is_reparse_point(path: Path) -> bool:
    """Detect symbolic links and Windows reparse attributes without Python 3.12 APIs."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _has_reparse(root: Path, target: Path) -> bool:
    """Detect reparse points in each existing component below an approved root."""
    if not target.is_relative_to(root):
        return True
    current = root
    if _is_reparse_point(current):
        return True
    for part in target.relative_to(root).parts:
        current = current / part
        if _is_reparse_point(current):
            return True
    return False


def _safe_trace(root: Path, candidate: Path) -> Path:
    """Resolve a run trace only when it stays below the selected trace root."""
    if ".." in candidate.parts:
        raise ValueError("fact_trace_path_outside_root")
    path = candidate.absolute()
    if not path.is_relative_to(root) or _has_reparse(root, path):
        raise ValueError("fact_trace_path_outside_root")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("fact_trace_path_outside_root")
    return resolved


def _safe_file(root: Path, relative: str) -> Path:
    """Resolve an archive under its approved root and reject reparse points."""
    candidate = Path(relative)
    if candidate.is_absolute() or not relative or "\\" in relative or ":" in relative:
        raise ValueError("fact_archive_path_invalid")
    if ".." in candidate.parts:
        raise ValueError("fact_archive_path_escape")
    target = (root / candidate).resolve(strict=True)
    if not target.is_relative_to(root) or target == root:
        raise ValueError("fact_archive_path_escape")
    current = root
    for part in candidate.parts:
        current = current / part
        if _is_reparse_point(current):
            raise ValueError("fact_archive_reparse_point")
    if not target.is_file():
        raise ValueError("fact_archive_not_file")
    return target


def _validate_archive_reference(root: Path, reference: Mapping[str, object], identity: FactScopeV2,
                                kind: str) -> tuple[Path, str]:
    """Validate a complete archive reference and return its canonical file identity."""
    if (set(reference) != {"path", "sha256"} or not isinstance(reference.get("path"), str)
            or not isinstance(reference.get("sha256"), str)):
        raise ValueError("fact_archive_reference_invalid")
    relative, digest = reference["path"], reference["sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("fact_archive_reference_invalid")
    parts = Path(relative).parts
    run_matches = len(parts) == 3 and (parts[1].casefold() == identity.run_id.casefold()
        if os.name == "nt" else parts[1] == identity.run_id)
    filename = parts[2] if len(parts) == 3 else ""
    suffix = f"_{kind}.json"
    suffix_matches = filename.casefold().endswith(suffix.casefold()) if os.name == "nt" else filename.endswith(suffix)
    if not run_matches or not suffix_matches:
        raise ValueError("fact_archive_run_path_mismatch")
    return _safe_file(root, relative), digest


def _read_archive(path: Path, expected_digest: str, identity: FactScopeV2,
                  kind: str) -> tuple[dict[str, object], str]:
    """Read one prevalidated identity-bound archive and verify its original bytes."""
    raw = path.read_bytes()
    digest = _digest(raw)
    if digest != expected_digest:
        raise ValueError("fact_archive_sha_mismatch")
    document = json.loads(raw)
    if (not isinstance(document, dict) or document.get("kind") != kind
            or document.get("run_id") != identity.run_id
            or document.get("sample_id") != identity.sample_id or document.get("agent") != identity.agent
            or not isinstance(document.get("payload"), dict)):
        raise ValueError("fact_archive_identity_mismatch")
    return document["payload"], digest

class ArchiveReadSession:
    """Cache verified archive payloads for exactly one public material read."""

    def __init__(self, root: Path) -> None:
        """Start an isolated archive cache rooted in one approved trace directory."""
        self.root = root
        self._cache: dict[tuple[Path, str, str, tuple[object, ...]], tuple[dict[str, object], str]] = {}
        self._references: dict[Path, tuple[str, str, tuple[object, ...]]] = {}
        self.read_count = 0

    def read(self, reference: Mapping[str, object], identity: FactScopeV2,
             kind: str) -> tuple[dict[str, object], str]:
        """Read and validate an archive once, rejecting conflicting references."""
        path, digest = _validate_archive_reference(self.root, reference, identity, kind)
        identity_key = (identity.case_id, identity.sample_id, identity.run_id, identity.agent,
                        identity.fact_round_no, identity.claim_sha256)
        prior = self._references.get(path)
        if prior is not None and prior != (digest, kind, identity_key):
            raise ValueError("fact_archive_reference_conflict")
        self._references[path] = (digest, kind, identity_key)
        key = (path, digest, kind, identity_key)
        if key not in self._cache:
            self._cache[key] = _read_archive(path, digest, identity, kind)
            self.read_count += 1
        payload, actual_digest = self._cache[key]
        return deepcopy(payload), actual_digest


def _message_tool_calls(message: object, *, error_code: str) -> list[dict[str, object]]:
    """Read an archived message's tool calls, allowing only absent or null as empty."""
    if not isinstance(message, dict):
        raise ValueError(error_code)
    calls = message.get("tool_calls")
    if calls is None:
        return []
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise ValueError(error_code)
    return calls


def _response_tool_calls(response: Mapping[str, object]) -> list[dict[str, object]]:
    """Extract tool calls from a verified SDK response with strict container checks."""
    body = response.get("body")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list):
        raise ValueError("fact_response_body_invalid")
    calls: list[dict[str, object]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("fact_response_choice_invalid")
        calls.extend(_message_tool_calls(choice.get("message"),
            error_code="fact_response_message_invalid"))
    return calls


def _check_call_identity(result: AgentResultV2, task: TaskInput,
                         contract: FactRunContextV2, scope: FactScopeV2) -> None:
    """Require all caller objects to describe the same explicit V2 run."""
    if (contract.scope != scope or contract.mode not in {"initial", "followup"} or result.run_id != scope.run_id
            or task.sample_id != scope.sample_id or scope.claim_sha256 != _digest(task.claim.encode())
            or scope.agent != "fact_investigator"):
        raise ValueError("fact_material_scope_mismatch")


def _check_context_identity(context: RunContext, task: TaskInput,
                            contract: FactRunContextV2, scope: FactScopeV2) -> None:
    """Bind a report-less delivery audit to the same program-owned V2 scope."""
    if (contract.scope != scope or contract.mode not in {"initial", "followup"} or context.run_id != scope.run_id
            or context.sample_id != task.sample_id or context.agent != scope.agent
            or scope.sample_id != task.sample_id or scope.claim_sha256 != _digest(task.claim.encode())):
        raise ValueError("fact_material_scope_mismatch")


def _check_scope_events(events: list[TraceEvent], contract: FactRunContextV2,
                        scope: FactScopeV2) -> None:
    """Require stored fingerprints and scope fields to match the program contract."""
    contract_event = _one(events, "fact_v2_request_contract")
    if (contract_event.data.get("schema_sha256") != contract.schema_sha256
            or contract_event.data.get("contract_sha256") != contract.contract_sha256):
        raise ValueError("fact_contract_fingerprint_mismatch")
    event = _one(events, "fact_v2_scope")
    if (any(event.data.get(key) != value for key, value in scope.model_dump(mode="json").items())
            or event.data.get("schema_sha256") != contract.schema_sha256
            or event.data.get("contract_sha256") != contract.contract_sha256):
        raise ValueError("fact_scope_event_mismatch")


def _load_manifest(path: Path, task: TaskInput,
                   events: list[TraceEvent]) -> tuple[InitialEvidenceManifestV2, str]:
    """Verify the frozen initial manifest against the task and recorded fingerprint."""
    manifest, digest = read_initial_evidence_manifest(path, task.sample_id, task.claim)
    if _one(events, "fact_v2_initial_manifest").data.get("manifest_sha256") != digest:
        raise ValueError("fact_initial_manifest_mismatch")
    return manifest, digest


def _load_events(path: Path, scope: FactScopeV2) -> tuple[list[TraceEvent], str]:
    """Parse and fingerprint one trace from the same exact byte sequence."""
    raw = path.read_bytes()
    events = [TraceEvent.model_validate_json(line, strict=True)
              for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not events or any((event.run_id, event.sample_id, event.agent) !=
                         (scope.run_id, scope.sample_id, scope.agent) for event in events):
        raise ValueError("fact_trace_identity_mismatch")
    return events, _digest(raw)


def _one(events: list[TraceEvent], kind: str) -> TraceEvent:
    """Return exactly one required event to avoid choosing among conflicting records."""
    matches = [event for event in events if event.type == kind]
    if len(matches) != 1:
        raise ValueError(f"fact_trace_{kind}_count_invalid")
    return matches[0]


def _load_request_chain(archives: ArchiveReadSession, events: list[TraceEvent], scope: FactScopeV2
                        ) -> tuple[dict[str, TraceEvent], dict[str, TraceEvent], list[dict[str, str]]]:
    """Verify unique request, known usage and response archive links for every model call."""
    requests: dict[str, TraceEvent] = {}
    responses: dict[str, TraceEvent] = {}
    hashes: list[dict[str, str]] = []
    for event in events:
        request_id = event.data.get("request_id")
        if event.type == "llm_request":
            if not isinstance(request_id, str) or request_id in requests:
                raise ValueError("fact_request_id_duplicate_or_missing")
            archive, digest = archives.read(_mapping(event.data.get("archive")), scope, "model_request")
            if archive.get("request_id") != request_id or archive.get("state") != "prepared":
                raise ValueError("fact_request_archive_mismatch")
            requests[request_id] = event
            hashes.append(_archive_record("model_request", event, digest))
        if event.type == "model_response_archive":
            if not isinstance(request_id, str) or request_id in responses:
                raise ValueError("fact_response_archive_duplicate_or_missing")
            archive, digest = archives.read(_mapping(event.data.get("archive")), scope, "model_response")
            if archive.get("request_id") != request_id:
                raise ValueError("fact_response_archive_mismatch")
            responses[request_id] = event
            hashes.append(_archive_record("model_response", event, digest))
    if not requests or requests.keys() != responses.keys():
        raise ValueError("fact_request_response_set_mismatch")
    return requests, responses, hashes


def _archive_record(kind: str, event: TraceEvent, digest: str) -> dict[str, str]:
    """Keep archive kind, relative path, and original byte digest together."""
    reference = _mapping(event.data.get("archive"))
    path = reference.get("path")
    if not isinstance(path, str):
        raise ValueError("fact_archive_reference_invalid")
    return {"kind": kind, "path": path, "sha256": digest}


def _mapping(value: object) -> Mapping[str, object]:
    """Require a JSON object where an archive reference is expected."""
    if not isinstance(value, dict):
        raise ValueError("fact_archive_reference_missing")
    return value


def _verify_usage_and_wallet(archives: ArchiveReadSession, events: list[TraceEvent], requests: dict[str, TraceEvent],
                             responses: dict[str, TraceEvent], wallet: ExperimentSession | None,
                             scope: FactScopeV2) -> None:
    """Verify each request has one known usage receipt and, when used, wallet settlement."""
    usage_events = [event for event in events if event.type == "llm_call"]
    by_id: dict[str, list[TraceEvent]] = {}
    for event in usage_events:
        by_id.setdefault(str(event.data.get("request_id")), []).append(event)
    if set(by_id) != set(requests) or any(len(items) != 1 for items in by_id.values()):
        raise ValueError("fact_usage_request_set_mismatch")
    for request_id, usage in by_id.items():
        data = usage[0].data
        if (data.get("usage_known") is not True or request_id not in responses
                or type(data.get("input_tokens")) is not int or data["input_tokens"] < 0
                or type(data.get("output_tokens")) is not int or data["output_tokens"] < 0
                or type(data.get("cost_cny")) not in {int, float}
                or not math.isfinite(data["cost_cny"]) or data["cost_cny"] < 0):
            raise ValueError("fact_usage_unknown")
        archived, _ = archives.read(_mapping(responses[request_id].data.get("archive")), scope, "model_response")
        body = _mapping(archived.get("body"))
        usage_body = _mapping(body.get("usage"))
        if (usage_body.get("prompt_tokens") != data["input_tokens"]
                or usage_body.get("completion_tokens") != data["output_tokens"]):
            raise ValueError("fact_usage_response_mismatch")
    dispatches = [event for event in events if event.type == "dual_dispatch"]
    if not dispatches:
        return
    dispatch_ids = [event.data.get("request_id") for event in dispatches
                    if event.data.get("state") == "marked"]
    if len(dispatch_ids) != len(dispatches) or set(dispatch_ids) != set(requests):
        raise ValueError("fact_dual_dispatch_set_mismatch")
    if wallet is None:
        raise ValueError("fact_shared_wallet_missing")
    _verify_wallet_history(wallet, requests, scope, by_id)


def _verify_wallet_history(wallet: ExperimentSession, requests: dict[str, TraceEvent],
                           scope: FactScopeV2, usage: dict[str, list[TraceEvent]]) -> None:
    """Require a healthy hash-verified wallet and settled request identities for this run."""
    history, snapshot = wallet.load_history(), wallet.snapshot()
    if snapshot.halted or snapshot.unknown_requests:
        raise ValueError("fact_shared_wallet_not_healthy")
    owned = {event.get("request_id") for event in history.events
             if event.get("event_type") == "reserved" and event.get("run_id") == scope.run_id}
    if owned != set(requests):
        raise ValueError("fact_shared_wallet_request_set_mismatch")
    by_id: dict[str, list[Mapping[str, object]]] = {}
    for event in history.events:
        request_id = event.get("request_id")
        if isinstance(request_id, str) and request_id in owned:
            by_id.setdefault(request_id, []).append(event)
    if set(by_id) != set(requests):
        raise ValueError("fact_shared_wallet_request_set_mismatch")
    for request_id, entries in by_id.items():
        if any(item.get("sample_id") not in (None, scope.sample_id) for item in entries):
            raise ValueError("fact_shared_wallet_identity_mismatch")
        kinds = [item.get("event_type") for item in entries]
        if any(kinds.count(kind) != 1 for kind in ("reserved", "dispatched", "settled")) or "unknown" in kinds:
            raise ValueError("fact_shared_wallet_unsettled_request")
        settled = next(item for item in entries if item.get("event_type") == "settled")
        if Decimal(str(settled.get("actual_cost_cny"))) != Decimal(str(usage[request_id][0].data["cost_cny"])):
            raise ValueError("fact_shared_wallet_cost_mismatch")
