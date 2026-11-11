"""One tightly bound continuation entry for the fixed M4 five-sample run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from experiments.archive import m4_five_runner as runner
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.experiment_budget_store import LedgerHistory, _digest, _event_data, _json, _manifest_data
from experiments.archive.m4_five_runner import M4RunPreflight, preflight_m4_run
from src.yuqing.settings import PROJECT_ROOT, get_m4_run_budget_config, get_settings

PRIOR_EXPERIMENT_ID = "b35364c324864c11a3f50b9f66a05644"
PRIOR_CONDITIONS_SHA256 = "cd3fa6af63d68929d1fe648b763d4378030b633e186979a0a4087d3f6dcb9d91"
PRIOR_RUN_DIR = "2026-10-22/M4-five-sample-2026-10-22-a9f07cb5be0c4d189c1509f302a0c72e"
PRIOR_TRACE = f"traces/{PRIOR_RUN_DIR}/traces/2026-10-21/a9426cab14774f33a2f926ed98417421.jsonl"
PRIOR_HASHES = {
    "claim": "9fe8c572641dfca00e0a4046b2e62b929f915e16804e1205c7896c28fb389ddb",
    "result": "8489d3a3898769df7782069702392f96cf78d1d48b59c2395a549231cbe965e4",
    "answers": "f9985898a7c1423a9dc6e4e3a16e94ace8657603dc01604b2e7a2964779a8e17",
    "trace": "847f7b75c4f906c7bf7282fae2146ac189e4c928b9c83cfa421d3b55e8206aec",
    "wallet": "5b7f6e89fb1e2d1b43b0962663769681b00c4530b46740c05087a7499bca5db7",
}
PRIOR_RESULT = "2026-10-22_m4_b35364c324864c11a3f50b9f66a05644.csv"
PRIOR_KNOWN_COST = Decimal("0.036154")
CONTINUATION_MAX_COST = Decimal("0.463846")
CONTINUATION_CLAIM = "traces/m4-five-sample-continuation-claim.json"


class PriorClaim(BaseModel):
    """Exact identity of the previously consumed M4 attempt."""

    model_config = ConfigDict(extra="forbid")
    conditions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_id: Literal["b35364c324864c11a3f50b9f66a05644"]
    run_dir: Literal["2026-10-22/M4-five-sample-2026-10-22-a9f07cb5be0c4d189c1509f302a0c72e"]
    status: Literal["claimed"]


@dataclass(frozen=True)
class PriorAttemptAudit:
    """Read-only verified facts needed before the new one-time attempt."""

    experiment_id: str
    known_cost_cny: Decimal
    claim_sha256: str


def audit_prior_m4_attempt(root: Path = PROJECT_ROOT) -> PriorAttemptAudit:
    """Verify pinned historical bytes, answer semantics, Trace, and wallet state."""
    paths = _prior_paths(root)
    _verify_prior_hashes(paths)
    claim = TypeAdapter(PriorClaim).validate_json(paths["claim"].read_bytes())
    if claim.conditions_sha256 != PRIOR_CONDITIONS_SHA256:
        raise ValueError("m4_prior_conditions_fingerprint_mismatch")
    _verify_prior_answers(paths["result"], paths["answers"])
    events, manifest = _read_wallet_history(paths["wallet"])
    _verify_prior_wallet(manifest, events, claim)
    _verify_prior_trace(paths["trace"], events)
    return PriorAttemptAudit(claim.experiment_id, PRIOR_KNOWN_COST, PRIOR_HASHES["claim"])


def run_m4_five_samples_continuation(run_dir: Path, *, root: Path = PROJECT_ROOT,
                                     results_dir: Path | None = None) -> Path:
    """Start one new full five-slot experiment after the immutable prior audit."""
    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("m4_run_root_must_be_project_root")
    ready = preflight_m4_run(root)
    if not ready.prompt_approved_for_dev:
        raise ValueError("m4_prompt_not_licensed_for_dev")
    prior = audit_prior_m4_attempt(root)
    if prior.known_cost_cny + CONTINUATION_MAX_COST != Decimal("0.50"):
        raise ValueError("m4_continuation_budget_invalid")
    target = runner._validate_run_dir(run_dir, root)
    output = results_dir or get_settings().results_dir
    runner._validate_results_dir(output, root)
    claim = root / CONTINUATION_CLAIM
    if claim.exists():
        raise ValueError("m4_continuation_already_claimed")
    _require_credentials()
    runner._require_provider_connection()
    runner._assert_conditions_unchanged(root, ready.conditions)
    target.mkdir(parents=True, exist_ok=False)
    experiment_id = uuid.uuid4().hex
    _create_continuation_claim(claim, experiment_id, target, ready, prior, root)
    return runner._run_prepared(ready, target, experiment_id, root, output,
                                CONTINUATION_MAX_COST)


def _prior_paths(root: Path) -> dict[str, Path]:
    """Resolve only the fixed historical claim, result, answer, Trace, and wallet."""
    run = root / "traces" / PRIOR_RUN_DIR
    return {"claim": root / "traces/m4-five-sample-claim.json",
        "result": root / "results" / PRIOR_RESULT, "answers": run / "answers.jsonl",
        "trace": root / PRIOR_TRACE, "wallet": run / "wallet.sqlite"}


def _verify_prior_hashes(paths: dict[str, Path]) -> None:
    """Reject missing, changed, or substituted historical artifacts by byte hash."""
    for name, expected in PRIOR_HASHES.items():
        path = paths[name]
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"m4_prior_{name}_fingerprint_mismatch")


def _verify_prior_answers(result_path: Path, answers_path: Path) -> None:
    """Check the immutable five-slot record still describes the known stopped run."""
    with result_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    journal = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines()]
    expected_ids = list(runner.M4_SAMPLE_IDS)
    if [row["sample_id"] for row in rows] != expected_ids:
        raise ValueError("m4_prior_result_order_invalid")
    if [row["sample_id"] for row in journal] != expected_ids:
        raise ValueError("m4_prior_answer_order_invalid")
    if rows[0]["experiment_id"] != PRIOR_EXPERIMENT_ID or rows[0]["status"] != "no_valid_result":
        raise ValueError("m4_prior_first_answer_invalid")
    if rows[0]["stop_reason"] != "request_capacity" or rows[0]["usage_known"] != "true":
        raise ValueError("m4_prior_first_answer_state_invalid")
    if abs(Decimal(rows[0]["known_cost_cny"]) - PRIOR_KNOWN_COST) > Decimal("0.00000000000001"):
        raise ValueError("m4_prior_answer_cost_mismatch")
    if any(row["status"] != "not_executed" for row in rows[1:]):
        raise ValueError("m4_prior_unexecuted_slots_changed")
    if any(row["experiment_id"] != PRIOR_EXPERIMENT_ID for row in rows):
        raise ValueError("m4_prior_result_identity_mismatch")
    if journal[0]["run_id"] != "a9426cab14774f33a2f926ed98417421":
        raise ValueError("m4_prior_answer_run_mismatch")


def _read_wallet_history(path: Path) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Read and verify the existing SQLite hash chain without opening it writable."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        raw_events = list(connection.execute(
            "SELECT revision, event_json, previous_hash, event_hash FROM events ORDER BY revision"))
    manifest_raw = metadata.get("manifest", "")
    if set(metadata) != {"format", "manifest", "manifest_hash", "revision", "head_hash"}:
        raise ValueError("m4_prior_wallet_metadata_invalid")
    if metadata["format"] != "1":
        raise ValueError("m4_prior_wallet_format_invalid")
    if hashlib.sha256(manifest_raw.encode("utf-8")).hexdigest() != metadata.get("manifest_hash"):
        raise ValueError("m4_prior_wallet_manifest_digest_mismatch")
    manifest = _manifest_data(json.loads(manifest_raw))
    if _json(manifest) != manifest_raw:
        raise ValueError("m4_prior_wallet_manifest_not_canonical")
    events, hashes, previous = [], [], "genesis"
    for expected, (revision, raw, prior, digest) in enumerate(raw_events, start=1):
        event = _event_data(json.loads(raw))
        if revision != expected or prior != previous or _json(event) != raw or _digest(revision, prior, raw) != digest:
            raise ValueError("m4_prior_wallet_chain_invalid")
        events.append(event)
        hashes.append(digest)
        previous = digest
    if int(metadata.get("revision", "-1")) != len(events) or metadata.get("head_hash") != previous:
        raise ValueError("m4_prior_wallet_revision_invalid")
    history = LedgerHistory(manifest, len(events), tuple(events), tuple(hashes))
    state = replay_history(history, manifest)
    status = state.snapshot()
    if (status.known_cost_subtotal_cny != PRIOR_KNOWN_COST or status.unknown_requests
            or status.pending_requests or status.halted):
        raise ValueError("m4_prior_wallet_not_fully_settled")
    return tuple(events), manifest


def _verify_prior_wallet(manifest: dict[str, object], events: tuple[dict[str, object], ...],
                         claim: PriorClaim) -> None:
    """Bind the old wallet to its claim, original cap, and exactly two settlements."""
    if (manifest["experiment_id"] != claim.experiment_id
            or manifest["conditions_fingerprint"] != claim.conditions_sha256
            or manifest["max_cost_cny"] != "0.5"):
        raise ValueError("m4_prior_wallet_manifest_mismatch")
    settled = [event for event in events if event["event_type"] == "settled"]
    reservations = [event for event in events if event["event_type"] == "reserved"]
    if len(settled) != 2 or sum((Decimal(event["actual_cost_cny"]) for event in settled), Decimal(0)) != PRIOR_KNOWN_COST:
        raise ValueError("m4_prior_wallet_expense_mismatch")
    if len(reservations) != 2 or any(
        (event.get("sample_id"), event.get("group_id"), event.get("run_id"), event.get("variant"))
        != ("chef:dev:0", "m4", "a9426cab14774f33a2f926ed98417421", "arbiter-v2-m4")
        for event in reservations
    ):
        raise ValueError("m4_prior_wallet_request_identity_mismatch")


def _verify_prior_trace(path: Path, events: tuple[dict[str, object], ...]) -> None:
    """Match the old Trace run identity, settled request IDs, and safe stop reason."""
    trace = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    calls = [event["data"] for event in trace if event.get("type") == "llm_call"]
    settled = {event["request_id"]: Decimal(event["actual_cost_cny"])
               for event in events if event["event_type"] == "settled"}
    if len(calls) != 2 or {call["request_id"]: Decimal(str(call["cost_cny"])) for call in calls} != settled:
        raise ValueError("m4_prior_trace_wallet_mismatch")
    if not any(event.get("type") == "arbiter_v2_stopped"
               and event.get("data", {}).get("reason") == "request_capacity" for event in trace):
        raise ValueError("m4_prior_trace_stop_reason_mismatch")


def _require_credentials() -> None:
    """Require configured provider credentials only after the historical audit."""
    current = get_settings()
    if current.deepseek_api_key is None or not current.deepseek_model:
        raise ValueError("m4_provider_credentials_missing")


def _create_continuation_claim(path: Path, experiment_id: str, run_dir: Path,
                               ready: M4RunPreflight, prior: PriorAttemptAudit,
                               root: Path) -> None:
    """Atomically claim the sole continuation with fixed source and hard cost cap."""
    payload = {"experiment_id": experiment_id,
        "run_dir": run_dir.relative_to((root / "traces").resolve()).as_posix(),
        "conditions_sha256": runner._mapping_sha256(ready.conditions),
        "prior_experiment_id": prior.experiment_id,
        "prior_claim_sha256": prior.claim_sha256,
        "max_run_cost_cny": str(CONTINUATION_MAX_COST), "status": "claimed"}
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m4_continuation_already_claimed") from exc


def _sha256(path: Path) -> str:
    """Hash exact historical file bytes without normalizing text."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
