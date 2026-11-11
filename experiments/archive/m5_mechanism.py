"""One-case M5 mechanism supplement, isolated from the immutable fixed-ten run."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.archive import m5_checkpoint
from src.yuqing.eval.experiment_session import build_experiment_manifest
from experiments.archive.m5_inputs import (
    SUMMARY, SELECTION, M5CaseInput, M5Inputs, _json_bytes,
    _load_archived_case, _questions, _sha, _source_shas,
)
from experiments.archive.m5_checkpoint_audit import (
    _audit_cost_rows, _audit_wallet, _read_answers, _read_slot_traces,
)
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.settings import (
    PROJECT_ROOT, get_cost_config, get_evaluation_config, get_graph_config,
    get_m5_case_budget_config, get_m5_run_budget_config, get_settings,
)

MECHANISM_SAMPLE = "chef:dev:826"
CLAIM = "traces/m5-mechanism-826.claim"
RUNNER_FILES = ("experiments/archive/m5_mechanism.py", "scripts/run_m5_mechanism_826.py")


def mechanism_cap_cny() -> Decimal:
    """Use the already approved M5 case ceiling from the single configuration source."""
    return Decimal(str(get_m5_case_budget_config().max_cost_cny))


def prepare_m5_mechanism(root: Path = PROJECT_ROOT) -> tuple[M5Inputs, M5CaseInput, dict[str, str]]:
    """Verify the frozen source case and return its current, committed run conditions."""
    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("m5_mechanism_root_invalid")
    inputs, case, conditions = _collect_conditions(root)
    if (root / CLAIM).exists():
        raise ValueError("m5_mechanism_attempt_already_claimed")
    return inputs, case, conditions


def _collect_conditions(root: Path) -> tuple[M5Inputs, M5CaseInput, dict[str, str]]:
    """Build the exact one-case identity without consulting its one-time claim."""
    inputs = _load_parent_case(root)
    _validate_runtime(root)
    case = _selected_case(inputs)
    _verify_runner_commit(root)
    conditions = m5_checkpoint.m5_condition_snapshot(root, inputs, require_committed=True)
    conditions.update({"run_kind": "m5_mechanism_supplement_v1",
        "sample_id": case.sample.sample_id, "sample_sha256": case.sample_sha256,
        "initial_evidence_sha256": case.initial_sha256 or "none",
        "mechanism_max_cost_cny": str(mechanism_cap_cny()),
        **{"runner:" + path: _file_sha(root / path) for path in RUNNER_FILES}})
    return inputs, case, conditions


def _load_parent_case(root: Path) -> M5Inputs:
    """Re-validate CHEF 826 data from the immutable v5 package without reusing its stale code hash."""
    summary_path, selection_path = root / SUMMARY, root / SELECTION
    raw, selection_raw = summary_path.read_bytes(), selection_path.read_bytes()
    summary, selection = json.loads(raw), json.loads(selection_raw)
    signature = summary.get("archive_summary_sha256")
    unsigned = dict(summary)
    unsigned.pop("archive_summary_sha256", None)
    archive = (root / summary["archive"]).resolve()
    review_root = (root / "traces/review-packages").resolve()
    archive_manifest_path = archive / "inputs.json"
    manifest_raw = archive_manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if (_sha(_json_bytes(unsigned) + b"\n") != signature
            or archive == review_root or not archive.is_relative_to(review_root)
            or manifest.get("version") != "m5-inputs-v5"
            or summary.get("selection_sha256") != _sha(selection_raw)
            or summary.get("archive_sha256") != _sha(manifest_raw)
            or manifest.get("cases") != summary.get("cases")
            or manifest.get("sources") != summary.get("source_sha256")
            or manifest.get("selection_sha256") != summary.get("selection_sha256")):
        raise ValueError("m5_mechanism_parent_registration_invalid")
    selected = [row for row in selection["samples"] if row.get("sample_id") == MECHANISM_SAMPLE]
    archived = [row for row in summary["cases"] if row.get("sample_id") == MECHANISM_SAMPLE]
    if len(selected) != 1 or len(archived) != 1:
        raise ValueError("m5_mechanism_sample_registration_invalid")
    case = _load_archived_case(root, archive, selected[0], archived[0], _questions(root))
    return M5Inputs(archive, summary["archive_sha256"], (case,), _source_shas(root))


def _validate_runtime(root: Path) -> None:
    """Reuse M5 model, prompt, configuration and full-corpus startup gates without the fixed-ten loader."""
    from src.yuqing.eval.preflight import validate_propagation_startup, validate_runtime_config
    from src.yuqing.graph.fact_adapter import FactNodeAdapter
    from experiments.archive.m5_checkpoint import _validate_arbiter_permission
    from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus

    validate_runtime_config()
    FactNodeAdapter(get_graph_config().fact_node_version, root=root)
    validate_propagation_startup(root / "prompts/propagation/v1.md", offline=True)
    _validate_arbiter_permission(root)
    get_m5_run_budget_config()
    runtime = get_settings()
    chunks = load_eval_corpus(runtime.chef_dev_path, [], corpus_name="chef_dev_full")
    validate_eval_corpus(chunks, runtime.dev_store_path, corpus_name="chef_dev_full")


def run_m5_mechanism(root: Path, run_dir: Path, *, execute: bool,
                     authorize_real_calls: bool, authorization: str) -> dict[str, Any]:
    """Run CHEF 826 once with a fresh claim and wallet after explicit authorization."""
    if (not execute or not authorize_real_calls
            or authorization != "M5-B-mechanism-826"):
        raise ValueError("m5_mechanism_requires_separate_authorization")
    inputs, case, conditions = prepare_m5_mechanism(root)
    target = m5_checkpoint._new_run_dir(run_dir, root)
    m5_checkpoint._require_credentials()
    m5_checkpoint._provider_connection()
    current_inputs, current_case, current = prepare_m5_mechanism(root)
    if current != conditions or current_case.sample_sha256 != case.sample_sha256:
        raise ValueError("m5_mechanism_conditions_changed_after_network_preflight")
    return _execute_one(root, target, current_inputs, current_case, current)


def _execute_one(root: Path, target: Path, inputs: M5Inputs, case: M5CaseInput,
                 conditions: dict[str, str]) -> dict[str, Any]:
    """Claim, create an independent wallet, run one production graph and audit it."""
    experiment_id = uuid4().hex
    _claim(root / CLAIM, experiment_id, target, conditions)
    target.mkdir(parents=True, exist_ok=False)
    manifest = build_experiment_manifest(experiment_id, conditions, get_cost_config(),
        get_evaluation_config(), mode="full", max_run_cost_cny=mechanism_cap_cny())
    wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest,
        get_evaluation_config())
    try:
        return _run_and_save(root, target, inputs, case, conditions, experiment_id, wallet)
    finally:
        wallet.close()


def _run_and_save(root: Path, target: Path, inputs: M5Inputs, case: M5CaseInput,
                  conditions: dict[str, str], experiment_id: str,
                  wallet: PersistentExperimentBudget) -> dict[str, Any]:
    """Persist the one-case response before auditing or reporting terminal status."""
    case_inputs = replace(inputs, cases=(case,))
    try:
        index = m5_checkpoint.build_bm25_index(m5_checkpoint._full_corpus(root))
    except Exception as exc:
        m5_checkpoint._halt_wallet_safely(wallet, "m5_mechanism_runtime_failure")
        rows = [m5_checkpoint._unexecuted(case, experiment_id, type(exc).__name__)]
        m5_checkpoint._append_jsonl(target / "answers.jsonl", rows[0])
    else:
        rows, _ = m5_checkpoint._run_slots(root, case_inputs, wallet, index, target,
            experiment_id, _base_conditions(conditions))
    condition_stable = False
    try:
        condition_stable = _collect_conditions(root)[2] == conditions
    except Exception as exc:
        wallet.halt("m5_mechanism_postcondition_check_failed:" + type(exc).__name__)
    if not condition_stable and not wallet.snapshot().halted:
        wallet.halt("m5_mechanism_conditions_changed_after_case")
    status = wallet.snapshot()
    try:
        audit = _audit_one_case(root, target, experiment_id, case_inputs, rows,
            wallet, conditions, status)
    except Exception as exc:
        audit = {"integrity": "failed", "reason": type(exc).__name__}
    summary = {"experiment_id": experiment_id, "sample_id": case.sample.sample_id,
        "status": rows[0]["status"], "stop_reason": rows[0]["stop_reason"],
        "audit": audit, "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_reserved_cny": str(status.pending_reserved_cny), "row": rows[0],
        "condition_stable": condition_stable,
        "condition_stop_reason": status.halt_reason if not condition_stable else None}
    _write_new(target / "summary.json", _json_bytes(summary) + b"\n")
    return summary


def _audit_one_case(root: Path, run_dir: Path, experiment_id: str,
                    inputs: M5Inputs, rows: list[dict[str, Any]], wallet: Any,
                    conditions: dict[str, str], status: Any) -> dict[str, object]:
    """Reuse the fixed runner's trace and wallet validators for one distinct claim."""
    expected = [(case.order, case.sample.sample_id) for case in inputs.cases]
    actual = [(row.get("order"), row.get("sample_id")) for row in rows]
    if len(expected) != 1 or actual != expected or len(rows) != 1:
        raise ValueError("m5_mechanism_result_slot_invalid")
    _audit_claim(root, run_dir, experiment_id, conditions)
    if rows[0].get("experiment_id") != experiment_id or rows[0].get("status") not in {
        "completed", "failed", "unexecuted"
    }:
        raise ValueError("m5_mechanism_result_identity_invalid")
    if _read_answers(run_dir / "answers.jsonl", experiment_id) != rows:
        raise ValueError("m5_mechanism_answer_journal_mismatch")
    requests, calls = _read_slot_traces(run_dir, rows)
    history, manifest = wallet.load_history(), wallet.manifest_snapshot()
    if (manifest.get("experiment_id") != experiment_id
            or Decimal(str(manifest.get("max_cost_cny"))) != mechanism_cap_cny()
            or manifest.get("conditions_fingerprint") != _sha(_json_bytes(conditions))):
        raise ValueError("m5_mechanism_wallet_manifest_mismatch")
    recovered_state = replay_history(history, manifest)
    recovered = recovered_state.snapshot()
    _audit_wallet(requests, calls, recovered_state.requests, inputs)
    if recovered != status or recovered != wallet.snapshot():
        raise ValueError("m5_mechanism_wallet_snapshot_mismatch")
    _audit_cost_rows(rows, recovered_state.requests)
    return {"integrity": "verified", "run_status": _run_status(rows[0], recovered),
        "planned": 1, "executed": int(rows[0]["status"] != "unexecuted"),
        "trace_requests": len(requests), "trace_receipts": len(calls),
        "unknown_requests": recovered.unknown_requests,
        "pending_requests": recovered.pending_requests,
        "known_cost_subtotal_cny": str(recovered.known_cost_subtotal_cny)}


def _audit_claim(root: Path, run_dir: Path, experiment_id: str,
                 conditions: dict[str, str]) -> None:
    """Bind the separate claim to this exact case, run directory and condition hash."""
    claim = json.loads((root / CLAIM).read_text(encoding="utf-8"))
    relative = run_dir.resolve().relative_to((root / "traces").resolve()).as_posix()
    expected = {"experiment_id": experiment_id, "run_dir": relative,
        "conditions_sha256": _sha(_json_bytes(conditions)), "status": "claimed"}
    if claim != expected:
        raise ValueError("m5_mechanism_claim_identity_mismatch")


def _run_status(row: dict[str, Any], wallet: Any) -> str:
    """Keep execution failure, unknown spend and unsettled requests visible."""
    if row["status"] == "unexecuted":
        return "not_started"
    if wallet.unknown_requests or wallet.pending_requests:
        return "stopped_with_unresolved_spend"
    return str(row["status"])


def _base_conditions(conditions: dict[str, str]) -> dict[str, str]:
    """Extract the original M5 snapshot expected by shared per-case pre/post checks."""
    return {key: value for key, value in conditions.items()
        if key not in {"run_kind", "sample_id", "sample_sha256", "initial_evidence_sha256",
                       "mechanism_max_cost_cny"}
        and not key.startswith("runner:")}


def _selected_case(inputs: M5Inputs) -> M5CaseInput:
    """Allow exactly the already frozen full-coverage CHEF mechanism sample."""
    matches = [case for case in inputs.cases if case.sample.sample_id == MECHANISM_SAMPLE]
    if len(matches) != 1 or matches[0].coverage != "full" or matches[0].sample.source != "chef":
        raise ValueError("m5_mechanism_sample_invalid")
    return matches[0]


def _verify_runner_commit(root: Path) -> None:
    """Require both newly introduced runner files to be tracked and clean in Git."""
    import subprocess

    for relative in RUNNER_FILES:
        tracked = subprocess.run(["git",
            "ls-files", "--error-unmatch", relative], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git",
            "diff", "--quiet", "HEAD", "--", relative], cwd=root)
        if tracked.returncode or dirty.returncode:
            raise ValueError("m5_mechanism_runner_not_committed:" + relative)


def _claim(path: Path, experiment_id: str, run_dir: Path,
           conditions: dict[str, str]) -> None:
    """Create one durable, non-reusable claim without touching the fixed-ten claim."""
    payload = {"experiment_id": experiment_id,
        "run_dir": run_dir.resolve().relative_to((PROJECT_ROOT / "traces").resolve()).as_posix(),
        "conditions_sha256": _sha(_json_bytes(conditions)), "status": "claimed"}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m5_mechanism_attempt_already_claimed") from exc


def _file_sha(path: Path) -> str:
    """Return a SHA-256 fingerprint for one committed runner source file."""
    return sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    """Write one new report atomically without replacing an existing artifact."""
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
