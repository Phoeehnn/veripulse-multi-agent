"""M6c-only paid runner for the frozen three-joint plus five-dev case set."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from experiments.archive import m6c_execution, m6c_readiness
from src.yuqing.eval.experiment_session import build_experiment_manifest
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing import settings

M6C_RUN_CAP = m6c_readiness.RUN_CAP


def run_m6c_live_experiment(root: Path, run_dir: Path, *, authorize_paid_run: bool) -> dict[str, Any]:
    """Run only the frozen eight after explicit authorization, same-process network and drift checks."""
    from experiments.archive.m5_checkpoint import (
        _close_worker_scopes, _full_corpus, _halt_wallet_safely, _provider_connection,
        _require_credentials,
    )
    from src.yuqing.retrieval.bm25 import build_bm25_index

    if not authorize_paid_run or root.resolve() != settings.PROJECT_ROOT.resolve():
        raise ValueError("m6c_live_authorization_or_root_invalid")
    selection = json.loads((root / m6c_readiness.SELECTION_PATH).read_bytes())
    preflight = m6c_readiness.m6c_preflight(root, selection)
    cases = m6c_readiness.prepare_m6c_cases(root, selection)
    target = _valid_target(root, run_dir)
    claim = root / m6c_execution.M6C_CLAIM
    if claim.exists():
        raise ValueError("m6c_fixed_eight_already_claimed")
    _require_credentials()
    _provider_connection()
    current = m6c_readiness.m6c_preflight(root, selection)
    if current["conditions_sha256"] != preflight["conditions_sha256"]:
        raise ValueError("m6c_conditions_changed_after_network_preflight")
    archive = m6c_execution.archive_inputs(root, selection, cases, current["conditions"])
    conditions = dict(current["conditions"], input_archive_sha256=archive["sha256"])
    run_id = uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    m6c_execution.claim_run(root, target, run_id, conditions)
    target.mkdir(parents=False, exist_ok=False)
    from experiments.archive.m6b_run import _copy_archive
    _copy_archive(archive["files"], target / "input_archive")
    wallet_config = m6c_readiness.m6c_wallet_evaluation_config()
    manifest = build_experiment_manifest(run_id, conditions, settings.get_cost_config(),
        wallet_config, mode="full", max_run_cost_cny=M6C_RUN_CAP)
    wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest, wallet_config)
    try:
        index = build_bm25_index(_full_corpus(root))
        return _execute_slots(root, selection, cases, target, run_id, conditions,
            archive["sha256"], wallet, index)
    except Exception:
        _halt_wallet_safely(wallet, "m6c_live_setup_or_audit_failure")
        raise
    finally:
        wallet.close()


def _valid_target(root: Path, run_dir: Path) -> Path:
    """Require a fresh child of the D-drive traces directory without creating it."""
    target, traces = run_dir.resolve(), (root / "traces").resolve()
    if target == traces or not target.is_relative_to(traces) or target.exists():
        raise ValueError("m6c_run_directory_invalid_or_exists")
    return target


def _execute_slots(root: Path, selection: dict[str, Any], cases: tuple[Any, ...],
                   target: Path, run_id: str, conditions: dict[str, str], archive_sha: str,
                   wallet: PersistentExperimentBudget, index: Any) -> dict[str, Any]:
    """Persist each fixed slot, reserving one full case envelope before each starts."""
    from experiments.archive.m6b_execution import append_row, write_run

    rows: list[dict[str, Any]] = []
    stopped = False
    (target / "answers.jsonl").write_bytes(b"")
    write_run(target, _summary(run_id, rows, wallet, archive_sha, False))
    for case in cases:
        row = _unexecuted(case, run_id, "prior_slot_stopped") if stopped else None
        if not stopped:
            row = _execute_case(root, case, target, run_id, wallet, index, selection, conditions)
            status = wallet.snapshot()
            if status.unknown_requests or status.pending_requests:
                if not status.halted:
                    wallet.halt("m6c_unknown_or_pending_request")
                status = wallet.snapshot()
            if row["status"] == "failed" and not status.halted:
                wallet.halt("m6c_live_case_failed")
                status = wallet.snapshot()
            stopped = (row["status"] == "failed" or status.halted or status.pending_requests > 0
                or status.unknown_requests > 0 or row["stop_reason"] == "experiment_cost_limit")
        rows.append(row)
        append_row(target / "answers.jsonl", row)
        write_run(target, _summary(run_id, rows, wallet, archive_sha, stopped))
    audit = _audit_run(root, target, run_id, selection, cases, rows, wallet, conditions, archive_sha)
    summary = _summary(run_id, rows, wallet, archive_sha, stopped) | {"audit": audit}
    write_run(target, summary)
    return summary


def _execute_case(root: Path, case: Any, target: Path, run_id: str,
                  wallet: PersistentExperimentBudget, index: Any, selection: dict[str, Any],
                  conditions: dict[str, str]) -> dict[str, Any]:
    """Run one slot only after conditions, archive, and complete per-case envelope pass."""
    from experiments.archive.m5_checkpoint import _close_worker_scopes, _has_experiment_budget_refusal, _halt_wallet_safely

    trace_root = target / "cases" / f"{case.order:02d}" / "traces"
    trace_root.mkdir(parents=True, exist_ok=False)
    before, started, composition = wallet.snapshot(), perf_counter(), None
    status, reason, state, thread_id = "failed", "m6c_case_setup_failed", {}, None
    try:
        if not m6c_readiness.m6c_preflight(root, selection)["conditions_sha256"] == _without_archive(conditions):
            raise ValueError("m6c_conditions_changed_before_slot")
        m6c_execution.audit_archive(target, conditions["input_archive_sha256"])
        current = wallet.snapshot()
        available = (Decimal(str(wallet.manifest_snapshot()["max_cost_cny"]))
            - current.known_cost_subtotal_cny - current.unknown_reserved_cny - current.pending_reserved_cny)
        if available < m6c_readiness.CASE_CAP:
            raise ValueError("m6c_case_envelope_not_available")
        composition = _compose(root, case, wallet, trace_root, index, target)
        thread_id = composition.context.thread_id
        state = composition.assemble(external_calls_authorized=True).invoke()
        if not state.get("brief") or not isinstance(state.get("risk"), dict):
            raise ValueError("m6c_graph_terminal_output_missing")
        status, reason = "completed", str(state.get("stop_reason") or "")
        if _has_experiment_budget_refusal(trace_root):
            reason = "experiment_cost_limit"
            _halt_wallet_safely(wallet, reason)
        if m6c_readiness.m6c_preflight(root, selection)["conditions_sha256"] != _without_archive(conditions):
            raise ValueError("m6c_conditions_changed_after_slot")
    except Exception as exc:
        status = "failed"
        reason = type(exc).__name__ if not isinstance(exc, ValueError) else str(exc)
        _halt_wallet_safely(wallet, "m6c_live_case_failure")
    finally:
        if composition is not None:
            error = _close_worker_scopes((composition.inputs.fact_worker_scope,
                composition.inputs.propagation_worker_scope), wallet)
            if error:
                status, reason = "failed", error
    after = wallet.snapshot()
    return {"run_id": run_id, "order": case.order, "sample_id": case.sample.sample_id,
        "group": case.group, "status": status, "stop_reason": reason, "thread_id": thread_id,
        "elapsed_ms": int((perf_counter() - started) * 1000),
        "known_cost_delta_cny": str(after.known_cost_subtotal_cny - before.known_cost_subtotal_cny),
        "usage_unknown": after.unknown_requests > before.unknown_requests,
        "pending_requests": after.pending_requests, "state": state,
        "trace_root": trace_root.relative_to(target).as_posix()}


def _compose(root: Path, case: Any, wallet: Any, trace_root: Path,
             index: Any, target: Path) -> Any:
    """Add only the licensed M6c review spec and flags to the existing case composition."""
    from dataclasses import replace
    from experiments.archive import m6b_run, m6c_review_prompt
    from experiments.archive.m6b_readiness import PROMPT
    from src.yuqing.graph.composition import CaseComposition
    from src.yuqing.schemas import AgentSpec

    initial = target / "input_archive" / f"case-{case.order:02d}" / "initial-evidence.json"
    if case.joint_input is not None:
        from src.yuqing.graph.composition import CaseComposition, CaseInputs
        from src.yuqing.harness.tools import EvidenceWorkerScope
        from src.yuqing.retrieval.bm25 import build_bm25_index

        prepared = case.joint_input
        fact_scope, propagation_scope = EvidenceWorkerScope(), EvidenceWorkerScope()
        inputs = CaseInputs(case_id=f"m6c:{case.order}:{case.sample.sample_id}",
            sample=case.sample, coverage="demo", trace_root=trace_root, wallet=wallet,
            fact_worker_scope=fact_scope, propagation_worker_scope=propagation_scope,
            index=build_bm25_index(list(prepared.chunks)), store_path=prepared.store,
            initial_manifest_path=prepared.store / "initial.json",
            propagation_spec=AgentSpec(agent="propagation_analyst", system_prompt=(root / PROMPT).read_text(encoding="utf-8")),
            arbiter_spec=AgentSpec(agent="arbiter", system_prompt=(root / "prompts/arbiter/v2.md").read_text(encoding="utf-8")),
            group_id="m6c_development", variant="m6c-review-v1",
            propagation_contract_version="v1.1", m6b=True, m6c=True,
            propagation_review_spec=AgentSpec(agent="propagation_analyst",
                system_prompt=m6c_review_prompt.validate_m6c_review_prompt(offline=True).text))
        try:
            base = CaseComposition(inputs)
        except Exception:
            fact_scope.close()
            propagation_scope.close()
            raise
    else:
        reused = m6b_run.M6bCase(case.order, "chef_dev_reuse", case.sample)
        base = m6b_run.assemble_m6b_case(root, reused, wallet, trace_root, index, initial)
    inputs = replace(base.inputs, m6b=True, m6c=True, propagation_contract_version="v1.1",
        propagation_spec=AgentSpec(agent="propagation_analyst", system_prompt=(root / PROMPT).read_text(encoding="utf-8")),
        propagation_review_spec=AgentSpec(agent="propagation_analyst",
            system_prompt=m6c_review_prompt.validate_m6c_review_prompt(offline=True).text),
        group_id="m6c_development", variant="m6c-review-v1")
    return CaseComposition(inputs)


def _audit_run(root: Path, target: Path, run_id: str, selection: dict[str, Any],
               cases: tuple[Any, ...], rows: list[dict[str, Any]], wallet: Any,
               conditions: dict[str, str], archive_sha: str) -> dict[str, Any]:
    """Reconcile fixed slots, archive, claim, trace identities, and persistent wallet history."""
    import json
    from types import SimpleNamespace
    from src.yuqing.eval.experiment_budget_recovery import replay_history
    from experiments.archive.m5_checkpoint_audit import _audit_cost_rows, _audit_wallet, _read_slot_traces

    m6c_execution.audit_archive(target, archive_sha)
    _audit_claim(root, target, run_id, conditions)
    expected = [(case.order, case.sample.sample_id) for case in cases]
    if len(rows) != len(cases) or [(r.get("order"), r.get("sample_id")) for r in rows] != expected:
        raise ValueError("m6c_result_slots_not_exact")
    if any(r.get("run_id") != run_id or r.get("status") not in {"completed", "failed", "unexecuted"}
           for r in rows):
        raise ValueError("m6c_result_identity_or_status_invalid")
    pending = [i for i, row in enumerate(rows) if row["status"] == "unexecuted"]
    if pending and pending != list(range(pending[0], len(rows))):
        raise ValueError("m6c_unexecuted_slots_not_final_suffix")
    raw = (target / "answers.jsonl").read_bytes()
    if not raw.endswith(b"\n") or [json.loads(line) for line in raw.splitlines()] != rows:
        raise ValueError("m6c_answer_journal_mismatch")
    requests, calls = _read_slot_traces(target, rows)
    history, manifest = wallet.load_history(), wallet.manifest_snapshot()
    if (manifest.get("experiment_id") != run_id
            or Decimal(str(manifest.get("max_cost_cny"))) != M6C_RUN_CAP
            or manifest.get("conditions_fingerprint") != _digest(conditions)):
        raise ValueError("m6c_wallet_manifest_mismatch")
    state = replay_history(history, manifest)
    _audit_wallet(requests, calls, state.requests, SimpleNamespace(cases=cases))
    if any(item.identity.get("group_id") != "m6c_development"
           or item.identity.get("variant") != "m6c-review-v1" for item in state.requests.values()):
        raise ValueError("m6c_wallet_request_group_mismatch")
    _audit_cost_rows(rows, state.requests)
    if state.snapshot() != wallet.snapshot():
        raise ValueError("m6c_wallet_snapshot_mismatch")
    status = state.snapshot()
    audit = "verified_stopped_with_unknown_cost" if status.unknown_requests else "verified"
    if status.pending_requests:
        audit = "verified_stopped_with_pending"
    return {"status": audit, "planned": len(cases), "executed": sum(r["status"] != "unexecuted" for r in rows),
        "trace_requests": len(requests), "trace_receipts": len(calls),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_requests": status.unknown_requests, "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_requests": status.pending_requests}


def _audit_claim(root: Path, target: Path, run_id: str, conditions: dict[str, str]) -> None:
    """Require the immutable one-time claim to match this precise M6c run."""
    import json
    claim = json.loads((root / m6c_execution.M6C_CLAIM).read_bytes())
    expected = {"run_id": run_id,
        "run_dir": target.relative_to((root / "traces").resolve()).as_posix(),
        "conditions_sha256": _digest(conditions), "status": "claimed"}
    if claim != expected:
        raise ValueError("m6c_claim_identity_mismatch")


def _summary(run_id: str, rows: list[dict[str, Any]], wallet: Any,
             archive_sha: str, stopped: bool) -> dict[str, Any]:
    """Summarize durable progress and never report unknown usage as zero cost."""
    status = wallet.snapshot()
    return {"run_id": run_id, "mode": "authorized_m6c_graph", "planned": 8,
        "executed": sum(row["status"] != "unexecuted" for row in rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unexecuted": sum(row["status"] == "unexecuted" for row in rows),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_requests": status.unknown_requests,
        "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_requests": status.pending_requests,
        "pending_reserved_cny": str(status.pending_reserved_cny), "halted": status.halted,
        "stop_reason": status.halt_reason, "stopped": stopped, "archive_sha256": archive_sha,
        "rows": rows}


def _unexecuted(case: Any, run_id: str, reason: str) -> dict[str, Any]:
    """Represent a slot not started after an earlier hard stop."""
    return {"run_id": run_id, "order": case.order, "sample_id": case.sample.sample_id,
        "group": case.group, "status": "unexecuted", "stop_reason": reason,
        "trace_root": None, "known_cost_delta_cny": "0", "usage_unknown": False,
        "pending_requests": 0}


def _without_archive(conditions: dict[str, str]) -> str:
    """Remove the post-preflight archive digest before comparing stable conditions."""
    from experiments.archive.m6c_readiness import _digest
    return _digest({key: value for key, value in conditions.items() if key != "input_archive_sha256"})


def _digest(value: object) -> str:
    """Hash canonical JSON for claim and conditions verification."""
    import json
    from hashlib import sha256
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
