"""M6b固定十案唯一真实执行入口；复用生产图、共享钱包与归档审计。"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Any

from experiments.archive import m6b_execution, m6b_run
from experiments.archive.m6b_readiness import m6b_preflight
from src.yuqing.eval.experiment_session import build_experiment_manifest
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing import settings

M6B_LIVE_CAP = Decimal("3")


def run_m6b_live_experiment(root: Path, selection: dict[str, Any], run_dir: Path,
                            *, authorize_paid_run: bool) -> dict[str, Any]:
    """Run the frozen ten once after authorization, provider, input and condition checks."""
    from experiments.archive.m5_checkpoint import (
        _full_corpus, _halt_wallet_safely, _provider_connection, _require_credentials,
    )
    from src.yuqing.retrieval.bm25 import build_bm25_index

    if not authorize_paid_run or root.resolve() != settings.PROJECT_ROOT.resolve():
        raise ValueError("m6b_live_run_authorization_or_root_invalid")
    preflight = m6b_preflight(root, selection)
    cases = m6b_run.prepare_m6b_cases(root, selection)
    target = _valid_target(root, run_dir)
    claim = root / m6b_execution.M6B_CLAIM
    if claim.exists():
        raise ValueError("m6b_fixed_ten_already_claimed")
    _require_credentials()
    _provider_connection()
    current = m6b_preflight(root, selection)
    if (current["conditions_sha256"] != preflight["conditions_sha256"]
            or current["selection_sha256"] != preflight["selection_sha256"]):
        raise ValueError("m6b_conditions_changed_after_network_preflight")
    archive = m6b_execution.archive_inputs(root, selection, cases, current["conditions"])
    conditions = dict(current["conditions"], input_archive_sha256=archive["sha256"])
    experiment_id = uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    m6b_execution.claim_run(root, target, experiment_id, conditions)
    target.mkdir(parents=False, exist_ok=False)
    m6b_run._copy_archive(archive["files"], target / "input_archive")
    manifest = build_experiment_manifest(experiment_id, conditions, settings.get_cost_config(),
        settings.get_evaluation_config(), mode="full", max_run_cost_cny=M6B_LIVE_CAP)
    wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest,
        settings.get_evaluation_config())
    try:
        index = build_bm25_index(_full_corpus(root))
        return _execute_slots(root, selection, cases, target, experiment_id,
            conditions, archive["sha256"], wallet, index)
    except Exception:
        _halt_wallet_safely(wallet, "m6b_live_setup_or_audit_failure")
        raise
    finally:
        wallet.close()


def _valid_target(root: Path, run_dir: Path) -> Path:
    """Validate a fresh D-drive run path without creating directories before network checks."""
    target, traces = run_dir.resolve(), (root / "traces").resolve()
    if target == traces or not target.is_relative_to(traces) or target.exists():
        raise ValueError("m6b_live_run_directory_invalid_or_exists")
    return target


def _execute_slots(root: Path, selection: dict[str, Any], cases: tuple[m6b_run.M6bCase, ...],
                   target: Path, experiment_id: str, conditions: dict[str, str], archive_sha: str,
                   wallet: PersistentExperimentBudget, index: Any) -> dict[str, Any]:
    """Run and durably record the fixed prefix, stopping on any uncertainty or drift."""
    rows, stopped = [], False
    (target / "answers.jsonl").write_bytes(b"")
    m6b_execution.write_run(target, _summary(experiment_id, rows, wallet, archive_sha, False))
    for case in cases:
        if stopped:
            row = m6b_run._unexecuted(case, experiment_id,
                wallet.snapshot().halt_reason or "prior_slot_stopped")
        else:
            row = _execute_case(root, case, target, experiment_id, wallet, index, selection, conditions)
            status = wallet.snapshot()
            if row["status"] == "failed" and not status.halted:
                wallet.halt("m6b_live_case_failed")
                status = wallet.snapshot()
            stopped = (row["status"] == "failed" or status.halted
                or status.pending_requests > 0 or status.unknown_requests > 0
                or row["stop_reason"] == "experiment_cost_limit")
        rows.append(row)
        m6b_execution.append_row(target / "answers.jsonl", row)
        m6b_execution.write_run(target, _summary(experiment_id, rows, wallet, archive_sha, stopped))
    audit = m6b_execution.audit_run(root, target, experiment_id, selection,
        cases, rows, wallet, conditions, archive_sha)
    summary = _summary(experiment_id, rows, wallet, archive_sha, stopped) | {"audit": audit}
    m6b_execution.write_run(target, summary)
    return summary


def _execute_case(root: Path, case: m6b_run.M6bCase, target: Path, experiment_id: str,
                  wallet: PersistentExperimentBudget, index: Any, selection: dict[str, Any],
                  conditions: dict[str, str]) -> dict[str, Any]:
    """Invoke the actual authorized graph once and preserve its slot trace and settlement state."""
    from experiments.archive.m5_checkpoint import _close_worker_scopes, _has_experiment_budget_refusal, _halt_wallet_safely

    traces = target / "cases" / f"{case.order:02d}" / "traces"
    traces.mkdir(parents=True, exist_ok=False)
    before, started, composition = wallet.snapshot(), perf_counter(), None
    state, status, reason, thread_id = {}, "failed", "m6b_case_setup_failed", None
    try:
        if not m6b_execution.conditions_unchanged(root, target, selection, conditions):
            raise ValueError("m6b_conditions_changed_before_slot")
        initial = target / "input_archive" / f"case-{case.order:02d}" / "initial-evidence.json"
        composition = m6b_run.assemble_m6b_case(root, case, wallet, traces, index,
            initial if initial.is_file() else None)
        thread_id = composition.context.thread_id
        state = composition.assemble(external_calls_authorized=True).invoke()
        if not state.get("brief") or not isinstance(state.get("risk"), dict):
            raise ValueError("m6b_graph_terminal_output_missing")
        status, reason = "completed", str(state.get("stop_reason") or "")
        if _has_experiment_budget_refusal(traces):
            reason = "experiment_cost_limit"
            _halt_wallet_safely(wallet, reason)
        if not m6b_execution.conditions_unchanged(root, target, selection, conditions):
            raise ValueError("m6b_conditions_changed_after_slot")
    except Exception as exc:
        status = "failed"
        reason = type(exc).__name__ if not isinstance(exc, ValueError) else str(exc)
        _halt_wallet_safely(wallet, "m6b_live_case_failure")
    finally:
        if composition is not None:
            close_error = _close_worker_scopes((composition.inputs.fact_worker_scope,
                composition.inputs.propagation_worker_scope), wallet)
            if close_error:
                status, reason = "failed", close_error
    after = wallet.snapshot()
    return {"experiment_id": experiment_id, "order": case.order,
        "sample_id": case.sample.sample_id, "status": status, "stop_reason": reason,
        "thread_id": thread_id, "elapsed_ms": int((perf_counter()-started)*1000),
        "known_cost_delta_cny": str(after.known_cost_subtotal_cny-before.known_cost_subtotal_cny),
        "usage_unknown": after.unknown_requests > 0, "pending_requests": after.pending_requests,
        "state": state, "trace_root": traces.relative_to(target).as_posix()}


def _summary(experiment_id: str, rows: list[dict[str, Any]], wallet: PersistentExperimentBudget,
             archive_sha: str, stopped: bool) -> dict[str, Any]:
    """Summarize live progress without treating unknown usage as free."""
    status = wallet.snapshot()
    return {"experiment_id": experiment_id, "mode": "authorized_live_graph",
        "planned": 10, "executed": sum(row["status"] != "unexecuted" for row in rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unexecuted": sum(row["status"] == "unexecuted" for row in rows),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_requests": status.unknown_requests,
        "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_requests": status.pending_requests,
        "pending_reserved_cny": str(status.pending_reserved_cny), "halted": status.halted,
        "stop_reason": status.halt_reason, "stopped": stopped,
        "archive_sha256": archive_sha, "rows": rows}
