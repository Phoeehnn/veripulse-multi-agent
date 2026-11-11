"""Prepare the fixed M6b inputs without creating a wallet or contacting a model."""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping
from uuid import uuid4

from experiments.archive import joint_demo as joint
from experiments.archive.m6b_readiness import PROMPT, m6b_composition, m6b_preflight
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.schemas import AgentSpec, Sample
from src.yuqing.graph.composition import CaseComposition, CaseInputs
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.eval.experiment_session import build_experiment_manifest
from src.yuqing import settings

M6B_CAP = Decimal("3")
M6B_CLAIM = "traces/m6b-fixed-ten.claim"


@dataclass(frozen=True)
class M6bCase:
    """One verified slot; joint evidence remains bound to its prepared case."""

    order: int
    group: str
    sample: Sample
    joint_input: joint.PreparedJointDemo | None = None


MockCaseExecutor = Callable[[M6bCase, CaseComposition, Path, str], Mapping[str, Any]]


def prepare_m6b_cases(root: Path, selection: dict[str, Any]) -> tuple[M6bCase, ...]:
    """Validate the frozen selection once and load its three distinct input groups."""
    m6b_preflight(root, selection)
    questions = {}
    for name in ("questions_30.jsonl", "dev_extra_60.jsonl"):
        for line in (root / "eval" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                questions[item["sample_id"]] = item
    cases = []
    for row in selection["samples"]:
        prepared = None
        if row["group"] == "joint_reuse":
            prepared = joint._validated_joint_demo(root, row["joint_case"])
            sample = prepared.sample
        elif row["group"] == "ced_new":
            _, category, stem = row["sample_id"].split(":", 2)
            sample = load_ced_sample(root / "data/raw/CED_Dataset", stem + ".json",
                rumor=category == "rumor")
        else:
            sample = Sample(sample_id=row["sample_id"], source="chef",
                claim=questions[row["sample_id"]]["claim"])
        sample = sample.model_copy(update={"gold_veracity": None, "gold_risk": None})
        if sample.sample_id != row["sample_id"]:
            raise ValueError("m6b_loaded_sample_identity_mismatch")
        if row.get("sample_sha256") and joint._digest(sample.model_dump(mode="json")) != row["sample_sha256"]:
            raise ValueError("m6b_loaded_sample_content_mismatch")
        cases.append(M6bCase(row["order"], row["group"], sample, prepared))
    return tuple(cases)


def assemble_m6b_case(root: Path, case: M6bCase, wallet: PersistentExperimentBudget,
                      trace_root: Path, index: BM25Index,
                      initial_path: Path | None = None) -> CaseComposition:
    """Bind the existing graph to one verified slot without invoking any Agent."""
    if case.joint_input is not None:
        return m6b_composition(case.joint_input, wallet, trace_root, initial_path)
    if case.group == "chef_dev_reuse" and initial_path is None:
        raise ValueError("m6b_chef_initial_evidence_missing")
    specs = (AgentSpec(agent="propagation_analyst", system_prompt=(root / PROMPT).read_text(encoding="utf-8")),
        AgentSpec(agent="arbiter", system_prompt=(root / "prompts/arbiter/v2.md").read_text(encoding="utf-8")))
    fact_scope, propagation_scope = EvidenceWorkerScope(), EvidenceWorkerScope()
    try:
        return CaseComposition(CaseInputs(case_id=f"m6b:{case.order}:{case.sample.sample_id}",
            sample=case.sample, coverage="full" if case.group == "chef_dev_reuse" else "none",
            trace_root=trace_root, wallet=wallet, fact_worker_scope=fact_scope,
            propagation_worker_scope=propagation_scope, index=index,
            store_path=settings.get_settings().dev_store_path,
            initial_manifest_path=initial_path or root, propagation_spec=specs[0], arbiter_spec=specs[1],
            as_of=case.sample.published_at, group_id="m6b_development", variant="m6b-offline-candidate",
            propagation_contract_version="v1.1", m6b=True))
    except Exception:
        fact_scope.close()
        propagation_scope.close()
        raise


def run_m6b_mock_experiment(root: Path, selection: dict[str, Any], run_dir: Path,
                            execute_case: MockCaseExecutor,
                            index: BM25Index | None = None) -> dict[str, Any]:
    """Run fixed ten through the existing composition with an injected offline executor."""
    from experiments.archive.m6b_execution import archive_inputs, claim_run
    from experiments.archive.m5_checkpoint import _full_corpus, _halt_wallet_safely
    from src.yuqing.retrieval.bm25 import build_bm25_index

    preflight = m6b_preflight(root, selection)
    cases = prepare_m6b_cases(root, selection)
    target = _new_run_dir(root, run_dir)
    archive = archive_inputs(root, selection, cases, preflight["conditions"])
    conditions = dict(preflight["conditions"], input_archive_sha256=archive["sha256"])
    experiment_id = uuid4().hex
    claim_run(root, target, experiment_id, conditions)
    target.mkdir(parents=True, exist_ok=False)
    (target / "input_archive").mkdir()
    _copy_archive(archive["files"], target / "input_archive")
    manifest = build_experiment_manifest(experiment_id, conditions, settings.get_cost_config(),
        settings.get_evaluation_config(), mode="full", max_run_cost_cny=M6B_CAP)
    wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest,
        settings.get_evaluation_config())
    try:
        runtime_index = index if index is not None else build_bm25_index(_full_corpus(root))
        return _execute_mock_slots(root, selection, cases, target, experiment_id,
            conditions, archive["sha256"], wallet, execute_case, runtime_index)
    except Exception:
        _halt_wallet_safely(wallet, "m6b_run_audit_or_setup_failure")
        raise
    finally:
        wallet.close()


def _new_run_dir(root: Path, run_dir: Path) -> Path:
    """Accept only a fresh child of this project’s D-drive traces directory."""
    target, traces = run_dir.resolve(), (root / "traces").resolve()
    if target == traces or not target.is_relative_to(traces) or target.exists():
        raise ValueError("m6b_run_directory_invalid_or_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _copy_archive(files: Mapping[str, bytes], target: Path) -> None:
    """Publish already fingerprinted model inputs with exclusive, durable writes."""
    for name, payload in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())


def _execute_mock_slots(root: Path, selection: dict[str, Any], cases: tuple[M6bCase, ...],
                        target: Path, experiment_id: str, conditions: dict[str, str],
                        archive_sha: str, wallet: PersistentExperimentBudget,
                        execute_case: MockCaseExecutor, index: BM25Index) -> dict[str, Any]:
    """Persist each fixed slot, stop on any uncertain spend or changed input, then audit."""
    from experiments.archive.m6b_execution import append_row, audit_run, write_run

    rows: list[dict[str, Any]] = []
    write_run(target, _summary(experiment_id, rows, wallet, archive_sha, False))
    stopped = False
    for case in cases:
        if stopped:
            row = _unexecuted(case, experiment_id, wallet.snapshot().halt_reason or "prior_slot_stopped")
        else:
            row = _execute_mock_case(root, case, target, experiment_id, wallet, index,
                execute_case, selection, conditions)
            status = wallet.snapshot()
            budget_refusal = row["stop_reason"] == "experiment_cost_limit"
            if not budget_refusal and row.get("trace_root"):
                from experiments.archive.m5_checkpoint import _has_experiment_budget_refusal
                budget_refusal = _has_experiment_budget_refusal(target / row["trace_root"])
            if budget_refusal and not status.halted:
                wallet.halt("experiment_cost_limit")
                status = wallet.snapshot()
            if status.pending_requests and not status.halted:
                wallet.halt("m6b_unsettled_request")
                status = wallet.snapshot()
            if status.pending_requests:
                row["stop_reason"] = "m6b_unsettled_request"
            stopped = (row["status"] == "failed" or status.halted or status.unknown_requests > 0
                or status.pending_requests > 0)
            if budget_refusal:
                row["stop_reason"] = "experiment_cost_limit"
        rows.append(row)
        append_row(target / "answers.jsonl", row)
        write_run(target, _summary(experiment_id, rows, wallet, archive_sha, stopped))
    audit = audit_run(root, target, experiment_id, selection, cases, rows, wallet, conditions, archive_sha)
    summary = _summary(experiment_id, rows, wallet, archive_sha, stopped) | {"audit": audit}
    write_run(target, summary)
    return summary


def _execute_mock_case(root: Path, case: M6bCase, target: Path, experiment_id: str,
                       wallet: PersistentExperimentBudget, index: BM25Index,
                       execute_case: MockCaseExecutor, selection: dict[str, Any],
                       conditions: dict[str, str]) -> dict[str, Any]:
    """Invoke only the injected executor and retain its terminal state and trace location."""
    from experiments.archive.m6b_execution import conditions_unchanged

    traces = target / "cases" / f"{case.order:02d}" / "traces"
    traces.mkdir(parents=True, exist_ok=False)
    before = wallet.snapshot()
    started = perf_counter()
    composition = None
    status, reason, state, thread_id = "failed", "m6b_case_setup_failed", {}, None
    try:
        if not conditions_unchanged(root, target, selection, conditions):
            raise ValueError("m6b_conditions_changed_before_slot")
        initial = target / "input_archive" / f"case-{case.order:02d}" / "initial-evidence.json"
        composition = assemble_m6b_case(root, case, wallet, traces, index,
            initial if initial.is_file() else None)
        thread_id = composition.context.thread_id
        output = dict(execute_case(case, composition, traces, experiment_id))
        if not conditions_unchanged(root, target, selection, conditions):
            raise ValueError("m6b_conditions_changed_after_slot")
        state = output.get("state") if isinstance(output.get("state"), dict) else {}
        status = "completed" if output.get("status") == "completed" else "failed"
        reason = str(output.get("stop_reason", ""))
        if status == "failed" and not wallet.snapshot().halted:
            from experiments.archive.m5_checkpoint import _halt_wallet_safely
            _halt_wallet_safely(wallet, "m6b_case_failed")
    except Exception as exc:
        reason = type(exc).__name__
        if not wallet.snapshot().halted:
            from experiments.archive.m5_checkpoint import _halt_wallet_safely
            _halt_wallet_safely(wallet, "m6b_case_failure")
    finally:
        if composition is not None:
            from experiments.archive.m5_checkpoint import _close_worker_scopes
            error = _close_worker_scopes((composition.inputs.fact_worker_scope,
                composition.inputs.propagation_worker_scope), wallet)
            if error:
                status, reason = "failed", error
    after = wallet.snapshot()
    return {"experiment_id": experiment_id, "order": case.order,
        "sample_id": case.sample.sample_id, "status": status, "stop_reason": reason,
        "thread_id": thread_id, "elapsed_ms": int((perf_counter()-started)*1000),
        "known_cost_delta_cny": str(after.known_cost_subtotal_cny-before.known_cost_subtotal_cny),
        "usage_unknown": after.unknown_requests > 0, "pending_requests": after.pending_requests,
        "state": state, "trace_root": traces.relative_to(target).as_posix()}


def _unexecuted(case: M6bCase, experiment_id: str, reason: str) -> dict[str, Any]:
    """Represent an untouched suffix slot without creating a case Trace."""
    return {"experiment_id": experiment_id, "order": case.order, "sample_id": case.sample.sample_id,
        "status": "unexecuted", "stop_reason": reason, "thread_id": None, "elapsed_ms": None,
        "known_cost_delta_cny": "0", "usage_unknown": False, "pending_requests": 0,
        "state": None, "trace_root": None}


def _summary(experiment_id: str, rows: list[dict[str, Any]],
             wallet: PersistentExperimentBudget, archive_sha: str, stopped: bool) -> dict[str, Any]:
    """Summarize all slots and distinguish known spend from uncertain reservations."""
    status = wallet.snapshot()
    return {"experiment_id": experiment_id, "mode": "mock_injected_executor",
        "planned": 10, "executed": sum(row["status"] != "unexecuted" for row in rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unexecuted": sum(row["status"] == "unexecuted" for row in rows),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_requests": status.unknown_requests, "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_requests": status.pending_requests, "pending_reserved_cny": str(status.pending_reserved_cny),
        "halted": status.halted, "stop_reason": status.halt_reason, "stopped": stopped,
        "paid_ready": False, "provider_calls": 0, "archive_sha256": archive_sha,
        "rows": rows}
