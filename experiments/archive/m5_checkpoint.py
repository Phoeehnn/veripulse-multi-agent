"""执行仅限固定十题的M5检查点，并绑定共享钱包和条件快照。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from src.yuqing import llm, settings
from src.yuqing.eval.experiment_session import build_experiment_manifest
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.graph.composition import CaseComposition, CaseInputs
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.retrieval.bm25 import BM25Index, build_bm25_index
from src.yuqing.schemas import AgentSpec, Sample
from experiments.archive.m5_inputs import (FIXED_CAP, SUMMARY, M5CaseInput, M5Inputs,
    _archive_shas, _json_bytes, _sha, _source_shas, _verify_committed_runtime,
    load_m5_inputs, prepare_m5_inputs)

logger = logging.getLogger(__name__)

def m5_condition_snapshot(root: Path, inputs: M5Inputs, *, require_committed: bool = False) -> dict[str, str]:
    """Fingerprint current runtime conditions and the exact fixed-ten input archive."""
    files = _source_shas(root)
    runtime = _verify_committed_runtime(root) if require_committed else {}
    config = settings.get_m5_run_budget_config()
    prices = settings.get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        raise ValueError("m5_prices_unknown")
    model = settings.get_settings()
    values = {"file:" + key: digest for key, digest in files.items()}
    values.update({"runtime_git:" + key: digest for key, digest in runtime.items()})
    values.update({"archive_sha256": inputs.archive_sha256,
        **{"archive_current:" + key: digest for key, digest in _archive_shas(root, inputs.archive).items()},
        "inputs_summary_sha256": _sha((root / SUMMARY).read_bytes()),
        "max_cost_cny": str(FIXED_CAP), "model": model.deepseek_model,
        "base_url_sha256": _sha(model.deepseek_base_url.encode()),
        "timeout_s": str(model.llm_timeout_s), "thinking": str(model.llm_thinking),
        "budget_sha256": _sha(_json_bytes(config.model_dump())),
        "prices_sha256": _sha(_json_bytes(prices.model_dump()))})
    corpus = settings.get_settings().dev_store_path
    for name in ("corpus.json", "READY"):
        path = corpus / name
        if not path.is_file():
            raise ValueError("m5_corpus_signature_missing:" + name)
        values["corpus:" + name] = _sha(path.read_bytes())
    return values


def preflight_m5(root: Path, *, offline: bool) -> tuple[M5Inputs, dict[str, str]]:
    """Validate fixed-ten inputs, all agent permissions, config and local corpus before effects."""
    from src.yuqing.eval.preflight import validate_propagation_startup, validate_runtime_config
    from src.yuqing.graph.fact_adapter import FactNodeAdapter

    validate_runtime_config()
    inputs = load_m5_inputs(root)
    FactNodeAdapter(settings.get_graph_config().fact_node_version, root=root)
    validate_propagation_startup(root / "prompts/propagation/v1.md", offline=offline)
    if not (root / "prompts/arbiter/v2.md").is_file():
        raise ValueError("m5_arbiter_prompt_missing")
    _validate_arbiter_permission(root)
    settings.get_m5_run_budget_config()
    from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus

    chunks = load_eval_corpus(settings.get_settings().chef_dev_path, [], corpus_name="chef_dev_full")
    validate_eval_corpus(chunks, settings.get_settings().dev_store_path, corpus_name="chef_dev_full")
    return inputs, m5_condition_snapshot(root, inputs, require_committed=not offline)


def _validate_arbiter_permission(root: Path) -> None:
    """Require the exact approved development contract for arbiter v2."""
    from experiments.archive.m4_five_runner import _prompt_dev_permission

    if not _prompt_dev_permission(root, root / "prompts/arbiter/v2.md"):
        raise ValueError("m5_arbiter_prompt_not_licensed")


def _runtime_case(case: M5CaseInput, inputs: M5Inputs, wallet: PersistentExperimentBudget,
                  root: Path, index: BM25Index, run_dir: Path,
                  specs: tuple[AgentSpec, AgentSpec]) -> tuple[dict[str, Any], str]:
    """Run one fixed case through the production composition root with shared wallet."""
    from src.yuqing.graph.composition import CaseComposition

    sample = case.sample
    sample_dir = run_dir / "cases" / f"{case.order:02d}"
    traces = sample_dir / "traces"
    fact_scope, propagation_scope = EvidenceWorkerScope(), EvidenceWorkerScope()
    composition = None
    state, status, reason, thread_id, before = {}, "failed", "m5_case_setup_failed", None, None
    started = time.perf_counter()
    try:
        traces.mkdir(parents=True, exist_ok=False)
        before = wallet.snapshot()
        composition = CaseComposition(CaseInputs(
            case_id=f"m5:{case.order}:{sample.sample_id}", sample=sample, coverage=case.coverage,
            trace_root=traces, wallet=wallet, fact_worker_scope=fact_scope,
            propagation_worker_scope=propagation_scope, index=index,
            store_path=settings.get_settings().dev_store_path, initial_manifest_path=case.initial_path or root,
            propagation_spec=specs[0], arbiter_spec=specs[1], as_of=sample.published_at))
        thread_id = composition.context.thread_id
        state = composition.assemble(external_calls_authorized=True).invoke()
        if not state.get("brief") or not isinstance(state.get("risk"), dict):
            raise ValueError("m5_graph_terminal_output_missing")
        status, reason = "completed", state.get("stop_reason") or ""
    except Exception as exc:
        status, reason = "failed", type(exc).__name__
        _halt_wallet_safely(wallet, "m5_case_runtime_failure")
    finally:
        close_error = _close_worker_scopes((fact_scope, propagation_scope), wallet)
        if close_error:
            status, reason = "failed", close_error
    elapsed = int((time.perf_counter() - started) * 1000)
    try:
        after = wallet.snapshot()
    except Exception as exc:
        after = None
        status, reason = "failed", "wallet_snapshot_" + type(exc).__name__
    before_cost = before.known_cost_subtotal_cny if before is not None else Decimal(0)
    after_cost = after.known_cost_subtotal_cny if after is not None else before_cost
    record = {"order": case.order, "sample_id": sample.sample_id, "status": status,
        "stop_reason": reason, "thread_id": thread_id,
        "elapsed_ms": elapsed, "known_cost_delta_cny": str(after_cost - before_cost),
        "usage_unknown": after is None or after.unknown_requests > 0,
        "pending_requests": after.pending_requests if after is not None else None,
        "state": state, "trace_root": traces.relative_to(run_dir).as_posix()}
    return record, thread_id or ""


def _close_worker_scopes(scopes: tuple[EvidenceWorkerScope, ...],
                         wallet: PersistentExperimentBudget) -> str | None:
    """Close every evidence worker and halt the run when cleanup fails."""
    first_error = None
    for scope in scopes:
        try:
            scope.close()
        except Exception as exc:
            _halt_wallet_safely(wallet, "m5_worker_scope_close_failure")
            logger.error("M5工作位关闭失败 type=%s", type(exc).__name__)
            first_error = first_error or "worker_scope_close_" + type(exc).__name__
    return first_error


def run_m5_checkpoint(root: Path, run_dir: Path, *, authorize_real_calls: bool) -> dict[str, object]:
    """Run exactly the fixed ten once; preflight and network checks precede all run side effects."""
    inputs, conditions, target = _preflight_real_run(root, run_dir, authorize_real_calls)
    experiment_id, wallet = _open_m5_wallet(root, target, conditions)
    try:
        return _execute_claimed_run(root, target, experiment_id, wallet, conditions, inputs)
    finally:
        wallet.close()


def _execute_claimed_run(root: Path, target: Path, experiment_id: str,
                         wallet: PersistentExperimentBudget,
                         conditions: dict[str, str], inputs: M5Inputs) -> dict[str, object]:
    """Keep setup, slot execution, audit and publication inside the wallet lifetime."""
    from experiments.archive.m5_checkpoint_audit import audit_m5_checkpoint

    rows: list[dict[str, object]] = []
    try:
        index = build_bm25_index(_full_corpus(root))
    except Exception as exc:
        _halt_wallet_safely(wallet, "m5_setup_failure")
        rows = [_unexecuted(case, experiment_id, "setup_" + type(exc).__name__)
                for case in inputs.cases]
        for row in rows:
            _append_jsonl(target / "answers.jsonl", row)
    else:
        rows, _ = _run_slots(root, inputs, wallet, index, target, experiment_id, conditions)
    rows = _complete_fixed_rows(inputs, rows, experiment_id, wallet.snapshot().halt_reason)
    summary = _run_summary(experiment_id, rows, wallet.snapshot())
    summary["audit"] = audit_m5_checkpoint(root, target, experiment_id, inputs, rows, wallet, conditions)
    _atomic_run_outputs(root, experiment_id, rows, summary)
    return summary


def _preflight_real_run(root: Path, run_dir: Path, authorized: bool
                        ) -> tuple[M5Inputs, dict[str, str], Path]:
    """Check authorization, fixed inputs, existing claim and network before creating state."""
    if not authorized or root.resolve() != settings.PROJECT_ROOT.resolve():
        raise ValueError("m5_real_run_root_or_authorization_invalid")
    inputs, conditions = preflight_m5(root, offline=False)
    _require_credentials()
    target = _new_run_dir(run_dir, root)
    if (root / "traces/m5-fixed-ten.claim").exists():
        raise ValueError("m5_fixed_ten_attempt_already_claimed")
    _provider_connection()
    current, repeated = preflight_m5(root, offline=False)
    if repeated != conditions or current.archive_sha256 != inputs.archive_sha256:
        raise ValueError("m5_conditions_changed_after_network_preflight")
    return current, conditions, target


def _open_m5_wallet(root: Path, target: Path, conditions: dict[str, str]
                    ) -> tuple[str, PersistentExperimentBudget]:
    """Claim the one run before creating its single shared six-yuan wallet."""
    experiment_id = uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    _claim_once(root / "traces/m5-fixed-ten.claim", experiment_id, target, conditions)
    target.mkdir(parents=False, exist_ok=False)
    manifest = build_experiment_manifest(experiment_id, conditions, settings.get_cost_config(),
        settings.get_evaluation_config(), mode="full", max_run_cost_cny=FIXED_CAP)
    wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest,
        settings.get_evaluation_config())
    try:
        (target / "answers.jsonl").write_bytes(b"")
    except Exception:
        wallet.close()
        raise
    return experiment_id, wallet


def _run_slots(root: Path, inputs: M5Inputs, wallet: PersistentExperimentBudget, index: BM25Index,
               target: Path, experiment_id: str, conditions: dict[str, str]
               ) -> tuple[list[dict[str, object]], bool]:
    """Execute fixed cases in order and stop on drift, unknown usage, pending spend or wallet halt."""
    rows, stopped = [], False
    for case in inputs.cases:
        record = None
        if stopped:
            record = _unexecuted(case, experiment_id,
                wallet.snapshot().halt_reason or "prior_slot_stopped")
        else:
            try:
                if m5_condition_snapshot(root, inputs, require_committed=True) != conditions:
                    raise ValueError("m5_conditions_changed_before_case")
                record, _ = _runtime_case(case, inputs, wallet, root, index, target, _agent_specs(root))
                record["experiment_id"] = experiment_id
                status = wallet.snapshot()
                stopped = (record["status"] == "failed" or status.halted
                    or status.pending_requests > 0 or status.unknown_requests > 0)
                if _has_experiment_budget_refusal(target / str(record["trace_root"])):
                    _halt_wallet_safely(wallet, "experiment_cost_limit")
                    record["stop_reason"], stopped = "experiment_cost_limit", True
                if m5_condition_snapshot(root, inputs, require_committed=True) != conditions:
                    _halt_wallet_safely(wallet, "m5_conditions_changed_after_case")
                    record["stop_reason"], stopped = "m5_conditions_changed_after_case", True
            except Exception as exc:
                _halt_wallet_safely(wallet, "m5_slot_postcheck_failure")
                stopped = True
                if record is None:
                    record = _unexecuted(case, experiment_id, "slot_" + type(exc).__name__)
                else:
                    record["status"] = "failed"
                    record["stop_reason"] = "postcheck_" + type(exc).__name__
        _append_jsonl(target / "answers.jsonl", record)
        rows.append(record)
    return rows, stopped


def _halt_wallet_safely(wallet: PersistentExperimentBudget, reason: str) -> None:
    """Try to persist a stop without hiding the original stage failure."""
    try:
        if not wallet.snapshot().halted:
            wallet.halt(reason)
    except Exception as exc:
        logger.error("M5停止账本未确认 type=%s", type(exc).__name__)


def _complete_fixed_rows(inputs: M5Inputs, rows: list[dict[str, object]],
                         experiment_id: str, reason: str | None) -> list[dict[str, object]]:
    """Fill only the unexecuted suffix so every final record has ten fixed slots."""
    if len(rows) > 10:
        raise ValueError("m5_slot_result_overflow")
    for case in inputs.cases[len(rows):]:
        rows.append(_unexecuted(case, experiment_id, reason or "prior_slot_stopped"))
    return rows


def _has_experiment_budget_refusal(trace_dir: Path) -> bool:
    """Find an unsent shared-wallet cap refusal in this slot's own Trace files."""
    for path in trace_dir.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            data = event.get("data", {})
            if isinstance(data, dict) and data.get("reason") == "experiment_cost_limit":
                return True
    return False


def _require_credentials() -> None:
    """Fail before network access when model identity or secret is absent."""
    runtime = settings.get_settings()
    key = runtime.deepseek_api_key.get_secret_value() if runtime.deepseek_api_key else ""
    if not runtime.deepseek_model or not key.strip():
        raise ValueError("m5_provider_credentials_missing")


def _provider_connection() -> None:
    """Perform the configured no-secret HTTPS reachability check before wallet creation."""
    runtime = settings.get_settings()
    reachable, stage, status = llm.check_provider_connection(runtime.deepseek_base_url, runtime.llm_timeout_s)
    if not reachable or status is None:
        raise ValueError("m5_provider_preflight_" + stage)


def _new_run_dir(path: Path, root: Path) -> Path:
    """Require a fresh experiment directory inside D-drive project traces."""
    target, traces = path.resolve(), (root / "traces").resolve()
    if not target.is_relative_to(traces) or target == traces or target.exists():
        raise ValueError("m5_run_directory_invalid_or_exists")
    return target


def _claim_once(path: Path, experiment_id: str, run_dir: Path,
                conditions: dict[str, str]) -> None:
    """Atomically claim the only fixed-ten attempt; an old claim is never replaced."""
    payload = {"experiment_id": experiment_id, "run_dir": run_dir.relative_to(
        (settings.PROJECT_ROOT / "traces").resolve()).as_posix(),
        "conditions_sha256": _sha(_json_bytes(conditions)), "status": "claimed"}
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m5_fixed_ten_attempt_already_claimed") from exc


def _agent_specs(root: Path) -> tuple[AgentSpec, AgentSpec]:
    """Load the two already licensed prompt bodies after preflight verified their contracts."""
    return tuple(AgentSpec(agent=agent, system_prompt=(root / path).read_text(encoding="utf-8"))
        for agent, path in (("propagation_analyst", "prompts/propagation/v1.md"),
            ("arbiter", "prompts/arbiter/v2.md")))


def _full_corpus(root: Path) -> list[object]:
    """Load the complete CHEF dev retrieval corpus used by the production graph."""
    from src.yuqing.retrieval.corpus import load_eval_corpus

    return load_eval_corpus(settings.get_settings().chef_dev_path, [], corpus_name="chef_dev_full")


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    """Durably append one finished slot before moving to the next selected sample."""
    with path.open("ab") as stream:
        stream.write(_json_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _unexecuted(case: M5CaseInput, experiment_id: str, reason: str) -> dict[str, object]:
    """Represent an unstarted fixed slot without fabricating a model answer."""
    return {"experiment_id": experiment_id, "order": case.order,
        "sample_id": case.sample.sample_id, "status": "unexecuted", "stop_reason": reason,
        "thread_id": None, "elapsed_ms": None, "known_cost_delta_cny": "0",
        "usage_unknown": False, "pending_requests": 0, "state": None, "trace_root": None}


def _run_summary(experiment_id: str, rows: list[dict[str, object]], status: object) -> dict[str, object]:
    """Summarize all ten slots and wallet certainty without treating failure as a score."""
    return {"experiment_id": experiment_id, "planned": 10,
        "executed": sum(row["status"] != "unexecuted" for row in rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unexecuted": sum(row["status"] == "unexecuted" for row in rows),
        "known_cost_subtotal_cny": str(status.known_cost_subtotal_cny),
        "unknown_reserved_cny": str(status.unknown_reserved_cny),
        "pending_reserved_cny": str(status.pending_reserved_cny),
        "halted": status.halted, "stop_reason": status.halt_reason}


def _atomic_run_outputs(root: Path, experiment_id: str, rows: list[dict[str, object]],
                        summary: dict[str, object]) -> tuple[Path, Path]:
    """Publish a new CSV and JSONL audit under unique immutable results names."""
    import csv
    import io

    run_id, day = experiment_id, datetime.now().date().isoformat()
    result_root = root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    fields = ("experiment_id", "order", "sample_id", "status", "stop_reason", "thread_id",
        "elapsed_ms", "known_cost_delta_cny", "usage_unknown", "pending_requests", "trace_root")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    outputs = [(result_root / f"{day}_m5_{run_id}.csv", stream.getvalue().encode("utf-8")),
        (result_root / f"{day}_audit_{run_id}.jsonl",
         (_json_bytes({"summary": summary, "rows": rows}).decode("utf-8") + "\n").encode("utf-8"))]
    for target, payload in outputs:
        temporary = target.with_name(target.name + "." + uuid4().hex + ".partial")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return outputs[0][0], outputs[1][0]
