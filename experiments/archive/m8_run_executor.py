"""Execute the fixed M8 slots through the existing graph with a shared durable wallet."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import time
from typing import Any, Callable, Literal

from pydantic import ValidationError

from src.yuqing.eval.experiment_budget_store import _manifest_data
from src.yuqing.eval.experiment_session import build_experiment_manifest
from experiments.archive.m8_cache import M8CacheIdentity, load_shared_decision
from experiments.archive.m8_inputs import (M8CaseInput, M8Inputs, SELECTION_PATH,
    _build_context, _read_jsonl, _source_shas, load_m8_inputs)
from experiments.archive.m8_run_journal import M8RunJournal, _read_manifest
from experiments.archive.m8_run_restart import (
    PARENT_MANIFEST_SHA256, audit_approved_parent, validate_retry_cap,
    validate_retry_input_identity,
)
from experiments.archive.m8_run_preflight import (
    M8ExecutionSlot, M8PreflightReport, _canonical, _digest, _runtime_conditions,
    build_m8_slot_plan, preflight_m8_run,
)
from experiments.archive.m8_runner import (assemble_m8_group, assemble_m8_shared_prefix,
    fork_m8_group, map_ced_shared_result, save_m8_shared_snapshot)
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.graph.composition import CaseComposition, CaseInputs
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.schemas import AgentSpec, Sample
from src.yuqing.settings import (
    get_cost_config, get_evaluation_config, get_settings,
)


SlotRunner = Callable[[M8ExecutionSlot, M8RunJournal, object], dict[str, Any]]


class M8SlotValidationFailure(RuntimeError):
    """Carry only safe validation locations from a failed cache stage."""

    def __init__(self, stage: str, issues: tuple[dict[str, Any], ...]) -> None:
        """Keep a controlled stage label and field/type pairs without input text."""
        self.stage, self.issues = stage, issues
        super().__init__("m8_slot_validation_failed")


class M8SlotRuntimeFailure(RuntimeError):
    """Carry a fixed stage and safe reason code for a non-schema slot failure."""

    def __init__(self, stage: str) -> None:
        """Accept only the shared snapshot stage exposed by the M8 runner."""
        if stage != "shared_snapshot_save":
            raise ValueError("m8_slot_failure_stage_invalid")
        self.stage, self.reason_code = stage, "m8_shared_snapshot_save_failed"
        super().__init__(self.reason_code)


@dataclass(frozen=True)
class M8RunReceipt:
    """Audited result for one run; a receipt never implies the full plan completed."""

    report: M8PreflightReport
    audit: dict[str, object]
    run_dir: Path


def run_m8_experiment(root: Path, summary_path: Path, summary_sha256: str,
                      run_dir: Path, *, operation: Literal["new", "resume", "retry"],
                      max_cost_cny: Decimal = Decimal("20"),
                      authorize_real_calls: bool = False,
                      slot_runner: SlotRunner | None = None,
                      retry_of: Path | None = None) -> M8RunReceipt:
    """Require live authorization, preflight, then claim and execute only durable next slots."""
    if slot_runner is None and not authorize_real_calls:
        raise ValueError("m8_real_run_requires_authorization")
    report = preflight_m8_run(root, summary_path, summary_sha256, max_cost_cny)
    inputs = load_m8_inputs(root, summary_path, summary_sha256)
    slots = build_m8_slot_plan(inputs)
    if ((operation == "retry" and retry_of is None)
            or (retry_of is not None and operation not in {"retry", "resume"})):
        raise ValueError("m8_retry_parent_argument_required")
    retry_parent_sha = None
    experiment_id = "m8-fixed-100"
    if retry_of is not None:
        if operation == "resume" and not _is_retry_run(run_dir):
            raise ValueError("m8_retry_resume_path_invalid")
        parent = audit_approved_parent(root, retry_of)
        validate_retry_cap(max_cost_cny, parent.known_cost_cny)
        validate_retry_input_identity(report, parent)
        retry_parent_sha = PARENT_MANIFEST_SHA256
        experiment_id = "m8-fixed-100-retry1"
    elif operation == "resume" and _is_retry_run(run_dir):
        raise ValueError("m8_retry_parent_argument_required")
    conditions = _wallet_conditions(report, max_cost_cny)
    budget_config = _m8_budget_config(max_cost_cny)
    manifest = build_experiment_manifest(experiment_id, conditions, get_cost_config(),
        budget_config, mode="full", max_run_cost_cny=max_cost_cny)
    manifest = _manifest_data(manifest)
    _validate_run_path(root, run_dir, operation, retry=retry_parent_sha is not None)
    if authorize_real_calls:
        _require_network_and_credentials()
    runner = slot_runner or M8GraphSlotExecutor(root, inputs, report, run_dir,
        authorize_real_calls)
    journal, wallet = _open_run(root, run_dir, report, slots, manifest,
        max_cost_cny, operation, budget_config, retry_parent_sha)
    try:
        _execute_slots(journal, wallet, slots, str(manifest["conditions_fingerprint"]), runner,
            lambda: _runtime_wallet_fingerprint(root, inputs, slots, report, max_cost_cny,
                slots[journal.completed_slots] if journal.completed_slots < len(slots) else None))
        audit = journal.audit(wallet)
        return M8RunReceipt(report, audit, journal.run_dir)
    finally:
        journal.close()
        wallet.close()


def _wallet_conditions(report: M8PreflightReport, cap: Decimal) -> dict[str, str]:
    """Bind the standard wallet fingerprint to the exact M8 archive and schedule."""
    return {"preflight_conditions_sha256": report.conditions_sha256,
        "summary_sha256": report.summary_sha256, "archive_sha256": report.archive_sha256,
        "slot_plan_sha256": report.slot_plan_sha256, "max_cost_cny": str(cap)}


def _runtime_wallet_fingerprint(root: Path, inputs: M8Inputs,
                                slots: tuple[M8ExecutionSlot, ...],
                                report: M8PreflightReport, cap: Decimal,
                                current_slot: M8ExecutionSlot | None = None) -> str:
    """Recheck source conditions and the next archived input before each slot."""
    summary = root / report.summary_path
    if (_sha_file(summary) != report.summary_sha256
            or _sha_file(inputs.archive / "inputs.json") != report.archive_sha256):
        raise ValueError("m8_run_archive_changed")
    pool = _read_jsonl(root / SELECTION_PATH)
    if _source_shas(root, pool) != inputs.source_sha256:
        raise ValueError("m8_run_source_conditions_changed")
    if current_slot is not None:
        case = next(item for item in inputs.cases if item.order == current_slot.order)
        _validate_case_archive(case)
    current = _digest(_canonical(_runtime_conditions(root, inputs, slots)))
    values = _wallet_conditions(report, cap)
    values["preflight_conditions_sha256"] = current
    return _digest(_canonical(values))


def _m8_budget_config(cap: Decimal) -> Any:
    """Apply the already approved M8 ceiling in-memory without changing shared config."""
    from src.yuqing.settings import EvaluationConfig

    base = get_evaluation_config().model_dump()
    base["max_run_cost_cny"] = float(cap)
    base["pilot_max_run_cost_cny"] = min(base["pilot_max_run_cost_cny"], float(cap))
    return EvaluationConfig.model_validate(base)


def _validate_case_archive(case: M8CaseInput) -> None:
    """Reject archived sample or initial-evidence bytes that changed after preflight."""
    if _sha_file(case.sample_path) != case.sample_sha256:
        raise ValueError("m8_slot_sample_changed")
    if case.initial_path is not None and _sha_file(case.initial_path) != case.initial_sha256:
        raise ValueError("m8_slot_initial_evidence_changed")


def _open_run(root: Path, run_dir: Path, report: M8PreflightReport,
              slots: tuple[M8ExecutionSlot, ...], manifest: dict[str, object],
              cap: Decimal, operation: Literal["new", "resume", "retry"], budget_config: Any,
              retry_parent_sha256: str | None = None,
              ) -> tuple[M8RunJournal, PersistentExperimentBudget]:
    """Create a one-time claim before its wallet, or strictly reopen matching records."""
    from src.yuqing.eval.experiment_session import _ledger_path
    run_path = run_dir.resolve(strict=False)
    wallet_path = run_path / "wallet.sqlite"
    if operation in {"new", "retry"}:
        experiment_id = "m8-fixed-100-retry1" if operation == "retry" else "m8-fixed-100"
        journal = M8RunJournal.create(root, run_path, experiment_id=experiment_id,
            summary_sha256=report.summary_sha256, archive_sha256=report.archive_sha256,
            conditions_sha256=str(manifest["conditions_fingerprint"]),
            slot_plan_sha256=report.slot_plan_sha256, max_cost_cny=cap, slots=slots,
            retry_parent_manifest_sha256=retry_parent_sha256)
        try:
            wallet = PersistentExperimentBudget.create(
                _ledger_path(wallet_path), manifest, budget_config)
            return journal, wallet
        except BaseException:
            journal.close()
            raise
    disk = _read_manifest(run_path / "manifest.json")
    _check_resume_manifest(disk, report, manifest, cap, slots)
    journal = M8RunJournal.resume(root, run_path, expected_manifest=disk)
    try:
        wallet = PersistentExperimentBudget.resume(
            _ledger_path(wallet_path), manifest, budget_config)
    except BaseException:
        journal.close()
        raise
    try:
        journal.audit(wallet)
        return journal, wallet
    except BaseException:
        wallet.close()
        journal.close()
        raise


def _validate_run_path(root: Path, run_dir: Path,
                       operation: Literal["new", "resume", "retry"],
                       *, retry: bool = False) -> None:
    """Reject unsafe, reused or unclaimed run paths before any network check."""
    traces = (root / "traces").resolve(strict=True)
    target = run_dir.resolve(strict=operation == "resume")
    if target == traces or not target.is_relative_to(traces):
        raise ValueError("m8_journal_path_escape")
    claim = traces / "m8-fixed-100.claim"
    retry_claim = traces / "m8-fixed-100-retry1.claim"
    if operation == "new" and (target.exists() or claim.exists() or retry):
        raise ValueError("m8_run_already_claimed")
    if operation == "retry" and (not retry or target.name != "m8-fixed-100-retry1"
            or target.exists() or not claim.is_file()
            or retry_claim.exists()):
        raise ValueError("m8_retry_not_available")
    if operation == "resume" and not (target / "manifest.json").is_file():
        raise ValueError("m8_resume_manifest_missing")


def _is_retry_run(run_dir: Path) -> bool:
    """Recognize only the one stable replacement directory for resume identity."""
    return run_dir.name == "m8-fixed-100-retry1"


def _check_resume_manifest(disk: dict[str, Any], report: M8PreflightReport,
                          wallet_manifest: dict[str, object], cap: Decimal,
                          slots: tuple[M8ExecutionSlot, ...]) -> None:
    """Reject resume before opening a wallet when any frozen identity differs."""
    expected_slots = [{"slot_index": index, "order": slot.order,
        "sample_id": slot.sample_id, "source": slot.source,
        "execution_group": slot.execution_group, "logical_groups": list(slot.logical_groups)}
        for index, slot in enumerate(slots, start=1)]
    if (disk.get("summary_sha256") != report.summary_sha256
            or disk.get("archive_sha256") != report.archive_sha256
            or disk.get("conditions_sha256") != wallet_manifest["conditions_fingerprint"]
            or disk.get("slot_plan_sha256") != report.slot_plan_sha256
            or disk.get("max_cost_cny") != str(cap) or disk.get("slots") != expected_slots):
        raise ValueError("m8_journal_conditions_changed")
    if (disk.get("experiment_id") == "m8-fixed-100-retry1"
            and disk.get("retry_parent_manifest_sha256") != PARENT_MANIFEST_SHA256):
        raise ValueError("m8_journal_retry_identity_invalid")


def _execute_slots(journal: M8RunJournal, wallet: object,
                   slots: tuple[M8ExecutionSlot, ...], conditions_sha256: str,
                   runner: SlotRunner, live_conditions: Callable[[], str]) -> None:
    """Run sequentially, save each terminal row, and stop on any unresolved spend."""
    while journal.completed_slots < len(slots):
        journal.require_conditions(live_conditions())
        status = wallet.snapshot()
        if status.halted or status.pending_requests or status.unknown_requests or status.cost_cny is None:
            journal.mark_unexecuted_suffix("wallet_unresolved")
            return
        slot = journal.start_next(conditions_sha256, status)
        before = _wallet_request_ids(wallet)
        started = time.perf_counter()
        terminal: Literal["completed", "failed"] = "completed"
        try:
            result = runner(slot, journal, wallet)
        except Exception as exc:
            result = _slot_failure_result(exc)
            terminal = "failed"
            _halt_wallet(wallet, "m8_slot_runtime_failure")
        if terminal == "completed":
            try:
                journal.require_conditions(live_conditions())
            except Exception as exc:
                result = {"status": "failed_conditions_changed", "completed_result": result,
                    "failure_type": type(exc).__name__, "failure_stage": "post_slot_condition_check",
                    "failure_code": "m8_conditions_changed_during_slot"}
                terminal = "failed"
                _halt_wallet(wallet, "m8_conditions_changed_during_slot")
        request_ids, trace_paths = _slot_trace_refs(journal.run_dir, slot, before, wallet)
        result = dict(result, elapsed_ms=int((time.perf_counter() - started) * 1000))
        journal.finish_slot(slot, terminal, result=result, request_ids=request_ids,
            trace_paths=trace_paths)
        try:
            from experiments.archive.m8_run_audit import audit_m8_slot

            audit_m8_slot(journal, wallet, journal.completed_slots)
        except Exception:
            _halt_wallet(wallet, "m8_slot_audit_failed")
            journal.mark_unexecuted_suffix("slot_audit_failed")
            return
        current = wallet.snapshot()
        if (terminal == "failed" or current.halted or current.pending_requests
                or current.unknown_requests or current.cost_cny is None):
            journal.mark_unexecuted_suffix("slot_failed_or_wallet_unresolved")
            return


def _slot_failure_result(exc: Exception) -> dict[str, Any]:
    """Convert a slot exception to safe, bounded diagnostic fields."""
    if isinstance(exc, M8SlotValidationFailure):
        return {"status": "failed", "failure_type": "ValidationError",
            "failure_stage": exc.stage, "failure_code": "m8_slot_validation_failed",
            "validation_errors": list(exc.issues)}
    if isinstance(exc, M8SlotRuntimeFailure):
        return {"status": "failed", "failure_type": type(exc).__name__,
            "failure_stage": exc.stage, "failure_code": exc.reason_code}
    return {"status": "failed", "failure_type": type(exc).__name__,
        "failure_stage": "slot_execution", "failure_code": "m8_slot_unexpected_error"}


def _wallet_request_ids(wallet: object) -> set[str]:
    """Snapshot request identities from the append-only budget history."""
    history = wallet.load_history()
    return {str(event["request_id"]) for event in history.events
        if event.get("event_type") in {"prepared", "reserved", "dispatched", "settled",
            "cancelled", "unknown"} and event.get("request_id")}


def _slot_trace_refs(run_dir: Path, slot: M8ExecutionSlot, before: set[str],
                     wallet: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Collect only this sample/group's traces and requests added by the current slot."""
    folder = run_dir / "slots" / f"{slot.order:03d}-{slot.execution_group}"
    trace_paths, request_ids = [], set()
    for path in sorted(folder.rglob("*.jsonl")) if folder.is_dir() else ():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = TraceEvent.model_validate_json(line, strict=True)
            if event.sample_id != slot.sample_id:
                raise ValueError("m8_slot_trace_sample_mismatch")
            if event.type == "llm_request":
                request_ids.add(str(event.data.get("request_id")))
        trace_paths.append(path.resolve().relative_to(run_dir.resolve()).as_posix())
    after = _wallet_request_ids(wallet)
    delta = after - before
    if request_ids != delta:
        raise ValueError("m8_slot_trace_wallet_request_mismatch")
    return tuple(sorted(request_ids)), tuple(trace_paths)


def _halt_wallet(wallet: object, reason: str) -> None:
    """Halt future requests without rewriting any already recorded spend."""
    wallet.halt(reason)


def _validation_error_summary(exc: ValidationError) -> tuple[dict[str, Any], ...]:
    """Expose bounded field paths and error codes without model or source values."""
    rows = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False)[:8]:
        location = tuple(item for item in error["loc"] if type(item) in (str, int))[:8]
        rows.append({"loc": location, "type": str(error["type"])})
    return tuple(rows)


class M8GraphSlotExecutor:
    """Build real case compositions while the model gateway remains injectable for offline tests."""

    def __init__(self, root: Path, inputs: M8Inputs, report: M8PreflightReport,
                 run_dir: Path, external_calls_authorized: bool) -> None:
        """Load one shared read-only retrieval index before the run creates durable state."""
        self.root, self.inputs, self.report = root, inputs, report
        self.run_dir, self.external_calls_authorized = run_dir, external_calls_authorized
        self.runtime = _build_context(root)
        self.propagation_spec = AgentSpec(agent="propagation_analyst",
            system_prompt=(root / "prompts/propagation/v1.md").read_text(encoding="utf-8"))
        self.arbiter_spec = AgentSpec(agent="arbiter",
            system_prompt=(root / "prompts/arbiter/v2.md").read_text(encoding="utf-8"))

    def __call__(self, slot: M8ExecutionSlot, journal: M8RunJournal,
                 wallet: object) -> dict[str, Any]:
        """Run one CED case once or one isolated branch from the shared CHEF prefix."""
        case = next(item for item in self.inputs.cases if item.order == slot.order)
        slot_dir = journal.run_dir / "slots" / f"{slot.order:03d}-{slot.execution_group}"
        slot_dir.mkdir(parents=True, exist_ok=False)
        sample = Sample.model_validate_json(case.sample_path.read_bytes(), strict=True)
        if _sha_file(case.sample_path) != case.sample_sha256:
            raise ValueError("m8_slot_sample_changed")
        if slot.source == "ced":
            state, composition = self._invoke(sample, case, slot_dir / "CED-shared", wallet,
                "CED-shared", "CED-shared")
            try:
                return {"status": "completed", "state": state,
                    "shared_result": map_ced_shared_result(state, "m8:" + sample.sample_id),
                    "thread_id": composition.context.thread_id}
            finally:
                self._close_scopes(composition)
        return self._run_chef_branch(sample, case, slot, slot_dir, journal, wallet)

    def _run_chef_branch(self, sample: Sample, case: M8CaseInput, slot: M8ExecutionSlot,
                         slot_dir: Path, journal: M8RunJournal, wallet: object) -> dict[str, Any]:
        """Create or reload one verified shared prefix before isolating this group."""
        prior_group = "B4-fixed" if slot.execution_group == "B4-nf" else "B4-nf"
        prior = journal.finished_result(slot.order, prior_group)
        identity = self._cache_identity(sample, case)
        cache_path = journal.run_dir / "caches" / f"{slot.order:03d}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if prior is None:
            shared_state, shared = self._invoke(sample, case, slot_dir / "shared-first-round", wallet,
                "shared-first-round", "shared-first-round", shared=True)
            try:
                try:
                    digest = save_m8_shared_snapshot(cache_path, identity, shared_state, shared)
                except ValidationError as exc:
                    raise M8SlotValidationFailure("shared_snapshot_save",
                        _validation_error_summary(exc)) from exc
                except Exception as exc:
                    raise M8SlotRuntimeFailure("shared_snapshot_save") from exc
            finally:
                self._close_scopes(shared)
        else:
            digest = str(prior["cache_sha256"])
            cache_path = self._safe_cache_path(journal.run_dir, str(prior["cache_path"]))
        snapshot = load_shared_decision(cache_path, identity, wallet, digest)
        branch_root = slot_dir / slot.execution_group
        branch_root.mkdir(parents=True, exist_ok=False)
        branch = fork_m8_group(snapshot, slot.execution_group, wallet, branch_root)
        inputs = self._case_inputs(sample, case, branch_root, wallet, slot.execution_group)
        from experiments.archive.m8_runner import assemble_m8_group

        try:
            composition = CaseComposition(inputs)
        except BaseException:
            self._close_inputs(inputs)
            raise
        try:
            state = assemble_m8_group(composition, branch,
                external_calls_authorized=self.external_calls_authorized).invoke()
            return {"status": "completed", "state": state,
                "cache_path": cache_path.resolve().relative_to(journal.run_dir.resolve()).as_posix(),
                "cache_sha256": digest, "thread_id": composition.context.thread_id}
        finally:
            self._close_scopes(composition)

    def _invoke(self, sample: Sample, case: M8CaseInput, trace_root: Path, wallet: object,
                group: str, variant: str, *, shared: bool = False
                ) -> tuple[dict[str, Any], CaseComposition]:
        """Execute the existing shared-prefix or single-case graph with isolated worker scopes."""
        trace_root.mkdir(parents=True, exist_ok=False)
        inputs = self._case_inputs(sample, case, trace_root, wallet, group, variant)
        try:
            composition = CaseComposition(inputs)
            assembly = (assemble_m8_shared_prefix(composition,
                external_calls_authorized=self.external_calls_authorized) if shared else
                composition.assemble(external_calls_authorized=self.external_calls_authorized))
            state = assembly.invoke()
            return state, composition
        except BaseException:
            self._close_inputs(inputs)
            raise

    def _case_inputs(self, sample: Sample, case: M8CaseInput, trace_root: Path,
                     wallet: object, group: str, variant: str | None = None) -> CaseInputs:
        """Bind archive evidence, full test retrieval, licensed prompts and fresh worker scopes."""
        settings = get_settings()
        return CaseInputs(case_id=f"m8:{case.order}:{case.sample_id}", sample=sample,
            coverage=case.coverage, trace_root=trace_root, wallet=wallet,
            fact_worker_scope=EvidenceWorkerScope(), propagation_worker_scope=EvidenceWorkerScope(),
            index=self.runtime.index, store_path=settings.test_store_path,
            initial_manifest_path=case.initial_path or case.sample_path,
            propagation_spec=self.propagation_spec, arbiter_spec=self.arbiter_spec,
            as_of=None, group_id=group, variant=variant or group)

    def _cache_identity(self, sample: Sample, case: M8CaseInput) -> M8CacheIdentity:
        """Bind cache identity to the preflight, exact input bytes, prompts and runtime snapshot."""
        from experiments.archive.m8_inputs import _json_bytes

        files = self.report.conditions_sha256
        payload = {"conditions": files, "sample": case.sample_sha256,
            "initial": case.initial_sha256, "claim": sample.claim}
        fields = {name: _digest(_json_bytes({"purpose": name, **payload})) for name in (
            "claim_sha256", "model_params_sha256", "system_messages_sha256",
            "tool_schemas_sha256", "inputs_sha256", "corpus_sha256", "code_sha256",
            "config_sha256", "licenses_sha256", "first_round_contract_sha256")}
        return M8CacheIdentity(sample_id=sample.sample_id,
            case_id=f"m8:{case.order}:{sample.sample_id}", coverage=case.coverage,
            fingerprints=fields)

    @staticmethod
    def _close_scopes(composition: CaseComposition) -> None:
        """Release the two per-case worker scopes after graph/cache use is complete."""
        M8GraphSlotExecutor._close_inputs(composition.inputs)

    @staticmethod
    def _close_inputs(inputs: CaseInputs) -> None:
        """Attempt to close both evidence workers and surface cleanup failure."""
        first_error = None
        for scope in (inputs.fact_worker_scope, inputs.propagation_worker_scope):
            try:
                scope.close()
            except Exception as exc:
                first_error = first_error or type(exc).__name__
        if first_error:
            raise ValueError("m8_worker_scope_close_failed:" + first_error)

    @staticmethod
    def _safe_cache_path(run_dir: Path, relative: str) -> Path:
        """Resolve a sibling slot's cache reference without allowing a path escape."""
        target = (run_dir / relative).resolve(strict=True)
        if not target.is_relative_to(run_dir.resolve(strict=True)) or target.suffix != ".json":
            raise ValueError("m8_resume_cache_path_invalid")
        return target


def _require_network_and_credentials() -> None:
    """Require a configured key and a no-secret transport check before any run side effects."""
    from src.yuqing.llm import check_provider_connection

    config = get_settings()
    if config.deepseek_api_key is None:
        raise ValueError("m8_provider_credential_missing")
    reachable, stage, _status = check_provider_connection(config.deepseek_base_url,
        config.llm_timeout_s)
    if not reachable:
        raise ValueError("m8_provider_unreachable:" + stage)


def _sha_file(path: Path) -> str:
    """Hash one verified input file without retaining its contents."""
    return sha256(path.read_bytes()).hexdigest()
