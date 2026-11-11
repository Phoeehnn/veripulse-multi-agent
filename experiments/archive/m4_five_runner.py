"""Run the single fixed five-sample M4 arbitration experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import uuid
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StrictBool, TypeAdapter

from src.yuqing import llm, settings
from src.yuqing.agents.arbiter_loop import run_arbiter_agent_v2
from src.yuqing.collab.arbiter_budget_v2 import ArbiterBudgetScope, RunActionBinding
from src.yuqing.collab.m4_readiness import (
    M4CapacityRow, M4PreparedCase, M4_SAMPLE_IDS, load_m4_prepared_cases,
    preflight_m4_five_samples,
)
from src.yuqing.collab.validate import CaseActionSnapshot, CaseReferenceSnapshot
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.harness.trace import RunContext, get_run_usage, use_run_context
from src.yuqing.schemas import AgentSpec
from src.yuqing.settings import PROJECT_ROOT, get_m4_run_budget_config

M4_PROMPT_SHA256 = "298cf72780715713e165c8231e24d7cae86308bb6354a6933e1db0156b0ab8e8"
M4_PROMPT_PATH = "prompts/arbiter/v2.md"
_CONDITION_FILES = (
    "config/weights.yaml", "prompts/arbiter/v2.md", "prompts/arbiter/contracts.json",
    "config/rules_compliance.yaml", "src/yuqing/agents/compliance.py",
    "src/yuqing/agents/propagation.py", "src/yuqing/features/propagation.py",
    "src/yuqing/ingest/chef.py", "src/yuqing/ingest/ced.py",
    "docs/cards/M4-five-sample-manifest.json", "docs/cards/M4-fact-history-import-v1.json",
    "src/yuqing/agents/arbiter_loop.py", "src/yuqing/collab/m4_readiness.py",
    "src/yuqing/collab/m4_history.py", "src/yuqing/collab/m4_fact_history_import.py",
    "src/yuqing/collab/arbiter_budget_v2.py", "src/yuqing/collab/arbiter_material_v2.py",
    "src/yuqing/collab/board.py", "src/yuqing/collab/validate.py",
    "experiments/archive/m4_five_runner.py", "src/yuqing/eval/preflight.py",
    "src/yuqing/eval/dual_budget.py", "src/yuqing/eval/experiment_session.py",
    "src/yuqing/eval/experiment_budget.py", "src/yuqing/eval/experiment_budget_store.py",
    "src/yuqing/eval/persistent_experiment_budget.py", "src/yuqing/harness/budget.py",
    "src/yuqing/harness/trace.py", "src/yuqing/harness/policy.py",
    "src/yuqing/harness/tools.py", "src/yuqing/harness/feedback.py",
    "src/yuqing/settings.py", "scripts/run_m4_five_samples.py",
    "experiments/archive/m4_continuation.py", "scripts/continue_m4_five_samples.py",
    "src/yuqing/schemas.py", "src/yuqing/llm.py",
)


class M4PromptContract(BaseModel):
    """The author-controlled development permission for the exact M4 prompt bytes."""

    model_config = ConfigDict(extra="forbid")
    contract_version: Literal["arbiter-decision-v2"]
    path: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_for_dev: StrictBool
    approved_for_test: StrictBool


@dataclass(frozen=True)
class M4RunPreflight:
    """Verified fixed inputs and condition fingerprints, without side effects."""

    cases: tuple[M4PreparedCase, ...]
    capacity: tuple[M4CapacityRow, ...]
    input_fingerprints: tuple[str, ...]
    conditions: Mapping[str, str]
    prompt_approved_for_dev: bool


@dataclass(frozen=True)
class M4RunRow:
    """One fixed sample's terminal result and auditable runtime identity."""

    sample_id: str
    experiment_id: str
    status: str
    run_id: str | None
    trace_path: str | None
    stop_reason: str
    action_execution: str
    decision_json: str
    action_allowed: str
    action_validation_issues: str
    known_cost_cny: str
    usage_known: str


def preflight_m4_run(root: Path = PROJECT_ROOT) -> M4RunPreflight:
    """Validate config, fixed sources, exact prompt bytes, and both retry capacities."""
    from src.yuqing.eval.preflight import validate_runtime_config

    validate_runtime_config()
    prompt_path = root / M4_PROMPT_PATH
    report = preflight_m4_five_samples(root, prompt_path)
    prices = settings.get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        raise ValueError("m4_prices_unconfirmed")
    if tuple(item.material.sample_id for item in report.cases) != M4_SAMPLE_IDS:
        raise ValueError("m4_fixed_sample_order_invalid")
    conditions = _collect_conditions(root)
    approved = _prompt_dev_permission(root, prompt_path)
    fingerprints = _prepared_case_fingerprints(report.cases)
    return M4RunPreflight(report.cases, report.capacity, fingerprints, conditions, approved)


def run_m4_five_samples(run_dir: Path, *, root: Path = PROJECT_ROOT,
                        results_dir: Path | None = None,
                        claim_path: Path | None = None) -> Path:
    """Run one fixed M4 attempt after permission, capacity, network, and path gates."""
    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("m4_run_root_must_be_project_root")
    ready = preflight_m4_run(root)
    if not ready.prompt_approved_for_dev:
        raise ValueError("m4_prompt_not_licensed_for_dev")
    _require_provider_credentials()
    target = _validate_run_dir(run_dir, root)
    output = results_dir or settings.get_settings().results_dir
    _validate_results_dir(output, root)
    claim = _validate_claim_path(claim_path or root / "traces" / "m4-five-sample-claim.json", root)
    if claim.exists():
        raise ValueError("m4_attempt_already_claimed")
    _require_provider_connection()
    _assert_conditions_unchanged(root, ready.conditions)
    target.mkdir(parents=True, exist_ok=False)
    experiment_id = uuid.uuid4().hex
    _create_one_time_claim(claim, experiment_id, target, ready.conditions)
    return _run_prepared(ready, target, experiment_id, root, output,
                         Decimal(str(get_m4_run_budget_config().experiment_max_cost_cny)))


def _run_prepared(ready: M4RunPreflight, run_dir: Path, experiment_id: str,
                  root: Path, results_dir: Path, max_run_cost_cny: Decimal) -> Path:
    """Create one shared wallet and execute each fixed sample at most once."""
    wallet_path = run_dir / "wallet.sqlite"
    session = ExperimentSession.create(wallet_path, experiment_id, ready.conditions,
        settings.get_cost_config(), settings.get_evaluation_config(), mode="full",
        max_run_cost_cny=max_run_cost_cny)
    rows: list[M4RunRow] = []
    stopped = False
    journal = run_dir / "answers.jsonl"
    try:
        for case in ready.cases:
            if stopped:
                row = _unexecuted_row(case.material.sample_id, experiment_id, "prior_slot_stopped")
            elif not _inputs_still_match(ready, root):
                session.halt("m4_conditions_changed")
                row, stopped = _unexecuted_row(case.material.sample_id, experiment_id,
                                                "m4_conditions_changed"), True
            else:
                row = _run_case(case, session, run_dir, experiment_id)
                stopped = not _run_can_continue(session, row)
                if not _inputs_still_match(ready, root):
                    session.halt("m4_conditions_changed_after_sample")
                    row = replace(row, stop_reason="m4_conditions_changed_after_sample")
                    stopped = True
            rows.append(row)
            _append_answer_journal(journal, row)
        return _write_results(rows, results_dir, experiment_id)
    finally:
        session.close()


def _run_case(case: M4PreparedCase, session: ExperimentSession,
              run_dir: Path, experiment_id: str) -> M4RunRow:
    """Execute one trusted sample under its own trace identity and the shared wallet."""
    run_id, sample_id = uuid.uuid4().hex, case.material.sample_id
    context = RunContext(run_id=run_id, sample_id=sample_id, agent="arbiter",
                         trace_root=run_dir / "traces")
    coverage_name = "chef_dev" if case.material.fact_reports else "ced"
    source_coverage = settings.get_evidence_sources_config()[coverage_name].coverage
    scope = ArbiterBudgetScope(experiment_id=experiment_id, case_id=case.material.case_id,
        sample_id=sample_id, group_id="m4", variant="arbiter-v2-m4", current_run_id=run_id,
        decision_round=case.material.decision_round,
        bindings=(RunActionBinding(run_id, "decision", case.material.decision_round, ()),),
        current_run_cost=Decimal(0), fact_coverage=source_coverage,
        propagation_has_data=bool(case.supplemental.stats and case.supplemental.stats.has_data),
        propagation_report_available=case.supplemental.propagation_report is not None,
        unreturned_comment_count=0)
    identity = {"run_id": run_id, "sample_id": sample_id, "group_id": "m4",
                "variant": "arbiter-v2-m4"}
    with use_run_context(context):
        result = run_arbiter_agent_v2(_m4_spec(), case.material, case.supplemental, context,
            node_config=settings.get_m4_arbiter_harness_config(), snapshot=case.snapshot,
            references=case.references, budget_scope=scope,
            coordinator_factory=session.coordinator_factory(), experiment_identity=identity,
            budget_profile="m4")
        usage = get_run_usage(run_id)
    return _result_row(context, result, usage, experiment_id)


def _result_row(context: RunContext, result: object, usage: object,
                experiment_id: str) -> M4RunRow:
    """Serialize only the validated arbiter result and trusted usage summary."""
    if result is None:
        return M4RunRow(context.sample_id, experiment_id, "no_valid_result", context.run_id,
            _relative_trace(context.trace_path), _trace_stop_reason(context.trace_path), "not_executed", "",
            "",
            "", str(usage.known_cost_subtotal_cny), str(usage.usage_known).lower())
    return M4RunRow(context.sample_id, experiment_id, "completed", context.run_id,
        _relative_trace(context.trace_path), result.stop_reason, result.action_execution,
        result.decision.model_dump_json(),
        "" if result.action_allowed is None else str(result.action_allowed).lower(),
        json.dumps(list(result.action_validation_issues), ensure_ascii=False),
        str(usage.known_cost_subtotal_cny), str(usage.usage_known).lower())


def _trace_stop_reason(path: Path) -> str:
    """Use only a short program event code and never copy model text into the result."""
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return "trace_unavailable"
    for event in reversed(events):
        if event.get("type") not in {"arbiter_v2_stopped", "arbiter_stopped"}:
            continue
        reason = event.get("data", {}).get("reason")
        if isinstance(reason, str) and reason.replace("_", "").isalnum():
            return reason[:80]
    for event in reversed(events):
        if event.get("type") != "run_outcome":
            continue
        reason = event.get("data", {}).get("reason")
        if isinstance(reason, str) and reason.replace("_", "").isalnum():
            return reason[:80]
    return "no_valid_decision"


def _run_can_continue(session: ExperimentSession, row: M4RunRow) -> bool:
    """Stop after unknown, pending, halted, or untrusted wallet state."""
    state = session.snapshot()
    return (row.status == "completed" and row.usage_known == "true"
            and state.unknown_requests == 0 and state.pending_requests == 0
            and not state.halted and state.committed_cost_cny <= state.max_cost_cny)


def _m4_spec() -> AgentSpec:
    """Load only the exact prompt bytes admitted by the M4 prompt registry."""
    path = PROJECT_ROOT / M4_PROMPT_PATH
    raw = path.read_bytes()
    if not raw or sha256(raw).hexdigest() != M4_PROMPT_SHA256:
        raise ValueError("m4_prompt_fingerprint_changed")
    return AgentSpec(agent="arbiter", system_prompt=raw.decode("utf-8"))


def _prompt_dev_permission(root: Path, prompt_path: Path, *,
                           normalize_checkout_newlines: bool = False) -> bool:
    """Require one exact, strict registry record; draft permission remains author controlled."""
    registry = root / "prompts/arbiter/contracts.json"
    records = TypeAdapter(list[M4PromptContract]).validate_json(registry.read_bytes())
    paths = [item.path for item in records]
    if len(paths) != len(set(paths)):
        raise ValueError("m4_prompt_registry_duplicate_path")
    matches = [item for item in records if (root / item.path).resolve() == prompt_path.resolve()]
    raw = prompt_path.read_bytes()
    admitted = raw.replace(b"\r\n", b"\n") if normalize_checkout_newlines else raw
    digest = sha256(admitted).hexdigest()
    if len(matches) != 1 or matches[0].prompt_sha256 != digest or digest != M4_PROMPT_SHA256:
        raise ValueError("m4_prompt_registry_fingerprint_mismatch")
    if not matches[0].approved_for_dev and matches[0].approved_for_test:
        raise ValueError("m4_test_permission_without_dev_permission")
    return matches[0].approved_for_dev


def _collect_conditions(root: Path) -> dict[str, str]:
    """Hash fixed M4 inputs, budget/config bytes, and executable logic."""
    if any(not (root / name).is_file() for name in _CONDITION_FILES):
        raise ValueError("m4_condition_file_missing")
    conditions = {f"file:{name}": _file_sha(root / name) for name in _CONDITION_FILES}
    model = settings.get_settings().deepseek_model
    if not model:
        raise ValueError("m4_model_not_configured")
    conditions["model"] = model
    runtime = settings.get_settings()
    conditions["base_url_sha256"] = sha256(runtime.deepseek_base_url.encode("utf-8")).hexdigest()
    conditions["timeout_s"] = str(runtime.llm_timeout_s)
    conditions["thinking"] = str(runtime.llm_thinking)
    conditions["sample_order_sha256"] = sha256("\n".join(M4_SAMPLE_IDS).encode()).hexdigest()
    return conditions


def _assert_conditions_unchanged(root: Path, expected: Mapping[str, str]) -> None:
    """Refuse a run if code, prompt, prices, configuration, or source manifests drift."""
    if _collect_conditions(root) != dict(expected):
        raise ValueError("m4_conditions_changed")


def _inputs_still_match(ready: M4RunPreflight, root: Path) -> bool:
    """Revalidate code, sources, all model inputs and snapshots at each sample boundary."""
    try:
        _assert_conditions_unchanged(root, ready.conditions)
        current = load_m4_prepared_cases(root)
    except (OSError, ValueError):
        return False
    return _prepared_case_fingerprints(current) == ready.input_fingerprints


def _prepared_case_fingerprints(cases: tuple[M4PreparedCase, ...]) -> tuple[str, ...]:
    """Fingerprint every delivered report and program validation snapshot in fixed order."""
    fingerprints = []
    for case in cases:
        payload = {"material": case.material.model_dump(mode="json"),
                   "supplemental": case.supplemental.model_dump(mode="json"),
                   "snapshot": _snapshot_payload(case.snapshot),
                   "references": _reference_payload(case.references)}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprints.append(sha256(encoded.encode("utf-8")).hexdigest())
    return tuple(fingerprints)


def _snapshot_payload(snapshot: CaseActionSnapshot) -> dict[str, object]:
    """Serialize the dataclass action snapshot with sorted sets and exact money strings."""
    return {"case_id": snapshot.case_id, "decision_round": snapshot.decision_round,
        "issues": [item.__dict__ for item in snapshot.issues],
        "executed_actions": [item.__dict__ for item in snapshot.executed_actions],
        "previous_progress": snapshot.previous_progress,
        "config": {"max_actions": snapshot.config.max_actions,
                   "targets": sorted(snapshot.config.targets),
                   "review_enabled": snapshot.config.review_enabled},
        "target": {"fact_coverage": snapshot.target.fact_coverage,
                   "propagation_has_data": snapshot.target.propagation_has_data,
                   "visible_comment_ids": sorted(snapshot.target.visible_comment_ids),
                   "returned_comment_ids": sorted(snapshot.target.returned_comment_ids),
                   "propagation_report_exists": snapshot.target.propagation_report_exists},
        "budget": {"settled": _decimal_text(snapshot.budget.settled),
                   "reserved": _decimal_text(snapshot.budget.reserved),
                   "action_cap": _decimal_text(snapshot.budget.action_cap),
                   "final_decision_reserve": _decimal_text(snapshot.budget.final_decision_reserve),
                   "case_limit": _decimal_text(snapshot.budget.case_limit)},
        "human_paused": snapshot.human_paused}


def _reference_payload(snapshot: CaseReferenceSnapshot) -> dict[str, object]:
    """Serialize mapping proxies and reference sets in deterministic order."""
    return {"case_id": snapshot.case_id, "decision_round": snapshot.decision_round,
        "leads": dict(sorted(snapshot.leads.items())),
        "returned_comment_ids": sorted(snapshot.returned_comment_ids),
        "trusted_refs": sorted(snapshot.trusted_refs),
        "forwarded_refs": sorted(snapshot.forwarded_refs)}


def _decimal_text(value: Decimal | None) -> str | None:
    """Keep decimal snapshot values stable while preserving unknown as null."""
    return None if value is None else str(value)


def _require_provider_credentials() -> None:
    """Require configured model and secret without exposing either value."""
    current = settings.get_settings()
    if current.deepseek_api_key is None or not current.deepseek_model:
        raise ValueError("m4_provider_credentials_missing")


def _require_provider_connection() -> None:
    """Check DNS, TLS, and HTTPS reachability before creating run state or wallet."""
    current = settings.get_settings()
    reachable, stage, status = llm.check_provider_connection(
        current.deepseek_base_url, current.llm_timeout_s)
    if not reachable:
        raise ValueError(f"m4_provider_preflight_{stage}")
    if status is None:
        raise ValueError("m4_provider_preflight_status_missing")


def _validate_run_dir(run_dir: Path, root: Path) -> Path:
    """Require one new output directory inside the project's D-drive traces tree."""
    target, traces = run_dir.resolve(), (root / "traces").resolve()
    if target == traces or not target.is_relative_to(traces):
        raise ValueError("m4_run_dir_outside_traces")
    if target.exists():
        raise ValueError("m4_run_dir_must_be_new")
    return target


def _create_one_time_claim(claim: Path, experiment_id: str, run_dir: Path,
                           conditions: Mapping[str, str]) -> None:
    """Atomically reserve the single M4 attempt without touching a wallet on duplicates."""
    target = _validate_claim_path(claim, PROJECT_ROOT)
    traces = (PROJECT_ROOT / "traces").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_id": experiment_id, "run_dir": run_dir.relative_to(traces).as_posix(),
        "conditions_sha256": _mapping_sha256(conditions), "status": "claimed"}
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m4_attempt_already_claimed") from exc


def _validate_claim_path(claim: Path, root: Path) -> Path:
    """Keep the one-time claim inside the project D-drive traces directory."""
    target, traces = claim.resolve(), (root / "traces").resolve()
    if target == traces or not target.is_relative_to(traces):
        raise ValueError("m4_claim_path_outside_traces")
    return target


def _write_results(rows: list[M4RunRow], results_dir: Path, experiment_id: str) -> Path:
    """Create one immutable five-row CSV with pending rows preserved after a stop."""
    if [item.sample_id for item in rows] != list(M4_SAMPLE_IDS):
        raise ValueError("m4_result_sample_order_invalid")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{date.today().isoformat()}_m4_{experiment_id}.csv"
    fields = list(M4RunRow.__dataclass_fields__)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item.__dict__ for item in rows)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _validate_results_dir(results_dir: Path, root: Path) -> None:
    """Keep experiment output on the project drive, never in a user cache."""
    target = results_dir.resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("m4_results_dir_outside_project")


def _unexecuted_row(sample_id: str, experiment_id: str, reason: str) -> M4RunRow:
    """Retain the fixed denominator for a slot stopped before model execution."""
    return M4RunRow(sample_id, experiment_id, "not_executed", None, None, reason, "not_executed",
                    "", "", "", "0", "true")


def _append_answer_journal(path: Path, row: M4RunRow) -> None:
    """Append and flush one row so completed samples survive an interrupted process."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _relative_trace(path: Path) -> str:
    """Return a project-relative trace path for the five-row result."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _file_sha(path: Path) -> str:
    """Hash file bytes without normalizing newlines or text encoding."""
    return sha256(path.read_bytes()).hexdigest()


def _mapping_sha256(values: Mapping[str, str]) -> str:
    """Hash a stable JSON encoding of the complete experiment conditions."""
    raw = json.dumps(dict(values), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()
