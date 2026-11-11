"""Prepare and, after separate approval, run the fixed two-case M9 demonstration."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.yuqing.eval.experiment_session import build_experiment_manifest
from experiments.archive.m5_inputs import M5Inputs
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.settings import PROJECT_ROOT, get_cost_config, get_evaluation_config

DEMO_CASES = ("chef:dev:826", "ced:rumor:334_zvWw03RaC_1768909161")
DEMO_CAP_CNY = Decimal("1.00")
DEMO_CLAIM = "traces/m9-demo.claim"
EXPECTED_PRIOR_COSTS = {DEMO_CASES[0]: Decimal("0.114504"),
                        DEMO_CASES[1]: Decimal("0.040616")}


def prepare_m9_demo(root: Path = PROJECT_ROOT, *, require_committed: bool = False
                    ) -> tuple[M5Inputs, dict[str, str]]:
    """Verify exactly two archived dev cases and exclude frozen evaluation pools."""
    from src.yuqing import settings
    from experiments.archive import m5_checkpoint
    from experiments.archive.m5_inputs import (
        SELECTION, SUMMARY, _json_bytes, _load_archived_case, _questions, _sha, _source_shas,
    )
    from src.yuqing.eval.preflight import validate_propagation_startup, validate_runtime_config
    from src.yuqing.graph.fact_adapter import FactNodeAdapter
    from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus

    if root.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("m9_demo_root_invalid")
    inputs = _load_demo_archive(root)
    selected = tuple(case for case in inputs.cases if case.sample.sample_id in DEMO_CASES)
    if (tuple(case.sample.sample_id for case in selected) != DEMO_CASES
            or len(selected) != 2 or selected[0].coverage != "full"
            or selected[1].coverage != "none"
            or selected[0].sample.source != "chef" or selected[1].sample.source != "ced"):
        raise ValueError("m9_demo_fixed_samples_invalid")
    _reject_frozen_overlap(root, DEMO_CASES)
    validate_runtime_config()
    FactNodeAdapter(settings.get_graph_config().fact_node_version, root=root)
    validate_propagation_startup(root / "prompts/propagation/v1.md",
        offline=not require_committed)
    m5_checkpoint._validate_arbiter_permission(root)
    settings.get_m5_run_budget_config()
    chunks = load_eval_corpus(settings.get_settings().chef_dev_path, [], corpus_name="chef_dev_full")
    validate_eval_corpus(chunks, settings.get_settings().dev_store_path, corpus_name="chef_dev_full")
    conditions = m5_checkpoint.m5_condition_snapshot(root, inputs,
        require_committed=require_committed)
    conditions.update({"run_kind": "m9_two_case_demo_v1", "sample_ids": ",".join(DEMO_CASES),
        "shared_cap_cny": str(DEMO_CAP_CNY),
        **{"m9_code:" + name: _file_sha(root / name) for name in (
            "src/yuqing/eval/m9_demo.py", "src/yuqing/eval/m9_demo_replay.py",
            "src/yuqing/report/case_usage.py", "scripts/run_m9_demo.py",
            "scripts/replay_m9_demo.py")},
        **{"sample_sha256:" + case.sample.sample_id: case.sample_sha256 for case in selected},
        **{"initial_sha256:" + case.sample.sample_id: case.initial_sha256 or "none"
           for case in selected}})
    return M5Inputs(inputs.archive, inputs.archive_sha256, selected, inputs.source_sha256), conditions


def _load_demo_archive(root: Path) -> M5Inputs:
    """Validate the two archived samples without treating the old M5 code hash as current."""
    from experiments.archive.m5_inputs import (
        SELECTION, SUMMARY, _json_bytes, _load_archived_case, _questions, _sha, _source_shas,
    )

    summary_raw = (root / SUMMARY).read_bytes()
    summary = json.loads(summary_raw)
    unsigned = dict(summary)
    signature = unsigned.pop("archive_summary_sha256", None)
    archive = (root / summary["archive"]).resolve()
    package_root = (root / "traces/review-packages").resolve()
    archive_raw = (archive / "inputs.json").read_bytes()
    archive_manifest = json.loads(archive_raw)
    selection_raw = (root / SELECTION).read_bytes()
    selection = json.loads(selection_raw)
    if (_sha(_json_bytes(unsigned) + b"\n") != signature
            or not archive.is_relative_to(package_root) or _sha(archive_raw) != summary["archive_sha256"]
            or _sha(selection_raw) != summary["selection_sha256"]
            or archive_manifest.get("version") != "m5-inputs-v5"
            or archive_manifest.get("cases") != summary.get("cases")
            or archive_manifest.get("sources") != summary.get("source_sha256")
            or archive_manifest.get("selection_sha256") != summary.get("selection_sha256")):
        raise ValueError("m9_demo_archive_registration_invalid")
    case_rows, selection_rows = {}, {}
    for row in summary["cases"]:
        if row["sample_id"] in DEMO_CASES:
            case_rows[row["sample_id"]] = row
    for row in selection["samples"]:
        if row["sample_id"] in DEMO_CASES:
            selection_rows[row["sample_id"]] = row
    if set(case_rows) != set(DEMO_CASES) or set(selection_rows) != set(DEMO_CASES):
        raise ValueError("m9_demo_cases_missing_from_source_archive")
    questions = _questions(root)
    loaded = []
    for sample_id in DEMO_CASES:
        loaded.append(_load_archived_case(root, archive, selection_rows[sample_id],
            case_rows[sample_id], questions))
    return M5Inputs(archive, summary["archive_sha256"], tuple(loaded), _source_shas(root))


def _reject_frozen_overlap(root: Path, sample_ids: tuple[str, ...]) -> None:
    """Fail closed if either demo case appears in a protected evaluation pool."""
    protected = ("eval/label_pool_100.jsonl", "eval/test_60.jsonl",
        "eval/dev_val_120.jsonl", "eval/dev_val_120_v2.jsonl")
    selected = set(sample_ids)
    for relative in protected:
        path = root / relative
        if not path.is_file():
            raise ValueError("m9_demo_protected_manifest_missing:" + relative)
        found = {json.loads(line).get("sample_id") for line in path.read_text(
            encoding="utf-8").splitlines() if line.strip()}
        overlap = selected & found
        if overlap:
            raise ValueError("m9_demo_protected_sample_overlap:" + relative)


def _claim(root: Path, target: Path, conditions: dict[str, str], run_id: str) -> None:
    """Create one durable claim so the exact demo experiment cannot be repeated."""
    from experiments.archive.m5_inputs import _json_bytes, _sha

    payload = {"run_id": run_id,
        "run_dir": target.relative_to((root / "traces").resolve()).as_posix(),
        "conditions_sha256": _sha(_json_bytes(conditions)), "status": "claimed"}
    path = root / DEMO_CLAIM
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ValueError("m9_demo_attempt_already_claimed") from exc


def run_m9_demo(root: Path, run_dir: Path, *, execute: bool,
                authorize_real_calls: bool, authorization: str) -> dict[str, Any]:
    """Run two archived cases under one wallet only after explicit authorization."""
    from src.yuqing import llm, settings
    from experiments.archive import m5_checkpoint
    if not execute or not authorize_real_calls or authorization != "M9-two-case-demo":
        raise ValueError("m9_demo_requires_separate_authorization")
    inputs, conditions = prepare_m9_demo(root, require_committed=True)
    if (root / DEMO_CLAIM).exists():
        raise ValueError("m9_demo_attempt_already_claimed")
    settings.get_m5_run_budget_config()
    settings.get_cost_config()
    m5_checkpoint._require_credentials()
    reachable, stage, status = llm.check_provider_connection(
        settings.get_settings().deepseek_base_url, settings.get_settings().llm_timeout_s)
    if not reachable or status is None:
        raise ValueError("m9_demo_provider_preflight_" + stage)
    repeated_inputs, repeated = prepare_m9_demo(root, require_committed=True)
    if repeated != conditions or repeated_inputs.cases != inputs.cases:
        raise ValueError("m9_demo_conditions_changed_after_network_preflight")
    target = _new_run_dir(root, run_dir)
    run_id = uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=False, exist_ok=False)
    _claim(root, target, conditions, run_id)
    manifest = build_experiment_manifest(run_id, conditions, get_cost_config(),
        get_evaluation_config(), mode="full", max_run_cost_cny=DEMO_CAP_CNY)
    wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest,
        get_evaluation_config())
    try:
        return _execute(root, target, inputs, conditions, run_id, wallet)
    finally:
        wallet.close()


def _new_run_dir(root: Path, path: Path) -> Path:
    """Keep the new demo artifacts under a fresh directory inside project traces."""
    target, traces = path.resolve(), (root / "traces").resolve()
    if not target.is_relative_to(traces) or target == traces or target.exists():
        raise ValueError("m9_demo_run_directory_invalid")
    return target


def _file_sha(path: Path) -> str:
    """Hash one current demo path so the experiment condition binds its exact bytes."""
    if not path.is_file():
        raise ValueError("m9_demo_code_file_missing:" + path.name)
    return sha256(path.read_bytes()).hexdigest()


def _execute(root: Path, target: Path, inputs: M5Inputs, conditions: dict[str, str],
             run_id: str, wallet: PersistentExperimentBudget) -> dict[str, Any]:
    """Run the selected production graph, save briefs, and reconcile Trace with wallet."""
    from experiments.archive import m5_checkpoint
    from experiments.archive.m5_checkpoint_audit import (
        _audit_cost_rows, _audit_wallet, _read_slot_traces,
    )
    from src.yuqing.eval.experiment_budget_recovery import replay_history
    from experiments.archive.m5_inputs import _json_bytes, _sha

    (target / "answers.jsonl").write_bytes(b"")
    index = m5_checkpoint.build_bm25_index(m5_checkpoint._full_corpus(root))
    runtime_conditions = m5_checkpoint.m5_condition_snapshot(root,
        _load_demo_archive(root), require_committed=True)
    rows, stopped = m5_checkpoint._run_slots(root, inputs, wallet, index, target,
        run_id, runtime_conditions)
    for case, row in zip(inputs.cases, rows, strict=True):
        state = row.get("state")
        brief = state.get("brief") if isinstance(state, dict) else None
        if isinstance(brief, str):
            (target / f"brief-{case.order:02d}.md").write_text(brief, encoding="utf-8")
    requests, calls = _read_slot_traces(target, rows)
    recovered = replay_history(wallet.load_history(), wallet.manifest_snapshot())
    _audit_wallet(requests, calls, recovered.requests, inputs)
    _audit_cost_rows(rows, recovered.requests)
    if recovered.snapshot() != wallet.snapshot():
        raise ValueError("m9_demo_wallet_snapshot_mismatch")
    try:
        source_now = prepare_m9_demo(root, require_committed=True)[1]
        stable = source_now == conditions
    except (OSError, TypeError, ValueError):
        stable = False
    if not stable and not wallet.snapshot().halted:
        wallet.halt("m9_demo_conditions_changed_after_run")
    summary = {"run_id": run_id, "status": "stopped" if stopped else "completed",
        "planned": len(inputs.cases), "executed": sum(row["status"] != "unexecuted" for row in rows),
        "samples": [row["sample_id"] for row in rows], "rows": rows,
        "audit": "verified", "answers_sha256": _sha((target / "answers.jsonl").read_bytes()),
        "trace_requests": len(requests), "trace_receipts": len(calls),
        "known_cost_cny": str(recovered.snapshot().known_cost_subtotal_cny),
        "unknown_requests": recovered.snapshot().unknown_requests,
        "pending_requests": recovered.snapshot().pending_requests,
        "condition_stable": stable, "conditions_sha256": _sha(_json_bytes(conditions)),
        "shared_cap_cny": str(DEMO_CAP_CNY), "stop_reason": wallet.snapshot().halt_reason}
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
        sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary
