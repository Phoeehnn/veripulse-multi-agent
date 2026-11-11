"""固定658联合案的一次运行入口；只在作者另行许可后才能调用。"""
from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from src.yuqing import settings
from experiments.archive import joint_demo as joint
from experiments.archive import m5_checkpoint as shared
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.experiment_session import build_experiment_manifest
from experiments.archive.m5_checkpoint_audit import (
    _audit_cost_rows, _audit_wallet, _read_answers, _read_slot_traces,
)
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.eval.preflight import validate_propagation_startup


def _write_new(path: Path, value: object) -> None:
    """只持久写入新文件，失败现场和旧文件不被覆盖。"""
    _write_bytes(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  allow_nan=False, indent=2).encode() + b"\n")


def _write_bytes(path: Path, payload: bytes) -> None:
    """独占保存原字节并落盘，不转换材料和初始清单换行。"""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _record_start_failure(target: Path, run_id: str, error: Exception) -> None:
    """认领后目录未能创建时，也在其父目录留下不可覆盖的失败回执。"""
    record = {"run_id": run_id, "stage": "runner",
        "error_type": type(error).__name__, "claim_retained": True}
    path = target / "failure.json" if target.is_dir() else target.parent / f"{run_id}.failure.json"
    _write_new(path, record)


def _input_files(prepared: joint.PreparedJointDemo) -> dict[str, tuple[Path, str]]:
    """固定五份实际输入文件与本次已核验原字节SHA。"""
    root, conditions = prepared.root, prepared.conditions
    inventory = json.loads((root / joint.INVENTORY_PATH).read_bytes())
    pairs = {"material.json": joint.MATERIAL_PATH, "sources.json": joint.INVENTORY_PATH,
        "original.json": inventory["original_path"], "propagation.json": inventory["propagation_path"]}
    return {**{name: (root / relative, conditions[relative]) for name, relative in pairs.items()},
            "initial.json": (prepared.store / "initial.json", conditions["initial.json"])}


def _archive_inputs(prepared: joint.PreparedJointDemo, target: Path) -> None:
    """钱包创建前冻结实际输入副本，拷贝期间发生漂移则直接拒绝。"""
    folder = target / "inputs"
    folder.mkdir()
    signature = {}
    for name, (path, expected) in _input_files(prepared).items():
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError("joint_input_changed_during_archive")
        _write_bytes(folder / name, raw)
        signature[name] = expected
    _write_new(folder / "manifest.json", signature)


def _verify_committed(root: Path, conditions: dict[str, str]) -> str:
    """只核对运行闭包的受跟踪文件，允许作者保留无关展示和文档改动。"""
    paths = [key for key in conditions if key == ".gitattributes" or key.startswith(
        ("src/", "scripts/", "prompts/", "config/", "docs/cards/", "data/demo_cases/"))]
    command = ["git"]
    try:
        tracked = subprocess.run([*command, "ls-files", "--error-unmatch", *paths],
            cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
        if set(tracked) != set(paths):
            raise ValueError("joint_runtime_not_committed")
        subprocess.run([*command, "diff", "--quiet", "HEAD", "--", *paths],
                       cwd=root, check=True, capture_output=True)
        return subprocess.run([*command, "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("joint_runtime_not_committed") from exc


def _prepare(root: Path, *, claimed: bool = False
             ) -> tuple[joint.PreparedJointDemo, str]:
    """核对固定身份、保护池、开发许可、价格及已提交闭包，不联网。"""
    if root.resolve() != settings.PROJECT_ROOT.resolve():
        raise ValueError("joint_root_invalid")
    prepared = joint._validated_joint_demo(root) if claimed else joint.preflight_joint_demo(root)
    inventory = json.loads((root / joint.INVENTORY_PATH).read_bytes())
    for item in inventory["protected_pool_checks"]:
        path = root / item["path"]
        if joint._sha(path) != item["sha256"]:
            raise ValueError("joint_protected_pool_drift")
        if any(json.loads(line).get("sample_id") == joint.SAMPLE_ID
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
            raise ValueError("joint_protected_sample_overlap")
    validate_propagation_startup(root / "prompts/propagation/v1.md", offline=False)
    prices = settings.get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        raise ValueError("joint_prices_unknown")
    return prepared, _verify_committed(root, prepared.conditions)


def _new_target(root: Path, run_dir: Path) -> Path:
    """只在本联合案的新子目录写产物，不借用已有实验或目录链接。"""
    base = root.resolve() / "traces/joint-ced-658"
    target = run_dir.absolute()
    if (base.resolve() != base or target.resolve() != target or target.exists()
            or target == base or not target.is_relative_to(base)):
        raise ValueError("joint_run_directory_invalid")
    return target


def run_joint_demo(root: Path, run_dir: Path, *, execute: bool,
                   authorize_real_calls: bool, authorization: str) -> dict[str, Any]:
    """获准后只运行固定一案；认领、正式钱包和网络均在许可检查之后。"""
    if not execute or not authorize_real_calls or authorization != "M9-joint-658":
        raise ValueError("joint_requires_separate_authorization")
    target = _new_target(root, run_dir)
    prepared, commit = _prepare(root)
    shared._require_credentials()
    shared._provider_connection()
    repeated, repeated_commit = _prepare(root)
    if repeated.conditions != prepared.conditions or repeated_commit != commit:
        raise ValueError("joint_conditions_changed_after_network")
    run_id = uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    shared._claim_once(root / joint.CLAIM_PATH, run_id, target, prepared.conditions)
    wallet = None
    try:
        target.mkdir(exist_ok=False)
        _write_new(target / "conditions.json", {"conditions": prepared.conditions, "commit": commit})
        _archive_inputs(prepared, target)
        manifest = build_experiment_manifest(run_id, prepared.conditions, settings.get_cost_config(),
            settings.get_evaluation_config(), mode="full",
            max_run_cost_cny=Decimal(str(settings.get_m5_case_budget_config().max_cost_cny)))
        wallet = PersistentExperimentBudget.create(target / "wallet.sqlite", manifest,
                                                   settings.get_evaluation_config())
        return _execute(root, target, prepared, run_id, commit, wallet)
    except Exception as exc:
        if wallet is not None:
            shared._halt_wallet_safely(wallet, "joint_runner_failure")
        _record_start_failure(target, run_id, exc)
        raise
    finally:
        if wallet is not None:
            wallet.close()


def _case_row(prepared: joint.PreparedJointDemo, target: Path, run_id: str,
              wallet: PersistentExperimentBudget) -> dict[str, Any]:
    """运行生产图并保存受控失败原因，独立关闭两个证据工作池。"""
    traces = target / "traces"
    traces.mkdir(exist_ok=False)
    composition, state, thread_id = None, {}, None
    status, reason = "failed", "joint_setup_failed"
    started = time.perf_counter()
    try:
        composition = prepared.composition(wallet, traces)
        thread_id = composition.context.thread_id
        state = composition.assemble(external_calls_authorized=True,
            initial_state=prepared.state(composition.inputs.case_id)).invoke()
        if not state.get("brief") or not isinstance(state.get("risk"), dict):
            raise ValueError("joint_terminal_output_missing")
        status, reason = "completed", state.get("stop_reason") or ""
    except Exception as exc:
        reason = "runtime_" + type(exc).__name__
        shared._halt_wallet_safely(wallet, "joint_case_runtime_failure")
    finally:
        if composition is not None:
            close_error = shared._close_worker_scopes((composition.inputs.fact_worker_scope,
                composition.inputs.propagation_worker_scope), wallet)
            if close_error:
                status, reason = "failed", close_error
    snapshot = wallet.snapshot()
    if shared._has_experiment_budget_refusal(traces):
        shared._halt_wallet_safely(wallet, "experiment_cost_limit")
        reason = "experiment_cost_limit"
    return {"experiment_id": run_id, "order": 1, "sample_id": prepared.sample.sample_id,
        "status": status, "stop_reason": reason, "thread_id": thread_id, "state": state,
        "elapsed_ms": int((time.perf_counter() - started) * 1000), "trace_root": "traces",
        "known_cost_delta_cny": str(snapshot.known_cost_subtotal_cny),
        "usage_unknown": snapshot.unknown_requests > 0, "pending_requests": snapshot.pending_requests}


def _audit(root: Path, target: Path, prepared: joint.PreparedJointDemo, run_id: str,
           row: dict[str, Any], wallet: PersistentExperimentBudget) -> dict[str, Any]:
    """复用既有逐请求审计，额外约束本案唯一身份、claim和钱包总额。"""
    claim = json.loads((root / joint.CLAIM_PATH).read_bytes())
    expected = {"experiment_id": run_id, "run_dir": target.relative_to(root / "traces").as_posix(),
        "conditions_sha256": joint._digest(prepared.conditions), "status": "claimed"}
    if claim != expected or _read_answers(target / "answers.jsonl", run_id) != [row]:
        raise ValueError("joint_claim_or_answers_mismatch")
    inputs = {name: digest for name, (_, digest) in _input_files(prepared).items()}
    if (json.loads((target / "inputs/manifest.json").read_bytes()) != inputs
            or any(joint._sha(target / "inputs" / name) != digest for name, digest in inputs.items())):
        raise ValueError("joint_input_archive_mismatch")
    if row["sample_id"] != joint.SAMPLE_ID or row["experiment_id"] != run_id:
        raise ValueError("joint_answer_identity_mismatch")
    requests, calls = _read_slot_traces(target, [row])
    manifest = wallet.manifest_snapshot()
    expected_manifest = build_experiment_manifest(run_id, prepared.conditions, settings.get_cost_config(),
        settings.get_evaluation_config(), mode="full",
        max_run_cost_cny=Decimal(str(settings.get_m5_case_budget_config().max_cost_cny)))
    if manifest != expected_manifest:
        raise ValueError("joint_wallet_manifest_mismatch")
    recovered = replay_history(wallet.load_history(), manifest)
    _audit_wallet(requests, calls, recovered.requests,
                  SimpleNamespace(cases=(SimpleNamespace(sample=prepared.sample),)))
    _audit_cost_rows([row], recovered.requests)
    if any(r.identity.get("group_id") != "joint_demo" or r.identity.get("variant") != "fact-v2_prop-v1.1"
           for r in recovered.requests.values()):
        raise ValueError("joint_wallet_group_mismatch")
    if recovered.snapshot() != wallet.snapshot():
        raise ValueError("joint_wallet_replay_mismatch")
    return {"status": "verified", "trace_requests": len(requests), "trace_receipts": len(calls),
            "answers_sha256": joint._sha(target / "answers.jsonl")}


def _execute(root: Path, target: Path, prepared: joint.PreparedJointDemo, run_id: str,
             commit: str, wallet: PersistentExperimentBudget) -> dict[str, Any]:
    """无论模型结果是否合格都保存一份答卷；审计失败明确停场。"""
    row = _case_row(prepared, target, run_id, wallet)
    try:
        current, current_commit = _prepare(root, claimed=True)
        if current.conditions != prepared.conditions or current_commit != commit:
            raise ValueError("joint_conditions_changed_after_case")
    except Exception as exc:
        shared._halt_wallet_safely(wallet, "joint_conditions_changed_after_case")
        row.update(status="failed", stop_reason="postcheck_" + type(exc).__name__)
    shared._append_jsonl(target / "answers.jsonl", row)
    state = row["state"]
    if isinstance(state.get("brief"), str):
        with (target / "brief.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(state["brief"])
    try:
        audit = _audit(root, target, prepared, run_id, row, wallet)
    except Exception as exc:
        shared._halt_wallet_safely(wallet, "joint_audit_failed")
        audit = {"status": "failed", "error_type": type(exc).__name__}
    snapshot = wallet.snapshot()
    summary = {"run_id": run_id, "runtime_commit": commit, "sample_id": joint.SAMPLE_ID,
        "status": "completed" if row["status"] == "completed" and not snapshot.halted
            and audit["status"] == "verified" and not snapshot.pending_requests else "stopped",
        "run_outcome": row["status"], "stop_reason": snapshot.halt_reason or row["stop_reason"],
        "planned": 1, "executed": 1, "risk": state.get("risk"), "thread_id": row["thread_id"],
        "is_demo": True, "analysis_mode": "retrospective", "human_gate_mode": "eval",
        "shared_cap_cny": str(snapshot.max_cost_cny), "known_cost_cny": str(snapshot.known_cost_subtotal_cny),
        "unknown_requests": snapshot.unknown_requests, "unknown_reserved_cny": str(snapshot.unknown_reserved_cny),
        "pending_requests": snapshot.pending_requests, "pending_reserved_cny": str(snapshot.pending_reserved_cny),
        "audit": audit, "conditions_sha256": joint._digest(prepared.conditions)}
    _write_new(target / "summary.json", summary)
    return summary
