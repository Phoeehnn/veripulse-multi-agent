"""M8 首轮快照缓存与隔离分组的窄适配。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.yuqing.agents.propagation import PropagationResult
from src.yuqing.collab.arbiter_material_v2 import FactArchiveInputV2
from src.yuqing.collab.fact_material import FactArchiveReaderV2
from src.yuqing.graph.fact_adapter import FactRuntimeArtifact, _result_summary
from src.yuqing.graph.runtime import CaseRuntimeContext
from src.yuqing.harness.fact_v2_contract import build_fact_run_context_v2
from src.yuqing.harness.trace import RunContext
from src.yuqing.schemas import AgentResultV2, ArbiterResultV2, FactScopeV2, TaskInput
from src.yuqing.settings import PROJECT_ROOT
from experiments.archive.m8_cache_validation import (_validate_fact_round_rows,
    _validate_propagation_rows, verify_cached_propagation)


_FINGERPRINT_FIELDS = frozenset({
    "claim_sha256", "model_params_sha256", "system_messages_sha256",
    "tool_schemas_sha256", "inputs_sha256", "corpus_sha256", "code_sha256",
    "config_sha256", "licenses_sha256", "first_round_contract_sha256",
})
_HEX = frozenset("0123456789abcdef")


class M8CacheIdentity(BaseModel):
    """绑定一次共享首轮的题目身份及所有会影响结果的条件指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    sample_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    coverage: Literal["full", "none"]
    fingerprints: dict[str, str]

    def fingerprint(self) -> str:
        """返回规范化实验条件指纹；缺项或额外字段均拒绝。"""
        if set(self.fingerprints) != _FINGERPRINT_FIELDS:
            raise ValueError("m8_cache_fingerprint_fields_invalid")
        if any(not _is_sha256(value) for value in self.fingerprints.values()):
            raise ValueError("m8_cache_fingerprint_invalid")
        return _sha256(_canonical(self.model_dump(mode="json")))


class M8TraceFile(BaseModel):
    """缓存依赖的只读 Trace 文件及原始字节指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class M8RunBinding(BaseModel):
    """首轮共享钱包中每个程序运行与请求集合的可验证映射。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    trace_path: str = Field(min_length=1)
    role: Literal["initial", "decision"]
    decision_round: int = Field(ge=1)
    request_ids: tuple[str, ...]
    target: Literal["fact", "propagation"] | None = None
    shared_prefix: bool = False


class M8CacheRecord(BaseModel):
    """可持久化的首次合格仲裁后 State 与可信来源，运行时句柄不进入 JSON。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    identity: M8CacheIdentity
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: dict[str, Any]
    fact_result_json: str | None = None
    fact_scope: dict[str, Any] | None = None
    manifest_path: str | None = None
    propagation_result_json: str | None = None
    arbiter_result_json: str | None = None
    run_bindings: tuple[M8RunBinding, ...]
    trace_files: tuple[M8TraceFile, ...]
    wallet_request_ids: tuple[str, ...]
    wallet_experiment_id: str = Field(min_length=1)
    wallet_path: str = Field(min_length=1)
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class M8VerifiedSnapshot:
    """缓存重验后可交给隔离分组的首次仲裁后快照。"""

    identity: M8CacheIdentity
    state: dict[str, Any]
    fact_artifact: FactRuntimeArtifact | None
    propagation_result: PropagationResult | None
    arbiter_result: ArbiterResultV2 | None
    run_bindings: tuple[M8RunBinding, ...]
    wallet_request_ids: tuple[str, ...]
    cache_sha256: str
    wallet_experiment_id: str
    wallet_path: str




def save_shared_decision(path: Path, identity: M8CacheIdentity, state: dict[str, Any],
                         context: CaseRuntimeContext, wallet: object,
                         wallet_request_ids: tuple[str, ...],
                         run_bindings: tuple[M8RunBinding, ...]) -> str:
    """保存可信merge与首次仲裁决策；仅允许写入项目Trace根且拒绝覆盖。"""
    root = (PROJECT_ROOT / "traces").resolve(strict=True)
    target = path.resolve(strict=False)
    if not target.is_relative_to(root) or target.suffix != ".json" or target.exists():
        raise ValueError("m8_cache_path_invalid")
    record = _build_record(identity, state, context, wallet, wallet_request_ids,
                           run_bindings, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve(strict=False)
    if not target.is_relative_to(root) or target.exists():
        raise ValueError("m8_cache_path_invalid")
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.partial")
    payload = _record_bytes(record)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            temporary.rename(target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        return sha256(payload).hexdigest()
    except FileExistsError as exc:
        raise ValueError("m8_cache_already_exists") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_shared_decision(path: Path, identity: M8CacheIdentity, wallet: object,
                         expected_file_sha256: str) -> M8VerifiedSnapshot:
    """核验首次仲裁决策、Trace归档、条件和钱包后返回不可变快照。"""
    root = (PROJECT_ROOT / "traces").resolve(strict=True)
    target = path.resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("m8_cache_path_invalid")
    raw = target.read_bytes()
    if not _is_sha256(expected_file_sha256) or sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError("m8_cache_external_hash_mismatch")
    record = M8CacheRecord.model_validate_json(raw, strict=True)
    if record.identity != identity or record.identity_sha256 != identity.fingerprint():
        raise ValueError("m8_cache_condition_drift")
    if _wallet_identity(wallet) != (record.wallet_experiment_id, record.wallet_path):
        raise ValueError("m8_cache_wallet_identity_mismatch")
    if record.record_sha256 != _record_digest(record):
        raise ValueError("m8_cache_record_hash_mismatch")
    traces = _verify_trace_files(record.trace_files, root)
    _verify_run_trace_identities(record.run_bindings, identity.sample_id, traces, root)
    artifact = _load_fact(record, traces, wallet)
    propagation = _load_propagation(record, traces)
    _verify_wallet_requests(wallet, record.wallet_request_ids, identity)
    _verify_run_bindings(wallet, record.run_bindings, record.wallet_request_ids, identity,
                        record.state)
    decision = _load_arbiter_result(record, traces)
    _validate_shared_decision_state(record.state, identity, artifact, propagation, decision)
    return M8VerifiedSnapshot(identity=identity, state=record.state,
        fact_artifact=artifact, propagation_result=propagation, arbiter_result=decision,
        run_bindings=record.run_bindings,
        wallet_request_ids=record.wallet_request_ids, cache_sha256=record.record_sha256,
        wallet_experiment_id=record.wallet_experiment_id, wallet_path=record.wallet_path)








def _build_record(identity: M8CacheIdentity, state: dict[str, Any], context: CaseRuntimeContext,
                  wallet: object, request_ids: tuple[str, ...],
                  run_bindings: tuple[M8RunBinding, ...], root: Path) -> M8CacheRecord:
    """提取首轮输出、首次仲裁结果和每笔费用对应的程序运行。"""
    fact = context._fact_artifacts.get(1)
    prop = context._propagation_result
    decision = _decision_from_trace(state, run_bindings, root)
    _validate_shared_decision_state(state, identity, fact, prop, decision)
    traces: list[M8TraceFile] = []
    fact_json, fact_scope, manifest = None, None, None
    if fact is not None:
        archive = fact.archive_input
        FactArchiveReaderV2(archive.reader.trace_root, archive.reader.manifest_path,
                            archive.reader.wallet).read(archive.result, archive.task,
                            archive.contract, archive.scope)
        fact_json = _fact_result_json(archive.result)
        fact_scope = archive.scope.model_dump(mode="json")
        manifest = _relative_project_path(archive.reader.manifest_path)
        traces.append(_trace_file(Path(archive.result.trace_path), root))
    prop_json = None
    if prop is not None:
        verify_cached_propagation(prop, root)
        prop_json = prop.model_dump_json()
        traces.append(_trace_file(Path(prop.trace_path), root))
    decision_run = next((item for item in run_bindings if item.role == "decision"), None)
    if decision_run is None and not _is_early_failure_state(state):
        raise ValueError("m8_cache_decision_binding_missing")
    traces.extend(_trace_file(_binding_trace_path(item, root), root)
        for item in run_bindings)
    if len(set(request_ids)) != len(request_ids) or any(not item for item in request_ids):
        raise ValueError("m8_cache_wallet_request_ids_invalid")
    if {key for item in run_bindings for key in item.request_ids} != set(request_ids):
        raise ValueError("m8_cache_run_binding_request_set_mismatch")
    _verify_wallet_requests(wallet, request_ids, identity)
    _verify_run_bindings(wallet, run_bindings, request_ids, identity, state)
    _verify_run_trace_identities(run_bindings, identity.sample_id,
        {item.path: (root / item.path).resolve(strict=True) for item in traces}, root)
    experiment_id, ledger_path = _wallet_identity(wallet)
    return _seal_record(M8CacheRecord(identity=identity, identity_sha256=identity.fingerprint(),
        state=deepcopy(state), fact_result_json=fact_json, fact_scope=fact_scope,
        manifest_path=manifest, propagation_result_json=prop_json,
        arbiter_result_json=None if decision is None else decision.model_dump_json(),
        run_bindings=run_bindings,
        trace_files=tuple({item.path: item for item in traces}.values()),
        wallet_request_ids=request_ids, wallet_experiment_id=experiment_id,
        wallet_path=ledger_path, record_sha256="0" * 64))


def _fact_result_json(result: AgentResultV2) -> str:
    """Serialize the fact result without computed fields that strict reload rejects."""
    return result.model_dump_json(exclude={"validation": {"accepted"}})


def _load_fact(record: M8CacheRecord, traces: dict[str, Path], wallet: object) -> FactRuntimeArtifact | None:
    """以原有V2归档阅读器重建事实运行时工件。"""
    if record.fact_result_json is None:
        if record.fact_scope is not None or record.manifest_path is not None:
            raise ValueError("m8_cache_fact_fields_incomplete")
        return None
    result = AgentResultV2.model_validate_json(record.fact_result_json, strict=True)
    scope = FactScopeV2.model_validate(record.fact_scope, strict=True)
    manifest = (PROJECT_ROOT / record.manifest_path).resolve(strict=True)
    if not manifest.is_relative_to(PROJECT_ROOT.resolve()) or not manifest.is_file():
        raise ValueError("m8_cache_manifest_path_invalid")
    task = TaskInput(sample_id=scope.sample_id,
        claim=(record.state.get("sample") or {}).get("claim", ""))
    contract = build_fact_run_context_v2("initial", initial_manifest_path=manifest, scope=scope)
    run_context = _run_context(result.run_id, scope.sample_id, Path(result.trace_path), traces)
    reader = FactArchiveReaderV2(run_context.trace_root, manifest, wallet)
    snapshot = reader.read(result, task, contract, scope)
    summary = _result_summary(result, contract, run_context)
    return FactRuntimeArtifact(summary, FactArchiveInputV2(reader=reader, result=result,
        task=task, contract=contract, scope=scope), snapshot)


def _load_propagation(record: M8CacheRecord, traces: dict[str, Path]) -> PropagationResult | None:
    """重建传播结果，并校验Trace中的提交归档、材料及费用摘要。"""
    if record.propagation_result_json is None:
        return None
    result = PropagationResult.model_validate_json(record.propagation_result_json, strict=True)
    verify_cached_propagation(result, PROJECT_ROOT / "traces")
    if Path(result.trace_path).resolve() not in traces.values():
        raise ValueError("m8_cache_propagation_trace_missing")
    return result


def _decision_from_trace(state: dict[str, Any], bindings: tuple[M8RunBinding, ...],
                         root: Path) -> ArbiterResultV2 | None:
    """只从已绑定的仲裁Trace归档恢复决策，不信任State单独保存的标签。"""
    decision_runs = [item for item in bindings if item.role == "decision"]
    initial_runs = [item for item in bindings if item.role == "initial"]
    required_targets = {"propagation"} if identity_coverage_from_state(state) == "none" else {
        "fact", "propagation"}
    if (any(not item.shared_prefix for item in (*initial_runs, *decision_runs))
            or not {item.target for item in initial_runs}.issubset(required_targets)):
        raise ValueError("m8_cache_decision_binding_ambiguous")
    if not decision_runs and _is_early_failure_state(state):
        _verify_early_failure_runs(state, initial_runs, root)
        return None
    if (len(decision_runs) != 1 or decision_runs[0].decision_round != 1
            or {item.target for item in initial_runs} != required_targets):
        raise ValueError("m8_cache_decision_binding_ambiguous")
    path = _binding_trace_path(decision_runs[0], root)
    events = _read_run_events(path, decision_runs[0].run_id,
                              (state.get("sample") or {}).get("sample_id", ""), "arbiter")
    decisions = [event for event in events if event.type == "arbiter_v2_decision"]
    state_rows = state.get("decisions", [])
    if not decisions:
        if (state_rows or not state.get("stop_reason")
                or sum(event.type == "run_outcome" for event in events) != 1):
            raise ValueError("m8_cache_arbiter_decision_missing")
        return None
    if len(decisions) != 1 or len(state_rows) != 1:
        raise ValueError("m8_cache_arbiter_decision_ambiguous")
    if not isinstance(decisions[0].data.get("path"), str):
        raise ValueError("m8_cache_arbiter_archive_missing")
    document = _read_archive(decisions[0].data, root, decisions[0], path)
    raw_result = (document.get("payload") or {}).get("result")
    result = ArbiterResultV2.model_validate_json(
        json.dumps(raw_result, ensure_ascii=False), strict=True)
    submissions = [event for event in events if event.type == "arbiter_v2_submission"
        and event.data.get("accepted") is True]
    if len(submissions) == 1:
        submitted = _read_archive(submissions[0].data, root, submissions[0], path)
        payload = submitted.get("payload") or {}
        submitted_decision = payload.get("decision")
    else:
        submitted_decision = None
    if (len(submissions) != 1 or payload.get("accepted") is not True
            or submitted_decision != result.decision.model_dump(mode="json")
            or result.run_id != decision_runs[0].run_id
            or Path(result.trace_path).resolve() != path
            or result.decision_round != 1 or result.decision.model_dump(mode="json") != state_rows[0]):
        raise ValueError("m8_cache_arbiter_decision_mismatch")
    return result


def _verify_early_failure_runs(state: dict[str, Any], bindings: tuple[M8RunBinding, ...],
                               root: Path) -> None:
    """Require every initial expert run to have one identity-matched terminal trace event."""
    sample_id = (state.get("sample") or {}).get("sample_id", "")
    for binding in bindings:
        agent = "fact_investigator" if binding.target == "fact" else "propagation_analyst"
        events = _read_run_events(_binding_trace_path(binding, root), binding.run_id,
            sample_id, agent)
        if sum(event.type == "run_outcome" for event in events) != 1:
            raise ValueError("m8_cache_early_failure_trace_incomplete")


def _load_arbiter_result(record: M8CacheRecord, traces: dict[str, Path]) -> ArbiterResultV2 | None:
    """重放Trace核验并与缓存副本逐字节语义比对。"""
    runs = [item for item in record.run_bindings if item.role == "decision"]
    if not runs and _is_early_failure_state(record.state):
        return None
    if len(runs) != 1:
        raise ValueError("m8_cache_decision_binding_ambiguous")
    path = _binding_trace_path(runs[0], (PROJECT_ROOT / "traces").resolve(strict=True))
    if path not in traces.values():
        raise ValueError("m8_cache_decision_trace_missing")
    result = _decision_from_trace(record.state, record.run_bindings,
                                  (PROJECT_ROOT / "traces").resolve(strict=True))
    if (record.arbiter_result_json is None) != (result is None):
        raise ValueError("m8_cache_arbiter_result_missing")
    if result is not None and ArbiterResultV2.model_validate_json(
            record.arbiter_result_json, strict=True) != result:
        raise ValueError("m8_cache_arbiter_result_mismatch")
    return result


def _validate_shared_decision_state(state: dict[str, Any], identity: M8CacheIdentity,
                                    fact: Any | None, prop: PropagationResult | None,
                                    decision: ArbiterResultV2 | None) -> None:
    """只接受可信merge与一份仲裁后、尚未执行动作的共同前缀State。"""
    forbidden = {"actions", "task_results", "pending_task", "action_validation",
        "risk", "approval", "review", "alert", "brief", "_m8_capture_shared_decision",
        "_m8_resume_group"}
    if (forbidden.intersection(state) or (state.get("case_id"),
            (state.get("sample") or {}).get("sample_id")) !=
            (identity.case_id, identity.sample_id) or state.get("coverage") != identity.coverage
            or (not isinstance(state.get("compliance"), dict)
                or (not _is_early_failure_state(state)
                    and not isinstance(state.get("issues"), list)))):
        raise ValueError("m8_cache_state_not_shared_decision")
    if identity.coverage == "none" and fact is not None:
        raise ValueError("m8_cache_ced_fact_must_not_run")
    if fact is None and identity.coverage != "none" and state.get("fact") is not None:
        raise ValueError("m8_cache_unverified_fact_result")
    if fact is not None and state.get("fact") != fact.summary.report:
        raise ValueError("m8_cache_fact_state_mismatch")
    early_failure = _is_early_failure_state(state)
    if not early_failure or state.get("fact_rounds"):
        _validate_fact_round_rows(state, identity, fact)
    if prop is not None and state.get("propagation") != prop.report.model_dump(mode="json"):
        raise ValueError("m8_cache_propagation_state_mismatch")
    if prop is None and state.get("propagation") is not None:
        raise ValueError("m8_cache_unverified_propagation_result")
    if not early_failure or state.get("prop_rounds"):
        _validate_propagation_rows(state, prop)
    if (decision is None and state.get("decisions") or decision is not None
            and state.get("decisions") != [decision.decision.model_dump(mode="json")]):
        raise ValueError("m8_cache_decision_state_mismatch")
    _validate_propagation_leads(state, prop)


def _is_early_failure_state(state: dict[str, Any]) -> bool:
    """Accept only a recorded merge failure that stopped before arbitration."""
    reason, failures = state.get("stop_reason"), state.get("failures")
    if not isinstance(reason, str) or not isinstance(failures, list):
        return False
    return (reason == "merge_round_failed" and not state.get("decisions")
        and any(isinstance(item, dict) and item.get("node") == "merge_round"
                and isinstance(item.get("type"), str) for item in failures))


def _validate_propagation_leads(state: dict[str, Any], prop: PropagationResult | None) -> None:
    """确认传播Lead只来自Trace验证过的同一份报告。"""
    expected = [] if prop is None else [item.model_dump(mode="json")
        for item in prop.leads_by_id.values()]
    if state.get("leads") != expected:
        raise ValueError("m8_cache_propagation_leads_mismatch")


def _read_run_events(path: Path, run_id: str, sample_id: str, agent: str) -> list[Any]:
    """严格解析仲裁Trace并核对运行身份。"""
    from src.yuqing.harness.trace import TraceEvent

    events = [TraceEvent.model_validate_json(line, strict=True)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or any((item.run_id, item.sample_id, item.agent) !=
            (run_id, sample_id, agent) for item in events):
        raise ValueError("m8_cache_decision_trace_identity_mismatch")
    return events


def _verify_run_trace_identities(bindings: tuple[M8RunBinding, ...], sample_id: str,
                                 traces: dict[str, Path], root: Path) -> None:
    """逐个验证缓存中的Agent运行Trace身份与缓存角色一致。"""
    agents = {"fact": "fact_investigator", "propagation": "propagation_analyst"}
    for binding in bindings:
        path = _binding_trace_path(binding, root)
        if path not in traces.values():
            raise ValueError("m8_cache_bound_trace_missing")
        agent = "arbiter" if binding.role == "decision" else agents[binding.target or ""]
        _read_run_events(path, binding.run_id, sample_id, agent)


def identity_coverage_from_state(state: dict[str, Any]) -> str:
    """读取已由快照身份校验的覆盖字段，供运行绑定合同检查。"""
    return str(state.get("coverage", ""))


def _read_archive(reference: dict[str, Any], root: Path, event: Any,
                  trace_path: Path) -> dict[str, Any]:
    """核对仲裁提交归档在Trace根内、哈希一致且来自同一次请求。"""
    relative = Path(str(reference.get("path", "")))
    candidates = []
    parent = trace_path.parent.parent
    while parent.is_relative_to(root.resolve(strict=True)):
        candidate = (parent / relative).resolve(strict=False)
        if candidate.is_relative_to(root.resolve(strict=True)) and candidate.is_file():
            candidates.append(candidate)
        if parent == root.resolve(strict=True):
            break
        parent = parent.parent
    matches = [item for item in candidates if item.is_file()
        and sha256(item.read_bytes()).hexdigest() == reference.get("sha256")]
    if len(matches) != 1:
        raise ValueError("m8_cache_arbiter_archive_outside_root")
    raw = matches[0].read_bytes()
    document = json.loads(raw)
    if (document.get("kind") != "submission" or document.get("run_id") != event.run_id
            or document.get("sample_id") != event.sample_id or document.get("agent") != "arbiter"):
        raise ValueError("m8_cache_arbiter_archive_identity_mismatch")
    return document


def _binding_trace_path(binding: M8RunBinding, root: Path) -> Path:
    """按缓存绑定的Trace相对路径定位，并在使用前重验项目根边界。"""
    candidate = (root / binding.trace_path).resolve(strict=True)
    if (not candidate.is_relative_to(root.resolve(strict=True))
            or candidate.name != f"{binding.run_id}.jsonl" or not candidate.is_file()):
        raise ValueError("m8_cache_run_trace_missing")
    return candidate




















def _trace_file(path: Path, root: Path) -> M8TraceFile:
    """保存源Trace的安全相对路径和原字节散列。"""
    resolved = path.resolve(strict=True)
    base = root.resolve(strict=True)
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise ValueError("m8_cache_trace_outside_root")
    return M8TraceFile(path=resolved.relative_to(base).as_posix(),
        sha256=sha256(resolved.read_bytes()).hexdigest())


def _verify_trace_files(items: tuple[M8TraceFile, ...], root: Path) -> dict[str, Path]:
    """逐个重算缓存引用的源文件指纹。"""
    result = {}
    for item in items:
        path = (root / item.path).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest() != item.sha256:
            raise ValueError("m8_cache_source_trace_drift")
        if item.path in result:
            raise ValueError("m8_cache_trace_duplicate")
        result[item.path] = path
    return result


def _run_context(run_id: str, sample_id: str, trace: Path,
                 traces: dict[str, Path]) -> RunContext:
    """从缓存中的Trace引用还原只读归档读取器所需运行上下文。"""
    path = trace.resolve(strict=True)
    matches = [item for item in traces.values() if item == path]
    if len(matches) != 1:
        raise ValueError("m8_cache_fact_trace_missing")
    return RunContext(run_id=run_id, sample_id=sample_id, agent="fact_investigator",
        trace_root=path.parents[1], trace_day=date.fromisoformat(path.parent.name))


def _relative_project_path(path: Path | None) -> str:
    """拒绝把项目外清单路径写入快照。"""
    if path is None:
        raise ValueError("m8_cache_manifest_missing")
    resolved = path.resolve(strict=True)
    root = PROJECT_ROOT.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("m8_cache_manifest_outside_project")
    return resolved.relative_to(root).as_posix()


def _wallet_ready(wallet: object) -> bool:
    """仅接受已打开的持久总钱包，不接受每组另建账本的替身。"""
    manifest = getattr(wallet, "manifest_snapshot", None)
    storage = getattr(wallet, "storage_path", None)
    try:
        return (callable(manifest) and isinstance(storage, Path)
                and bool(manifest().get("experiment_id"))
                and storage.resolve(strict=True).is_file())
    except (OSError, ValueError):
        return False


def _wallet_identity(wallet: object) -> tuple[str, str]:
    """返回经过校验的实验编号与D盘项目Trace目录内账本路径。"""
    if not _wallet_ready(wallet):
        raise ValueError("m8_shared_wallet_invalid")
    experiment_id = str(wallet.manifest_snapshot()["experiment_id"])
    ledger = wallet.storage_path.resolve(strict=True)
    root = (PROJECT_ROOT / "traces").resolve(strict=True)
    if not ledger.is_relative_to(root):
        raise ValueError("m8_shared_wallet_outside_trace_root")
    return experiment_id, str(ledger)


def _verify_wallet_requests(wallet: object, request_ids: tuple[str, ...],
                            identity: M8CacheIdentity) -> None:
    """确认快照列出的每笔费用都在同一钱包、同一题的共享首轮组内。"""
    from src.yuqing.collab.arbiter_budget_v2 import _request_states

    recovered, _ = _request_states(wallet)
    expected = {key for key, item in recovered.requests.items()
        if item.identity.get("sample_id") == identity.sample_id
        and item.identity.get("group_id") == "shared-first-round"
        and item.identity.get("variant") == "shared-first-round"}
    if expected != set(request_ids):
        raise ValueError("m8_cache_wallet_request_set_mismatch")
    for request_id in request_ids:
        item = recovered.requests.get(request_id)
        if (item is None or not item.identity.get("run_id")
                or item.identity.get("variant") != "shared-first-round"
                or item.state not in {"settled", "cancelled"}):
            raise ValueError("m8_cache_wallet_request_mismatch")


def _verify_run_bindings(wallet: object, bindings: tuple[M8RunBinding, ...],
                         request_ids: tuple[str, ...], identity: M8CacheIdentity,
                         state: dict[str, Any]) -> None:
    """确认每个共享钱包请求只属于其缓存登记的运行与角色。"""
    from src.yuqing.collab.arbiter_budget_v2 import _request_states

    recovered, _ = _request_states(wallet)
    ids = [item.run_id for item in bindings]
    decisions = [item for item in bindings if item.role == "decision"]
    initial = [item for item in bindings if item.role == "initial"]
    targets = {item.target for item in initial}
    expected_targets = {"propagation"} if identity.coverage == "none" else {
        "fact", "propagation"}
    early_failure = _is_early_failure_state(state)
    if (len(ids) != len(set(ids)) or any(not item.shared_prefix for item in bindings)
            or (early_failure and (decisions or not targets.issubset(expected_targets)))
            or (not early_failure and (len(decisions) != 1
                or decisions[0].decision_round != 1 or targets != expected_targets))):
        raise ValueError("m8_cache_run_binding_contract_invalid")
    mapped = {request_id: binding for binding in bindings for request_id in binding.request_ids}
    if set(mapped) != set(request_ids):
        raise ValueError("m8_cache_run_binding_request_set_mismatch")
    for request_id, binding in mapped.items():
        request = recovered.requests.get(request_id)
        if (request is None or request.identity.get("run_id") != binding.run_id
                or request.identity.get("sample_id") != identity.sample_id
                or request.identity.get("group_id") != "shared-first-round"
                or request.identity.get("variant") != "shared-first-round"):
            raise ValueError("m8_cache_run_binding_wallet_mismatch")


def _trace_root_valid(path: Path, group_id: str) -> bool:
    """要求每组Trace目录位于项目traces下，并以组名作为独立末级目录。"""
    root = (PROJECT_ROOT / "traces").resolve(strict=True)
    if ".." in path.parts:
        return False
    candidate = path.resolve()
    return candidate.is_relative_to(root) and candidate.parent != root and candidate.name == group_id


def _seal_record(record: M8CacheRecord) -> M8CacheRecord:
    """以排除自引用字段的规范JSON生成整份缓存内容指纹。"""
    return record.model_copy(update={"record_sha256": _record_digest(record)})


def _record_digest(record: M8CacheRecord) -> str:
    """哈希除record_sha256外的所有缓存字段。"""
    data = record.model_dump(mode="json")
    data["record_sha256"] = ""
    return _sha256(_canonical(data))


def _record_bytes(record: M8CacheRecord) -> bytes:
    """生成稳定UTF-8 JSON文件字节。"""
    return _canonical(record.model_dump(mode="json"))


def _canonical(value: Any) -> bytes:
    """生成排序键、无NaN的规范JSON字节。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    """计算SHA-256十六进制指纹。"""
    return sha256(value).hexdigest()


def _is_sha256(value: str) -> bool:
    """检查值是否为小写64位SHA-256。"""
    return len(value) == 64 and set(value) <= _HEX
