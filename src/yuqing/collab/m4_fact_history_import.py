"""只为M4固定CHEF来源重验历史事实答卷并建立独立导入身份。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Mapping
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from src.yuqing.collab.arbiter_material_v2 import FactArchiveInputV2
from src.yuqing.collab.fact_archive import (ArchiveReadSession, _canonical, _digest, _load_events,
    _load_request_chain, _one, _verify_usage_and_wallet)
from src.yuqing.collab.fact_import_contract import M4FactImportBindingV1, M4FactImportProvenanceV1
from src.yuqing.collab.fact_material import (FactArchiveReaderV2, _reconstruct_delivered,
    _load_delivery_material, _read_final_submission, _snapshot)
from experiments.archive.e2_artifacts import validate_checkpoint_artifacts
from src.yuqing.eval.experiment_budget import ExperimentBudgetStatus
from src.yuqing.eval.experiment_budget_recovery import RecoveredState, replay_history
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore, LedgerHistory, _manifest_data
from src.yuqing.harness.fact_v2_contract import FactRunContextV2, build_fact_run_context_v2, validate_fact_v2_core
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.ingest.chef import load_chef
from src.yuqing.schemas import (AgentResultV2, FactScopeV2, FactV2ProgramValidation, LedgerEntry,
    TaskInput, ValidatedFactSnapshotV2)
from src.yuqing.settings import PROJECT_ROOT
from src.yuqing.collab.m4_history import _under_root

MANIFEST_PATH = "docs/cards/M4-fact-history-import-v1.json"
_MANIFEST_SHA256 = "bc79f8171333280a01bf91d36a94cf3705d045c4e92eea2596688a2131706d8e"
_EXPECTED_IDS = ("chef:dev:0", "chef:dev:1", "chef:dev:10")
_SOURCE_CONTRACT_FIELDS = ("mode", "contract_sha256", "system_sha256", "schema_sha256",
                           "tool_schema_sha256", "limits_sha256")
_SOURCE_EXPERIMENT = "6c5c209565e744178a676fbe4cd7e940"
_SOURCE_CONDITION = "215d6ff075b9bf9874a465870ec0699e05262dd8f69014fa7fdd0911daa1b439"


class M4FactImportItem(BaseModel):
    """固定五样本中一条CHEF历史事实运行与新增M4身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    case_id: str
    fact_round_no: int = Field(gt=0, strict=True)
    group_id: str
    run_id: str
    row_sha256: str
    trace_path: str
    trace_sha256: str
    manifest_path: str
    manifest_sha256: str
    claim_sha256: str


class M4FactImportManifest(BaseModel):
    """固定来源和来源运行合同，不接受调用者自选历史样本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    source_answers: str
    source_answers_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_state: str
    source_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_wallet: str
    source_wallet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_id: str
    condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_contract: dict[str, str]
    items: tuple[M4FactImportItem, ...]


@dataclass(frozen=True)
class M4ImportedFactSource:
    """供程序建板与仲裁重读的事实快照及其来源凭据。"""

    archive_input: FactArchiveInputV2
    snapshot: ValidatedFactSnapshotV2
    provenance: M4FactImportProvenanceV1


@dataclass(frozen=True)
class _ReadOnlyWallet:
    """只读重放已结算的钱包账本，不提供写入或恢复接口。"""

    history: LedgerHistory
    state: RecoveredState

    def snapshot(self) -> ExperimentBudgetStatus:
        """返回由验证过的账本事件重放出的费用状态。"""
        return self.state.snapshot()

    def load_history(self) -> LedgerHistory:
        """返回只读连接取得的哈希核验账本事件。"""
        return self.history


def load_m4_fact_history_imports(root: Path = PROJECT_ROOT) -> tuple[M4ImportedFactSource, ...]:
    """按固定清单重验三条CHEF历史事实来源并分配M4首轮身份。"""
    manifest, manifest_sha = _load_import_manifest(root)
    answers = _read_answers(root, manifest)
    samples = {item.sample_id: item for item in load_chef(_under_root(root, "data/CHEF/dev.json"))}
    loaded: list[M4ImportedFactSource] = []
    for item in manifest.items:
        task = _task_for(item, samples)
        archive_input = _archive_input(root, manifest, manifest_sha, item, task, answers[item.sample_id])
        snapshot, provenance = read_m4_historical_fact(archive_input, root=root)
        loaded.append(M4ImportedFactSource(archive_input, snapshot, provenance))
    return tuple(loaded)


def read_m4_historical_fact(item: FactArchiveInputV2, *,
                            root: Path = PROJECT_ROOT
                            ) -> tuple[ValidatedFactSnapshotV2, M4FactImportProvenanceV1]:
    """重读固定答卷、Trace、初始证据、归档与只读钱包后生成M4快照。"""
    binding = item.historical_import
    if binding is None:
        raise ValueError("m4_fact_import_binding_missing")
    manifest, manifest_sha = _load_import_manifest(root)
    entry = _find_manifest_item(manifest, binding.sample_id)
    _check_binding(binding, manifest_sha, entry, item, root)
    expected_contract = build_fact_run_context_v2("initial", initial_manifest_path=item.reader.manifest_path,
                                                   scope=item.scope)
    if (item.contract.mode != expected_contract.mode
            or item.contract.contract_sha256 != expected_contract.contract_sha256
            or item.contract.schema_sha256 != expected_contract.schema_sha256
            or item.contract.initial_manifest_path != expected_contract.initial_manifest_path
            or item.contract.scope != expected_contract.scope):
        raise ValueError("m4_fact_import_runtime_contract_mismatch")
    provenance = _provenance(binding, manifest_sha)
    events, trace_sha, _ = _load_bound_trace(root, item, binding)
    source_contract = build_fact_run_context_v2("initial", initial_manifest_path=item.reader.manifest_path)
    _check_source_contract(events, manifest, source_contract)
    import_material, initial_sha = _load_delivery_material(item.reader.manifest_path, item.task,
                                                           item.contract, events)
    if initial_sha != binding.initial_manifest_sha256:
        raise ValueError("m4_fact_import_initial_manifest_mismatch")
    snapshot = _rebuild_snapshot(item, events, trace_sha, initial_sha, import_material, provenance, root)
    return snapshot, provenance


def _load_import_manifest(root: Path) -> tuple[M4FactImportManifest, str]:
    """核对导入清单固定字节、顺序和仅三条CHEF事实来源。"""
    path = _under_root(root, MANIFEST_PATH)
    raw = path.read_bytes()
    digest = _digest(raw)
    if digest != _MANIFEST_SHA256:
        raise ValueError("m4_fact_import_manifest_fingerprint_mismatch")
    manifest = M4FactImportManifest.model_validate_json(raw, strict=True)
    ids = tuple(item.sample_id for item in manifest.items)
    if (manifest.version != "m4-fact-history-import-v1" or ids != _EXPECTED_IDS
            or manifest.experiment_id != _SOURCE_EXPERIMENT or manifest.condition_sha256 != _SOURCE_CONDITION):
        raise ValueError("m4_fact_import_manifest_scope_invalid")
    if set(manifest.source_contract) != set(_SOURCE_CONTRACT_FIELDS):
        raise ValueError("m4_fact_import_source_contract_invalid")
    for item in manifest.items:
        if item.group_id != "iterative" or item.fact_round_no != 1 or item.case_id != f"m4:{item.sample_id}":
            raise ValueError("m4_fact_import_entry_identity_invalid")
    return manifest, digest


def _read_answers(root: Path, manifest: M4FactImportManifest) -> dict[str, object]:
    """只从指纹固定的R2答卷中提取三个iterative行并逐行核验。"""
    path = _under_root(root, manifest.source_answers)
    if _digest(path.read_bytes()) != manifest.source_answers_sha256:
        raise ValueError("m4_fact_import_answers_fingerprint_mismatch")
    matches: dict[str, list[str]] = {sample_id: [] for sample_id in _EXPECTED_IDS}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        run = value.get("run", {})
        if run.get("sample_id") in matches and run.get("group_id") == "iterative":
            matches[run["sample_id"]].append(line)
    if any(len(rows) != 1 for rows in matches.values()):
        raise ValueError("m4_fact_import_answer_row_count_invalid")
    rows = {sample_id: validate_checkpoint_artifacts([lines[0]], experiment_id=manifest.experiment_id,
        condition_sha256=manifest.condition_sha256)[0] for sample_id, lines in matches.items()}
    return rows


def _task_for(item: M4FactImportItem, samples: Mapping[str, object]) -> TaskInput:
    """从固定CHEF开发数据重载声明，绝不从历史答卷自由拼接声明。"""
    sample = samples.get(item.sample_id)
    if sample is None or _digest(sample.claim.encode("utf-8")) != item.claim_sha256:
        raise ValueError("m4_fact_import_claim_mismatch")
    return TaskInput(sample_id=item.sample_id, claim=sample.claim)


def _archive_input(root: Path, manifest: M4FactImportManifest, manifest_sha: str,
                   entry: M4FactImportItem, task: TaskInput, row: object) -> FactArchiveInputV2:
    """构造带显式M4导入绑定的首轮事实读取请求。"""
    trace_root = _under_root(root, entry.trace_path).parents[1]
    manifest_path = _under_root(root, entry.manifest_path)
    scope = FactScopeV2(case_id=entry.case_id, sample_id=entry.sample_id, run_id=entry.run_id,
        agent="fact_investigator", fact_round_no=entry.fact_round_no, claim_sha256=entry.claim_sha256)
    contract = build_fact_run_context_v2("initial", initial_manifest_path=manifest_path, scope=scope)
    binding = M4FactImportBindingV1(manifest_sha256=manifest_sha, entry_sha256=_entry_sha(entry),
        sample_id=entry.sample_id, case_id=entry.case_id, fact_round_no=entry.fact_round_no,
        experiment_id=manifest.experiment_id, condition_sha256=manifest.condition_sha256,
        group_id="iterative", run_id=entry.run_id, row_sha256=entry.row_sha256,
        trace_path=entry.trace_path, trace_sha256=entry.trace_sha256,
        initial_manifest_path=entry.manifest_path, initial_manifest_sha256=entry.manifest_sha256,
        claim_sha256=entry.claim_sha256)
    result = _result_for(row, entry, root)
    if (row.run.run_id != entry.run_id or row.run.sample_id != entry.sample_id
            or row.run.group_id != entry.group_id or row.row_sha256 != entry.row_sha256):
        raise ValueError("m4_fact_import_answer_identity_mismatch")
    return FactArchiveInputV2(reader=FactArchiveReaderV2(trace_root, manifest_path), result=result,
        task=task, contract=contract, scope=scope, historical_import=binding)


def _result_for(row: object, entry: M4FactImportItem, root: Path) -> AgentResultV2:
    """从指纹核验的答卷行还原标准结果对象供归档比较。"""
    payload = row.payload
    result = payload["result"]
    report = dict(result["report"])
    if not report.get("proposed_issues"):
        report.pop("proposed_issues", None)
    trace_path = _under_root(root, entry.trace_path)
    events = [TraceEvent.model_validate_json(line, strict=True)
              for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    outcome = _one(events, "run_outcome").data
    usages = [event.data for event in events if event.type == "llm_call"]
    validation_data = dict(result["validation"])
    validation_data.pop("accepted", None)
    validation = FactV2ProgramValidation.model_validate(validation_data, strict=True)
    evidence = [LedgerEntry.model_validate(data, strict=True) for data in result["evidence"]]
    return AgentResultV2(run_id=entry.run_id, report_json=_canonical(report), validation=validation,
        evidence=evidence, stop_reason=payload["stop_reason"], steps=outcome["steps"],
        input_output_tokens=sum(item["input_tokens"] + item["output_tokens"] for item in usages),
        cost_cny=sum(item["cost_cny"] for item in usages), trace_path=str(trace_path))


def _check_binding(binding: M4FactImportBindingV1, manifest_sha: str,
                   entry: M4FactImportItem, item: FactArchiveInputV2, root: Path) -> None:
    """拒绝调用者改写清单身份、跨案件、补查轮或普通读取路径。"""
    expected = M4FactImportBindingV1(manifest_sha256=manifest_sha, entry_sha256=_entry_sha(entry),
        sample_id=entry.sample_id, case_id=entry.case_id, fact_round_no=entry.fact_round_no,
        experiment_id=_SOURCE_EXPERIMENT, condition_sha256=_SOURCE_CONDITION,
        group_id="iterative", run_id=entry.run_id, row_sha256=entry.row_sha256,
        trace_path=entry.trace_path, trace_sha256=entry.trace_sha256,
        initial_manifest_path=entry.manifest_path, initial_manifest_sha256=entry.manifest_sha256,
        claim_sha256=entry.claim_sha256)
    if binding != expected:
        raise ValueError("m4_fact_import_binding_mismatch")
    if (not isinstance(item.scope, FactScopeV2) or item.contract.mode != "initial"
            or item.contract.scope != item.scope or item.scope.case_id != entry.case_id
            or item.scope.fact_round_no != entry.fact_round_no or item.task.sample_id != entry.sample_id
            or item.task.claim is None or _digest(item.task.claim.encode()) != entry.claim_sha256
            or item.reader.trace_root != (root / "traces/F2-B1-R2-2026-10-21-run1").resolve()
            or item.reader.manifest_path != (root / entry.manifest_path).resolve()):
        raise ValueError("m4_fact_import_input_scope_invalid")
    if item.result.run_id != entry.run_id or item.result.trace_path != str(_under_root(root, entry.trace_path)):
        raise ValueError("m4_fact_import_result_identity_invalid")


def _load_bound_trace(root: Path, item: FactArchiveInputV2, binding: M4FactImportBindingV1
                      ) -> tuple[list[TraceEvent], str, Path]:
    """校验Trace原字节指纹和程序事件身份，禁止补写或伪造scope事件。"""
    path = _under_root(root, binding.trace_path)
    events, digest = _load_events(path, item.scope)
    if digest != binding.trace_sha256 or any(event.type == "fact_v2_scope" for event in events):
        raise ValueError("m4_fact_import_trace_identity_invalid")
    return events, digest, path


def _check_source_contract(events: list[TraceEvent], manifest: M4FactImportManifest,
                           source: FactRunContextV2) -> None:
    """验证原始未绑定合同完整匹配，M4新身份不回写到来源合同。"""
    event = _one(events, "fact_v2_request_contract").data
    actual = {key: event.get(key) for key in _SOURCE_CONTRACT_FIELDS}
    expected = manifest.source_contract
    if source.scope is not None or actual != expected or source.contract_sha256 != expected["contract_sha256"]:
        raise ValueError("m4_fact_import_source_contract_mismatch")


def _rebuild_snapshot(item: FactArchiveInputV2, events: list[TraceEvent], trace_sha: str,
                      initial_manifest_sha: str, material: object,
                      provenance: M4FactImportProvenanceV1, root: Path) -> ValidatedFactSnapshotV2:
    """重验请求费用、钱包、成功提交、实际交付和事实V2规则。"""
    wallet = _load_read_only_wallet(root)
    archives = ArchiveReadSession(item.reader.trace_root)
    requests, responses, hashes = _load_request_chain(archives, events, item.scope)
    _verify_usage_and_wallet(archives, events, requests, responses, wallet, item.scope)
    outcome = _one(events, "run_outcome")
    if outcome.data.get("outcome_class") != "submitted_judgment":
        raise ValueError("fact_run_has_no_accepted_outcome")
    final, report_json, archived_validation, submission_sha = _read_final_submission(
        archives, events, requests, responses, outcome, item.result, item.scope)
    delivered, tool_hashes = _reconstruct_delivered(final, material, item.task, events,
        requests, responses, archives, item.scope)
    check = validate_fact_v2_core(json.loads(report_json), delivered=delivered,
        limits=item.contract.limits, plan_max_text_chars=item.contract.plan_max_text_chars,
        mode="initial", assigned_issue_ids=frozenset())
    if (check.report is None or not check.validation.accepted
            or check.validation.model_dump(mode="json") != archived_validation):
        raise ValueError("fact_archived_validation_mismatch")
    all_hashes = hashes + tool_hashes + [{"kind": "submission", "path": final.data["archive"]["path"],
        "sha256": submission_sha}]
    return _snapshot(item.scope, report_json, check.validation, delivered, initial_manifest_sha,
        trace_sha, tuple(requests), all_hashes, historical_import_sha256=_digest(
            _canonical(provenance.model_dump(mode="json")).encode("utf-8")))


def _load_read_only_wallet(root: Path) -> _ReadOnlyWallet:
    """核验固定状态和钱包数据库，并以只读SQLite重放完整费用历史。"""
    manifest, _ = _load_import_manifest(root)
    state_path = _under_root(root, manifest.source_state)
    wallet_path = _under_root(root, manifest.source_wallet)
    if (_digest(state_path.read_bytes()) != manifest.source_state_sha256
            or _digest(wallet_path.read_bytes()) != manifest.source_wallet_sha256):
        raise ValueError("m4_fact_import_wallet_fingerprint_mismatch")
    state_doc = json.loads(state_path.read_text(encoding="utf-8"))
    ledger_manifest = _manifest_data(state_doc["manifest"])
    if (ledger_manifest["experiment_id"] != manifest.experiment_id
            or ledger_manifest["conditions_fingerprint"] != manifest.condition_sha256
            or _digest(_canonical(state_doc["conditions"]).encode()) != manifest.condition_sha256):
        raise ValueError("m4_fact_import_wallet_identity_mismatch")
    history = _read_sqlite_history(wallet_path)
    state = replay_history(history, ledger_manifest)
    status = state.snapshot()
    if status.halted or status.pending_requests or status.unknown_requests or status.cost_cny is None:
        raise ValueError("fact_shared_wallet_not_healthy")
    return _ReadOnlyWallet(history, state)


def _read_sqlite_history(path: Path) -> LedgerHistory:
    """Open the historical wallet with SQLite mode=ro and verify its hash chain."""
    uri = f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    store = ExperimentBudgetStore(path, connection)
    try:
        return store.load()
    finally:
        store.close()


def _provenance(binding: M4FactImportBindingV1, manifest_sha: str) -> M4FactImportProvenanceV1:
    """Build separate immutable provenance for original run and assigned M4 scope."""
    return M4FactImportProvenanceV1(manifest_sha256=manifest_sha, entry_sha256=binding.entry_sha256,
        sample_id=binding.sample_id, case_id=binding.case_id, fact_round_no=binding.fact_round_no,
        experiment_id=binding.experiment_id, condition_sha256=binding.condition_sha256,
        run_id=binding.run_id, row_sha256=binding.row_sha256, trace_sha256=binding.trace_sha256,
        initial_manifest_sha256=binding.initial_manifest_sha256,
        source_contract_sha256=_source_contract_sha256())


def _source_contract_sha256() -> str:
    """Return the pinned unscoped source contract identity."""
    return "e81e37ba791b3dd8075f2cf35559419c05beab63cf13d80e4292b909839e41b0"


def _find_manifest_item(manifest: M4FactImportManifest, sample_id: str) -> M4FactImportItem:
    """Return exactly one allowlisted source entry by its fixed sample identity."""
    matches = [item for item in manifest.items if item.sample_id == sample_id]
    if len(matches) != 1:
        raise ValueError("m4_fact_import_sample_not_allowed")
    return matches[0]


def _entry_sha(entry: M4FactImportItem) -> str:
    """Hash the exact program-owned source and assignment mapping."""
    return _digest(_canonical(entry.model_dump(mode="json")).encode("utf-8"))
