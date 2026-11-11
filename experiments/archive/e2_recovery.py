"""仅承接作者核验的两段dev90记录，生成唯一受限收尾计划。"""

from dataclasses import dataclass
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

from src.yuqing.eval.dataset import load_questions
from experiments.archive.e2_artifacts import E2CheckpointRow, checkpoint_row_sha256, validate_checkpoint_artifacts
from experiments.archive.e2_continuation import ContinuationParent, inspect_continuation_parent
from experiments.archive.e2_resume import audit_e2_resume
from experiments.archive.e2_workspace import _State, _acquire, _unique_object, _check_previous_owner, _publish
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore
from src.yuqing.eval.experiment_budget_recovery import replay_history
from experiments.archive.e2_checkpoint import FINAL_RECOVERY_FREEZE_SHA256, PlannedRun
from src.yuqing.settings import PROJECT_ROOT

_FREEZE_SHA256 = FINAL_RECOVERY_FREEZE_SHA256
_FREEZE_PATH = PROJECT_ROOT / "docs/cards/E2-dev90-recovery-freeze.json"


@dataclass(frozen=True)
class FinalRecovery:
    """双来源校验快照及仅含未执行槽位的最终计划。"""

    parent: ContinuationParent
    child_path: Path
    child_state: _State
    child_rows: tuple[E2CheckpointRow, ...]
    remaining: tuple[PlannedRun, ...]
    known_cost: Decimal
    unknown_reserved: Decimal
    remaining_budget: Decimal
    source_sha256: str


_FINAL_ALLOWED_CODE = {
    "code_revision", "file:experiments/archive/e2_checkpoint.py", "file:experiments/archive/e2_conditions.py",
    "file:experiments/archive/e2_journal.py", "file:experiments/archive/e2_recovery.py",
    "file:experiments/archive/e2_resume.py", "file:experiments/archive/e2_runner.py",
    "file:experiments/archive/e2_workspace.py", "file:scripts/run_dev_e2.py", "file:src/yuqing/llm.py",
    "file:experiments/archive/e2_execute.py",
}
_FINAL_CHILD_LLM_SHA256 = "6e980d441a356ec79041b002839a23722b4623049e2804583d4a187138c37834"
_FINAL_READY_8_LLM_SHA256 = "b55463f338a5ef8566737ebc6f69ba1347a426e15592d17dae9740e180aaffc1"
_FINAL_OLD_EXECUTOR_SHA256 = "0cb9d349b1cc6055e9f9369fa55991f257021a92ae11f3b0d55650fb98592924"
_FINAL_FIXED_EXECUTOR_SHA256 = "8043e6c4247c35bcf97ce05b11aa4a18fd0884d1f2aef4050bc70128406ead23"
_ZERO_START_FREEZE = PROJECT_ROOT / "docs/cards/E2-dev90-zero-start-freeze.json"
_ZERO_START_SHA256 = "680cb4dac0218d594c06b0ffe318063a4dd8114986c396de3128542a422e2ace"


def zero_start_freeze(folder: Path) -> dict[str, object]:
    """仅接受配置规定的零请求现场，清单指纹按换行归一化绑定。"""
    raw = _ZERO_START_FREEZE.read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != _ZERO_START_SHA256:
        raise ValueError("zero_start_freeze_changed")
    data = json.loads(raw, object_pairs_hook=_unique_object)
    actual = folder if folder.is_absolute() else PROJECT_ROOT / folder
    if actual.resolve() != _relative_path(PROJECT_ROOT.resolve(), data["folder"]):
        raise ValueError("zero_start_not_authorized")
    return data


def _check_zero_start(recovery: FinalRecovery, folder: Path, frozen: dict[str, object], handle: BinaryIO) -> None:
    """锁内证明原场零事件、零答案、无归档且仍绑定原来源与认领。"""
    root = PROJECT_ROOT.resolve()
    if {item.name for item in folder.iterdir()} != {"state.json", "owner.json", "answers.jsonl", "wallet.sqlite", "workspace.lock"}:
        raise ValueError("zero_start_unexpected_files")
    lock_key = (folder / "workspace.lock").relative_to(root).as_posix()
    files = {name: digest for name, digest in frozen["files"].items() if name != lock_key}
    handle.seek(0)
    if sha256(handle.read()).hexdigest() != frozen["files"][lock_key]:
        raise ValueError("zero_start_lock_changed")
    _check_frozen_files(root, files)
    state_raw = (folder / "state.json").read_bytes()
    state = _State.model_validate_json(state_raw)
    if state.row_sha256 != checkpoint_row_sha256(json.loads(state_raw)) or state.manifest != frozen["manifest"]:
        raise ValueError("zero_start_state_mismatch")
    if state.plan != recovery.remaining or frozen["source_sha256"] != recovery.source_sha256 or (
            frozen["plan_sha256"] != recovery_plan_sha256(recovery.remaining)):
        raise ValueError("zero_start_plan_mismatch")
    old_claim = _final_claim(recovery, folder, state.manifest["experiment_id"],
        state.manifest["conditions_fingerprint"], Decimal(state.manifest["max_cost_cny"]))
    if (recovery.parent.folder / "final_segment.json").read_bytes() != old_claim or (folder / "answers.jsonl").read_bytes():
        raise ValueError("zero_start_claim_or_answers_mismatch")
    store = ExperimentBudgetStore.open_existing(folder / "wallet.sqlite", state.manifest)
    try:
        history = store.load()
        replay_history(history, state.manifest)
        if history.events or history.revision:
            raise ValueError("zero_start_has_wallet_events")
    finally:
        store.close()
    owner_key = (folder / "owner.json").relative_to(root).as_posix()
    _check_previous_owner(folder, frozen["files"][owner_key])
    _check_frozen_files(root, files)


@contextmanager
def _claim_locks(recovery: FinalRecovery, replacement: Path | None) -> Iterator[Path]:
    """来源锁与零启动锁共同保护一次追加认领，活动进程或partial均拒绝。"""
    frozen = zero_start_freeze(replacement) if replacement is not None else None
    old = _relative_path(PROJECT_ROOT.resolve(), frozen["folder"]) if frozen else None
    target = recovery.parent.folder / ("final_segment_replacement.json" if old else "final_segment.json")
    with ExitStack() as locks:
        for folder in (recovery.parent.folder, recovery.child_path, *((old,) if old else ())):
            handle = locks.enter_context((folder / "workspace.lock").open("r+b"))
            _acquire(handle)
        _assert_recovery_sources(recovery)
        replaced = recovery.parent.folder / "final_segment_replacement.json"
        if replaced.with_suffix(".partial").exists() or (old is None and replaced.exists()):
            raise ValueError("final_recovery_replaced_or_partial")
        if old is not None:
            _check_zero_start(recovery, old, frozen, handle)
        yield target


def _digest(path: Path) -> str:
    """按原始字节计算来源文件指纹。"""
    return sha256(path.read_bytes()).hexdigest()


def _relative_path(root: Path, value: str) -> Path:
    """仅接受项目内的相对文件路径。"""
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("recovery_source_path_escape")
    return path


def _check_frozen_files(root: Path, files: dict[str, str]) -> None:
    """比对冻结清单中的每个源文件，不容许缺失或字节变化。"""
    if not files:
        raise ValueError("recovery_source_files_missing")
    for name, expected in files.items():
        if _digest(_relative_path(root, name)) != expected:
            raise ValueError("recovery_source_changed")


def _frozen_manifest(path: Path) -> dict[str, object]:
    """读取经代码指纹固定的配置规定来源清单。"""
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != _FREEZE_SHA256:
        raise ValueError("recovery_freeze_manifest_changed")
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if data.get("version") != 1 or data.get("exception") != "E2-dev90-2026-10-16-final-segment-only":
        raise ValueError("recovery_freeze_manifest_invalid")
    return data


def _fixed_questions(root: Path) -> list[str]:
    """按原始dev90冻结顺序读取题号。"""
    paths = (root / "eval/questions_30.jsonl", root / "eval/dev_extra_60.jsonl")
    return [item.sample_id for path in paths for item in load_questions(path)]


def _validate_child_state(root: Path, folder: Path, child: dict[str, object],
                          parent: ContinuationParent) -> _State:
    """校验续评工作区关闭状态、指纹、父子关系和冻结计划。"""
    owner = json.loads((folder / "owner.json").read_bytes(), object_pairs_hook=_unique_object)
    if owner.get("status") != "closed" or (folder / "owner.partial").exists():
        raise ValueError("recovery_child_not_closed")
    raw = (folder / "state.json").read_bytes()
    state, data = _State.model_validate_json(raw), json.loads(raw, object_pairs_hook=_unique_object)
    if state.row_sha256 != checkpoint_row_sha256(data):
        raise ValueError("recovery_child_state_digest")
    conditions = json.dumps(state.conditions, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if sha256(conditions).hexdigest() != state.manifest["conditions_fingerprint"]:
        raise ValueError("recovery_child_conditions_digest")
    if state.manifest["experiment_id"] != child["experiment_id"]:
        raise ValueError("recovery_child_identity_changed")
    if state.conditions.get("continuation:parent") != parent.folder.relative_to(root).as_posix():
        raise ValueError("recovery_parent_child_link_changed")
    if json.loads(state.conditions["continuation:parent_files"]) != parent.files:
        raise ValueError("recovery_parent_snapshot_changed")
    expected = parent.state.plan[len(parent.rows):]
    if [(slot.sample_id, slot.group_id) for slot in state.plan] != [
            (slot.sample_id, slot.group_id) for slot in expected]:
        raise ValueError("recovery_child_plan_changed")
    return state


def _child_snapshot(root: Path, child: dict[str, object], parent: ContinuationParent,
                    sample_ids: Sequence[str]) -> tuple[Path, _State, tuple[E2CheckpointRow, ...], Decimal, Decimal]:
    """只读核对现有续评场答卷、Trace、钱包、计划和父场认领。"""
    folder = _relative_path(root, str(child["path"]))
    lock_path = folder / "workspace.lock"
    if not lock_path.is_file() or lock_path.stat().st_size == 0:
        raise ValueError("recovery_child_lock_missing")
    with lock_path.open("r+b") as handle:
        _acquire(handle)
        state = _validate_child_state(root, folder, child, parent)
        rows = validate_checkpoint_artifacts((folder / "answers.jsonl").read_bytes().decode("utf-8").splitlines(),
            experiment_id=str(state.manifest["experiment_id"]),
            condition_sha256=str(state.manifest["conditions_fingerprint"]))
        history_store = ExperimentBudgetStore.open_existing(folder / "wallet.sqlite", state.manifest)
        try:
            history = history_store.load()
        finally:
            history_store.close()
        audit = audit_e2_resume(state.plan, rows, history, state.manifest)
        wallet = replay_history(history, state.manifest).snapshot()
        if len(rows) != 1 or (rows[0].run.sample_id, rows[0].run.group_id) != (
                parent.state.plan[len(parent.rows)].sample_id, parent.state.plan[len(parent.rows)].group_id):
            raise ValueError("recovery_child_answer_mismatch")
        if set(issue.code for issue in audit.issues) != {"ledger_halted", "unresolved_request"}:
            raise ValueError("recovery_child_audit_mismatch")
        if audit.half_pairs != ("chef:dev:334",) or wallet.halt_reason != "usage_unknown":
            raise ValueError("recovery_child_stop_mismatch")
        if wallet.pending_requests or wallet.unknown_requests != 1:
            raise ValueError("recovery_child_wallet_mismatch")
        expected_child_cap = parent.remaining_cap
        if Decimal(state.manifest["max_cost_cny"]) != expected_child_cap:
            raise ValueError("recovery_child_budget_mismatch")
        trace_ref = rows[0].payload["trace"]
        trace_path = _relative_path(root, str(trace_ref["path"]))
        if _digest(trace_path) != trace_ref["sha256"]:
            raise ValueError("recovery_child_trace_mismatch")
        return folder, state, tuple(rows), wallet.known_cost_subtotal_cny, wallet.unknown_reserved_cny


def _remaining_slots(parent: ContinuationParent, child_rows: Sequence[E2CheckpointRow]) -> tuple[PlannedRun, ...]:
    """从两个来源已保存的运行编号中剔除旧槽位，保持原计划顺序。"""
    executed = {(row.run.sample_id, row.run.group_id) for row in (*parent.rows, *child_rows)}
    remaining = tuple(slot for slot in parent.state.plan if (slot.sample_id, slot.group_id) not in executed)
    if len(executed) != len(parent.rows) + len(child_rows) or len(remaining) != 157:
        raise ValueError("recovery_remaining_slot_count")
    if remaining[0].sample_id != "chef:dev:334" or remaining[0].group_id != "iterative":
        raise ValueError("recovery_first_slot_mismatch")
    return remaining


def recovery_plan_sha256(plan: Sequence[PlannedRun]) -> str:
    """绑定来源推导出的槽位次序与运行编号。"""
    canonical = json.dumps([slot.model_dump(mode="json") for slot in plan], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(canonical).hexdigest()


def final_recovery_conditions(recovery: FinalRecovery, current: dict[str, str]) -> dict[str, str]:
    """以第二段条件为基线，只允许本收尾适配涉及的代码差异。"""
    old, original = recovery.child_state.conditions, recovery.parent.state.conditions
    executor_key = "file:experiments/archive/e2_execute.py"
    if executor_key in old or executor_key in current:
        if (old.get(executor_key) != _FINAL_OLD_EXECUTOR_SHA256 or
                current.get(executor_key) != _FINAL_FIXED_EXECUTOR_SHA256):
            raise ValueError(f"recovery_conditions_changed: {executor_key}")
    llm_key = "file:src/yuqing/llm.py"
    if (old.get(llm_key) != _FINAL_CHILD_LLM_SHA256 or
            current.get(llm_key) != _FINAL_READY_8_LLM_SHA256):
        raise ValueError(f"recovery_conditions_changed: {llm_key}")
    if current.get("questions") != original.get("questions"):
        raise ValueError("recovery_frozen_questions_changed")
    changed = {key for key in old.keys() | current.keys() if not key.startswith("continuation:")
        and old.get(key) != current.get(key)}
    if changed - (_FINAL_ALLOWED_CODE | {"questions"}):
        raise ValueError("recovery_conditions_changed: " + ", ".join(sorted(changed - _FINAL_ALLOWED_CODE - {"questions"})))
    return {"recovery:final_segment": "true", "recovery:source_sha256": recovery.source_sha256,
        "recovery:plan_sha256": recovery_plan_sha256(recovery.remaining)}


def _final_claim(recovery: FinalRecovery, folder: Path, experiment_id: str,
                condition_sha256: str, budget: Decimal, replacement: Path | None = None) -> bytes:
    """绑定最终场目录、实验条件、固定来源与预算。"""
    data = dict(child=folder.relative_to(PROJECT_ROOT).as_posix(), experiment_id=experiment_id,
        condition_sha256=condition_sha256, max_cost_cny=str(budget), source_sha256=recovery.source_sha256,
        plan_sha256=recovery_plan_sha256(recovery.remaining))
    if replacement is not None:
        data["abandoned_startup"] = zero_start_freeze(replacement)
        if budget > Decimal(data["abandoned_startup"]["manifest"]["max_cost_cny"]) or (
                folder.resolve() == _relative_path(PROJECT_ROOT.resolve(), data["abandoned_startup"]["folder"])):
            raise ValueError("zero_start_budget_or_target_invalid")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _assert_recovery_sources(recovery: FinalRecovery) -> None:
    """认领前后再次核对两段来源原字节和原始父子关系。"""
    manifest = _frozen_manifest(_FREEZE_PATH)
    _check_frozen_files(PROJECT_ROOT.resolve(), manifest["parent"]["files"])
    _check_frozen_files(PROJECT_ROOT.resolve(), manifest["parent"]["results"])
    _check_frozen_files(PROJECT_ROOT.resolve(), manifest["child"]["files"])
    _check_frozen_files(PROJECT_ROOT.resolve(), manifest["child"]["results"])


def claim_final_segment(recovery: FinalRecovery, folder: Path, experiment_id: str,
                        condition_sha256: str, budget: Decimal, *, resume: bool, replacement: Path | None = None) -> None:
    """父子来源双锁内独占认领最终段，不改旧答卷、Trace或钱包。"""
    payload = _final_claim(recovery, folder, experiment_id, condition_sha256, budget, replacement)
    with _claim_locks(recovery, replacement) as claim_path:
        if claim_path.resolve().parent != recovery.parent.folder.resolve():
            raise ValueError("recovery_claim_path_escape")
        if resume:
            if claim_path.read_bytes() != payload:
                raise ValueError("recovery_claim_mismatch")
        else:
            if claim_path.exists():
                raise FileExistsError("final_recovery_already_claimed")
            partial = claim_path.with_suffix(".partial") if replacement is not None else claim_path
            with partial.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if replacement is not None:
                _publish(partial, claim_path)


def assert_final_segment_claim_available(recovery: FinalRecovery, folder: Path, experiment_id: str,
                                         condition_sha256: str, budget: Decimal, *, resume: bool,
                                         replacement: Path | None = None) -> None:
    """在新建工作区前拒绝重复启动，恢复仅接受原认领身份。"""
    payload = _final_claim(recovery, folder, experiment_id, condition_sha256, budget, replacement)
    with _claim_locks(recovery, replacement) as claim_path:
        if resume:
            if claim_path.read_bytes() != payload:
                raise ValueError("recovery_claim_mismatch")
        elif claim_path.exists():
            raise FileExistsError("final_recovery_already_claimed")


def check_final_segment_claim(recovery: FinalRecovery, folder: Path, experiment_id: str,
                              condition_sha256: str, budget: Decimal, replacement: Path | None = None) -> None:
    """每个槽位前核对认领内容与两个来源指纹，变化即停止。"""
    with _claim_locks(recovery, replacement) as claim_path:
        if claim_path.read_bytes() != _final_claim(recovery, folder, experiment_id, condition_sha256, budget, replacement):
            raise ValueError("recovery_claim_changed")


def _segment_details(folder: Path, state: _State, rows: Sequence[E2CheckpointRow], history: object) -> list[dict]:
    """从来源段的原始Trace和钱包提取同口径答卷明细。"""
    from experiments.archive.e2_report import _detail

    requests = replay_history(history, state.manifest).requests
    source = type("Segment", (), {"folder": folder, "conditions": state.conditions})()
    return [_detail(row, source, requests) for row in rows]


def _child_history(recovery: FinalRecovery) -> object:
    """只读重新读取已冻结续评段的SQLite账本。"""
    store = ExperimentBudgetStore.open_existing(recovery.child_path / "wallet.sqlite", recovery.child_state.manifest)
    try:
        return store.load()
    finally:
        store.close()


def publish_final_recovery_summary(recovery: FinalRecovery, workspace: object, journal: object,
                                   session: object, files: dict[str, str], replacement: Path | None = None) -> str:
    """新建三段汇总，逐槽去重并保留未知费用与全部未执行项。"""
    from experiments.archive.e2_report import _report_rows, _write_result, _json, summarize_e2
    from src.yuqing.eval.experiment_budget_recovery import replay_history

    check_final_segment_claim(recovery, workspace.folder, workspace.manifest["experiment_id"],
        workspace.manifest["conditions_fingerprint"], Decimal(workspace.manifest["max_cost_cny"]), replacement)
    child_history = _child_history(recovery)
    parent_rows = _segment_details(recovery.parent.folder, recovery.parent.state, recovery.parent.rows,
        recovery.parent.history)
    child_rows = _segment_details(recovery.child_path, recovery.child_state, recovery.child_rows, child_history)
    final_audit, final_rows = _report_rows(workspace, journal, session)
    rows = parent_rows + child_rows + final_rows
    keys = [(row["sample_id"], row["variant"]) for row in rows]
    if len(keys) != 180 or len(set(keys)) != len(keys) or len({key[0] for key in keys}) != 90:
        raise ValueError("recovery_three_segment_duplicate_or_missing_slot")
    summary = summarize_e2(rows)
    final_wallet = session.snapshot()
    child_wallet = replay_history(child_history, recovery.child_state.manifest).snapshot()
    report = dict(kind="audit", scope="three_segment_dev90_recovery", source_freeze_sha256=recovery.source_sha256,
        source_experiments=[recovery.parent.state.manifest["experiment_id"], recovery.child_state.manifest["experiment_id"]],
        final_experiment=workspace.manifest["experiment_id"], summary=summary, cost_cny=None,
        abandoned_zero_start=zero_start_freeze(replacement) if replacement is not None else None,
        known_cost_subtotal_cny=str(recovery.parent.known_cost + child_wallet.known_cost_subtotal_cny +
            final_wallet.known_cost_subtotal_cny),
        unknown_reserved_cny=str(recovery.parent.unknown_reserved + child_wallet.unknown_reserved_cny +
            final_wallet.unknown_reserved_cny), parent_unknown_retained=True,
        child_unknown_retained=child_wallet.unknown_requests > 0, final_audit=final_audit.model_dump(mode="json"),
        source_files={"freeze": recovery.source_sha256, "parent": recovery.parent.files,
            "parent_results": _frozen_manifest(_FREEZE_PATH)["parent"]["results"],
            "child": _frozen_manifest(_FREEZE_PATH)["child"]["files"],
            "child_results": _frozen_manifest(_FREEZE_PATH)["child"]["results"]},
        final_files={key: {"path": path, "sha256": sha256(Path(path).read_bytes()).hexdigest()}
            for key, path in files.items()},
        interpretation="三段dev记录，保留停场、未知费用及未执行槽位；不是独立test成绩，总实际费用未知。")
    path = _write_result(workspace, "audit", "jsonl", (_json(report) + "\n").encode(),
        "E2 dev90三段恢复汇总；来源逐槽去重，未知费用保留")
    return str(path)


def inspect_final_recovery(path: Path = _FREEZE_PATH) -> FinalRecovery:
    """核验两段冻结来源并返回唯一的157槽最终计划，不创建运行文件。"""
    root = PROJECT_ROOT.resolve()
    manifest = _frozen_manifest(path)
    parent_meta, child_meta = manifest["parent"], manifest["child"]
    sample_ids = _fixed_questions(root)
    parent_folder = _relative_path(root, str(parent_meta["path"]))
    parent = inspect_continuation_parent(parent_folder, sample_ids)
    if parent.state.manifest["experiment_id"] != parent_meta["experiment_id"]:
        raise ValueError("recovery_parent_identity_changed")
    _check_frozen_files(root, parent_meta["files"])
    _check_frozen_files(root, parent_meta["results"])
    claim = json.loads((parent_folder / "continuation.json").read_bytes(), object_pairs_hook=_unique_object)
    if claim.get("child") != str(child_meta["path"]):
        raise ValueError("recovery_parent_claim_changed")
    folder, state, rows, child_known, child_unknown = _child_snapshot(root, child_meta, parent, sample_ids)
    _check_frozen_files(root, child_meta["files"])
    _check_frozen_files(root, child_meta["results"])
    remaining = _remaining_slots(parent, rows)
    known = parent.known_cost + child_known
    unknown = parent.unknown_reserved + child_unknown
    cap = Decimal(parent.state.manifest["max_cost_cny"]) - known - unknown
    expected = manifest["expected"]
    if (len(parent.state.plan), len(parent.rows), parent.finished_pairs, len(rows), len(remaining)) != (
            expected["planned_slots"], expected["parent_rows"], expected["parent_pairs"],
            expected["child_rows"], expected["remaining_slots"]):
        raise ValueError("recovery_source_counts_changed")
    if (known, unknown, cap) != tuple(Decimal(expected[key]) for key in (
            "known_cost_cny", "unknown_reserved_cny", "remaining_budget_cny")):
        raise ValueError("recovery_source_budget_changed")
    return FinalRecovery(parent, folder, state, rows, remaining, known, unknown, cap, _FREEZE_SHA256)
