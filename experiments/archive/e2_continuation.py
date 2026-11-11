"""只读核验停场父实验；未知消费保留，不解除旧钱包停止状态。"""

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from experiments.archive.e2_artifacts import E2CheckpointRow, checkpoint_row_sha256, validate_checkpoint_artifacts
from experiments.archive.e2_resume import audit_e2_resume
from experiments.archive.e2_report import _read_events
from experiments.archive.e2_workspace import _State, _acquire, _unique_object
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore, LedgerHistory
from src.yuqing.settings import PROJECT_ROOT


@dataclass(frozen=True)
class ContinuationParent:
    """父场只读快照；remaining_cap只是预算余额，不是服务商账单保证。"""

    folder: Path
    state: _State
    finished_pairs: int
    known_cost: Decimal
    unknown_reserved: Decimal
    remaining_cap: Decimal
    files: dict[str, str]
    rows: tuple[E2CheckpointRow, ...]
    history: LedgerHistory


_CONTINUATION_CODE = {
    "file:src/yuqing/llm.py", "file:experiments/archive/e2_runner.py",
    "file:experiments/archive/e2_conditions.py", "file:experiments/archive/e2_continuation.py",
    "file:experiments/archive/e2_continuation_run.py", "file:scripts/run_dev_e2.py", "code_revision",
}


def continuation_conditions(parent: ContinuationParent, current: dict[str, str]) -> dict[str, str]:
    """只容许本卡接线代码变化；提示词/配置/模型/语料/判分结构必须一致。"""
    old = parent.state.conditions
    changed = {key: [old.get(key), current.get(key)] for key in old.keys() | current.keys()
               if old.get(key) != current.get(key)}
    if set(changed) - _CONTINUATION_CODE:
        raise ValueError("continuation_conditions_changed: " + ", ".join(sorted(set(changed) - _CONTINUATION_CODE)))
    return {"continuation:parent": parent.folder.relative_to(PROJECT_ROOT).as_posix(),
        "continuation:parent_files": json.dumps(parent.files, sort_keys=True),
        "continuation:finished_pairs": str(parent.finished_pairs),
        "continuation:known_cost": str(parent.known_cost),
        "continuation:unknown_reserved": str(parent.unknown_reserved),
        "continuation:remaining_cap": str(parent.remaining_cap),
        "continuation:code_changes": json.dumps(changed, sort_keys=True)}


def _digest(path: Path) -> str:
    """读取原字节指纹，文件缺失不跳过。"""
    return sha256(path.read_bytes()).hexdigest()


def _parent_state(folder: Path, sample_ids: Sequence[str]) -> _State:
    """核对状态内容指纹、条件指纹及冻结完整配对计划。"""
    raw = (folder / "state.json").read_bytes()
    json.loads(raw, object_pairs_hook=_unique_object)
    state = _State.model_validate_json(raw)
    if state.row_sha256 != checkpoint_row_sha256(state.model_dump(mode="json")):
        raise ValueError("continuation_parent_state_digest")
    encoded = json.dumps(state.conditions, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if state.manifest["conditions_fingerprint"] != sha256(encoded).hexdigest():
        raise ValueError("continuation_parent_conditions_digest")
    expected = [(sample, group) for sample in sample_ids for group in ("single_shot", "iterative")]
    if state.manifest["mode"] != "full" or any(k.startswith("continuation:") for k in state.conditions):
        raise ValueError("continuation_requires_original_full")
    if [(slot.sample_id, slot.group_id) for slot in state.plan] != expected or (
        len({slot.run_id for slot in state.plan}) != len(state.plan)):
        raise ValueError("continuation_parent_plan_mismatch")
    return state


def _inspect_locked(folder: Path, sample_ids: Sequence[str]) -> ContinuationParent:
    """复用答卷/Trace/账本审计，只接纳未知消费导致的完整配对停场。"""
    paths = [folder / name for name in ("state.json", "answers.jsonl", "wallet.sqlite", "owner.json")]
    if any(not path.resolve().is_relative_to(folder) for path in paths):
        raise ValueError("continuation_parent_file_escape")
    before = {str(path.relative_to(PROJECT_ROOT)): _digest(path) for path in paths}
    owner = json.loads((folder / "owner.json").read_bytes(), object_pairs_hook=_unique_object)
    if owner.get("status") != "closed" or any(folder.rglob("*.partial")):
        raise ValueError("continuation_parent_not_closed")
    state = _parent_state(folder, sample_ids)
    store = ExperimentBudgetStore.open_existing(folder / "wallet.sqlite", state.manifest)
    try:
        history = store.load()
    finally:
        store.close()
    raw = (folder / "answers.jsonl").read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("continuation_parent_answers_partial")
    rows = validate_checkpoint_artifacts(raw.decode("utf-8").splitlines(),
        experiment_id=state.manifest["experiment_id"], condition_sha256=state.manifest["conditions_fingerprint"])
    audit = audit_e2_resume(state.plan, rows, history, state.manifest)
    recovered = replay_history(history, state.manifest)
    wallet = recovered.snapshot()
    if {issue.code for issue in audit.issues} - {"ledger_halted", "unresolved_request"} or audit.half_pairs:
        raise ValueError("continuation_parent_audit_failed")
    if wallet.halt_reason != "usage_unknown" or not wallet.unknown_requests or wallet.pending_requests:
        raise ValueError("continuation_parent_not_unknown_terminal")
    if any(request.state not in {"settled", "cancelled", "unknown"} for request in recovered.requests.values()):
        raise ValueError("continuation_parent_incomplete_request")
    if audit.finished_runs != len(rows) or not 0 < audit.completed_pairs < len(sample_ids):
        raise ValueError("continuation_parent_not_complete_prefix")
    for slot in state.plan[len(rows):]:
        if any(folder.glob(f"*/{slot.run_id}.jsonl")):
            raise ValueError("continuation_parent_orphan_trace")
    for row in rows:
        _read_events(row, SimpleNamespace(folder=folder))
        path = (PROJECT_ROOT / row.payload["trace"]["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("continuation_parent_trace_outside_workspace")
        before[str(path.relative_to(PROJECT_ROOT))] = _digest(path)
        if before[str(path.relative_to(PROJECT_ROOT))] != row.payload["trace"]["sha256"]:
            raise ValueError("continuation_parent_trace_changed")
    if any(_digest(PROJECT_ROOT / path) != digest for path, digest in before.items()):
        raise ValueError("continuation_parent_changed")
    remaining = wallet.max_cost_cny - wallet.committed_cost_cny
    if remaining <= 0:
        raise ValueError("continuation_no_remaining_budget")
    return ContinuationParent(folder, state, audit.completed_pairs, wallet.known_cost_subtotal_cny,
        wallet.unknown_reserved_cny, remaining, before, rows, history)


def inspect_continuation_parent(folder: Path, sample_ids: Sequence[str]) -> ContinuationParent:
    """在项目E2目录持有原系统锁，只读核验后立即释放，不接管或恢复父场。"""
    root, actual = (PROJECT_ROOT / "traces/e2").resolve(), folder.resolve()
    if actual == root or not actual.is_relative_to(root):
        raise ValueError("continuation_parent_outside_e2")
    with (actual / "workspace.lock").open("r+b") as handle:
        _acquire(handle)
        return _inspect_locked(actual, sample_ids)
