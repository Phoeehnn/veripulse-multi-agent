"""实验目录排他与固定计划存储，不创建钱包或执行模型。"""

from hashlib import sha256
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter

from experiments.archive.e2_artifacts import Fingerprint, checkpoint_row_sha256
from experiments.archive.e2_checkpoint import (FINAL_RECOVERY_FREEZE_SHA256, FINAL_RECOVERY_PLAN_SHA256,
    Identifier, PlannedRun, PAIR_GROUPS, THREE_ARM_GROUPS,
    _final_recovery_layout_issues)
from src.yuqing.eval.experiment_budget_store import _manifest_data
from src.yuqing.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class _State(BaseModel):
    """只接收明确实验栏目，内容指纹不是防恶意重写签名。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    manifest: dict[str, object]
    conditions: dict[str, str]
    plan: tuple[PlannedRun, ...]
    row_sha256: Fingerprint


class _Owner(BaseModel):
    """占用诊断记录；旧格式默认未确认，不凭心跳时长授予接管权。"""

    model_config = ConfigDict(strict=True, extra="forbid")
    pid: int = Field(gt=0)
    started_at: AwareDatetime
    heartbeat_at: AwareDatetime
    status: Literal["active", "closed", "legacy"] = "legacy"
    confirmed_previous_sha256: Fingerprint | None = None


def _check_previous_owner(folder: Path, confirmed_owner_sha256: str | None) -> None:
    """仅在持有系统锁后检查前次占用；人工确认绑定具体原字节。"""
    if (folder / "owner.partial").exists():
        raise ValueError("workspace_owner_partial_requires_review")
    path = folder / "owner.json"
    if not path.is_file():
        raise ValueError("workspace_owner_missing_requires_review")
    raw = path.read_bytes()
    # 拒绝重复字段，损坏记录不能靠提供其哈希自动放行。
    json.loads(raw, object_pairs_hook=_unique_object)
    owner = _Owner.model_validate_json(raw)
    digest = sha256(raw).hexdigest()
    if confirmed_owner_sha256 is not None and confirmed_owner_sha256 != digest:
        raise ValueError("workspace_owner_confirmation_mismatch")
    if owner.status != "closed" and confirmed_owner_sha256 is None:
        raise ValueError("workspace_owner_confirmation_required: " + digest)


def _write_owner(folder: Path, *, status: Literal["active", "closed"], started_at: str,
                 confirmed_previous_sha256: str | None) -> None:
    """原子替换诊断记录，失败保留partial供核对，不改变答卷与钱包。"""
    target, partial = folder / "owner.json", folder / "owner.partial"
    if any(path.resolve().parent != folder for path in (target, partial)):
        raise ValueError("workspace_owner_path_escape")
    data = dict(pid=os.getpid(), started_at=started_at, heartbeat_at=datetime.now(timezone.utc).isoformat(),
        status=status, confirmed_previous_sha256=confirmed_previous_sha256)
    raw = _Owner.model_validate_json(json.dumps(data)).model_dump_json().encode("utf-8")
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        # 活动认领须同步；关闭标记丢失只会留下旧active，保守要求再次确认。
        if status == "active":
            os.fsync(stream.fileno())
    os.replace(partial, target)


def _samples(sample_ids: Sequence[str]) -> list[str]:
    """题号非空、严格合法且不重复，保持已冻结顺序。"""
    if isinstance(sample_ids, (str, bytes)) or not sample_ids:
        raise ValueError("invalid_workspace_samples")
    ids = [TypeAdapter(Identifier).validate_python(value, strict=True) for value in sample_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_workspace_samples")
    return ids


def build_e2_plan(sample_ids: Sequence[str], *, groups: Sequence[str] = PAIR_GROUPS) -> tuple[PlannedRun, ...]:
    """只为原双组或明确三臂顺序生成计划，恢复不调用本函数。"""
    if tuple(groups) not in (PAIR_GROUPS, THREE_ARM_GROUPS):
        raise ValueError("unsupported_e2_group_layout")
    return tuple(PlannedRun(sample_id=sample, group_id=group, run_id=uuid4().hex)
        for sample in _samples(sample_ids) for group in groups)


def _acquire(handle: BinaryIO) -> None:
    """非阻塞取得操作系统锁，不能用锁文件存在性判断是否占用。"""
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise ValueError("experiment_workspace_busy") from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """状态JSON重复字段拒绝，不静默采用最后一份。"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_workspace_state_key")
        result[key] = value
    return result


def _publish(partial: Path, target: Path) -> None:
    """平台分别用不覆盖重命名或独占硬链接发布，失败保留临时痕迹。"""
    if os.name == "nt":
        partial.rename(target)
    else:
        os.link(partial, target)
        partial.unlink()


class E2Workspace:
    """持有目录锁与独立状态快照，关闭释放句柄而不删除运行记录。"""

    def __init__(self, folder: Path, handle: BinaryIO, state: _State) -> None:
        """接收已加锁且校验通过的工作区。"""
        self.folder, self._handle, self._state = folder, handle, state
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._confirmed_owner_sha256: str | None = None

    @property
    def plan(self) -> tuple[PlannedRun, ...]:
        """返回不可变运行槽位，不重新生成编号。"""
        return self._state.plan

    @property
    def conditions(self) -> dict[str, str]:
        """返回独立条件副本，外部修改不改变已保存状态。"""
        return dict(self._state.conditions)

    @property
    def manifest(self) -> dict[str, object]:
        """返回已规范的钱包清单副本，不创建实际钱包。"""
        return dict(self._state.manifest)

    def close(self) -> None:
        """幂等关闭句柄，系统释放锁，不删除锁文件或状态。"""
        if self._handle.closed:
            return
        try:
            _write_owner(self.folder, status="closed", started_at=self._started_at,
                confirmed_previous_sha256=self._confirmed_owner_sha256)
        finally:
            self._handle.close()

    def require_open(self) -> None:
        """要求工作区仍持有句柄锁，关闭后的对象不得继续执行读写。"""
        if self._handle.closed:
            raise ValueError("experiment_workspace_closed")

    def heartbeat(self) -> None:
        """锁内写诊断时间；心跳从不授予抢锁、退款或重跑权限。"""
        self.require_open()
        _write_owner(self.folder, status="active", started_at=self._started_at,
            confirmed_previous_sha256=self._confirmed_owner_sha256)

    def __enter__(self) -> "E2Workspace":
        """允许用with明确管理锁的生命周期，关闭后不能重新使用。"""
        self.require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """正常退出记关闭；异常只释放锁，保留待确认记录和原始异常。"""
        if exc_type is None:
            self.close()
            return
        try:
            self._handle.close()
        except BaseException as cleanup_error:
            logger.error("异常退出释放锁失败 type=%s；原始异常保留", type(cleanup_error).__name__)


def _state(folder: Path, *, mode: Literal["new", "resume"], manifest: dict[str, object],
            conditions: dict[str, str], sample_ids: list[str],
            planned_runs: Sequence[PlannedRun] | None = None) -> _State:
    """新建状态独占发布；恢复只读并核对内容、条件和配对顺序。"""
    path, partial = folder / "state.json", folder / "state.partial"
    if mode == "new":
        if set(p.name for p in folder.iterdir()) - {"workspace.lock"}:
            raise ValueError("workspace_not_empty")
        groups = _groups_for_conditions(conditions)
        plan = tuple(planned_runs) if planned_runs is not None else build_e2_plan(sample_ids, groups=groups)
        data = dict(manifest=manifest, conditions=conditions, plan=[p.model_dump() for p in plan])
        data["row_sha256"] = checkpoint_row_sha256(data)
        with partial.open("xb") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        _publish(partial, path)
    if partial.exists():
        raise ValueError("workspace_partial_state")
    raw = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    state = _State.model_validate_json(json.dumps(raw, ensure_ascii=False, allow_nan=False))
    if state.row_sha256 != checkpoint_row_sha256(state.model_dump(mode="json")):
        raise ValueError("workspace_state_digest_mismatch")
    _manifest_data(state.manifest)
    if state.manifest != manifest or state.conditions != conditions:
        changed = ["manifest:" + k for k in set(state.manifest) | set(manifest)
            if state.manifest.get(k) != manifest.get(k)]
        changed += ["condition:" + k for k in set(state.conditions) | set(conditions)
            if state.conditions.get(k) != conditions.get(k)]
        raise ValueError("workspace_conditions_mismatch: " + ", ".join(sorted(changed)))
    if planned_runs is None:
        groups = _groups_for_conditions(conditions)
        plan_matches = ([(p.sample_id, p.group_id) for p in state.plan] ==
            [(sample, group) for sample in sample_ids for group in groups])
    else:
        expected = tuple(planned_runs)
        plan_matches = ([(p.sample_id, p.group_id, p.run_id) for p in state.plan] ==
            [(p.sample_id, p.group_id, p.run_id) for p in expected])
    if not plan_matches or len({p.run_id for p in state.plan}) != len(state.plan):
        raise ValueError("workspace_plan_mismatch")
    return state


def _groups_for_conditions(conditions: Mapping[str, str]) -> tuple[str, ...]:
    """仅在带专用条件标记时接受三组验证布局。"""
    layout = conditions.get("evaluation:group_layout", "pair-v1")
    if layout == "pair-v1":
        return PAIR_GROUPS
    if layout == "fact-v2-three-arm-v1":
        return THREE_ARM_GROUPS
    raise ValueError("unsupported_e2_group_layout")


def _validate_custom_plan(conditions: Mapping[str, str], sample_ids: list[str],
                          planned_runs: Sequence[PlannedRun]) -> tuple[PlannedRun, ...]:
    """自定义槽位只允许本次冻结来源绑定的精确收尾计划。"""
    plan = tuple(PlannedRun.model_validate(slot) for slot in planned_runs)
    encoded = json.dumps([slot.model_dump(mode="json") for slot in plan], sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = sha256(encoded).hexdigest()
    authorized = (conditions.get("recovery:final_segment") == "true" and
        conditions.get("recovery:source_sha256") == FINAL_RECOVERY_FREEZE_SHA256 and
        conditions.get("recovery:plan_sha256") == FINAL_RECOVERY_PLAN_SHA256 == digest)
    if not authorized or _final_recovery_layout_issues(plan, (),
            source_sha256=FINAL_RECOVERY_FREEZE_SHA256, plan_sha256=digest):
        raise ValueError("custom_workspace_plan_not_authorized")
    if list(dict.fromkeys(slot.sample_id for slot in plan)) != sample_ids:
        raise ValueError("custom_workspace_plan_samples_mismatch")
    return plan


def open_e2_workspace(path: Path, *, mode: Literal["new", "resume"], manifest: Mapping[str, object],
                       conditions: Mapping[str, str], sample_ids: Sequence[str],
                       confirmed_owner_sha256: str | None = None,
                       planned_runs: Sequence[PlannedRun] | None = None) -> E2Workspace:
    """取得实验目录排他后新建或恢复固定状态，失败释放锁且不覆盖原数据。"""
    if mode not in {"new", "resume"}:
        raise ValueError("invalid_workspace_mode")
    if confirmed_owner_sha256 is not None:
        TypeAdapter(Fingerprint).validate_python(confirmed_owner_sha256, strict=True)
        if mode != "resume":
            raise ValueError("owner_confirmation_requires_resume")
    clean, ids = _manifest_data(manifest), _samples(sample_ids)
    copied = dict(conditions)
    if not copied or any(type(k) is not str or not k.strip() or type(v) is not str for k, v in copied.items()):
        raise ValueError("invalid_workspace_conditions")
    if planned_runs is not None:
        planned_runs = _validate_custom_plan(copied, ids, planned_runs)
    frozen = json.dumps(copied, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if sha256(frozen.encode("utf-8")).hexdigest() != clean["conditions_fingerprint"]:
        raise ValueError("workspace_manifest_conditions_mismatch")
    project = PROJECT_ROOT.resolve()
    root, folder = (project / "traces").resolve(), Path(path).resolve()
    if not root.is_relative_to(project) or folder == root or not folder.is_relative_to(root):
        raise ValueError("workspace_path_escape")
    if mode == "new":
        folder.mkdir(parents=True, exist_ok=True)
    for name in ("workspace.lock", "state.json", "state.partial", "owner.json", "owner.partial"):
        if (folder / name).resolve().parent != folder:
            raise ValueError("workspace_file_escape")
    handle = (folder / "workspace.lock").open("a+b")
    try:
        _acquire(handle)
        state = _state(folder, mode=mode, manifest=clean, conditions=copied, sample_ids=ids,
            planned_runs=planned_runs)
        if mode == "resume":
            _check_previous_owner(folder, confirmed_owner_sha256)
        opened = E2Workspace(folder, handle, state)
        opened._confirmed_owner_sha256 = confirmed_owner_sha256
        opened.heartbeat()
        return opened
    except BaseException:
        handle.close()
        raise
