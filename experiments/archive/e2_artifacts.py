"""纯检查答卷内容指纹和实验归属，不读写文件或许可付费。"""

from hashlib import sha256
import json
import math
from typing import Annotated, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from experiments.archive.e2_checkpoint import CheckpointRun, Identifier

Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class E2CheckpointRow(BaseModel):
    """新答卷封装；payload仅保存数据，不证明模型判断或Trace可信。"""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, revalidate_instances="always")
    experiment_id: Identifier
    condition_sha256: Fingerprint
    run: CheckpointRun
    payload: dict[str, JsonValue]
    row_sha256: Fingerprint


def _check_json(value: object) -> None:
    """只接受原生有限JSON数据，拒绝隐式改键名和非标准数值。"""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json(item)
        return
    raise ValueError("invalid_checkpoint_json")


def _canonical(data: Mapping[str, object]) -> str:
    """固定编码和字段排序，映射键顺序不影响指纹，列表顺序仍影响。"""
    copied = dict(data)
    _check_json(copied)
    return json.dumps(copied, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def checkpoint_row_sha256(row: Mapping[str, object]) -> str:
    """计算全部行内容的指纹，仅排除顶层row_sha256自身。"""
    content = {key: value for key, value in row.items() if key != "row_sha256"}
    return sha256(_canonical(content).encode("utf-8")).hexdigest()


def build_checkpoint_row(run: CheckpointRun, payload: Mapping[str, object], *,
                         experiment_id: str, condition_sha256: str) -> E2CheckpointRow:
    """构造新记录并复制嵌套数据，不为历史答卷补造指纹。"""
    checked = CheckpointRun.model_validate(run)
    data = dict(experiment_id=experiment_id, condition_sha256=condition_sha256,
        run=checked.model_dump(mode="json"), payload=dict(payload))
    data["row_sha256"] = checkpoint_row_sha256(data)
    return E2CheckpointRow.model_validate_json(_canonical(data))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝原始JSON任意层级的重复键，而非保留最后一个值。"""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_checkpoint_key")
        result[key] = value
    return result


def _decode_row(row: Mapping[str, object] | E2CheckpointRow | str) -> E2CheckpointRow:
    """先核查原始JSON类型，再严格验证封装，已建模型也重新检查。"""
    if isinstance(row, str):
        raw = json.loads(row, object_pairs_hook=_unique_object)
    elif isinstance(row, E2CheckpointRow):
        raw = row.model_dump(mode="json")
    elif isinstance(row, Mapping):
        raw = dict(row)
    else:
        raise ValueError("invalid_checkpoint_row")
    _check_json(raw)
    if not isinstance(raw, dict):
        raise ValueError("invalid_checkpoint_row")
    return E2CheckpointRow.model_validate_json(_canonical(raw))


def validate_checkpoint_artifacts(rows: Sequence[Mapping[str, object] | E2CheckpointRow | str], *,
                                  experiment_id: str, condition_sha256: str) -> tuple[E2CheckpointRow, ...]:
    """全批验证后返回独立完整封装，供恢复检查读取受指纹保护的Trace引用。"""
    if isinstance(rows, (str, bytes)):
        raise ValueError("invalid_checkpoint_rows")
    # 空批次也必须检查预期实验身份，不能绕过启动条件校验。
    expected_id = TypeAdapter(Identifier).validate_python(experiment_id, strict=True)
    expected_hash = TypeAdapter(Fingerprint).validate_python(condition_sha256, strict=True)
    checked: list[E2CheckpointRow] = []
    for row in rows:
        item = _decode_row(row)
        if item.experiment_id != expected_id or item.condition_sha256 != expected_hash:
            raise ValueError("checkpoint_experiment_mismatch")
        if item.row_sha256 != checkpoint_row_sha256(item.model_dump(mode="json")):
            raise ValueError("checkpoint_row_digest_mismatch")
        checked.append(item)
    return tuple(checked)


def validate_checkpoint_rows(rows: Sequence[Mapping[str, object] | E2CheckpointRow | str], *,
                             experiment_id: str, condition_sha256: str) -> tuple[CheckpointRun, ...]:
    """兼容原元信息接口，复用完整封装校验，不重复实现指纹规则。"""
    return tuple(item.run for item in validate_checkpoint_artifacts(rows,
        experiment_id=experiment_id, condition_sha256=condition_sha256))
