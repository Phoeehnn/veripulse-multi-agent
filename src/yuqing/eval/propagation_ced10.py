"""固定CED-10传播验证样本及其来源审计。"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.schemas import Sample

SELECTION_PATH = Path("docs/cards/F1-CED10-selection-v1.json")
CED10_CODE_PATHS = (
    "src/yuqing/eval/propagation_ced10.py",
    "src/yuqing/ingest/ced.py",
)
REGRESSION_CSVS = (
    "results/2026-10-16_propagation_8073ac09f2f648c09d673f1b01d351e8.csv",
    "results/2026-10-17_propagation_93874a16414c4522b5fef4c6cf7a5070.csv",
)
REGRESSION_SHA256 = (
    "7f5bf5773e0166595dc813b156a57653cba7fc895a650017ec2bb2d00d138ab2",
    "c857d1e583e681934afffdae5a911d86c58d8367958f254e885f1d021871af2e",
)
ANNOTATION_PATH = "docs/review_history/风险补充50条_清单_2026-10-18.jsonl"
LABEL_POOL_PATH = "eval/label_pool_100.jsonl"
TEST_PATH = "eval/test_60.jsonl"


class CED10SelectionItem(BaseModel):
    """一条固定样本的分组、身份和原始输入指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    category: Literal["rumor", "non_rumor"]
    group: Literal["regression", "new_validation"]
    rank_sha256: str
    input_sha256: str
    original_sha256: str
    repost_sha256: str


class CED10CandidateAudit(BaseModel):
    """记录一个原始候选为何入选或被排除。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    category: Literal["rumor", "non_rumor"]
    rank_sha256: str
    disposition: Literal["selected_new_validation", "not_selected_by_rank", "excluded"]
    exclusion_reasons: list[Literal[
        "regression_and_debugged", "label_pool", "supplemental_annotation", "test_overlap",
        "ced_parser_rejected",
    ]]


class CED10Selection(BaseModel):
    """可复核的CED-10固定清单及候选排除审计。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["f1-ced10-selection-v1"]
    selection_rule: str
    dataset_catalog_sha256: str
    source_artifact_sha256: dict[str, str]
    debug_scope_note: str
    debugged_sample_ids: list[str]
    regression_sample_ids: list[str]
    new_validation_sample_ids: list[str]
    selected: list[CED10SelectionItem]
    candidate_audit: list[CED10CandidateAudit]


def _file_sha256(path: Path) -> str:
    """计算文件原始字节的SHA-256。"""

    return sha256(path.read_bytes()).hexdigest()


def _jsonl_ids(path: Path) -> set[str]:
    """读取JSONL中的样本身份，拒绝损坏或重复记录。"""

    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("ced10_exclusion_file_blank_line")
        value = json.loads(line)
        sample_id = value.get("sample_id") if isinstance(value, dict) else None
        if not isinstance(sample_id, str) or sample_id in ids:
            raise ValueError("ced10_exclusion_file_identity_invalid")
        ids.add(sample_id)
    return ids


def _regression_ids(root: Path) -> tuple[set[str], dict[str, str]]:
    """从两份历史M3答卷读取并核对原CED回归组。"""

    snapshots: dict[str, str] = {}
    id_sets: list[set[str]] = []
    for relative, expected_sha in zip(REGRESSION_CSVS, REGRESSION_SHA256):
        path = root / relative
        if _file_sha256(path) != expected_sha:
            raise ValueError("ced10_regression_result_fingerprint_changed")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            ids = {row["sample_id"] for row in csv.DictReader(stream) if row.get("source") == "ced"}
        if len(ids) != 5 or any(not value.startswith("ced:") for value in ids):
            raise ValueError("ced10_regression_result_invalid")
        id_sets.append(ids)
        snapshots[relative] = expected_sha
    if id_sets[0] != id_sets[1]:
        raise ValueError("ced10_regression_sources_disagree")
    return id_sets[0], snapshots


def _candidate_catalog(dataset_root: Path) -> list[tuple[str, str, str]]:
    """枚举CED原帖与对应传播文件，发现任何缺失配对就停止。"""

    originals = {item.stem for item in (dataset_root / "original-microblog").glob("*.json")}
    rows: list[tuple[str, str, str]] = []
    repost_catalog: set[str] = set()
    for category, folder in (("non_rumor", "non-rumor-repost"), ("rumor", "rumor-repost")):
        reposts = {item.stem for item in (dataset_root / folder).glob("*.json")}
        if reposts & repost_catalog or not reposts <= originals:
            raise ValueError("ced10_original_repost_pair_missing")
        repost_catalog.update(reposts)
        rows.extend((f"ced:{category}:{stem}", category, stem) for stem in reposts)
    if repost_catalog != originals:
        raise ValueError("ced10_original_repost_pair_missing")
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ValueError("ced10_candidate_catalog_invalid")
    return sorted(rows)


def _catalog_fingerprint(rows: list[tuple[str, str, str]]) -> str:
    """指纹固定候选样本身份与所属类别。"""

    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _rank(sample_id: str) -> str:
    """按固定盐值生成可复现的候选排序。"""

    return sha256(f"F1-CED10-new-v1|{sample_id}".encode("utf-8")).hexdigest()


def _source_paths(dataset_root: Path, category: str, stem: str) -> tuple[Path, Path]:
    """返回一条CED样本对应的原帖文件与传播文件。"""

    folder = "rumor-repost" if category == "rumor" else "non-rumor-repost"
    return dataset_root / "original-microblog" / f"{stem}.json", dataset_root / folder / f"{stem}.json"


def _selected_item(root: Path, sample: Sample, category: str, stem: str,
                   group: str) -> CED10SelectionItem:
    """读取固定样本并计算原帖、传播树和专家输入指纹。"""

    original, repost = _source_paths(root / "data/raw/CED_Dataset", category, stem)
    from src.yuqing.eval.propagation_eval import _input_fingerprint

    return CED10SelectionItem(sample_id=sample.sample_id, category=category, group=group,
        rank_sha256=_rank(sample.sample_id), input_sha256=_input_fingerprint(sample),
        original_sha256=_file_sha256(original), repost_sha256=_file_sha256(repost))


def _select_new_validation(catalog: list[tuple[str, str, str]], blocked: set[str],
                           dataset_root: Path) -> tuple[list[str], dict[str, Sample], set[str]]:
    """按类别额度和固定哈希顺序选题，记录无法解析的候选。"""

    selected: list[str] = []
    parsed: dict[str, Sample] = {}
    parse_failed: set[str] = set()
    ranks = {sample_id: _rank(sample_id) for sample_id, _, _ in catalog}
    for category, quota in (("rumor", 2), ("non_rumor", 3)):
        candidates = sorted((row for row in catalog if row[1] == category and row[0] not in blocked),
                            key=lambda row: ranks[row[0]])
        count = 0
        for sample_id, _, stem in candidates:
            try:
                parsed[sample_id] = load_ced_sample(dataset_root, f"{stem}.json",
                                                    rumor=category == "rumor")
            except (OSError, ValueError, KeyError, TypeError, ValidationError):
                parse_failed.add(sample_id)
                continue
            selected.append(sample_id)
            count += 1
            if count == quota:
                break
        if count != quota:
            raise ValueError("ced10_new_validation_candidates_insufficient")
    return selected, parsed, parse_failed


def build_ced10_selection(root: Path) -> CED10Selection:
    """按预登记分层哈希规则生成CED回归5条加新增验证5条清单。"""

    dataset_root = root / "data/raw/CED_Dataset"
    catalog = _candidate_catalog(dataset_root)
    regression_ids, regression_sources = _regression_ids(root)
    pool = _jsonl_ids(root / LABEL_POOL_PATH)
    test = _jsonl_ids(root / TEST_PATH)
    annotations = _jsonl_ids(root / ANNOTATION_PATH)
    blocked = regression_ids | pool | test | annotations
    ranks = {sample_id: _rank(sample_id) for sample_id, _, _ in catalog}
    new_ids, parsed_new, parse_failed = _select_new_validation(catalog, blocked, dataset_root)
    if len(new_ids) != 5 or set(new_ids) & blocked:
        raise ValueError("ced10_new_validation_candidates_insufficient")
    selected = []
    by_id = {sample_id: (category, stem) for sample_id, category, stem in catalog}
    for sample_id in sorted(regression_ids):
        category, stem = by_id[sample_id]
        sample = load_ced_sample(dataset_root, f"{stem}.json", rumor=category == "rumor")
        selected.append(_selected_item(root, sample, category, stem, "regression"))
    for sample_id in new_ids:
        category, stem = by_id[sample_id]
        selected.append(_selected_item(root, parsed_new[sample_id], category, stem, "new_validation"))
    audit = [_audit_row(sample_id, category, ranks[sample_id], regression_ids,
        pool, annotations, test, set(new_ids), parse_failed) for sample_id, category, _ in catalog]
    artifacts = {relative: _file_sha256(root / relative) for relative in
        (LABEL_POOL_PATH, TEST_PATH, ANNOTATION_PATH)}
    artifacts.update(regression_sources)
    rule = "按类别分别取SHA256(F1-CED10-new-v1|sample_id)最小者：rumor 2条、non_rumor 3条；排除原5条、两份历史M3答卷中的CED题、冻结标注池、test及补充人工标注样本。"
    selection = CED10Selection(version="f1-ced10-selection-v1", selection_rule=rule,
        dataset_catalog_sha256=_catalog_fingerprint(catalog), source_artifact_sha256=artifacts,
        debug_scope_note="按作者2026-10-20口径，真实M3答卷中的CED题作为已调试样本排除。归档M3提示词检查记录SHA d81b0dc9dae17b6946bd4eb44d15aab0934dc66d53a5dc273c3e5d076b8e2ae1显示model_calls=0且仅解析2条样本；未记录样本ID，纯读取解析不算已调试，因此不扩大排除范围。",
        debugged_sample_ids=sorted(regression_ids),
        regression_sample_ids=sorted(regression_ids), new_validation_sample_ids=new_ids,
        selected=selected, candidate_audit=audit)
    validate_exclusion_boundary(selection)
    return selection


def _audit_row(sample_id: str, category: str, rank: str, regression: set[str],
        pool: set[str], annotations: set[str], test: set[str], new_ids: set[str],
        parse_failed: set[str]) -> CED10CandidateAudit:
    """给每个CED候选记录入选或排除原因。"""

    reasons = []
    for blocked, reason in ((regression, "regression_and_debugged"), (pool, "label_pool"),
            (annotations, "supplemental_annotation"), (test, "test_overlap"),
            (parse_failed, "ced_parser_rejected")):
        if sample_id in blocked:
            reasons.append(reason)
    disposition = "excluded" if reasons else (
        "selected_new_validation" if sample_id in new_ids else "not_selected_by_rank")
    return CED10CandidateAudit(sample_id=sample_id, category=category, rank_sha256=rank,
                               disposition=disposition, exclusion_reasons=reasons)


def load_ced10_selection(root: Path) -> tuple[CED10Selection, list[Sample]]:
    """重算来源、排除和指纹，只有与固定清单完全一致才返回样本。"""

    path = root / SELECTION_PATH
    selection = CED10Selection.model_validate_json(path.read_text(encoding="utf-8"))
    validate_exclusion_boundary(selection)
    expected = build_ced10_selection(root)
    if selection != expected:
        raise ValueError("ced10_selection_manifest_drift")
    dataset_root = root / "data/raw/CED_Dataset"
    samples = []
    for item in selection.selected:
        category = item.category
        stem = item.sample_id.rsplit(":", 1)[1]
        sample = load_ced_sample(dataset_root, f"{stem}.json", rumor=category == "rumor")
        samples.append(sample)
    validate_selected_groups(selection, samples)
    return selection, samples


def validate_selected_groups(selection: CED10Selection, samples: list[Sample]) -> None:
    """检查固定清单严格由五条原回归和五条新增CED组成。"""

    groups = [item.group for item in selection.selected]
    if len(samples) != 10 or any(item.source != "ced" for item in samples) or \
            len({item.sample_id for item in samples}) != 10:
        raise ValueError("ced10_selection_requires_ten_unique_ced_samples")
    if groups.count("regression") != 5 or groups.count("new_validation") != 5:
        raise ValueError("ced10_selection_requires_five_plus_five_groups")
    if [item.sample_id for item in selection.selected] != [item.sample_id for item in samples]:
        raise ValueError("ced10_selection_sample_order_mismatch")
    if set(selection.regression_sample_ids) != {item.sample_id for item in selection.selected
                                               if item.group == "regression"} or \
            set(selection.new_validation_sample_ids) != {item.sample_id for item in selection.selected
                                                         if item.group == "new_validation"}:
        raise ValueError("ced10_selection_group_identity_mismatch")
    new_categories = [item.category for item in selection.selected if item.group == "new_validation"]
    if new_categories.count("rumor") != 2 or new_categories.count("non_rumor") != 3:
        raise ValueError("ced10_validation_strata_mismatch")
    regression_categories = [item.category for item in selection.selected if item.group == "regression"]
    if regression_categories != ["non_rumor"] * 5:
        raise ValueError("ced10_regression_strata_mismatch")


def validate_exclusion_boundary(selection: CED10Selection) -> None:
    """确保新验证题未进入任何排除集合，原回归题确属已调试集合。"""

    audit = {item.sample_id: item for item in selection.candidate_audit}
    if len(audit) != len(selection.candidate_audit):
        raise ValueError("ced10_candidate_audit_duplicate_id")
    if set(selection.regression_sample_ids) != set(selection.debugged_sample_ids):
        raise ValueError("ced10_regression_debug_provenance_mismatch")
    for sample_id in selection.new_validation_sample_ids:
        row = audit.get(sample_id)
        if row is None or row.disposition != "selected_new_validation" or row.exclusion_reasons:
            raise ValueError("ced10_new_validation_exclusion_overlap")
    for sample_id in selection.regression_sample_ids:
        row = audit.get(sample_id)
        if row is None or "regression_and_debugged" not in row.exclusion_reasons:
            raise ValueError("ced10_regression_provenance_missing")


def save_ced10_selection(root: Path) -> Path:
    """首次生成版本化固定清单，禁止覆盖已有清单。"""

    selection = build_ced10_selection(root)
    target = root / SELECTION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(selection.model_dump_json(indent=2) + "\n")
    return target


def validate_run_rows(rows: list[dict[str, str]], expected: list[dict[str, str]]) -> None:
    """对比运行清单与当前提示词、预算、样本及代码配置指纹。"""

    if rows != expected or len(rows) != 10:
        raise ValueError("ced10_run_manifest_drift")


def valid_sha256(value: str) -> bool:
    """验证SHA-256十六进制字段格式。"""

    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def ced10_code_fingerprint(root: Path) -> str:
    """单独指纹化CED-10选样校验与CED原始数据解析实现。"""

    files = {name: _file_sha256(root / name) for name in CED10_CODE_PATHS}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def ced10_run_rows(selection: CED10Selection, prompt_sha256: str, max_cost_cny: float,
                   fingerprints: dict[str, str], selection_sha256: str) -> list[dict[str, str]]:
    """生成绑定十条样本、分组、许可目标和完整运行条件的清单。"""

    required = {"tools_sha256", "schema_sha256", "config_sha256", "code_sha256",
                "ced10_code_sha256"}
    if set(fingerprints) != required or any(not valid_sha256(value) for value in fingerprints.values()):
        raise ValueError("ced10_run_fingerprints_invalid")
    if not valid_sha256(prompt_sha256) or not valid_sha256(selection_sha256) or max_cost_cny <= 0:
        raise ValueError("ced10_run_manifest_inputs_invalid")
    return [{"sample_id": item.sample_id, "source": "ced", "category": item.category,
        "evaluation_group": item.group, "variant": "propagation_v1", "split": "dev",
        "as_of": "None", "input_sha256": item.input_sha256,
        "original_sha256": item.original_sha256, "repost_sha256": item.repost_sha256,
        "selection_sha256": selection_sha256, "prompt_sha256": prompt_sha256,
        "max_cost_cny": str(max_cost_cny), **fingerprints} for item in selection.selected]


def read_run_manifest(path: Path) -> list[dict[str, str]]:
    """读取十行运行清单并拒绝重复JSON键或非字符串字段。"""

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            raise ValueError("ced10_run_manifest_blank_line")
        row = json.loads(line, object_pairs_hook=_unique_pairs)
        if not isinstance(row, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in row.items()):
            raise ValueError("ced10_run_manifest_invalid_row")
        rows.append(row)
    if len(rows) != 10 or len({row.get("sample_id") for row in rows}) != 10:
        raise ValueError("ced10_run_manifest_requires_ten_unique_rows")
    return rows


def save_run_manifest(rows: list[dict[str, str]], path: Path) -> Path:
    """独占保存运行条件清单，禁止覆盖一次性实验凭据。"""

    if len(rows) != 10 or len({row.get("sample_id") for row in rows}) != 10:
        raise ValueError("ced10_run_manifest_requires_ten_unique_rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """拒绝JSON清单中的重复键。"""

    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError("ced10_run_manifest_duplicate_key")
        row[key] = value
    return row
