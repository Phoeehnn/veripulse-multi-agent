"""检索评测题的结构和确定性选样。"""

from __future__ import annotations

import json
import re
import csv
import unicodedata
from random import Random
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.yuqing.ingest.chef import load_chef, load_chef_evidence
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.schemas import EvidenceChunk, Sample
from src.yuqing.settings import get_evaluation_config, get_retrieval_config


class RetrievalQuestion(BaseModel):
    """一条带可检索 gold 证据 ID 的 CHEF 检索题。"""

    sample_id: str
    claim: str
    gold_veracity: Literal["真", "假", "无法判定"]
    gold_evidence_ids: list[str] = Field(default_factory=list)
    gold_alignment: Literal["unverified", "matched", "partial", "unmatched"] = "unverified"
    gold_total: int = Field(default=0, ge=0)
    gold_matched: int = Field(default=0, ge=0)
    original_gold_evidence_ids: list[str] = Field(default_factory=list)


def normalize_evidence_text(text: str) -> str:
    """忽略空白和中文引号格式，保留数字、否定词及其他标点。"""

    return re.sub(r"[\s「」『』“”‘’]", "", text)


def match_gold_documents(record: dict, sample_id: str) -> dict[str, list[str]]:
    """以去空白后的原文包含关系定位 gold 来源，禁止按相同编号猜测。"""

    candidates = {
        key: normalize_evidence_text(value["text"])
        for key, value in record["evidence"].items()
    }
    matches: dict[str, list[str]] = {}
    for gold_key, value in record["gold evidence"].items():
        text = normalize_evidence_text(value.get("text", ""))
        if text:
            matches[gold_key] = [
                f"{sample_id}:evidence:{key}"
                for key, candidate in candidates.items() if text in candidate
            ]
    return matches


def load_gold_reviews(path: Path, records: dict[str, dict]) -> dict[tuple[str, str], list[str]]:
    """读取逐段复核映射，拒绝标注或来源原文已变更的记录。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    reviews: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["sample_id"], row["gold_key"])
        if key in reviews:
            raise ValueError(f"复核记录重复：{key}")
        record = records[key[0]]
        gold = record["gold evidence"][key[1]]["text"]
        if sha256(gold.encode()).hexdigest() != row["gold_sha256"]:
            raise ValueError(f"复核 gold 原文已变更：{key}")
        documents = json.loads(row["source_docs"])
        hashes = json.loads(row["source_sha256"])
        if row["decision"] not in {"format_only", "equivalent_passage", "unavailable"}:
            raise ValueError(f"未知复核状态：{key}")
        if row["decision"] == "unavailable" and documents:
            raise ValueError(f"未匹配记录不能指定来源：{key}")
        for document in documents:
            prefix = f"{key[0]}:evidence:"
            if not document.startswith(prefix):
                raise ValueError(f"复核来源属于其他样本：{document}")
            text = record["evidence"][document[len(prefix):]]["text"]
            if sha256(text.encode()).hexdigest() != hashes[document]:
                raise ValueError(f"复核来源原文已变更：{document}")
            if row["decision"] == "format_only" and \
                    normalize_evidence_text(gold) not in normalize_evidence_text(text):
                raise ValueError(f"格式差异记录实际不匹配：{key}")
        reviews[key] = documents
    return reviews


def align_question_gold(
    path: Path, questions: list[RetrievalQuestion], review_path: Path | None = None
) -> list[RetrievalQuestion]:
    """保留冻结题目，在内存中重建来源并显式记录未匹配的标注。"""

    records = json.loads(path.read_text(encoding="utf-8"))
    source = {sample.sample_id: (sample, record)
              for sample, record in zip(load_chef(path), records, strict=True)}
    reviews = load_gold_reviews(review_path, {key: pair[1] for key, pair in source.items()}) \
        if review_path is not None else {}
    aligned: list[RetrievalQuestion] = []
    for question in questions:
        sample, record = source[question.sample_id]
        if question.claim != sample.claim or question.gold_veracity != sample.gold_veracity:
            raise ValueError(f"冻结题目与原始样本不一致：{question.sample_id}")
        matches = match_gold_documents(record, sample.sample_id)
        for gold_key in matches:
            if (sample.sample_id, gold_key) in reviews:
                matches[gold_key] = reviews[(sample.sample_id, gold_key)]
        ids = sorted({doc for found in matches.values() for doc in found})
        matched = sum(bool(found) for found in matches.values())
        status = "matched" if matched == len(matches) else "partial"
        aligned.append(question.model_copy(update={
            "gold_evidence_ids": ids,
            "original_gold_evidence_ids": question.gold_evidence_ids,
            "gold_alignment": status if ids else "unmatched",
            "gold_total": len(matches), "gold_matched": matched,
        }))
    return aligned


def select_chef_questions(path: Path, per_label: int | None = None) -> list[RetrievalQuestion]:
    """按原始顺序从 CHEF 每类选出有可检索 gold 证据的题目。"""

    target = get_retrieval_config().questions_per_label if per_label is None else per_label
    if target <= 0:
        raise ValueError("每类题目数量必须大于零")
    records = json.loads(path.read_text(encoding="utf-8"))
    samples = load_chef(path)
    counts = {label: 0 for label in ("真", "假", "无法判定")}
    questions: list[RetrievalQuestion] = []
    for sample, record in zip(samples, records, strict=True):
        label = sample.gold_veracity
        if counts[label] >= target:
            continue
        matches = match_gold_documents(record, sample.sample_id)
        gold_ids = sorted({doc for found in matches.values() for doc in found})
        if not gold_ids:
            continue
        questions.append(RetrievalQuestion(
            sample_id=sample.sample_id,
            claim=sample.claim,
            gold_veracity=label,
            gold_evidence_ids=gold_ids,
            gold_alignment="matched" if all(matches.values()) else "partial",
            gold_total=len(matches),
            gold_matched=sum(bool(found) for found in matches.values()),
        ))
        counts[label] += 1
        if all(count == target for count in counts.values()):
            break
    if any(count != target for count in counts.values()):
        raise ValueError(f"CHEF 可检索 gold 证据不足：{counts}")
    return questions


def build_label_pool(
    chef_test_path: Path, test_questions_path: Path, ced_root: Path, *, seed: int
) -> list[Sample]:
    """固定抽出不与 test60 重叠的 CHEF 60 条及分层 CED 40 条。"""

    rng = Random(seed)
    test_ids = {question.sample_id for question in load_questions(test_questions_path)}
    chef_candidates = [sample for sample in load_chef(chef_test_path) if sample.sample_id not in test_ids]
    if len(chef_candidates) < 60:
        raise ValueError("排除 test60 后，CHEF test 可选题不足60条")
    selected = rng.sample(chef_candidates, 60)
    ced_picks: list[Sample] = []
    for rumor, folder in ((True, "rumor-repost"), (False, "non-rumor-repost")):
        filenames = sorted(path.name for path in (ced_root / folder).glob("*.json"))
        if len(filenames) < 20:
            raise ValueError(f"CED {folder} 可选原帖不足20条")
        ced_picks.extend(load_ced_sample(ced_root, name, rumor=rumor)
                         for name in rng.sample(filenames, 20))
    pool = selected + ced_picks
    rng.shuffle(pool)
    if len({sample.sample_id for sample in pool}) != 100:
        raise ValueError("标注池包含重复样本")
    return pool


def load_questions(path: Path, *, content: bytes | None = None) -> list[RetrievalQuestion]:
    """从文件或已捕获字节逐行校验检索评测题并拒绝重复样本ID。"""

    source = path.read_bytes() if content is None else content
    questions = [RetrievalQuestion.model_validate_json(line)
                 for line in source.decode("utf-8").splitlines()]
    if not questions:
        raise ValueError(f"检索评测题文件为空：{path}")
    ids = [question.sample_id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"检索评测题存在重复 sample_id：{path}")
    return questions


def _question_pools(
    path: Path, excluded_ids: set[str], per_label: int,
    samples: list[Sample] | None = None,
) -> dict[str, list[RetrievalQuestion]]:
    """只按类别和排除名单建立题池，不按证据覆盖或预测表现筛题。"""

    if type(per_label) is not int or per_label <= 0:
        raise ValueError("每类题目数量必须是正整数")
    pools: dict[str, list[RetrievalQuestion]] = {label: [] for label in ("真", "假", "无法判定")}
    samples = samples if samples is not None else load_chef(path)
    if excluded_ids - {sample.sample_id for sample in samples}:
        raise ValueError("排除名单含不属于当前分片的样本")
    for sample in samples:
        if sample.sample_id not in excluded_ids:
            pools[sample.gold_veracity].append(RetrievalQuestion(
                sample_id=sample.sample_id, claim=sample.claim, gold_veracity=sample.gold_veracity))
    if any(len(pool) < per_label for pool in pools.values()):
        raise ValueError(f"各类剩余题目不足：{ {label: len(pool) for label, pool in pools.items()} }")
    return pools


def select_test_questions(path: Path, per_label: int) -> list[RetrievalQuestion]:
    """按分片原顺序每类选题后再对齐证据，保留无法对应证据的题。"""

    pools = _question_pools(path, set(), per_label)
    selected = [question for pool in pools.values() for question in pool[:per_label]]
    selected.sort(key=lambda question: int(question.sample_id.rsplit(":", 1)[1]))
    return align_question_gold(path, selected)


def sample_extra_dev_questions(
    path: Path, excluded_ids: set[str], per_label: int, seed: int
) -> list[RetrievalQuestion]:
    """使用局部随机数生成器抽取均衡的新开发题，并按原始位置固定运行顺序。"""

    if type(seed) is not int:
        raise ValueError("随机种子必须是整数")
    pools = _question_pools(path, excluded_ids, per_label)
    generator = Random(seed)
    selected = [question for pool in pools.values() for question in generator.sample(pool, per_label)]
    selected.sort(key=lambda question: int(question.sample_id.rsplit(":", 1)[1]))
    return align_question_gold(path, selected)


def normalized_claim_key(claim: str) -> str:
    """生成精确文本去重键，只统一 Unicode 兼容形式和连续空白。"""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", claim)).strip().casefold()


def sample_fact_v2_validation_questions(
    path: Path, excluded_ids: set[str], per_label: int, seed: int,
    source_samples: list[Sample] | None = None,
) -> tuple[list[RetrievalQuestion], dict[str, str]]:
    """排除已暴露题与完全重复声明后按标签定种子抽样，并返回重复排除原因。"""

    if type(seed) is not int:
        raise ValueError("随机种子必须是整数")
    samples = source_samples if source_samples is not None else load_chef(path)
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        grouped.setdefault(normalized_claim_key(sample.claim), []).append(sample.sample_id)
    duplicate_ids = {
        sample_id for group in grouped.values() if len(group) > 1 for sample_id in group
    }
    all_excluded = excluded_ids | duplicate_ids
    pools = _question_pools(path, all_excluded, per_label, samples=samples)
    generator = Random(seed)
    selected = [question for pool in pools.values() for question in generator.sample(pool, per_label)]
    selected.sort(key=lambda question: int(question.sample_id.rsplit(":", 1)[1]))
    questions = align_question_gold(path, selected)
    duplicate_reasons = {sample_id: "exact_normalized_claim_duplicate" for sample_id in duplicate_ids}
    return questions, duplicate_reasons


def freeze_questions(path: Path, questions: list[RetrievalQuestion]) -> None:
    """验证题目后以独占创建方式写入 UTF-8/LF 文件，拒绝覆盖任何已有文件。"""

    if not questions or len({question.sample_id for question in questions}) != len(questions):
        raise ValueError("冻结题目不能为空或含重复样本")
    lines = [RetrievalQuestion.model_validate(question).model_dump_json() for question in questions]
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def load_question_corpus(
    chef_path: Path, questions: list[RetrievalQuestion]
) -> list[EvidenceChunk]:
    """只取冻结评测题所属样本的候选证据片段作为共同语料。"""

    sample_ids = {question.sample_id for question in questions}
    return [
        piece for piece in load_chef_evidence(chef_path)
        if piece.source_doc.partition(":evidence:")[0] in sample_ids
    ]


def select_acceptance_questions(questions: list[RetrievalQuestion]) -> list[RetrievalQuestion]:
    """按固定顺序和配置中的各类配额选题，禁止依据预测结果筛选。"""

    quotas = get_evaluation_config().fact_check_counts
    counts = {label: 0 for label in quotas}
    selected: list[RetrievalQuestion] = []
    for question in questions:
        label = question.gold_veracity
        if label in quotas and counts[label] < quotas[label]:
            selected.append(question)
            counts[label] += 1
    if counts != quotas or not selected:
        raise ValueError(f"验收样本不能满足固定配额：{counts}，要求 {quotas}")
    return selected
