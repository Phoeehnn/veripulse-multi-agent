"""完整候选语料的选择、建库指纹和恢复检查。"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Literal

from src.yuqing.eval.dataset import RetrievalQuestion, load_question_corpus
from src.yuqing.ingest.chef import load_chef_evidence
from src.yuqing.retrieval.vector_store import index_chunks, validate_vector_corpus
from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_retrieval_config, get_settings

logger = logging.getLogger(__name__)
CorpusName = Literal["question_only", "chef_dev_full", "chef_test_full"]


def load_eval_corpus(
    chef_path: Path, questions: list[RetrievalQuestion], *, corpus_name: CorpusName | None = None
) -> list[EvidenceChunk]:
    """按明确语料范围读取候选正文，不把标签或参考答案放入索引。"""

    selected = corpus_name or get_retrieval_config().corpus
    if selected == "question_only":
        return load_question_corpus(chef_path, questions)
    split = {"chef_dev_full": "dev", "chef_test_full": "test"}.get(selected)
    if split is None or chef_path.stem != split:
        raise ValueError(f"{selected} 只能读取对应的 CHEF {split or 'dev/test'} 分片；test 需显式选择 chef_test_full")
    return load_chef_evidence(chef_path)


def _corpus_digest(chunks: list[EvidenceChunk]) -> str:
    """按顺序给片段身份、正文和位置计算指纹，排除查询分数。"""

    data = [(item.chunk_id, item.text, item.source_doc, item.position) for item in chunks]
    return sha256(json.dumps(data, ensure_ascii=False).encode("utf-8")).hexdigest()


def _build_signature(
    chef_path: Path, chunks: list[EvidenceChunk], *, corpus_name: CorpusName | None = None
) -> dict[str, object]:
    """绑定原始分片、片段正文和影响编码的参数，拒绝混用建库条件。"""

    if hasattr(get_retrieval_config, "cache_clear"):
        get_retrieval_config.cache_clear()
    config = get_retrieval_config()
    selected = corpus_name or config.corpus
    model_root = get_settings().model_cache_dir / "huggingface/hub" / ("models--" + config.embedding_model.replace("/", "--"))
    revision = model_root / "refs/main"
    fingerprinted_config = {
        "corpus": selected,
        **{name: getattr(config, name) for name in (
            "chunk_size",
            "chunk_overlap",
            "embedding_model",
            "embedding_use_fp16",
            "embedding_batch_size",
            "collection_name",
        )},
    }
    return {
        "source_path": str(chef_path.resolve()),
        "source_sha256": sha256(chef_path.read_bytes()).hexdigest(),
        "corpus_sha256": _corpus_digest(chunks),
        "chunks": len(chunks),
        "embedding_revision": revision.read_text().strip() if revision.exists() else None,
        "config": fingerprinted_config,
    }


def validate_eval_corpus(
    chunks: list[EvidenceChunk], path: Path, *, corpus_name: CorpusName | None = None
) -> None:
    """完整库必须具备完成标记和匹配的建库指纹，再核对每条持久化片段。"""

    validate_vector_corpus(chunks, path)
    selected = corpus_name or get_retrieval_config().corpus
    if selected in {"chef_dev_full", "chef_test_full"}:
        if not (path / "corpus.json").is_file() or not (path / "READY").is_file():
            raise ValueError("完整索引尚未完成或缺少建库记录")
        manifest = json.loads((path / "corpus.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("source_path"), str):
            raise ValueError("建库记录缺少有效来源路径")
        if manifest != _build_signature(Path(manifest["source_path"]), chunks, corpus_name=selected):
            raise ValueError("完整索引尚未完成或来源/编码配置变化，请检查建库记录")


def build_eval_corpus(
    chef_path: Path, store_path: Path, *, corpus_name: CorpusName | None = None
) -> int:
    """在隔离目录建完整开发或测试库；中断后核对指纹再续建。"""

    settings = get_settings()
    selected = corpus_name or get_retrieval_config().corpus
    target = store_path.resolve()
    protected = [
        settings.project_root / "chroma_db",
        settings.project_root / "data/processed/questions30_chroma",
    ]
    other_store = settings.test_store_path if selected == "chef_dev_full" else settings.dev_store_path
    protected.append(other_store)
    outside_data = not target.is_relative_to((settings.project_root / "data/processed").resolve())
    overlaps_protected = any(
        target == old.resolve() or target.is_relative_to(old.resolve()) for old in protected
    )
    if outside_data or overlaps_protected:
        raise ValueError("新索引必须位于项目 data/processed 下且不能使用旧索引目录")
    if selected not in {"chef_dev_full", "chef_test_full"}:
        raise ValueError("建完整 CHEF 库需要明确选择 dev 或 test 范围")
    chunks = load_eval_corpus(chef_path, [], corpus_name=selected)
    if not chunks:
        raise ValueError("完整候选语料为空")
    signature = _build_signature(chef_path, chunks, corpus_name=selected)
    manifest = target / "corpus.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError("来源或编码配置变化，拒绝续建，请另用新目录")
    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError("目标目录已有未登记材料，拒绝混用索引")
        target.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("完整语料 %s 片段，开始编码入库：%s", len(chunks), target)
    added = index_chunks(chunks, target)
    current_chunks = load_eval_corpus(chef_path, [], corpus_name=selected)
    current_signature = _build_signature(chef_path, current_chunks, corpus_name=selected)
    if signature != current_signature:
        raise ValueError("建库期间源数据或配置变化，拒绝登记完成")
    validate_vector_corpus(chunks, target)
    (target / "READY").touch()
    logger.info("完整索引核对通过：新增 %s，总计 %s 片段", added, len(chunks))
    return len(chunks)
