"""持久化 Chroma 证据库及其检索入口。"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import chromadb

from src.yuqing.retrieval.embed import embed
from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_retrieval_config, get_settings


@cache
def get_vector_collection(path: Path | None = None) -> chromadb.Collection:
    """按路径复用持久化 Chroma 集合。"""

    directory = path or get_settings().project_root / "chroma_db"
    client = chromadb.PersistentClient(path=str(directory))
    return client.get_or_create_collection(name=get_retrieval_config().collection_name)


def validate_vector_corpus(chunks: list[EvidenceChunk], path: Path | None = None) -> None:
    """校验向量库与评测语料的 ID、正文及来源位置完全一致。"""

    expected = {item.chunk_id: item for item in chunks}
    if len(expected) != len(chunks):
        raise ValueError("评测语料存在重复片段 ID")
    stored = get_vector_collection(path).get(include=["documents", "metadatas"])
    indexed = set(stored["ids"])
    if indexed != set(expected):
        raise ValueError(
            f"向量库与 BM25 语料不一致：缺少 {len(set(expected) - indexed)} 条，"
            f"多出 {len(indexed - set(expected))} 条"
        )
    for key, text, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"], strict=True
    ):
        item = expected[key]
        if text != item.text or metadata is None or \
                metadata.get("source_doc") != item.source_doc or metadata.get("position") != item.position:
            raise ValueError(f"向量库与 BM25 语料不一致：片段 {key} 的正文或来源位置变化")


def index_chunks(chunks: list[EvidenceChunk], path: Path | None = None) -> int:
    """只编码新片段，拒绝静默复用同 ID 下已变化的正文或来源。"""

    if not chunks:
        return 0
    collection = get_vector_collection(path)
    batch_size = get_retrieval_config().embedding_batch_size
    added = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        stored = collection.get(ids=[item.chunk_id for item in batch], include=["documents", "metadatas"])
        existing = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"], strict=True), strict=True))
        for item in batch:
            if item.chunk_id in existing:
                text, metadata = existing[item.chunk_id]
                if text != item.text or metadata is None or \
                        metadata.get("source_doc") != item.source_doc or metadata.get("position") != item.position:
                    raise ValueError(f"已入库片段 {item.chunk_id} 的正文或来源变化，请使用新索引目录")
        fresh = [item for item in batch if item.chunk_id not in existing]
        if not fresh:
            continue
        vectors = embed([item.text for item in fresh])
        collection.upsert(
            ids=[item.chunk_id for item in fresh],
            embeddings=vectors.tolist(),
            documents=[item.text for item in fresh],
            metadatas=[
                {"source_doc": item.source_doc, "position": item.position}
                for item in fresh
            ],
        )
        added += len(fresh)
    return added


def vector_search(
    query: str, k: int | None = None, path: Path | None = None
) -> list[EvidenceChunk]:
    """用 BGE-M3 查询持久化证据库并恢复来源元数据。"""

    collection = get_vector_collection(path)
    count = collection.count()
    if not query.strip() or count == 0:
        return []
    limit = get_retrieval_config().vector_top_k if k is None else k
    if limit <= 0:
        raise ValueError("向量检索 k 必须大于零")
    result = collection.query(
        query_embeddings=embed([query]).tolist(),
        n_results=min(limit, count),
        include=["documents", "metadatas", "distances"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        EvidenceChunk(
            chunk_id=chunk_id,
            text=text,
            source_doc=metadata["source_doc"],
            position=metadata["position"],
            score=1 / (1 + float(distance)),
        )
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]
