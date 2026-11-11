"""jieba 分词后的 BM25 关键词检索。"""

from __future__ import annotations

from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_retrieval_config, get_settings


@dataclass
class BM25Index:
    """保持原始证据顺序及其 BM25 索引。"""

    chunks: list[EvidenceChunk]
    engine: BM25Okapi | None


def build_bm25_index(chunks: list[EvidenceChunk]) -> BM25Index:
    """对证据片段执行 jieba 分词并构建 BM25 索引。"""

    if not chunks:
        return BM25Index(chunks=[], engine=None)
    cache_dir = get_settings().model_cache_dir / "jieba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(cache_dir)
    tokens = [[word for word in jieba.lcut(item.text) if word.strip()] for item in chunks]
    return BM25Index(chunks=chunks, engine=BM25Okapi(tokens))


def bm25_search(query: str, index: BM25Index, k: int | None = None) -> list[EvidenceChunk]:
    """按 BM25 得分返回中文关键词相关的证据片段。"""

    limit = get_retrieval_config().bm25_top_k if k is None else k
    if limit <= 0:
        raise ValueError("BM25 检索 k 必须大于零")
    words = [word for word in jieba.lcut(query) if word.strip()]
    if not words or index.engine is None:
        return []
    scores = index.engine.get_scores(words)
    ranked = sorted(
        (i for i, score in enumerate(scores) if score != 0.0),
        key=lambda i: (-scores[i], i),
    )[:limit]
    return [
        index.chunks[i].model_copy(update={"score": float(scores[i])})
        for i in ranked
    ]
