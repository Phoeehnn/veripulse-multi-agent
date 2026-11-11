"""正式检索与评测共用的候选检索、融合和重排入口。"""

from __future__ import annotations

from pathlib import Path

from src.yuqing.retrieval.bm25 import BM25Index, bm25_search
from src.yuqing.retrieval.fusion import rrf_fuse, select_diverse_evidence
from src.yuqing.retrieval.rerank import rerank
from src.yuqing.retrieval.vector_store import vector_search
from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_retrieval_config


def retrieve_candidates(
    query: str, index: BM25Index, path: Path | None = None
) -> dict[str, list[EvidenceChunk]]:
    """各执行一次向量与关键词检索，保留两路和完整融合排名。"""

    config = get_retrieval_config()
    vector = vector_search(query, k=config.vector_top_k, path=path)
    keyword = bm25_search(query, index, k=config.bm25_top_k)
    return {"vector": vector, "bm25": keyword, "hybrid": rrf_fuse(vector, keyword)}


def rank_candidates(
    query: str, candidates: list[EvidenceChunk], k: int, use_rerank: bool
) -> list[EvidenceChunk]:
    """共用候选筛选和最终排序规则，关闭重排时直接选择融合结果。"""

    if k <= 0:
        raise ValueError("检索 k 必须大于零")
    config = get_retrieval_config()
    candidate_k = max(k, config.rerank_candidate_k) if use_rerank else k
    selected = (select_diverse_evidence(candidates, candidate_k) if config.diversify_results
                else candidates[:candidate_k])
    if not use_rerank:
        return selected
    ranked = rerank(query, selected, top_k=len(selected)) if selected else []
    return select_diverse_evidence(ranked, k) if config.diversify_results else ranked[:k]


def hybrid_search(
    query: str, index: BM25Index, k: int | None = None, path: Path | None = None
) -> list[EvidenceChunk]:
    """分别检索向量和关键词候选，再用共用规则返回前 k 条。"""

    limit = get_retrieval_config().final_top_k if k is None else k
    if limit <= 0:
        raise ValueError("混合检索 k 必须大于零")
    candidates = retrieve_candidates(query, index, path)
    return rank_candidates(query, candidates["hybrid"], limit, use_rerank=False)


def search(
    query: str,
    index: BM25Index,
    k: int | None = None,
    use_rerank: bool = True,
    path: Path | None = None,
) -> list[EvidenceChunk]:
    """按开关选择共用的混合检索或混合检索后重排。"""

    limit = get_retrieval_config().final_top_k if k is None else k
    if limit <= 0:
        raise ValueError("检索 k 必须大于零")
    candidates = retrieve_candidates(query, index, path)
    return rank_candidates(query, candidates["hybrid"], limit, use_rerank)
